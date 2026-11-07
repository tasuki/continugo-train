import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import test_loop as helpers
import torch
from test_acceptance import ArenaJobs

from continugo_train import archive, loop, rate_saved, ratings, saved_models


class MilestoneTests(helpers.LoopHarness):
    def test_saves_only_completed_multiples_as_independent_copies(self):
        output = self.root / "campaign"
        state = self.run_loop(output, extra=["--iterations", "5", "--save-model-every", "2"])
        store = Path(state["config"]["model_store"])
        snapshots = sorted(store.glob("campaign-r*"))
        self.assertEqual([p.name for p in snapshots], ["campaign-r0002", "campaign-r0004"])
        for directory in snapshots:
            metadata = saved_models.verify(directory)
            self.assertEqual(metadata["campaign"], "campaign")
            self.assertFalse(metadata["accepted"])
            self.assertEqual(
                {p.name for p in directory.iterdir()}, {*saved_models.FILES, "metadata.json"}
            )
            self.assertTrue(all(p.stat().st_nlink == 1 for p in directory.iterdir()))
            source = output / state["iterations"][metadata["round"] - 1]["checkpoint"]["path"]
            self.assertNotEqual(source.stat().st_ino, (directory / "checkpoint.pt").stat().st_ino)
            self.assertEqual(archive.digest(source), metadata["checkpoint_sha256"])
        before = {p: p.read_bytes() for d in snapshots for p in d.iterdir()}
        calls = len(self.jobs.calls)
        self.run_loop(output, resume=True)
        self.assertEqual(len(self.jobs.calls), calls)
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_historical_v5_backfill_is_idempotent_and_does_not_run_jobs(self):
        output = self.root / "historical"
        state = self.run_loop(output, extra=["--iterations", "2", "--save-model-every", "0"])
        state["version"] = 5
        loop.atomic_checkpoint(output / "loop.pt", state)
        before = (output / "loop.pt").read_bytes()
        calls = len(self.jobs.calls)
        first = saved_models.save_campaign(output, every=2)
        self.assertEqual(saved_models.save_campaign(output, every=2), first)
        self.assertEqual(len(first), 1)
        self.assertEqual(before, (output / "loop.pt").read_bytes())
        self.assertEqual(len(self.jobs.calls), calls)
        graph = first[0] / "encoder.onnx"
        graph.chmod(0o644)
        graph.write_text("changed")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            saved_models.save_campaign(output, every=2)

    def test_corrupt_sources_and_name_conflicts_are_not_overwritten(self):
        output = self.root / "conflict"
        state = self.run_loop(output, extra=["--save-model-every", "0"])
        saved = saved_models.save_campaign(output, iteration=1)[0]
        metadata = saved / "metadata.json"
        original = metadata.read_bytes()
        metadata.chmod(0o644)
        value = json.loads(original)
        value["campaign"] = "different"
        metadata.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "conflicts"):
            saved_models.save_campaign(output, iteration=1)
        metadata.write_bytes(original)
        source = output / state["iterations"][0]["model"]["encoder"]["path"]
        source.write_text("changed")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            saved_models.save_campaign(output, iteration=1)

    def test_milestone_evaluation_is_automatic_even_with_a_different_interval(self):
        self.jobs = ArenaJobs()
        output = self.root / "automatic"
        state = self.run_loop(
            output,
            extra=[
                "--iterations",
                "2",
                "--save-model-every",
                "2",
                "--evaluation-interval",
                "99",
                "--reference-games",
                "2",
            ],
        )
        self.assertIsNone(state["iterations"][0]["reference"])
        self.assertEqual(state["iterations"][1]["reference"]["report"]["games"], 2)
        store = Path(state["config"]["model_store"])
        self.assertTrue((store / "automatic-r0002/metadata.json").is_file())
        self.assertTrue((store / "ratings.png").is_file())

    def test_defaults_and_legacy_protocol_round_trip(self):
        config = loop.LoopConfig()
        self.assertEqual((config.save_model_every, config.evaluation_interval), (10, 10))
        self.assertEqual((config.adaptive_target_rd, config.adaptive_min_pairs), (75, 20))
        settings = ratings.protocol(config, legacy=True)
        self.assertEqual(settings["version"], 1)
        self.assertNotIn("score_weight", settings["search"])
        self.assertEqual(ratings._protocol_from_row(ratings._protocol_columns(settings)), settings)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            loop.LoopConfig(save_model_every=-1).validate()

    def test_crash_after_completion_repairs_missing_save_without_retraining(self):
        output = self.root / "save-gap"
        with (
            patch.object(saved_models, "save_round", side_effect=KeyboardInterrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.run_loop(output, extra=["--save-model-every", "1"])
        calls = len(self.jobs.calls)
        state = self.run_loop(output, resume=True)
        self.assertEqual(len(self.jobs.calls), calls)
        self.assertTrue(state["iterations"][0]["completed"])
        self.assertTrue(
            (Path(state["config"]["model_store"]) / "save-gap-r0001/metadata.json").is_file()
        )

    def test_fresh_background_schedule_is_adaptive_not_all_pairs_and_resumes(self):
        self.jobs = ArenaJobs()
        campaign = self.root / "adaptive-campaign"
        state = self.run_loop(campaign, extra=["--iterations", "2", "--save-model-every", "1"])
        output = self.root / "fresh-ratings"
        args = rate_saved.parse_args(
            [
                "--output",
                str(output),
                "--model-store",
                state["config"]["model_store"],
                "--reference-games",
                "2",
                "--target-rd",
                "350",
                "--min-pairs",
                "1",
                "--min-opponents",
                "1",
                "--min-opponent-pairs",
                "1",
                "--max-games",
                "2",
                "--batch-games",
                "2",
            ]
        )
        with patch.object(loop, "run_process", self.jobs), redirect_stdout(io.StringIO()):
            manifest = rate_saved.run(args)
        self.assertEqual(manifest["models"], 2)
        cohort = output / manifest["cohorts"][0]
        calibration = torch.load(cohort / "loop.pt", weights_only=True)
        self.assertTrue(
            all("reference" in job and "gate" not in job for job in calibration["jobs"])
        )
        adaptive = torch.load(cohort / "adaptive/loop.pt", weights_only=True)
        self.assertEqual(adaptive["stop_reason"], "confidence-target-met")
        self.assertEqual(len(adaptive["jobs"]), 1)
        calls = len(self.jobs.calls)
        with patch.object(loop, "run_process", self.jobs), redirect_stdout(io.StringIO()):
            rate_saved.run(rate_saved.parse_args(["--output", str(output), "--resume"]))
        self.assertEqual(len(self.jobs.calls), calls)
