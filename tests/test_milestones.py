import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import test_loop as helpers
from test_acceptance import ArenaJobs

from continugo_train import loop, rate_saved, ratings, saved_models
from continugo_train.evaluation import EvaluationConfig


class MilestoneTests(helpers.LoopHarness):
    def test_saves_only_completed_multiples_as_independent_copies(self):
        output = self.root / "campaign"
        state = self.run_loop(output, extra=["--iterations", "5", "--save-model-every", "2"])
        store = Path(state["config"]["model_store"])
        snapshots = sorted((store / "models/test1").glob("r*"))
        self.assertEqual([p.name for p in snapshots], ["r0002", "r0004"])
        for directory in snapshots:
            metadata = saved_models.verify(directory)
            self.assertEqual(metadata["source"]["run"], "campaign")
            self.assertEqual(metadata["family"], "test1")
            self.assertEqual(
                {p.name for p in directory.iterdir()}, {*saved_models.FILES, "metadata.json"}
            )
            self.assertTrue(all(p.stat().st_nlink == 1 for p in directory.iterdir()))
            source = output / state["iterations"][metadata["round"] - 1]["checkpoint"]["path"]
            self.assertNotEqual(source.stat().st_ino, (directory / "checkpoint.pt").stat().st_ino)
            self.assertEqual(source.read_bytes(), (directory / "checkpoint.pt").read_bytes())
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
        with self.assertRaisesRegex(ValueError, "changed data"):
            saved_models.save_campaign(output, every=2)

    def test_corrupt_sources_and_name_conflicts_are_not_overwritten(self):
        output = self.root / "conflict"
        state = self.run_loop(output, extra=["--save-model-every", "0"])
        saved = saved_models.save_campaign(output, iteration=1)[0]
        metadata = saved / "metadata.json"
        original = metadata.read_bytes()
        metadata.chmod(0o644)
        value = json.loads(original)
        value["source"]["run"] = "different"
        metadata.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "conflicts"):
            saved_models.save_campaign(output, iteration=1)
        metadata.write_bytes(original)
        source = output / state["iterations"][0]["model"]["encoder"]["path"]
        source.write_text("changed")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            saved_models.save_campaign(output, iteration=1)

    def test_milestones_only_save_models_without_evaluation(self):
        output = self.root / "automatic"
        state = self.run_loop(output, extra=["--iterations", "10"])
        store = Path(state["config"]["model_store"])
        self.assertTrue((store / "models/test1/r0010/metadata.json").is_file())
        self.assertFalse((store / "ratings/ratings.png").exists())
        self.assertFalse(any("--candidate" in c for c in self.jobs.calls))
        for name in ["gate", "reference", "adaptive-ratings"]:
            self.assertFalse(list(output.rglob(name)))
        self.assertNotIn("rating_protocol", state)
        self.assertTrue(all("gate" not in job for job in state["iterations"]))

    def test_defaults_and_legacy_protocol_round_trip(self):
        self.assertEqual(loop.LoopConfig().save_model_every, 10)
        settings = ratings.protocol(EvaluationConfig(), legacy=True)
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
            (Path(state["config"]["model_store"]) / "models/test1/r0001/metadata.json").is_file()
        )

    def test_saved_models_are_evaluated_separately_starting_against_random(self):
        self.jobs = ArenaJobs()
        campaign = self.root / "champion-campaign"
        state = self.run_loop(campaign, extra=["--iterations", "2", "--save-model-every", "1"])
        before = (campaign / "loop.pt").read_bytes()
        store = Path(state["config"]["model_store"])
        with patch.object(loop, "run_process", self.jobs), redirect_stdout(io.StringIO()):
            manifest = rate_saved.run(
                rate_saved.parse_args(
                    [
                        "--family",
                        "test1",
                        "--model-store",
                        str(store),
                    ]
                )
            )
            self.assertEqual(manifest["status"], "complete")
            calls = len(self.jobs.calls)
            rate_saved.run(rate_saved.parse_args(["--model-store", str(store), "--resume"]))
        self.assertEqual(len(self.jobs.calls), calls)
        self.assertEqual((campaign / "loop.pt").read_bytes(), before)
        entry = rate_saved.load_champions(store)
        self.assertEqual(entry["champion"], "random")
        candidate = "test1-r0002"
        self.assertIn(candidate, entry["skipped"])
        self.assertFalse((store / "LATEST-ACCEPTED").exists())
