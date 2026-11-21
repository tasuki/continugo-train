"""Shared campaign policy: freezing, forks, cache compatibility and recovery."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from continugo_train import campaign, loop
from continugo_train.feature_cache import sidecar_path


class CampaignTests(unittest.TestCase):
    def test_fork_preserves_state_without_mutating_parent(self):
        parent = {
            "progress": {"family": "old", "samples_seen": 123, "data_positions": 12},
            "model": {"weight": torch.tensor([1.0, 2.0])},
            "optimizer": {"state": {0: {"momentum_buffer": torch.tensor([3.0])}}},
            "options": {"augmentation": "d4"},
            "game_splits": {"game": "validation"},
        }
        fork = campaign.fork_checkpoint(parent, "new", Path("parent.pt"), "digest")
        self.assertEqual(parent["progress"]["family"], "old")
        self.assertEqual(fork["progress"], dict(parent["progress"], family="new"))
        self.assertTrue(torch.equal(parent["model"]["weight"], fork["model"]["weight"]))
        self.assertTrue(
            torch.equal(
                parent["optimizer"]["state"][0]["momentum_buffer"],
                fork["optimizer"]["state"][0]["momentum_buffer"],
            )
        )
        self.assertEqual(fork["game_splits"], parent["game_splits"])
        fork["model"]["weight"][0] = 9
        self.assertEqual(parent["model"]["weight"][0].item(), 1)

    def test_freeze_builds_once_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs/test"
            run.mkdir(parents=True)
            (root / "python/continugo_train").mkdir(parents=True)
            (root / "python/continugo_train/test.py").write_text("frozen")
            (root / "target/release").mkdir(parents=True)
            for name in campaign.BINARIES:
                (root / "target/release" / name).write_text(name)
            recipe = {"family": "test"}
            with (
                patch.object(campaign.subprocess, "run") as build,
                patch.object(campaign.subprocess, "check_output", return_value="revision\n"),
            ):
                campaign.freeze(root, run, recipe)
                (root / "python/continugo_train/test.py").write_text("edited")
                campaign.freeze(root, run, recipe)
                build.assert_called_once()
                self.assertEqual((run / "runtime/continugo_train/test.py").read_text(), "frozen")
                with self.assertRaisesRegex(ValueError, "Recipe changed"):
                    campaign.freeze(root, run, {"family": "different"})
                (run / "bin/features").write_text("changed")
                with self.assertRaisesRegex(ValueError, "Frozen campaign file changed"):
                    campaign.freeze(root, run, recipe)

    def test_incompatible_seed_cache_is_rebuilt_locally(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, target = root / "old/game.sgf", root / "new/game.sgf"
            target.parent.mkdir()
            record = {"source": str(original), "path": str(target), "sha256": "digest"}
            with (
                patch("continugo_train.feature_cache.CacheReader") as reader,
                patch("continugo_train.feature_cache.build_cache") as build,
            ):
                reader.side_effect = [
                    ValueError("incompatible extractor"),
                    unittest.mock.MagicMock(),
                ]
                result = campaign.seed_cache(record, root / "features", 4)
                build.assert_called_once_with(target, density=4, binary=root / "features")
                self.assertEqual(result, str(sidecar_path(target)))
                self.assertFalse(sidecar_path(original).exists())

    def test_shared_runner_preserves_bootstrap_order_and_disabled_ratings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = {"rating_every": 0, "runtime_hashes": {}}
            state = {
                "config": {},
                "bootstrap": [{"source": "rank-000002.sgf"}, {"source": "rank-000001.sgf"}],
            }
            runner = campaign.campaign_runner(root, receipt)(root, state, None)
            self.assertEqual(state["bootstrap"][0]["source"], "rank-000001.sgf")
            with (
                patch.object(loop.LoopRunner, "save_milestone"),
                patch.object(runner, "execute") as execute,
            ):
                runner.save_milestone({"number": 5})
                execute.assert_not_called()

    def test_fresh_preparation_uses_defaults_and_commits_independent_checkpoint(self):
        from dataclasses import asdict

        from continugo_train.model import NetworkConfig
        from continugo_train.train import Options, feature_spec

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs/fresh"
            run.mkdir(parents=True)
            recipe = {"fresh": True, "family": "cg11", "lineage": "new",
                      "bootstrap": {"games": 2, "rollouts": 1}, "rating_every": 5}
            campaign.dump(run / "runtime.json", {"hashes": {}})
            saved = {
                "initialization": {"mode": "fresh"}, "progress": {"family": "cg11"},
                "feature_spec": feature_spec(Options().samples_per_diameter),
                "options": asdict(Options()), "config": asdict(NetworkConfig()),
                "model": {"weight": torch.tensor([1.0])},
                "metrics": {"train": {"loss": 1.0}, "validation": {"loss": 1.0}},
            }
            commands = []

            def execute(command, logfile, lock_fd, environment):
                commands.append(command)
                self.assertEqual(lock_fd, 123)
                if "--engine" in command:
                    (run / "bootstrap").mkdir()
                    campaign.dump(run / "bootstrap-stats.json", {"completed": 2})
                else:
                    (run / "initial-training").mkdir()
                    (run / "initial-training/latest.pt").write_bytes(b"fresh")

            with (
                patch.object(loop, "run_process", side_effect=execute),
                patch("continugo_train.data.inspect_corpus", return_value=(
                    [object(), object()], {"unlabeled": 0, "no_policy": 0, "duplicate_game": 0}
                )),
                patch.object(torch, "load", return_value=saved),
            ):
                receipt = campaign.prepare(root, run, recipe, 123)
                self.assertTrue(receipt["fresh"])
                self.assertIsNone(receipt["source_checkpoint"])
                self.assertEqual(receipt["inherited_corpus"], [])
                self.assertEqual(receipt["config"]["model_store"], str(root / "artifacts/new"))
                self.assertEqual(receipt["config"]["games_per_iteration"], loop.LoopConfig().games_per_iteration)
                self.assertEqual(len(commands), 2)
                for flag in ("--epochs", "--batch-size", "--channels", "--warm-start", "--resume"):
                    self.assertNotIn(flag, commands[1])
                # Committed preparation no longer launches generation/training, even
                # after the loop has initialized the formerly empty lineage store.
                (root / "artifacts/new").mkdir(parents=True)
                self.assertEqual(campaign.prepare(root, run, recipe, 123), receipt)
                self.assertEqual(len(commands), 2)

    def test_fresh_preparation_rejects_existing_store_or_partial_bootstrap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs/fresh"
            run.mkdir(parents=True)
            store = root / "artifacts/new"
            store.mkdir(parents=True)
            recipe = {"fresh": True, "family": "cg11", "lineage": "new"}
            campaign.dump(run / "runtime.json", {"hashes": {}})
            with patch.object(loop, "run_process") as execute:
                with self.assertRaisesRegex(ValueError, "store already exists"):
                    campaign.prepare(root, run, recipe)
                store.rmdir()
                (run / "bootstrap").mkdir()
                with self.assertRaisesRegex(ValueError, "Partial bootstrap"):
                    campaign.prepare(root, run, recipe)
                execute.assert_not_called()

    def test_first_lineage_evaluation_initializes_then_resumes(self):
        from continugo_train.rate_saved import parse_args

        with tempfile.TemporaryDirectory() as directory:
            store = Path(directory)
            first = parse_args(campaign.rating_arguments(store, "cg11", 5))
            self.assertFalse(first.resume)
            self.assertEqual(first.family, "cg11")
            self.assertEqual(first.include_family, "cg11")
            self.assertEqual(first.family_evaluation_every, 5)
            self.assertEqual(first.model_store, store)
            (store / "ratings").mkdir()
            (store / "ratings/evaluator.pt").touch()
            resumed = parse_args(campaign.rating_arguments(store, "cg11", 5))
            self.assertTrue(resumed.resume)
            self.assertIsNone(resumed.family)
            self.assertEqual(resumed.include_family, "cg11")
            self.assertEqual(resumed.family_evaluation_every, 5)

    def test_frozen_environment_resolves_viewer_outside_campaign(self):
        root = Path("/project/continugo-train")
        run = root / "runs/test"
        with patch.dict(campaign.os.environ, {}, clear=True):
            environment = campaign.frozen_environment(root, run)
        self.assertEqual(environment["PYTHONPATH"], str(run / "runtime"))
        self.assertEqual(
            environment["CONTINUGO_VIEWER_BUNDLE"],
            "/project/continugo/frontend/viewer/dist/continugo-viewer.js",
        )
        with patch.dict(
            campaign.os.environ, {"CONTINUGO_VIEWER_BUNDLE": "custom/viewer.js"}, clear=True
        ):
            environment = campaign.frozen_environment(root, run)
        self.assertEqual(
            environment["CONTINUGO_VIEWER_BUNDLE"], str(Path("custom/viewer.js").resolve())
        )

    def test_prepare_resume_is_independent_of_source_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "parent.pt").write_bytes(b"checkpoint")
            receipt = {"boundary_checkpoint_sha256": campaign.sha(root / "parent.pt")}
            campaign.dump(root / "launch.json", receipt)
            campaign.dump(root / "runtime.json", {"hashes": {}})
            self.assertEqual(campaign.prepare(root, root, {}), receipt)
            (root / "parent.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "boundary checkpoint changed"):
                campaign.prepare(root, root, {})


if __name__ == "__main__":
    unittest.main()
