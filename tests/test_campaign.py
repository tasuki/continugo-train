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
