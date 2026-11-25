"""Loop option forwarding and inheritance, without pinning tunable defaults."""

import unittest
import io
from contextlib import redirect_stderr
from dataclasses import asdict

from continugo_train.evaluation import EvaluationConfig
from continugo_train import loop, rate_saved, rating_store, ratings
from continugo_train.loop import LoopConfig
from continugo_train.train import Options


class LoopOptionTests(unittest.TestCase):
    def test_model_store_has_no_implicit_lineage(self):
        self.assertIsNone(LoopConfig().model_store)
        with self.assertRaisesRegex(ValueError, "--model-store is required"):
            loop.run_loop(loop.parse_args([]))

    def test_rating_commands_require_explicit_store_selection(self):
        for command in (rate_saved.parse_args, ratings.main, rating_store.main):
            with self.subTest(command=command.__module__):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    command([])
                self.assertEqual(error.exception.code, 2)
        args = rate_saved.parse_args(["--model-store", "artifacts/20261122"])
        self.assertEqual(str(args.model_store), "artifacts/20261122")

    def test_all_new_game_caps_default_to_512_plies(self):
        config = EvaluationConfig()
        self.assertEqual(
            (LoopConfig().max_moves, config.gate_max_moves, config.reference_max_moves),
            (512, 512, 512),
        )

    def test_loop_options_override_parent_io_and_caps_but_preserve_optimizer_and_losses(self):
        parent = Options(
            batch_size=2,
            feature_cache="auto",
            steps_per_epoch=20,
            positions_per_game=8,
            validation_positions_per_game=4,
            size_weights="9:2,13:2,19:1",
            score_loss_weight=1.0,
            ownership_loss_weight=0.25,
            lr=0.02,
        )
        config = LoopConfig(
            batch_size=7,
            feature_cache="off",
            steps_per_epoch=3,
            positions_per_game=5,
            validation_positions_per_game=2,
            size_weights="uniform",
        )
        actual = config.training_options({"options": asdict(parent)}, 1)
        for name in (
            "batch_size",
            "feature_cache",
            "steps_per_epoch",
            "positions_per_game",
            "validation_positions_per_game",
            "size_weights",
        ):
            self.assertEqual(getattr(actual, name), getattr(config, name), name)
        for name in ("score_loss_weight", "ownership_loss_weight", "lr"):
            self.assertEqual(getattr(actual, name), getattr(parent, name), name)


if __name__ == "__main__":
    unittest.main()
