"""Loop option forwarding and inheritance, without pinning tunable defaults."""

import unittest
from dataclasses import asdict

from continugo_train.loop import LoopConfig
from continugo_train.train import Options


class LoopOptionTests(unittest.TestCase):
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
