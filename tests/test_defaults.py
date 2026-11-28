"""Loop option forwarding and inheritance, without pinning tunable defaults."""

import io
import unittest
from contextlib import redirect_stderr
from dataclasses import asdict
from unittest.mock import patch

from continugo_train import loop, rate_saved, rating_store, ratings
from continugo_train.evaluation import EvaluationConfig
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

    def test_useless_move_penalty_is_forwarded_to_selfplay_evaluation_and_human_diagnostics(self):
        from continugo_train.selfplay import SelfplayConfig, diagnostic_settings

        config = SelfplayConfig()
        for settings in (config.search(), config.evaluation_search(), config.native_settings(), diagnostic_settings()):
            self.assertEqual(settings["useless_move_penalty_points"], 0.5)
        self.assertEqual(SelfplayConfig(useless_move_penalty_points=0).evaluation_search()["useless_move_penalty_points"], 0)
        for invalid in (-0.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                LoopConfig(useless_move_penalty_points=invalid).validate()

    def test_rating_protocol_roundtrips_useless_move_cost_and_preserves_old_protocols(self):
        from continugo_train.selfplay import SelfplayConfig

        settings = {
            "version": 3, "algorithm": "static-glicko-v1", "reference": ratings.REFERENCE_ID,
            "board_mix": {"9": 1}, "search": SelfplayConfig().evaluation_search(),
            "neural_max_moves": 512, "reference_max_moves": 512,
            "selection": "maximum-visits-random-ties", "arena_sha256": "a" * 64,
            "rating_sha256": "b" * 64,
        }
        restored = ratings._protocol_from_row(ratings._protocol_columns(settings))
        self.assertEqual(restored, settings)
        disabled = {**settings, "search": {**settings["search"], "useless_move_penalty_points": 0}}
        self.assertNotEqual(ratings.identity(settings), ratings.identity(disabled))
        for version in (1, 2):
            search = {k: v for k, v in settings["search"].items() if k != "useless_move_penalty_points"}
            if version == 1:
                search.pop("score_weight")
            old = {**settings, "version": version, "search": search}
            self.assertEqual(ratings._protocol_from_row(ratings._protocol_columns(old)), old)

    def test_frozen_rating_protocol_does_not_adopt_new_search_fields(self):
        from continugo_train.evaluation import source_protocol

        config = EvaluationConfig()
        with patch.object(ratings.archive, "digest", return_value="a" * 64):
            current = ratings.protocol(config)
            self.assertEqual(current["version"], 3)
            self.assertIn("useless_move_penalty_points", current["search"])
            for version in (1, 2):
                frozen = ratings.protocol(config, version=version)
                self.assertEqual(frozen["version"], version)
                self.assertNotIn("useless_move_penalty_points", frozen["search"])
                self.assertEqual("score_weight" in frozen["search"], version == 2)
                self.assertEqual(source_protocol({"rating_protocol": frozen}, config), frozen)
            with self.assertRaises(ValueError):
                ratings.protocol(config, version=99)

    def test_new_game_caps_match_current_selfplay_and_evaluation_usage(self):
        config = EvaluationConfig()
        self.assertEqual(
            (LoopConfig().max_moves, config.gate_max_moves, config.reference_max_moves),
            (1024, 1024, 1024),
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
