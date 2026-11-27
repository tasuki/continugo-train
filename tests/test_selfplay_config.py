import unittest
from dataclasses import asdict
from unittest.mock import patch

from continugo_train import runtime
from continugo_train.evaluation import EvaluationConfig
from continugo_train.loop import LoopConfig
from continugo_train.selfplay import SelfplayConfig


class SelfplayConfigTests(unittest.TestCase):
    def test_selfplay_uses_the_same_settings_inside_and_outside_the_loop(self):
        base, training = SelfplayConfig(), LoopConfig()
        for key in base.__dataclass_fields__:
            self.assertEqual(getattr(training, key), getattr(base, key), key)

    def test_launchers_use_workflow_settings(self):
        for command, config, settings in (
            ("selfplay", LoopConfig(), LoopConfig().native_settings()),
            ("arena", EvaluationConfig(), EvaluationConfig().arena_settings()),
        ):
            with self.subTest(command=command), patch.object(
                runtime, "inference_environment", return_value=(None, {}),
            ), patch.object(runtime.os, "execvpe") as execute:
                runtime.main([command, "--device", "cpu", "--output", "unused"])
                argv = execute.call_args.args[1]
                for name, value in settings.items():
                    self.assertEqual(argv[argv.index("--" + name.replace("_", "-")) + 1], str(value))
                self.assertEqual(config.widening, SelfplayConfig.widening)
                self.assertEqual(config.independent_exploration, SelfplayConfig.independent_exploration)

    def test_recorded_settings_win_over_defaults(self):
        frozen = asdict(EvaluationConfig(gate_simulations=91, gate_max_moves=611, komi=-2.5))
        self.assertEqual(asdict(EvaluationConfig.from_config(frozen)), frozen)
