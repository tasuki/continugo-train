"""Explicit standalone evaluation fixtures; the training loop never launches matches."""

import copy
import io
from contextlib import redirect_stdout
from dataclasses import asdict
from unittest.mock import patch

import test_loop as helpers
import torch

from continugo_train import gate, loop, ratings
from continugo_train.evaluation import EvaluationConfig, EvaluationRunner


class FixtureRunner(EvaluationRunner):
    def save(self):
        loop.atomic_checkpoint(self.root / "evaluation-fixture.pt", self.state)


class EvaluationHarness(helpers.LoopHarness):
    def run_loop(self, output, resume=False, extra=()):
        # These tests exercise the rating engine directly, not training scheduling.
        options = {}
        training = []
        evaluation_names = {
            "gate_games",
            "reference_games",
            "gate_max_moves",
            "gate_simulations",
            "reference_max_moves",
        }
        it = iter(extra)
        for flag in it:
            if flag.removeprefix("--").replace("-", "_") in evaluation_names:
                name = flag[2:].replace("-", "_")
                options[name] = int(next(it))
            else:
                training.append(flag)
        fixture = output / "evaluation-fixture.pt"
        state = super().run_loop(output, resume, training)
        if "--migrate-storage" in extra or not options and not fixture.exists():
            return state
        if fixture.exists():
            evaluated = torch.load(fixture, weights_only=True)
            for name, value in options.items():
                if value != evaluated["config"][name]:
                    raise ValueError("conflicting evaluation fixture setting")
            # Training may have been extended explicitly by a test.
            evaluated["iterations"] += copy.deepcopy(
                state["iterations"][len(evaluated["iterations"]) :]
            )
        else:
            evaluated = copy.deepcopy(state)
            config = EvaluationConfig.from_config(state["config"])
            evaluated["config"] = {
                **asdict(config),
                "gate_games": 0,
                "reference_games": 0,
                **options,
            }
            evaluated["rating_protocol"] = ratings.protocol(
                EvaluationConfig.from_config(evaluated["config"])
            )
            evaluated["initial_reference"] = {
                "number": 1,
                "initial": True,
                "model": state["initial_model"],
                "reference": None,
            }
        # Keep the actual training checkpoint read-only during evaluation.
        runner = FixtureRunner(output, evaluated, None)
        with patch.object(loop, "run_process", self.jobs), redirect_stdout(io.StringIO()):
            initial = evaluated["initial_reference"]
            if runner.config.reference_games:
                ratings.register_checkpoint(
                    runner.config.model_store,
                    runner.check(state["initial_actor_checkpoint"]),
                    model=runner.check_model(state["initial_model"]),
                )
                gate.evaluate(runner, initial, reference=True)
                runner.rate_evaluation(initial, "reference", [state["initial_actor_checkpoint"]])
            for job in evaluated["iterations"]:
                job.setdefault("gate", None)
                job.setdefault("reference", None)
                for key, enabled, checkpoints in [
                    (
                        "gate",
                        runner.config.gate_games,
                        [job["checkpoint"], job["parent_actor_checkpoint"]],
                    ),
                    ("reference", runner.config.reference_games, [job["checkpoint"]]),
                ]:
                    if not enabled:
                        continue
                    ratings.register_checkpoint(
                        runner.config.model_store,
                        runner.check(job["checkpoint"]),
                        model=runner.check_model(job["model"]),
                    )
                    gate.evaluate(runner, job, reference=key == "reference")
                    runner.rate_evaluation(job, key, checkpoints)
            runner.save()
        return evaluated
