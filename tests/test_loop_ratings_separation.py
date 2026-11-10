import copy

import test_loop as helpers

from continugo_train import loop
from continugo_train.evaluation import EvaluationConfig, validate_output


class TrainingOnlyTests(helpers.LoopHarness):
    def test_resume_detaches_legacy_jobs_without_requiring_evidence_or_rating_binaries(self):
        output = self.root / "legacy-evaluations"
        state = self.run_loop(output, extra=["--save-model-every", "1"])
        legacy = copy.deepcopy(state)
        legacy["config"].update(
            gate_games=20,
            reference_games=10,
            adaptive_max_games=360,
            evaluation_interval=10,
            actor_policy="gate",
            arena_binary="/missing/arena",
            rating_binary="/missing/rating",
        )
        legacy["rating_protocol"] = {"version": 2, "arena_sha256": "old"}
        legacy["initial_reference"] = {"reference": {"games": [{"path": "gone.sgf"}]}}
        job = legacy["iterations"][0]
        job.update(
            gate={"games": [{"path": "gone.sgf"}]},
            reference=None,
            adaptive_rating={"path": "gone"},
            decision="failed",
        )
        loop.atomic_checkpoint(output / "loop.pt", legacy)
        before = (output / "loop.pt").read_bytes()
        calls = len(self.jobs.calls)
        migrated = self.run_loop(output, resume=True)
        self.assertEqual((output / "before-standalone-ratings.pt").read_bytes(), before)
        self.assertEqual(migrated["config"], state["config"])
        self.assertNotIn("rating_protocol", migrated)
        self.assertEqual(migrated["iterations"], state["iterations"])
        self.assertEqual(len(self.jobs.calls), calls)
        self.run_loop(output, resume=True)
        self.assertEqual((output / "before-standalone-ratings.pt").read_bytes(), before)

    def test_interrupted_post_training_match_is_skipped_without_retraining_or_export(self):
        output = self.root / "interrupted-evaluation"
        state = self.run_loop(output, extra=["--save-model-every", "1"])
        state["iterations"][0].update(
            completed=False, gate={"games": [None] * 20}, adaptive_rating=None, decision=None
        )
        state["config"].update(gate_games=20, adaptive_max_games=360)
        state["actor"] = {
            "checkpoint": state["initial_actor_checkpoint"],
            "model": state["initial_model"],
        }
        loop.atomic_checkpoint(output / "loop.pt", state)
        calls = len(self.jobs.calls)
        resumed = self.run_loop(output, resume=True)
        self.assertEqual(len(self.jobs.calls), calls)
        self.assertTrue(resumed["iterations"][0]["completed"])
        self.assertEqual(resumed["actor"]["checkpoint"], resumed["iterations"][0]["checkpoint"])
        self.assertNotIn("gate", resumed["iterations"][0])

    def test_standalone_config_and_output_live_separately_from_training(self):
        config = EvaluationConfig.from_config(
            {**loop.LoopConfig().__dict__, "actor_policy": "gate"}
        )
        config.validate()
        self.assertEqual(config.gate_games, 20)
        store = self.root / "models"
        validate_output(store / "ratings/jobs/study", store)
        validate_output(self.root / "separate-study", store)
        for output in [store, store / "ratings", store / "ratings/games", store / "accepted"]:
            with self.assertRaises(ValueError):
                validate_output(output, store)
