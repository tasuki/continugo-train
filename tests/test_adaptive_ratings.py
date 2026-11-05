import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import test_loop as helpers
import torch
from test_acceptance import ArenaJobs

from continugo_train import adaptive_ratings as adaptive
from continugo_train import loop, ratings

POLICY = {
    "target_rd": 25,
    "min_pairs": 60,
    "min_opponents": 2,
    "min_opponent_pairs": 10,
    "batch_games": 4,
    "max_games": 600,
}


def player(rating=400, deviation=75):
    return {"rating": rating, "deviation": deviation, "anchored": 1}


class MatchingTests(unittest.TestCase):
    def test_near_rating_opponent_and_reselection_after_results(self):
        states = {"a": player(), "b": player(410, 35), "c": player(850, 35)}
        first = adaptive.select_match(states, ["a"], {}, POLICY)
        self.assertEqual(first["players"], ["a", "b"])
        self.assertAlmostEqual(first["expected_score_first"], 0.5, delta=0.03)
        states["a"] = player(820)
        second = adaptive.select_match(states, ["a"], {}, POLICY)
        self.assertEqual(second["players"], ["a", "c"])

    def test_certain_opponents_and_opponent_diversity(self):
        states = {"a": player(), "b": player(400, 30), "c": player(400, 300)}
        self.assertEqual(adaptive.select_match(states, ["a"], {}, POLICY)["players"], ["a", "b"])
        pairs = {("a", "b"): {"pairs": 10, "completed": 20, "attempted": 20}}
        self.assertEqual(adaptive.select_match(states, ["a"], pairs, POLICY)["players"], ["a", "c"])

    def test_completed_targets_are_not_matched_against_each_other(self):
        states = {"a": player(deviation=20), "b": player(deviation=20), "c": player()}
        pairs = {
            (a, b): {"pairs": 30, "completed": 60, "attempted": 60}
            for a, b in [("a", "b"), ("a", "c"), ("b", "c")]
        }
        self.assertIsNone(adaptive.select_match(states, ["a", "b"], pairs, POLICY))
        states["c"] = player(deviation=100)
        selection = adaptive.select_match(states, ["a", "b", "c"], pairs, POLICY)
        self.assertIn("c", selection["players"])

    def test_confidence_requires_uncertainty_pairs_and_substantial_diversity(self):
        states = {"a": player(deviation=20), "b": player(), "c": player()}
        pairs = {("a", "b"): {"pairs": 60}, ("a", "c"): {"pairs": 1}}
        self.assertFalse(adaptive.progress(["a"], states, pairs, POLICY)["a"]["done"])
        pairs[("a", "c")]["pairs"] = 10
        self.assertTrue(adaptive.progress(["a"], states, pairs, POLICY)["a"]["done"])
        states["a"]["deviation"] = 55
        self.assertFalse(adaptive.progress(["a"], states, pairs, POLICY)["a"]["done"])
        states["a"]["deviation"] = 20
        states["a"]["anchored"] = 0
        self.assertFalse(adaptive.progress(["a"], states, pairs, POLICY)["a"]["done"])

    def test_pairs_require_both_colors_same_size_seed_and_completed_outcomes(self):
        def game(black, seed=1, size=9, result="0"):
            return {
                "black": black,
                "white": "b" if black == "a" else "a",
                "seed": seed,
                "board_size": size,
                "result": result,
            }

        evidence = {
            "1": game("a"),
            "2": game("a"),
            "3": game("b", seed=2),
            "4": game("b", result=""),
            "5": game("b", size=13),
        }
        self.assertEqual(adaptive.pair_statistics(evidence)[("a", "b")]["pairs"], 0)
        evidence["6"] = game("b")
        stats = adaptive.pair_statistics(evidence)[("a", "b")]
        self.assertEqual(stats["pairs"], 1)
        self.assertEqual(stats["completed"], 5)
        self.assertEqual(stats["attempted"], 6)

    def test_truncated_matchups_are_less_useful_and_extreme_ratings_are_stable(self):
        states = {"a": player(), "b": player(400, 40), "c": player(420, 40)}
        pairs = {("a", "b"): {"pairs": 0, "completed": 0, "attempted": 100}}
        self.assertEqual(adaptive.select_match(states, ["a"], pairs, POLICY)["players"], ["a", "c"])
        gain, p = adaptive.information(player(1e9), player(-1e9))
        self.assertEqual((gain, p), (0.0, 1.0))


class AdaptiveTests(helpers.LoopHarness):
    def setUp(self):
        super().setUp()
        self.jobs = ArenaJobs()
        self.source_loop = self.root / "training"
        self.source_state = self.run_loop(
            self.source_loop, extra=["--iterations", "2", "--reference-games", "2"]
        )
        self.output = self.root / "adaptive"

    def evaluate(self, resume=False, extra=()):
        args = ["--output", str(self.output)]
        args += (
            ["--resume"]
            if resume
            else [
                "--loop",
                str(self.source_loop),
                "--latest",
                "2",
                "--target-rd",
                "350",
                "--min-pairs",
                "1",
                "--min-opponents",
                "1",
                "--min-opponent-pairs",
                "1",
                "--batch-games",
                "2",
                "--max-games",
                "6",
            ]
        )
        with patch.object(loop, "run_process", self.jobs), redirect_stdout(io.StringIO()):
            return adaptive.run(adaptive.parse_args([*args, *extra]))

    def test_achieves_explicit_goal_and_resume_does_not_replay_or_change_training(self):
        before = (self.source_loop / "loop.pt").read_bytes()
        state = self.evaluate()
        self.assertEqual(state["stop_reason"], "confidence-target-met")
        self.assertEqual(len(state["jobs"]), 1)
        self.assertTrue(all(item["done"] for item in state["progress"].values()))
        self.assertEqual((self.source_loop / "loop.pt").read_bytes(), before)
        calls = len(self.jobs.calls)
        self.evaluate(resume=True)
        self.assertEqual(len(self.jobs.calls), calls)
        with ratings.transaction(self.source_state["config"]["model_store"]) as (tables, _):
            self.assertEqual(sum(r["kind"] == "gate" for r in tables.matches), 2)

    def test_explicit_target_does_not_depend_on_registration_order(self):
        target = self.source_state["initial_checkpoint"]["sha256"]
        state = self.evaluate(extra=["--targets", target, "--latest", "1"])
        self.assertEqual(state["targets"], [target])
        self.assertTrue(state["progress"][target]["done"])

    def test_future_loop_candidates_are_adaptively_rated_automatically(self):
        before = (self.source_loop / "loop.pt").read_bytes()
        calls = len(self.jobs.calls)
        state = self.run_loop(
            self.source_loop,
            resume=True,
            extra=["--enable-adaptive-ratings", "4", "--stop-after-iteration", "2"],
        )
        self.assertEqual(len(self.jobs.calls), calls)
        self.assertEqual((self.source_loop / "before-adaptive-ratings.pt").read_bytes(), before)
        self.assertEqual(state["config"]["adaptive_max_games"], 4)
        self.assertEqual(len(state["iterations"]), 2)
        state = self.run_loop(self.source_loop, resume=True, extra=["--iterations", "3"])
        iteration = state["iterations"][-1]
        report = iteration["adaptive_rating"]
        self.assertEqual(report["games"], 4)
        self.assertIn("NOT met", report["stop_reason"])
        self.assertEqual(set(report["progress"]), {iteration["checkpoint"]["sha256"]})
        self.assertEqual(state["actor"]["model"], iteration["model"])
        calls = len(self.jobs.calls)
        self.run_loop(self.source_loop, resume=True)
        self.assertEqual(len(self.jobs.calls), calls)

    def test_automatic_evaluation_interrupt_resumes_without_retraining(self):
        self.run_loop(
            self.source_loop,
            resume=True,
            extra=["--enable-adaptive-ratings", "4", "--stop-after-iteration", "2"],
        )
        self.jobs.arena_mode = "partial"
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop(self.source_loop, resume=True, extra=["--iterations", "3"])
        training_calls = len(self.stage_calls("continugo_train.train"))
        path = next(
            self.source_loop.glob(
                "iteration-0003/adaptive-ratings/iteration-0001/gate/*/game-000001-*.sgf"
            )
        )
        before = path.read_bytes()
        seen_locks = []

        def execute(command, logfile, locks, environment=None):
            if isinstance(locks, tuple):
                seen_locks.append(locks)
            return self.jobs(command, logfile, locks, environment)

        with patch.object(loop, "run_process", execute), redirect_stdout(io.StringIO()):
            state = loop.run_loop(self.args(self.source_loop, resume=True))
        self.assertEqual(len(self.stage_calls("continugo_train.train")), training_calls)
        self.assertEqual(path.read_bytes(), before)
        self.assertTrue(seen_locks)
        self.assertTrue(all(len(locks) == 2 and locks[0] != locks[1] for locks in seen_locks))
        self.assertEqual(state["iterations"][-1]["adaptive_rating"]["games"], 4)

    def test_enable_automatic_ratings_rejects_mid_iteration_and_invalid_budget(self):
        self.jobs.fault = "epoch_gap"
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop(self.source_loop, resume=True, extra=["--iterations", "3"])
        before = (self.source_loop / "loop.pt").read_bytes()
        for budget in ["4", "0", "3"]:
            with self.assertRaises(ValueError):
                self.run_loop(
                    self.source_loop, resume=True, extra=["--enable-adaptive-ratings", budget]
                )
            self.assertEqual((self.source_loop / "loop.pt").read_bytes(), before)

    def test_new_campaigns_enable_adaptive_ratings_without_an_opt_in(self):
        defaults = loop.LoopConfig()
        self.assertEqual(defaults.adaptive_max_games, 600)
        self.assertEqual(defaults.adaptive_target_rd, 25)
        self.assertEqual(defaults.adaptive_min_pairs, 60)
        self.assertEqual(defaults.adaptive_min_opponents, 2)
        self.assertEqual(defaults.adaptive_min_opponent_pairs, 10)
        self.assertEqual(defaults.adaptive_batch_games, 4)
        self.assertIsNone(loop.parse_args([]).adaptive_max_games)
        self.assertEqual(loop.parse_args(["--adaptive-max-games", "0"]).adaptive_max_games, 0)
        output = self.root / "default-adaptive-loop"
        args = self.args(
            output,
            extra=[
                "--reference-games",
                "2",
                "--adaptive-target-rd",
                "350",
                "--adaptive-min-pairs",
                "1",
                "--adaptive-min-opponents",
                "1",
                "--adaptive-min-opponent-pairs",
                "1",
            ],
        )
        # Remove the harness's explicit smoke-test opt-out. No enabling flag:
        # a genuinely omitted CLI setting must activate the production default.
        args.adaptive_max_games = None
        with (
            patch.object(loop, "run_process", self.jobs),
            patch.object(loop, "verify_export", self.jobs.bundle),
            redirect_stdout(io.StringIO()),
        ):
            state = loop.run_loop(args)
        self.assertEqual(state["config"]["adaptive_max_games"], 600)
        report = state["iterations"][0]["adaptive_rating"]
        self.assertGreater(report["games"], 0)
        self.assertEqual(report["stop_reason"], "confidence-target-met")
        saved = torch.load(output / report["path"] / "loop.pt", weights_only=True)
        self.assertEqual(saved["policy"]["max_games"], 600)

    def test_older_v5_loop_defaults_to_disabled_until_explicitly_enabled(self):
        state = torch.load(self.source_loop / "loop.pt", weights_only=True)
        for key in list(state["config"]):
            if key.startswith("adaptive_"):
                del state["config"][key]
        helpers.atomic_checkpoint(self.source_loop / "loop.pt", state)
        calls = len(self.jobs.calls)
        state = self.run_loop(self.source_loop, resume=True)
        self.assertEqual(state["config"]["adaptive_max_games"], 0)
        self.assertEqual(len(self.jobs.calls), calls)
        before = (self.source_loop / "loop.pt").read_bytes()
        with self.assertRaisesRegex(ValueError, "invalid adaptive"):
            self.run_loop(self.source_loop, resume=True, extra=["--enable-adaptive-ratings", "3"])
        self.assertEqual((self.source_loop / "loop.pt").read_bytes(), before)

    def test_too_few_opponents_reports_unmet_goal_without_blocking_latest_actor(self):
        state = self.run_loop(
            self.root / "small-loop",
            extra=[
                "--model-store",
                str(self.root / "small-store"),
                "--reference-games",
                "2",
                "--adaptive-max-games",
                "4",
            ],
        )
        iteration = state["iterations"][0]
        self.assertEqual(iteration["adaptive_rating"]["games"], 0)
        self.assertIn("insufficient-opponents", iteration["adaptive_rating"]["stop_reason"])
        self.assertEqual(state["actor"]["model"], iteration["model"])

    def test_budget_is_a_hard_cap_not_a_claim_of_confidence(self):
        state = self.evaluate(extra=["--target-rd", "1", "--max-games", "4"])
        self.assertEqual(len(state["jobs"]), 2)
        self.assertIn("NOT met", state["stop_reason"])
        self.assertFalse(all(item["done"] for item in state["progress"].values()))

    def test_interrupted_batch_keeps_completed_sgf_and_same_opponent_on_resume(self):
        self.jobs.arena_mode = "partial"
        with self.assertRaises(KeyboardInterrupt):
            self.evaluate()
        path = next(self.output.glob("iteration-0001/gate/*/game-000001-*.sgf"))
        before = path.read_bytes()
        state = self.evaluate(resume=True)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(len(state["jobs"]), 1)
        self.assertEqual(state["jobs"][0]["gate"]["attempts"][1]["indices"], [1])

    def test_rejects_conflicting_resume_policy_and_invalid_creation_policy(self):
        self.evaluate()
        with self.assertRaisesRegex(ValueError, "restores saved"):
            self.evaluate(resume=True, extra=["--target-rd", "25"])
        self.output = self.root / "bad"
        with self.assertRaisesRegex(ValueError, "positive uncertainty"):
            self.evaluate(extra=["--target-rd", "nan"])
