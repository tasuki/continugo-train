import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from rating_helpers import EvaluationHarness
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


class AdaptiveTests(EvaluationHarness):
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
