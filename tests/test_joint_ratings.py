import math
import random
import unittest
from unittest.mock import patch

from continugo_train.joint_ratings import fit
from continugo_train.ratings import REFERENCE_ID


def match(black, white, wins, losses):
    return [{"black": black, "white": white, "result": "B+1"}] * wins + [
        {"black": black, "white": white, "result": "W+1"}
    ] * losses


class JointRatingsTests(unittest.TestCase):
    def test_equal_records_against_same_opponent_have_equal_ratings(self):
        records = match("champion", REFERENCE_ID, 30, 10)
        records += match("r15", "champion", 5, 5)
        records += match("r20", "champion", 6, 4)
        records += match("r25", "champion", 6, 4)[::-1]
        records += match("r30", "champion", 5, 5)
        states = fit(records)
        for key in ("rating", "deviation"):
            self.assertAlmostEqual(states["r15"][key], states["r30"][key])
            self.assertAlmostEqual(states["r20"][key], states["r25"][key])
        self.assertGreater(states["r20"]["rating"], states["r15"]["rating"])
        for name in ("r15", "r30"):
            self.assertAlmostEqual(states[name]["rating"], states["champion"]["rating"])
            self.assertGreater(states[name]["deviation"], states["champion"]["deviation"])
        self.assertAlmostEqual(
            states["r20"]["rating"] - states["champion"]["rating"],
            400 * math.log10(6.5 / 4.5),
            delta=1,
        )
        random.Random(42).shuffle(records)
        self.assertEqual(states, fit(records))

    def test_cg11_round20_roundoff_stall(self):
        # Exact pair totals at the cg11 failure: the old Armijo search stalled
        # at gradient ~1.6e-9 despite a full correction of <1e-6 rating points.
        records = match("cg11-r0005", REFERENCE_ID, 86, 4)
        records += match("cg11-r0005", "cg11-r0010", 4, 86)
        records += match("cg11-r0010", REFERENCE_ID, 49, 1)
        records += match("cg11-r0010", "cg11-r0015", 4, 6)
        records += match("cg11-r0010", "cg11-r0020", 9, 30)
        records += [{"black": "cg11-r0010", "white": "cg11-r0020", "result": None}]
        states = fit(records)
        self.assertAlmostEqual(
            states["cg11-r0020"]["rating"] - states["cg11-r0010"]["rating"],
            400 * math.log10(30.5 / 9.5),
            delta=1,
        )
        self.assertAlmostEqual(
            states["cg11-r0015"]["rating"] - states["cg11-r0010"]["rating"],
            400 * math.log10(6.5 / 4.5),
            delta=1,
        )
        self.assertEqual(states["cg11-r0020"]["games"], 40)
        self.assertEqual(states["cg11-r0020"]["truncated"], 1)
        for state in states.values():
            self.assertTrue(math.isfinite(state["rating"]))
            self.assertTrue(math.isfinite(state["deviation"]))
        random.Random(42).shuffle(records)
        self.assertEqual(states, fit(records))

    def test_unbeaten_chain_retains_rating_accuracy(self):
        # A small gradient alone is not a rating-accuracy criterion when the
        # curvature is small. Check the full Newton correction instead.
        records = []
        opponent = REFERENCE_ID
        for i in range(30):
            player = f"r{i:02d}"
            records += match(player, opponent, 200, 0)
            opponent = player
        states = fit(records)
        difference = 400 * math.log10(200.5 / 0.5)
        opponent = REFERENCE_ID
        for i in range(30):
            player = f"r{i:02d}"
            self.assertAlmostEqual(
                states[player]["rating"] - states[opponent]["rating"],
                difference,
                delta=1,
            )
            opponent = player

    def test_one_point_tolerance_matches_strict_fit(self):
        rng = random.Random(20261122)
        names = [REFERENCE_ID, *[f"r{i}" for i in range(8)]]
        for _ in range(25):
            records = []
            for _ in range(25):
                black, white = rng.sample(names, 2)
                records += match(black, white, rng.randrange(40), rng.randrange(40))
            states = fit(records)
            with patch("continugo_train.joint_ratings.RATING_TOLERANCE", 0.01):
                strict = fit(records)
            for name in states:
                self.assertAlmostEqual(states[name]["rating"], strict[name]["rating"], delta=1)
                self.assertAlmostEqual(
                    states[name]["deviation"], strict[name]["deviation"], delta=1
                )

    def test_color_swaps_do_not_change_fit(self):
        records = match("a", REFERENCE_ID, 8, 2)
        swapped = [
            {
                "black": r["white"],
                "white": r["black"],
                "result": "W+1" if r["result"].startswith("B") else "B+1",
            }
            for r in records
        ]
        self.assertEqual(fit(records), fit(swapped))

    def test_new_evidence_updates_previous_opponents(self):
        records = match("a", REFERENCE_ID, 8, 2) + match("b", "a", 5, 5)
        before = fit(records)
        after = fit(records + match("a", REFERENCE_ID, 0, 20))
        self.assertLess(after["b"]["rating"], before["b"]["rating"])

    def test_counts_anchor_draws_and_unfinished_games(self):
        records = match("a", REFERENCE_ID, 2, 1)
        records += [
            {"black": "a", "white": REFERENCE_ID, "result": result} for result in ("0", None)
        ]
        states = fit(records)
        self.assertEqual(states[REFERENCE_ID]["rating"], 100)
        self.assertEqual(states[REFERENCE_ID]["deviation"], 0)
        self.assertEqual(
            {k: states["a"][k] for k in ("games", "wins", "losses", "draws", "truncated")},
            {"games": 5, "wins": 2, "losses": 1, "draws": 1, "truncated": 1},
        )
        self.assertEqual(fit([])[REFERENCE_ID]["rating"], 100)

    def test_unbeaten_disconnected_and_unrated_are_finite(self):
        states = fit(match("a", "b", 10, 0) + [{"black": "c", "white": "d", "result": None}])
        for state in states.values():
            self.assertTrue(math.isfinite(state["rating"]))
            self.assertTrue(math.isfinite(state["deviation"]))
        self.assertEqual(states["c"]["rating"], 100)
        self.assertAlmostEqual(states["c"]["deviation"], 350)

    def test_unbeaten_anchored_and_balanced_challenger(self):
        records = match("champion", REFERENCE_ID, 40, 0)
        baseline = fit(records)
        states = fit(records + match("challenger", "champion", 5, 5))
        self.assertAlmostEqual(states["champion"]["rating"], baseline["champion"]["rating"])
        self.assertAlmostEqual(states["challenger"]["rating"], states["champion"]["rating"])
        self.assertGreater(states["challenger"]["deviation"], states["champion"]["deviation"])
        self.assertAlmostEqual(
            states["champion"]["rating"], 100 + 400 * math.log10(40.5 / 0.5), delta=1
        )

    def test_disconnected_origin_prior_does_not_shrink_differences(self):
        states = fit(match("a", "b", 6, 4))
        self.assertAlmostEqual(states["a"]["rating"] + states["b"]["rating"], 200)
        self.assertAlmostEqual(
            states["a"]["rating"] - states["b"]["rating"], 400 * math.log10(6.5 / 4.5), delta=1
        )
