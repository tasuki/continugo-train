import math
import random
import unittest

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
        )
        random.Random(42).shuffle(records)
        self.assertEqual(states, fit(records))

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
        self.assertAlmostEqual(states["champion"]["rating"], 100 + 400 * math.log10(40.5 / 0.5))

    def test_disconnected_origin_prior_does_not_shrink_differences(self):
        states = fit(match("a", "b", 6, 4))
        self.assertAlmostEqual(states["a"]["rating"] + states["b"]["rating"], 200)
        self.assertAlmostEqual(
            states["a"]["rating"] - states["b"]["rating"], 400 * math.log10(6.5 / 4.5)
        )
