import io
import math
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from continugo_train.data import GameEntry, assign_splits, merge_split_assignments, scan_corpus
from continugo_train.loop import (
    LoopConfig,
    parse_args,
    select_position_replay,
    training_config,
    training_window_size,
)


def record(number):
    return {
        "game_id": f"{number:064x}",
        "board_size": 9,
        "eligible": True,
        "eligible_positions": 1,
        "path": f"game-{number}.sgf",
    }


class GrowingReplayTests(unittest.TestCase):
    def test_obsolete_window_controls_are_not_supported(self):
        for name in ("training_window_fraction", "window_games", "window_iterations"):
            self.assertFalse(hasattr(LoopConfig(), name))
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--" + name.replace("_", "-"), "1"])
        # Old checkpoint configuration is normalized to current settings; these
        # removed values cannot influence newly selected windows.
        config = {"training_window_fraction": 0.25, "window_games": 100, "window_iterations": 8}
        self.assertEqual(training_config({**config, "seed": 42}), {"seed": 42})
        for flag in ("--replay-training-fraction", "--replay-games", "--replay-iterations"):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args([flag, "1"])
        with self.assertRaisesRegex(ValueError, "obsolete replay configuration"):
            training_config({"replay_training_fraction": 0.25})

    def test_katago_formula_and_threshold(self):
        examples = {
            0: 0,
            100: 100,
            100_000: 100_000,
            250_000: 250_000,
            1_000_000: 474_968,
            10_000_000: 1_788_253,
            100_000_000: 7_654_481,
            1_000_000_000: 33_857_958,
        }
        for n, expected in examples.items():
            self.assertEqual(training_window_size(n), expected)
        for n in (249_999, 250_000, 250_001, 900_000, 2**63 - 1):
            exact = min(
                n,
                250_000 + (0.4 * 250_000**0.35 / 0.65) * (max(n, 250_000) ** 0.65 - 250_000**0.65),
            )
            self.assertEqual(training_window_size(n), math.ceil(exact))
        for invalid in (None, -1, True, 1.5):
            with self.assertRaisesRegex(ValueError, "known cumulative data positions"):
                training_window_size(invalid)

    def test_combined_window_precedes_splits_and_keeps_whole_games(self):
        records = [record(i) for i in range(4, 0, -1)]
        for r in records:
            r["eligible_positions"] = 200_000
        target = training_window_size(1_000_000)
        selected = select_position_replay(records, target, "uniform")
        self.assertEqual(selected, records[:3])
        self.assertEqual(sum(r["eligible_positions"] for r in selected), 600_000)
        games = [GameEntry(r["path"], r["game_id"], "hash", 9, 4, 1, (0,)) for r in selected]
        pins = {records[-1]["game_id"]: "validation", records[0]["game_id"]: "train"}
        train, validation, assigned = assign_splits(games, 7, 0.2, pins)
        self.assertEqual({g.game_id for g in train + validation}, {r["game_id"] for r in selected})
        self.assertEqual(assigned[records[0]["game_id"]], "train")
        self.assertEqual(len(validation), 1)
        pins.update({game.game_id: "train" for game in games})
        train, validation, _ = assign_splits(games, 99, 0.2, pins)
        self.assertEqual(len(train), 3)
        self.assertEqual(validation, [])  # Do not borrow the older held-out game.
        self.assertEqual(select_position_replay(records, 0), [])
        duplicated = [records[0], records[0], *records[1:]]
        self.assertEqual(select_position_replay(duplicated, target, "uniform"), selected)

    def test_deduplicates_and_caps_at_available_games(self):
        records = [record(2), record(2), record(1)]
        self.assertEqual(len(select_position_replay(records, training_window_size(1000))), 2)

    def test_historical_baseline_retains_initial_pool_until_it_fills(self):
        old, new = record(1), record(2)
        old["eligible_positions"] = 430_000
        new["eligible_positions"] = 10_000
        records = [new, old]
        self.assertEqual(select_position_replay(records, training_window_size(900_000)), records)
        new["eligible_positions"] = 900_000
        self.assertEqual(select_position_replay(records, training_window_size(1_800_000)), [new])
        old["eligible_positions"] = 0
        with self.assertRaisesRegex(ValueError, "positive eligible_positions"):
            select_position_replay([old], training_window_size(900_000))

    def test_pool_pins_do_not_inflate_cumulative_training_count(self):
        games = [GameEntry(str(i), f"{i:064x}", "hash", 9, 4, 1, (0,)) for i in (1, 2)]
        pool = {f"{i:064x}": "train" for i in range(1, 20)}
        pool[games[1].game_id] = "validation"
        with patch("continugo_train.data.inspect_corpus", return_value=(games, {})):
            corpus = scan_corpus([], 4, 999, 0.9, split_assignments=pool)
        self.assertEqual([g.game_id for g in corpus.train], [games[0].game_id])
        self.assertEqual([g.game_id for g in corpus.validation], [games[1].game_id])
        self.assertEqual(set(corpus.split_history), {g.game_id for g in games})
        with self.assertRaisesRegex(ValueError, "conflicts"):
            merge_split_assignments(pool, {games[1].game_id: "train"})

    def test_assignments_stay_pinned_when_seed_and_pool_change(self):
        games = [GameEntry(str(i), f"{i:064x}", "hash", 9, 4, 1, (0,)) for i in range(20)]
        _, _, initial = assign_splits(games, 12, 0.2, {})
        train, validation, later = assign_splits(games[:7], 999, 0.8, initial)
        self.assertEqual(initial, later)
        self.assertTrue(all(initial[g.game_id] == "train" for g in train))
        self.assertTrue(all(initial[g.game_id] == "validation" for g in validation))
