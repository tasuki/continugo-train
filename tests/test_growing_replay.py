import io
import math
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from continugo_train.data import GameEntry, assign_splits, merge_split_assignments, scan_corpus
from continugo_train.loop import (
    LoopConfig,
    parse_args,
    select_growing_replay,
    training_config,
)


def record(number):
    return {
        "game_id": f"{number:064x}",
        "board_size": 9,
        "eligible": True,
        "path": f"game-{number}.sgf",
    }


class GrowingReplayTests(unittest.TestCase):
    def test_default_and_validation(self):
        self.assertEqual(LoopConfig().training_window_fraction, 1 / 3)
        for invalid in (-1, 1.01, math.nan, math.inf):
            with self.assertRaisesRegex(ValueError, "training-window-fraction"):
                LoopConfig(training_window_fraction=invalid).validate()

    def test_only_window_names_are_accepted(self):
        public = parse_args(
            [
                "--training-window-fraction",
                "0.25",
                "--window-games",
                "100",
                "--window-iterations",
                "8",
            ]
        )
        self.assertEqual(public.training_window_fraction, 0.25)
        self.assertEqual(public.window_games, 100)
        self.assertEqual(public.window_iterations, 8)
        for flag in ("--replay-training-fraction", "--replay-games", "--replay-iterations"):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args([flag, "1"])
        config = {"training_window_fraction": 0.25, "window_games": 100, "window_iterations": 8}
        self.assertEqual(training_config(config), config)
        with self.assertRaisesRegex(ValueError, "obsolete replay configuration"):
            training_config({"replay_training_fraction": 0.25})

    def test_training_budget_excludes_separate_validation_pool(self):
        records = [record(i) for i in range(100, 0, -1)]
        splits = {
            r["game_id"]: "validation" if int(r["game_id"], 16) % 5 == 0 else "train"
            for r in records
        }
        selected, target = select_growing_replay(records, splits, 61, 1 / 3, 0.2, "uniform")
        train = [r for r in selected if splits[r["game_id"]] == "train"]
        validation = [r for r in selected if splits[r["game_id"]] == "validation"]
        self.assertEqual(target, 21)
        self.assertEqual(len(train), 21)
        self.assertEqual(len(validation), 6)
        self.assertEqual(train, [r for r in records if splits[r["game_id"]] == "train"][:21])
        self.assertEqual(
            validation, [r for r in records if splits[r["game_id"]] == "validation"][:6]
        )
        # Larger windows never change either pool's assignments.
        larger, _ = select_growing_replay(records, splits, 90, 1 / 3, 0.2, "uniform")
        self.assertTrue({r["game_id"] for r in selected}.issubset(r["game_id"] for r in larger))

    def test_deduplicates_and_caps_at_available_games(self):
        records = [record(2), record(2), record(1)]
        splits = {record(2)["game_id"]: "train", record(1)["game_id"]: "validation"}
        selected, target = select_growing_replay(records, splits, 1000, 1 / 3, 0.2, "uniform")
        self.assertEqual(target, 334)
        self.assertEqual(len(selected), 2)

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
