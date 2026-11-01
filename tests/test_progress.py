import unittest
from types import SimpleNamespace

from continugo_train.progress import checkpoint_progress, data_history, initialize, network_name


def game(identity, positions):
    return SimpleNamespace(game_id=identity, plies=tuple(range(positions)))


class ProgressTests(unittest.TestCase):
    def test_new_data_imports_resume_and_orchestrated_generation_total(self):
        games = [game("a", 2), game("b", 3)]
        progress = initialize(None, games, {}, family="cg2")
        self.assertEqual(progress, {"family": "cg2", "samples_seen": 0, "data_positions": 5})
        progress["samples_seen"] = 123
        saved = {"progress": progress, "config": {"blocks": 4, "channels": 32}}
        self.assertEqual(network_name(saved), "cg2-b4c32-s123-d5")
        history = {"a": "train", "b": "validation"}
        window = [game("a", 2), game("c", 4)]
        warmed = initialize(saved, window, history)
        self.assertEqual(warmed["samples_seen"], 123)
        self.assertEqual(warmed["data_positions"], 9)
        # Exact generation includes completed games that replay selection dropped.
        generated = initialize(saved, window, history, data_positions=100)
        self.assertEqual(generated["data_positions"], 100)
        self.assertEqual(initialize(saved, games, history, resuming=True), progress)
        self.assertEqual(progress["data_positions"], 5)  # Never mutate the parent.
        for kwargs in [
            {"data_positions": 8},
            {"resuming": True, "data_positions": 6},
            {"resuming": True, "family": "renamed"},
        ]:
            with self.assertRaises(ValueError):
                initialize(saved, window, history, **kwargs)

    def test_generated_but_untrained_data_is_not_imported_twice(self):
        trained, dropped, new = "a" * 64, "b" * 64, "c" * 64
        saved = {
            "progress": {"family": "cg1", "samples_seen": 1, "data_positions": 60},
            "data_game_ids": [trained, dropped],
        }
        known = data_history(saved, {trained: "train"})
        imported = initialize(saved, [game(dropped, 40), game(new, 30)], known)
        self.assertEqual(imported["data_positions"], 90)
        with self.assertRaises(ValueError):
            data_history({"data_game_ids": ["not-a-game-id"]}, {})

    def test_legacy_totals_stay_unknown_instead_of_resetting_to_zero(self):
        saved = {"config": {"blocks": 1, "channels": 4}, "global_step": 100}
        self.assertEqual(network_name(saved), "cg1-b1c4-sunknown-dunknown")
        continued = initialize(saved, [game("new", 100)], {})
        self.assertIsNone(continued["samples_seen"])
        self.assertIsNone(continued["data_positions"])
        with self.assertRaisesRegex(ValueError, "unknown ancestral"):
            initialize(saved, [], {}, data_positions=100)

    def test_invalid_counter_or_family_metadata_is_rejected(self):
        good = {"family": "cg1", "samples_seen": 0, "data_positions": 0}
        for changes in [
            {"family": "../escape"},
            {"family": 1},
            {"samples_seen": True},
            {"data_positions": -1},
            {"samples_seen": 1.5},
            {"samples_seen": 2**63},
        ]:
            with self.assertRaises(ValueError):
                checkpoint_progress({"progress": good | changes})
        with self.assertRaises(ValueError):
            checkpoint_progress({"progress": {}})
