import unittest
from unittest.mock import patch

from continugo_train.data import _subsample_plans, batches
from continugo_train.loop import LoopConfig
from continugo_train.train import Options


class PositionBudgetTests(unittest.TestCase):
    def setUp(self):
        # Unequal game lengths must not produce equal per-game quotas.
        self.plans = [("short", list(range(10))), ("long", list(range(10, 1000)))]

    def test_defaults_and_validation(self):
        self.assertEqual(LoopConfig().positions_per_epoch, 65_536)
        self.assertEqual(Options().positions_per_epoch, 0)
        for options in (Options(positions_per_epoch=-1), LoopConfig(positions_per_epoch=-1)):
            with self.assertRaises(ValueError):
                options.validate()
        options = LoopConfig().training_options({"options": {}}, 1)
        self.assertEqual(options.positions_per_epoch, 65_536)

    def test_exact_global_sample_is_reproducible_and_refreshes(self):
        selected = _subsample_plans(self.plans, 100, 42, 1)
        self.assertEqual(len(selected), 100)
        positions = [plies[0] for _, plies in selected]
        self.assertEqual(len(set(positions)), 100)
        self.assertEqual(selected, _subsample_plans(self.plans, 100, 42, 1))
        self.assertNotEqual(selected, _subsample_plans(self.plans, 100, 42, 2))
        self.assertGreater(sum(game == "long" for game, _ in selected), 80)

    def test_zero_and_small_pools_are_exhaustive(self):
        for budget in (0, 1000, 2000):
            self.assertEqual(_subsample_plans(self.plans, budget, 42, 1), self.plans)
        self.assertEqual(_subsample_plans([], 100, 42, 1), [])
        with self.assertRaises(ValueError):
            _subsample_plans(self.plans, -1, 42, 1)

    def test_preserve_order_keeps_same_membership(self):
        random_order = _subsample_plans(self.plans, 100, 42, 1)
        ordered = _subsample_plans(self.plans, 100, 42, 1, preserve_order=True)
        self.assertEqual(sorted(random_order), sorted(ordered))
        self.assertEqual([p[0] for _, p in ordered], sorted(p[0] for _, p in ordered))

    def test_only_training_is_capped_before_loading(self):
        for training, expected in ((True, 100), (False, 1000)):
            seen = []

            def reader(plans, *args, seen=seen):
                seen.extend((game, ply) for game, plies in plans for ply in plies)
                yield from ()

            with (
                patch("continugo_train.data._plans", return_value=self.plans),
                patch("continugo_train.data._parallel_samples", side_effect=reader),
            ):
                list(
                    batches(
                        [],
                        batch_size=64,
                        seed=42,
                        epoch=1,
                        training=training,
                        density=4,
                        workers=0,
                        shuffle_buffer=0,
                        position_budget=100,
                    )
                )
            self.assertEqual(len(seen), expected)


if __name__ == "__main__":
    unittest.main()
