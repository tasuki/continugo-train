import csv
import importlib.util
import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch

from continugo_train import archive, plot_ratings
from continugo_train.plot_ratings import collect_points, main, plot, training_games
from continugo_train.ratings import REFERENCE_ID


class PlotRatingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def table(self, **overrides):
        path = self.root / "ratings" / "networks.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = self.root / "fixture.pt"
        history = {f"{i:064x}": "train" if i <= 120 else "validation" for i in range(1, 151)}
        torch.save(
            {
                "split_history": history,
                "manifest": [
                    {"game_id": f"{i:064x}", "split": history[f"{i:064x}"]} for i in (120, 150)
                ],
            },
            checkpoint,
        )
        checksum = archive.digest(checkpoint)
        bundle = self.root / "training" / checksum
        bundle.mkdir(parents=True, exist_ok=True)
        checkpoint.replace(bundle / "checkpoint.pt")
        torch.save({"kind": "training", "checkpoint_sha256": checksum}, bundle / "recovery.pt")
        row = {
            "network_id": checksum,
            "network": "test-net",
            "status": "rated",
            "games": "20",
            "rating": "150",
            "deviation": "30",
        }
        row.update(overrides)
        rows = [
            row,
            {**row, "network_id": "c" * 64, "status": "unrated", "rating": "", "deviation": ""},
            {
                **row,
                "network_id": REFERENCE_ID,
                "network": REFERENCE_ID,
                "status": "fixed",
                "games": "1000",
                "rating": "100",
                "deviation": "0",
            },
        ]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_rated_only_training_ledger_not_eval_games_and_reference_at_zero(self):
        path = self.table()
        before = path.read_bytes()
        points = collect_points(self.root)
        self.assertEqual(len(points), 2)
        self.assertEqual(sorted(p.games for p in points), [0, 120])
        self.assertEqual(before, path.read_bytes())

    def test_champion_lineage_and_unrated_ladder_estimates_are_preserved(self):
        self.table()
        checksum = next((self.root / "training").iterdir()).name
        uncertain = "c" * 64
        (self.root / "ratings/champions.json").write_text(
            json.dumps(
                {
                    "cohorts": {
                        "test": {
                            "history": [{"checkpoint": REFERENCE_ID}, {"checkpoint": checksum}],
                            "candidates": [checksum, uncertain],
                            "evaluated": {},
                        }
                    }
                }
            )
        )
        directory = self.root / "ratings/inference"
        directory.mkdir()
        torch.save(
            {"checkpoint_sha256": uncertain, "training_games": 140}, directory / f"{uncertain}.pt"
        )

        class Prior:
            def __init__(self):
                self.networks = {uncertain: {}}

            def state(self, _network):
                return {"rating": 100.0, "deviation": 350.0}

        @contextmanager
        def transaction(_store):
            yield Prior(), None

        with patch.object(plot_ratings, "transaction", transaction):
            points = collect_points(self.root)
        champion = next(
            point for point in points if point.network == "test-net" and point.games == 120
        )
        estimate = next(point for point in points if point.games == 140)
        self.assertTrue(champion.champion)
        self.assertFalse(estimate.champion)
        self.assertTrue(estimate.estimated)
        self.assertEqual((estimate.rating, estimate.deviation), (100, 350))

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib not installed")
    def test_100_connected_points_with_muted_rejections_and_full_champions(self):
        from matplotlib.collections import PathCollection

        points = [
            plot_ratings.Point(
                str(i), i * 20, 100 + i, 100 / i, False, "campaign", i, champion=i % 10 == 0
            )
            for i in range(1, 101)
        ]
        points.append(plot_ratings.Point("random", 0, 100, 0, True))
        save = plot_ratings.save_figure

        def inspect(plt, figure, target):
            axis = figure.axes[0]
            dots = [item for item in axis.collections if isinstance(item, PathCollection)]
            self.assertEqual([len(item.get_offsets()) for item in dots], [10, 90, 1])
            self.assertNotEqual(
                list(dots[0].get_facecolors()[0]), list(dots[1].get_facecolors()[0])
            )
            from matplotlib.markers import MarkerStyle

            circle = MarkerStyle("o")
            for collection, marker, size in [(dots[0], circle, 32), (dots[1], circle, 32)]:
                expected = marker.get_path().transformed(marker.get_transform())
                self.assertTrue((collection.get_paths()[0].vertices == expected.vertices).all())
                self.assertEqual(list(collection.get_sizes()), [size])
            self.assertTrue(
                (dots[0].get_facecolors()[0] == axis.containers[0].lines[0].get_color()).all()
            )
            self.assertIsNone(axis.get_legend())
            self.assertEqual(dots[0].get_alpha(), 1.0)
            self.assertEqual(dots[1].get_alpha(), 0.4)
            for bars, dots_group in zip(axis.containers[1:3], dots[:2]):
                bar_color = bars.lines[2][0].get_colors()[0]
                self.assertTrue((bar_color == dots_group.get_facecolors()[0]).all())
            self.assertEqual(len(axis.containers[0].lines[0].get_xdata()), 100)
            self.assertEqual(axis.get_xscale(), "linear")
            save(plt, figure, target)

        with patch.object(plot_ratings, "save_figure", inspect):
            plot(points, self.root / "hundred.png")

    def test_missing_and_invalid_inputs(self):
        with self.assertRaises(OSError):
            collect_points(self.root)
        for field, value in [("rating", "nan"), ("deviation", "-1")]:
            self.table(**{field: value})
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "invalid"):
                collect_points(self.root)

    def test_missing_checkpoint_is_not_replaced_by_eval_count(self):
        self.table(network_id="b" * 64)
        with self.assertRaisesRegex(ValueError, "no archived checkpoint"):
            collect_points(self.root)

    def test_checkpoint_identity_is_verified(self):
        self.table()
        checkpoint = next((self.root / "training").glob("*/checkpoint.pt"))
        network_id = checkpoint.parent.name
        checkpoint.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            training_games(self.root, network_id)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib not installed")
    def test_plot_and_cli(self):
        self.table()
        points = collect_points(self.root)
        output = self.root / "plots" / "ratings.png"
        save = plot_ratings.save_figure

        def inspect(plt, figure, target):
            self.assertEqual(len(figure.axes), 1)
            axis = figure.axes[0]
            self.assertEqual(axis.get_xscale(), "linear")
            self.assertLess(axis.get_xlim()[0], 0)
            self.assertGreater(axis.get_xlim()[1], max(p.games for p in points))
            self.assertIn(0, axis.get_xticks())
            save(plt, figure, target)

        with patch.object(plot_ratings, "save_figure", inspect):
            plot(points, output, labels=True)
        self.assertGreater(output.stat().st_size, 1000)
        self.assertEqual(main(["--model-store", str(self.root), "--output", str(output)]), 0)
        self.assertEqual(main(["--model-store", str(self.root / "missing")]), 1)
        plot_ratings.update(self.root)
        self.assertTrue((self.root / "ratings/ratings.png").is_file())

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib not installed")
    def test_plot_accepts_networks_below_100_games(self):
        points = [
            plot_ratings.Point("small-net", 50, 150, 30, False),
            plot_ratings.Point("random", 0, 100, 0, True),
        ]
        output = self.root / "small.png"
        plot(points, output)
        self.assertGreater(output.stat().st_size, 1000)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib not installed")
    def test_campaigns_have_distinct_colors_and_lines_follow_round_order(self):
        points = [
            plot_ratings.Point("a20", 200, 160, 20, False, "campaign-a", 20),
            plot_ratings.Point("b10", 50, 120, 30, False, "campaign-b", 10),
            plot_ratings.Point("a10", 100, 140, 25, False, "campaign-a", 10),
            plot_ratings.Point("b20", 150, 150, 20, False, "campaign-b", 20),
            plot_ratings.Point("random", 0, 100, 0, True),
        ]
        save = plot_ratings.save_figure

        def inspect(plt, figure, target):
            curves = {
                item.get_label(): item.lines[0]
                for item in figure.axes[0].containers if item.lines[0] is not None
            }
            self.assertEqual(set(curves), {"campaign-a", "campaign-b"})
            self.assertEqual(list(curves["campaign-a"].get_xdata()), [100, 200])
            self.assertEqual(list(curves["campaign-b"].get_xdata()), [50, 150])
            self.assertEqual(curves["campaign-a"].get_linestyle(), "-")
            self.assertNotEqual(curves["campaign-a"].get_color(), curves["campaign-b"].get_color())
            save(plt, figure, target)

        with patch.object(plot_ratings, "save_figure", inspect):
            plot(points, self.root / "campaigns.png")

    def test_campaign_labels_include_distinct_network_families(self):
        def point(name):
            return plot_ratings.Point(name, 100, 150, 30, False)

        self.assertEqual(
            plot_ratings.campaign_label("campaign", [point("cg4-b4c32-s10-d20")]),
            "cg4 · campaign",
        )
        self.assertEqual(
            plot_ratings.campaign_label("campaign", [point("cg1-b4c32"), point("cg1-b4c64")]),
            "cg1 · campaign",
        )
        self.assertEqual(
            plot_ratings.campaign_label("campaign", [point("cg10-b4c32"), point("cg2-b4c32")]),
            "cg2/cg10 · campaign",
        )
        self.assertEqual(plot_ratings.campaign_label("campaign", [point("custom-net")]), "campaign")

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib not installed")
    def test_plot_omits_legend(self):
        points = [
            plot_ratings.Point("cg4-b4c32-s10-d20", 100, 150, 30, False, "score-campaign"),
            plot_ratings.Point("random", 0, 100, 0, True),
        ]
        save = plot_ratings.save_figure

        def inspect(plt, figure, target):
            self.assertIsNone(figure.axes[0].get_legend())
            save(plt, figure, target)

        with patch.object(plot_ratings, "save_figure", inspect):
            plot(points, self.root / "families.png")

    def test_auto_plot_failure_cannot_abort_learning(self):
        with patch.object(plot_ratings, "collect_points", side_effect=RuntimeError("plot failed")):
            plot_ratings.update(self.root)


if __name__ == "__main__":
    unittest.main()
