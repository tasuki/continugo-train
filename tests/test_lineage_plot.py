import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from continugo_train import plot_ratings


class LineagePlotTests(unittest.TestCase):
    def test_portable_champions_only_across_families_and_lineages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for lineage in ("first", "second"):
                store = root / lineage
                store.mkdir(parents=True)
                (store / "champions.json").write_text(
                    json.dumps(
                        {
                            "history": [
                                {"model": name} for name in ("random", "cg9-r0060", "cg10-r0005")
                            ]
                        }
                    )
                )
                (store / "networks.csv").write_text(
                    "network,rating,deviation,games\n"
                    "random,100,0,100\ncg9-r0060,200,20,40\n"
                    "rejected-r0001,150,30,10\ncg10-r0005,300,15,40\n"
                )
                for family, round_text, positions in (
                    ("cg9", "r0060", 100),
                    ("cg10", "r0005", 200),
                ):
                    directory = store / "models" / family / round_text
                    directory.mkdir(parents=True)
                    (directory / "metadata.json").write_text(
                        json.dumps({"progress": {"data_positions": positions}})
                    )
            (root / "empty").mkdir(parents=True)
            (root / "empty/champions.json").write_text(
                json.dumps({"history": [{"model": "random"}]})
            )
            points = plot_ratings.collect_lineage_champions(root)
            self.assertEqual(len(points), 6)
            self.assertTrue(all(point.champion for point in points))
            self.assertEqual([p.positions for p in points if not p.reference], [100, 200, 100, 200])
            with patch.object(plot_ratings, "plot") as render:
                self.assertEqual(plot_ratings.main(["--artifacts-root", str(root)]), 0)
            self.assertEqual(render.call_args.args[1], root / "ratings.png")
            self.assertTrue(render.call_args.kwargs["champions_only"])


if __name__ == "__main__":
    unittest.main()
