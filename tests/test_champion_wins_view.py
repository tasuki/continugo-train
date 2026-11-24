import csv
import json
import tempfile
import unittest
from pathlib import Path

from continugo_train.champion_wins_view import generate, percentage_style


class ChampionWinsViewTests(unittest.TestCase):
    def test_counts_mirror_tooltips_black_diagonal_and_safe_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "champion-wins.csv"
            names = ["random", 'champion <script> & "x"', "unplayed"]
            with path.open("w", newline="") as stream:
                csv.writer(stream).writerows(
                    [
                        ["champion", *names],
                        [names[0], "", "3", ""],
                        [names[1], "12", "", "0"],
                        [names[2], "", "0", ""],
                    ]
                )
            output = generate(path)
            page = output.read_text()
            self.assertEqual(output, path.with_suffix(".html"))
            self.assertIn(
                "random wins: 3 · champion &lt;script&gt; &amp; &quot;x&quot; wins: 12", page
            )
            self.assertIn("unplayed wins: 0", page)
            self.assertIn(">20.0%</td>", page)
            self.assertIn(">80.0%</td>", page)
            self.assertIn("of 15 decisive games", page)
            self.assertIn("percentage undefined", page)
            self.assertIn("excluding draws and unfinished games", page)
            self.assertIn("no recorded pairing", page)
            self.assertEqual(page.count('class="diagonal"'), 3)
            self.assertEqual(page.count('tabindex="0"'), 6)
            self.assertIn("td.diagonal{background:#000", page)
            self.assertIn("mirror.classList.add('mirror')", page)
            self.assertNotIn("champion <script>", page)
            self.assertNotIn("<script src=", page)
            self.assertNotIn("<link ", page)
            self.assertEqual(output.read_bytes(), generate(path).read_bytes())

    def test_piyg_scale_has_neutral_midpoint_and_opposing_endpoints(self):
        self.assertEqual(percentage_style(0), "background:rgb(142,1,82);color:#fff")
        self.assertEqual(percentage_style(0.5), "background:rgb(247,247,247);color:#111")
        self.assertEqual(percentage_style(1), "background:rgb(39,100,25);color:#fff")

    def test_compact_labels_and_real_lineage_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = ["random", "cg5-r0011", "cg10-r0070", "cg11-r0055"]
            for lineage, models in [("11", names[1:3]), ("22", names[3:])]:
                directory = root / lineage
                directory.mkdir()
                (directory / "champions.json").write_text(
                    json.dumps(
                        {
                            "history": [{"model": name} for name in ["random", *models]],
                        }
                    )
                )
            path = root / "champion-wins.csv"
            with path.open("w", newline="") as stream:
                csv.writer(stream).writerows(
                    [
                        ["champion", *names],
                        *[[name, "", "", "", ""] for name in names],
                    ]
                )
            page = generate(path).read_text()
            self.assertIn('<th scope="col">cg11-r55</th>', page)
            self.assertIn('<th scope="row">cg10-r70</th>', page)
            self.assertIn("cg11-r0055: no recorded pairing", page)
            self.assertIn("tr > :nth-child(5){border-left:3px", page)
            self.assertIn("tbody tr:nth-child(4) > *{border-top:3px", page)
            self.assertNotIn("tr > :nth-child(4){border-left:3px", page)
            self.assertNotIn("tbody tr:nth-child(3) > *{border-top:3px", page)

    def test_invalid_matrix_does_not_replace_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "champion-wins.csv"
            output = path.with_suffix(".html")
            output.write_text("previous report")
            for data in ["champion,a,b\na,,1\n", "champion,a\na,<script>\n"]:
                path.write_text(data)
                with self.assertRaises(ValueError):
                    generate(path)
                self.assertEqual(output.read_text(), "previous report")


if __name__ == "__main__":
    unittest.main()
