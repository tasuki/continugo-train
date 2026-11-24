import csv
import json
import tempfile
import unittest
from pathlib import Path

from continugo_train.champion_wins import root_result, update


class ChampionWinsTests(unittest.TestCase):
    def test_order_counts_cross_lineage_and_missing_pairs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for lineage, names in (("22", ["c", "d"]), ("11", ["a", "b"])):
                store = root / lineage
                store.mkdir()
                (store / "champions.json").write_text(json.dumps({
                    "history": [{"model": n} for n in ["random", *names]],
                }))
                (store / "sgfs/day").mkdir(parents=True)
            for seed, (black, white, result) in enumerate([
                ("a", "b", "B+1"), ("b", "a", "W+R"),
                ("b", "a", "B+2"), ("a", "b", "0"),
                ("a", "b", None), ("a", "random", "B+1"),
                ("a", "rejected", "B+1"), ("a", "c", "W+3"),
            ]):
                path = root / "11/sgfs/day" / f"{black}-vs-{white}-seed{seed}.sgf"
                path.write_text('(;C[fake RE[B+999\\] here]'
                                + (f"RE[{result}]" if result else "") + ';B[])')
            output = update(root)
            with output.open() as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
            self.assertEqual(reader.fieldnames, ["champion", "random", "a", "b", "c", "d"])
            self.assertEqual([r["champion"] for r in rows], reader.fieldnames[1:])
            self.assertEqual(rows[0]["a"], "0")
            self.assertEqual(rows[1]["random"], "1")
            self.assertEqual(rows[0]["random"], "")
            self.assertEqual(rows[1]["b"], "2")
            self.assertEqual(rows[2]["a"], "1")
            self.assertEqual(rows[1]["c"], "0")
            self.assertEqual(rows[3]["a"], "1")
            self.assertEqual(rows[1]["d"], "")
            self.assertEqual(rows[1]["a"], "")
            before = output.read_bytes()
            update(root)
            self.assertEqual(output.read_bytes(), before)

    def test_draw_only_and_truncation_only_have_zero_wins(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = root / "lineage"
            (store / "sgfs/day").mkdir(parents=True)
            (store / "champions.json").write_text(json.dumps({
                "history": [{"model": n} for n in ["a", "b", "c"]],
            }))
            for opponent, result in [("b", "RE[0]"), ("c", "")]:
                (store / f"sgfs/day/a-vs-{opponent}-seed1.sgf").write_text(f"(;{result};B[])")
            with update(root).open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[1]["b"], "0")
            self.assertEqual(rows[1]["c"], "0")

    def test_invalid_result_does_not_replace_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "game.sgf"
            path.write_text("(;RE[bad];B[])")
            with self.assertRaisesRegex(ValueError, "unrecognized SGF result"):
                root_result(path)


if __name__ == "__main__":
    unittest.main()
