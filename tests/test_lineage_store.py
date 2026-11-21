"""Lineage stores retain rating identities after model pruning."""

import json
import tempfile
import unittest
from pathlib import Path

from continugo_train.rate_saved import load_champions
from continugo_train.saved_models import saved_sources


class LineageStoreTests(unittest.TestCase):
    def test_independent_champions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = load_champions(root / "20261111")
            first["champion"] = "cg10-r0070"
            self.assertEqual(load_champions(root / "new")["champion"], "random")

    def test_pruned_models_are_history_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "ratings").mkdir()
            catalog = {"checksum": {"id": "cg5-r0001", "pruned": True}}
            (root / "ratings/pruned-models.json").write_text(json.dumps(catalog))
            self.assertEqual(saved_sources(root), {})
            self.assertEqual(saved_sources(root, include_pruned=True), catalog)


if __name__ == "__main__":
    unittest.main()
