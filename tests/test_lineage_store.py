"""Lineage stores retain rating identities after model pruning."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from continugo_train import saved_models
from continugo_train.rate_saved import load_champions
from continugo_train.saved_models import saved_sources


class LineageStoreTests(unittest.TestCase):
    def test_independent_champions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = load_champions(root / "20261111")
            first["champion"] = "cg10-r0070"
            self.assertEqual(load_champions(root / "new")["champion"], "random")

    def test_source_loop_found_from_store_with_frozen_package_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = "fresh-campaign"
            source = root / "runs" / run / "loop"
            source.mkdir(parents=True)
            (source / "loop.pt").touch()
            frozen_root = root / "runs" / run / "runtime"
            with patch.object(saved_models, "ROOT", frozen_root):
                for store in (root / "artifacts/20261122", root / "artifacts"):
                    with self.subTest(store=store):
                        self.assertEqual(saved_models.source_loop(store, run), source)
                self.assertIsNone(saved_models.source_loop(root / "artifacts/20261122", "missing"))

    def test_source_loop_keeps_legacy_store_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "legacy-campaign/loop"
            source.mkdir(parents=True)
            (source / "loop.pt").touch()
            with patch.object(saved_models, "ROOT", root / "frozen/runtime"):
                self.assertEqual(saved_models.source_loop(root / "models", "legacy-campaign"), source)

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
