"""Rating rebuilds must use the evaluator's frozen protocol, not new defaults."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from continugo_train import rating_store, ratings
from continugo_train.evaluation import EvaluationConfig


class FrozenRatingProtocolTests(unittest.TestCase):
    def test_refresh_preserves_v2_even_when_new_defaults_include_move_penalty(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Path(directory)
            (store / "ratings").mkdir()
            config = EvaluationConfig()
            with patch.object(ratings.archive, "digest", return_value="a" * 64):
                frozen = ratings.protocol(config, version=2)
                torch.save({
                    "rating_protocol": frozen,
                    "runtime_changes": [{"previous_protocol": frozen, "protocol": frozen}],
                }, store / "ratings/evaluator.pt")
                with (
                    patch.object(rating_store, "saved_sources", return_value={}),
                    patch("continugo_train.lineage_ratings.refresh", return_value={}),
                    patch("continugo_train.plot_ratings.update"),
                ):
                    cache, summary = rating_store.refresh(store, config, visualize=False)
                self.assertEqual(cache["protocol"], frozen)
                self.assertEqual(cache["protocols"], [frozen])
                self.assertNotIn("useless_move_penalty_points", cache["protocol"]["search"])
                self.assertEqual(summary, {})


if __name__ == "__main__":
    unittest.main()
