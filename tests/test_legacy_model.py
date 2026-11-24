import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from continugo_train.legacy_model import LegacyConfig, LegacyPolicyValueNet
from continugo_train.nn_view import load_checkpoint
from continugo_train.train import feature_spec


class LegacyModelTests(unittest.TestCase):
    def test_original_weights_load_without_migration_or_invented_uncertainty(self):
        config = LegacyConfig(
            channels=4,
            blocks=1,
            policy_hidden=4,
            value_hidden=4,
            score_hidden=4,
            ownership_hidden=4,
        )
        model = LegacyPolicyValueNet(config).eval()
        spec = {k: v for k, v in feature_spec(4, schema=6).items() if k not in ("trunk", "value")}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            torch.save(
                {"config": asdict(config), "feature_spec": spec, "model": model.state_dict()}, path
            )
            before = path.read_bytes()
            restored, density, _identity = load_checkpoint(path)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(density, 4)
            self.assertTrue(restored.legacy_separate_heads)
            self.assertEqual(restored.config.global_inputs, 0)
            coarse = torch.randn(1, 14, 9, 9)
            with torch.inference_mode():
                a = model._encode(coarse)
                b = restored._encode(coarse, torch.zeros(1))
                for expected, actual in zip(a, b, strict=True):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(restored.score_stdev(b[1]).item(), 0)
                self.assertEqual(restored.score_head(b[1]).shape, (1, 1))
                self.assertEqual(restored.ownership_head(b[0]).shape, (1, 1, 9, 9))


if __name__ == "__main__":
    unittest.main()
