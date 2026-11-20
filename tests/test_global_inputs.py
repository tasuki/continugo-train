"""Shape-derived global inputs survive normalization and dynamic ONNX export."""
import copy
import importlib.util
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from continugo_train.export import export_checkpoint
from continugo_train.features import FeatureClient
from continugo_train.model import NetworkConfig, PolicyValueNet
from continugo_train.train import feature_spec


def scalar_only_model():
    model = PolicyValueNet(NetworkConfig(
        channels=1, blocks=1, gpool_every=0, policy_hidden=1,
        value_hidden=1, value_channels=1,
    ))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.global_projection.weight.fill_(1)
        model.pass_head.weight.fill_(1)
    return model


class GlobalInputTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_width_is_unclipped_and_uses_actual_density_and_shape(self):
        model = scalar_only_model()
        with tempfile.TemporaryDirectory() as directory, FeatureClient() as client:
            for size, density, width in ((9, 4, 38), (13, 4, 54), (19, 4, 78), (19, 8, 155)):
                sgf = Path(directory) / f"empty-{size}-density{density}.sgf"
                sgf.write_text(f"(;FF[9071]GM[9071]SZ[{size}])")
                sample = client.load(sgf, 0, cells=[], density=density)
                self.assertEqual(sample.feature_schema, 7)
                self.assertEqual(sample.resolution, width)
                global_input = model.global_features(sample.coarse[None])
                torch.testing.assert_close(global_input, torch.tensor([[width / 100, 0.0]]))
                for training in (True, False):
                    model.train(training)
                    batch = sample.coarse[None].expand(2, -1, -1, -1)
                    _, pass_logit, *_ = model.encode_training(batch)
                    torch.testing.assert_close(pass_logit, torch.full((2, 1), width / 100))

    def test_projection_receives_gradient_in_size_homogeneous_training_batches(self):
        model = scalar_only_model().train()
        coarse = torch.zeros(2, model.config.input_planes, 38, 38)
        _, pass_logit, *_ = model.encode_training(coarse)
        pass_logit.sum().backward()
        torch.testing.assert_close(model.global_projection.weight.grad, torch.tensor([[0.76, 0.0]]))

    def test_global_conditioning_reaches_every_head_and_can_change_local_preferences(self):
        model = PolicyValueNet(NetworkConfig(channels=8, blocks=1, gpool_every=0)).eval()
        coarse = torch.rand(2, model.config.input_planes, 38, 38)
        before = model.encode_training(coarse)
        # Keep spatial data fixed; alter only the global-context projection.
        with torch.no_grad():
            model.global_projection.weight.zero_()
        after = model.encode_training(coarse)
        for left, right in zip(before, after, strict=True):
            self.assertFalse(torch.equal(left, right))
        cell_logits_before = model.cell_head(before[0]).flatten(1)
        cell_logits_after = model.cell_head(after[0]).flatten(1)
        logits_before = model.coarse_logits(before[0], before[1])
        logits_after = model.coarse_logits(after[0], after[1])
        self.assertFalse(torch.equal(logits_before.softmax(-1), logits_after.softmax(-1)))
        self.assertFalse(torch.equal(cell_logits_before.softmax(-1), cell_logits_after.softmax(-1)))

    def test_legacy_configs_have_no_new_weights_and_metadata_declares_globals(self):
        old = asdict(NetworkConfig(input_planes=14))
        old.pop("global_inputs")
        legacy = PolicyValueNet(NetworkConfig.from_checkpoint(old))
        self.assertIsNone(legacy.global_projection)
        self.assertFalse(any(k.startswith("global_projection.") for k in legacy.state_dict()))
        self.assertEqual(feature_spec(4)["globals"], ["coarse_width_div100", "komi_self_div_board_area"])
        for schema in (4, 5, 6):
            self.assertNotIn("globals", feature_spec(4, schema=schema))
            self.assertEqual(feature_spec(4, schema=schema)["trunk"], "global-pooling-v1")

    @unittest.skipUnless(importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime"), "install inference extra")
    def test_onnx_width_remains_dynamic_and_exceeds_one_and_bad_metadata_is_rejected(self):
        import onnxruntime as ort
        model = scalar_only_model().eval()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            saved = {"config": asdict(model.config), "model": model.state_dict(),
                     "feature_spec": feature_spec(4), "epoch": 1}
            torch.save(saved, checkpoint)
            export_checkpoint(checkpoint, root / "onnx")
            settings = ort.SessionOptions()
            settings.intra_op_num_threads = 1
            session = ort.InferenceSession(str(root / "onnx/encoder.onnx"), settings,
                                          providers=["CPUExecutionProvider"])
            for width in (38, 54, 78, 128):
                coarse = np.zeros((2, model.config.input_planes, width, width), dtype=np.float32)
                result = session.run(["pass_logit"], {"coarse": coarse, "komi": np.zeros(2, dtype=np.float32)})[0]
                np.testing.assert_allclose(result, width / 100, rtol=1e-4, atol=1e-5)
            bad = copy.deepcopy(saved)
            bad["feature_spec"]["globals"] = ["coarse_width_div50"]
            torch.save(bad, root / "bad.pt")
            with self.assertRaisesRegex(ValueError, "incompatible features"):
                export_checkpoint(root / "bad.pt", root / "bad-export")
            bad = copy.deepcopy(saved)
            bad["config"]["global_inputs"] = 0
            torch.save(bad, root / "wrong-config.pt")
            with self.assertRaisesRegex(ValueError, "architecture"):
                export_checkpoint(root / "wrong-config.pt", root / "wrong-export")
