import hashlib
import importlib.util
import json
import shutil
import subprocess
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from test_features import SGF

from continugo_train.export import export_checkpoint, runtime_path
from continugo_train.features import load_position
from continugo_train.model import NetworkConfig, PolicyValueNet
from continugo_train.train import feature_spec

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(
    importlib.util.find_spec("onnxruntime") and importlib.util.find_spec("onnx"),
    "install the inference extra",
)
class NativeInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(8)
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.root = Path(cls.directory.name)
        cls.model = PolicyValueNet(
            NetworkConfig(channels=4, blocks=1, policy_hidden=4, value_hidden=4)
        ).eval()
        cls.checkpoint = cls.root / "checkpoint.pt"
        torch.save(
            {
                "config": asdict(cls.model.config),
                "model": cls.model.state_dict(),
                "feature_spec": feature_spec(4),
                "epoch": 1,
            },
            cls.checkpoint,
        )
        cls.bundle = cls.root / "onnx"
        cls.error = export_checkpoint(cls.checkpoint, cls.bundle)
        cls.library = runtime_path()

    def infer(self, source, actions, model=None, value_only=False):
        return subprocess.run(
            [
                str(ROOT / "target/release/infer"),
                "--model",
                str(model or self.bundle),
                "--ort-library",
                str(self.library),
                "--sgf",
                str(source),
                "--ply",
                "1",
                "--actions",
                ",".join(map(str, actions)),
                *(["--value-only"] if value_only else []),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_real_full_domain_native_outputs_match_torch_across_sizes(self):
        self.assertLess(self.error, 1e-5)
        for size in [9, 13, 19]:
            source = self.root / f"size-{size}.sgf"
            source.write_text(SGF.replace("SZ[13]", f"SZ[{size}]").replace("RE[B+220.4]", ""))
            sample = load_position(source, 1)
            actions = [
                0,
                999999,
                200200,
                200200 + sample.diameter,
                200200 + sample.diameter - 1,
                500500,
                500501,
                65535,
                65536,
                900900,
                1000000,
            ]
            with torch.inference_mode():
                logits, value = self.model(*sample.network_inputs())
            native = self.infer(source, actions)
            self.assertEqual(native.returncode, 0, native.stderr)
            result = json.loads(native.stdout)
            self.assertAlmostEqual(result["value"], value.item(), places=5)
            value_only = self.infer(source, actions, value_only=True)
            self.assertEqual(value_only.returncode, 0, value_only.stderr)
            self.assertAlmostEqual(
                json.loads(value_only.stdout)["value"], result["value"], places=7
            )
            self.assertIn("calls: 0, value_calls: 1", value_only.stderr)
            for action, actual in zip(actions, result["logits"], strict=True):
                expected = logits[0, action].item()
                if not np.isfinite(expected):
                    self.assertIsNone(actual)
                else:
                    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    def test_native_search_generates_replayable_sparse_visit_targets(self):
        output = self.root / "games"
        result = subprocess.run(
            [
                str(ROOT / "target/release/continugo-train"),
                "--engine",
                "neural",
                "--model",
                str(self.bundle),
                "--ort-library",
                str(self.library),
                "--sizes",
                "9",
                "--simulations",
                "4",
                "--max-moves",
                "2",
                "--output",
                str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("calls: 2, value_calls: 8", result.stderr)
        source = next(output.glob("*.sgf"))
        # Rust replays and validates every move/policy before exposing tensors.
        sample = load_position(source, 0)
        self.assertEqual(sample.visits.sum().item(), 4)
        self.assertIsNone(sample.outcome)
        identity = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.assertIn(f"PB[neural-{identity}]", source.read_text())
        self.assertEqual(len(list(output.iterdir())), 1)  # SGF only, no sidecar.

    def test_rejects_mixed_or_incompatible_graphs_and_existing_export(self):
        import onnx

        with self.assertRaisesRegex(ValueError, "already exists"):
            export_checkpoint(self.checkpoint, self.bundle)
        bad = self.root / "bad"
        shutil.copytree(self.bundle, bad)
        source = self.root / "diagnostic.sgf"
        source.write_text(SGF)
        graph = onnx.load(bad / "encoder.onnx")
        metadata = json.loads(graph.metadata_props[0].value)
        metadata["feature_spec"]["moves"][0] = "wrong-feature"
        graph.metadata_props[0].value = json.dumps(metadata)
        onnx.save(graph, bad / "encoder.onnx")
        result = self.infer(source, [500500], bad)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mixed model", result.stderr)
        graph = onnx.load(bad / "policy.onnx")
        graph.metadata_props[0].value = json.dumps(metadata)
        onnx.save(graph, bad / "policy.onnx")
        result = self.infer(source, [500500], bad)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("incompatible feature", result.stderr)


if __name__ == "__main__":
    unittest.main()
