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
from continugo_train.runtime import gpu_library, gpu_root, inference_environment
from continugo_train.train import feature_spec

ROOT = Path(__file__).resolve().parents[1]
GPU_READY = (gpu_root() / "nvidia/cudnn/lib/libcudnn.so.8").is_file() and torch.cuda.is_available()


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

    def infer(
        self,
        source,
        actions,
        model=None,
        value_only=False,
        device="cpu",
        library=None,
        environment=None,
        extra=(),
    ):
        return subprocess.run(
            [
                str(ROOT / "target/release/infer"),
                "--model",
                str(model or self.bundle),
                "--ort-library",
                str(library or self.library),
                "--device",
                device,
                "--sgf",
                str(source),
                "--ply",
                "1",
                "--actions",
                ",".join(map(str, actions)),
                *(["--value-only"] if value_only else []),
                *extra,
            ],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_cached_parallel_queries_match_uncached_serial_results(self):
        source = self.root / "cache.sgf"
        source.write_text(SGF)
        common = ["--iterations", "3"]
        actions = [0, 200200, 500500, 500501, 65535, 65536, 900900, 1000000]
        serial = self.infer(
            source, actions, extra=[*common, "--feature-threads", "1", "--cache-positions", "0"]
        )
        cached = self.infer(
            source, actions, extra=[*common, "--feature-threads", "4", "--cache-positions", "2"]
        )
        self.assertEqual(serial.returncode, 0, serial.stderr)
        self.assertEqual(cached.returncode, 0, cached.stderr)
        self.assertEqual(json.loads(serial.stdout), json.loads(cached.stdout))
        self.assertIn("encoder_positions: 3", serial.stderr)
        self.assertIn("cache_hits: 0", serial.stderr)
        self.assertIn("encoder_positions: 1", cached.stderr)
        self.assertIn("cache_hits: 2", cached.stderr)
        self.assertIn("cache_misses: 1", cached.stderr)
        value = self.infer(
            source, actions, value_only=True, extra=[*common, "--cache-positions", "2"]
        )
        self.assertEqual(value.returncode, 0, value.stderr)
        self.assertIn("encoder_positions: 1", value.stderr)
        self.assertIn("cache_hits: 2", value.stderr)
        self.assertAlmostEqual(
            json.loads(value.stdout)["value"], json.loads(cached.stdout)["value"], places=7
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

    def check_search(self, device="cpu", library=None, environment=None):
        output = self.root / f"games-{device}"
        result = subprocess.run(
            [
                str(ROOT / "target/release/continugo-train"),
                "--engine",
                "neural",
                "--model",
                str(self.bundle),
                "--ort-library",
                str(library or self.library),
                "--device",
                device,
                "--sizes",
                "9",
                "--simulations",
                "4",
                "--max-moves",
                "2",
                "--output",
                str(output),
            ],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("calls: 2, value_calls: 8", result.stderr)
        # The chosen first move was already a value-only leaf: its next root
        # reuses that context rather than rebuilding coarse facts/encoding.
        self.assertIn("encoder_positions: 9", result.stderr)
        self.assertIn("cache_hits: 1", result.stderr)
        source = next(output.glob("*.sgf"))
        # Rust replays and validates every move/policy before exposing tensors.
        sample = load_position(source, 0)
        self.assertEqual(sample.visits.sum().item(), 4)
        self.assertIsNone(sample.outcome)
        identity = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.assertIn(f"PB[neural-{identity}]", source.read_text())
        self.assertEqual(len(list(output.iterdir())), 1)  # SGF only, no sidecar.

    def test_native_search_generates_replayable_sparse_visit_targets(self):
        self.check_search()

    def test_cuda_request_fails_instead_of_falling_back_on_cpu_runtime(self):
        source = self.root / "no-cuda.sgf"
        source.write_text(SGF)
        result = self.infer(source, [500500], device="cuda")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no CUDA execution provider", result.stderr)

    @unittest.skipUnless(GPU_READY, "requires setup-gpu and an NVIDIA GPU")
    def test_cuda_matches_cpu_and_runs_heavy_operators_on_gpu(self):
        library, environment = inference_environment("cuda", gpu_library())
        for size in [9, 13, 19]:
            source = self.root / f"cuda-{size}.sgf"
            source.write_text(SGF.replace("SZ[13]", f"SZ[{size}]").replace("RE[B+220.4]", ""))
            profile = self.root / f"profile-{size}"
            profile.mkdir()
            env = dict(environment, CONTINUGO_ORT_PROFILE=str(profile))
            actions = [0, 200200, 500500, 500501, 65535, 65536, 900900, 1000000]
            cpu = self.infer(source, actions)
            gpu = self.infer(source, actions, device="cuda", library=library, environment=env)
            self.assertEqual(cpu.returncode, 0, cpu.stderr)
            self.assertEqual(gpu.returncode, 0, gpu.stderr)
            expected, actual = json.loads(cpu.stdout), json.loads(gpu.stdout)
            self.assertEqual(actual["device"], "cuda")
            np.testing.assert_allclose(actual["value"], expected["value"], rtol=1e-4, atol=1e-5)
            for left, right in zip(expected["logits"], actual["logits"], strict=True):
                if left is None:
                    self.assertIsNone(right)
                else:
                    np.testing.assert_allclose(right, left, rtol=1e-4, atol=1e-5)
            value_only = self.infer(
                source,
                actions,
                value_only=True,
                device="cuda",
                library=library,
                environment=environment,
            )
            self.assertEqual(value_only.returncode, 0, value_only.stderr)
            np.testing.assert_allclose(
                json.loads(value_only.stdout)["value"], actual["value"], rtol=1e-5, atol=1e-6
            )
            events = [
                event for path in profile.glob("*.json") for event in json.loads(path.read_text())
            ]
            compute = [
                event["args"]
                for event in events
                if event.get("args", {}).get("provider")
                and event["args"].get("op_name")
                in {"Conv", "FusedConv", "Gemm", "MatMul", "GridSample"}
            ]
            self.assertTrue(compute)
            self.assertTrue(any(event["op_name"] == "GridSample" for event in compute))
            self.assertTrue(
                all(event["provider"] == "CUDAExecutionProvider" for event in compute), compute
            )

    @unittest.skipUnless(GPU_READY, "requires setup-gpu and an NVIDIA GPU")
    def test_cuda_selfplay_records_are_replayable(self):
        library, environment = inference_environment("cuda", gpu_library())
        self.check_search("cuda", library, environment)

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
