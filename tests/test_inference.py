import hashlib
import importlib.util
import json
import os
import re
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
from continugo_train.model import NetworkConfig, PolicyValueNet, cell_actions, cell_ids
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

    def test_structured_profile_excludes_warmup_and_preserves_outputs(self):
        source = self.root / "profile.sgf"
        source.write_text(SGF)
        for value_only in (False, True):
            result = self.infer(
                source,
                [1000000],
                value_only=value_only,
                extra=[
                    "--profile-json",
                    "--cache-positions",
                    "0",
                    "--warmup",
                    "1",
                    "--iterations",
                    "2",
                ],
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["iterations"], 2)
            self.assertGreater(report["seconds"], 0)
            self.assertEqual(report["stats"]["encoder_positions"], 2)
            self.assertEqual(report["stats"]["value_calls" if value_only else "calls"], 2)
            self.assertEqual(report["stats"]["cache_hits"], 0)
            stages = report["stats"]["coarse_stages"]
            self.assertEqual(stages["positions"], 2)
            self.assertEqual(stages["preparation"]["calls"], 2)
            self.assertNotIn("group_construction", stages)
            self.assertNotIn("spans_and_capturability", stages)
            self.assertEqual(stages["liberty_spans"]["calls"], stages["groups"])
            self.assertEqual(stages["capturability"]["calls"], stages["groups"])
            self.assertEqual(stages["liberty_area"]["calls"], stages["groups"])
            if value_only:
                self.assertEqual(report["stats"]["scorer_batches"], 0)
            plain = self.infer(source, [1000000], value_only=value_only)
            for key in ("iterations", "seconds", "stats"):
                report.pop(key)
            self.assertEqual(report, json.loads(plain.stdout))
            cached = self.infer(
                source,
                [1000000],
                value_only=value_only,
                extra=[
                    "--profile-json",
                    "--cache-positions",
                    "2",
                    "--warmup",
                    "1",
                    "--iterations",
                    "2",
                ],
            )
            self.assertEqual(cached.returncode, 0, cached.stderr)
            stats = json.loads(cached.stdout)["stats"]
            self.assertEqual(stats["cache_hits"], 2)
            self.assertEqual(stats["coarse_stages"]["positions"], 0)
            self.assertEqual(stats["coarse_seconds"], 0)

    def test_real_selected_cell_native_outputs_match_torch_across_sizes(self):
        self.assertLess(self.error, 1e-5)
        for size in [9, 13, 19]:
            source = self.root / f"size-{size}.sgf"
            source.write_text(SGF.replace("SZ[13]", f"SZ[{size}]").replace("RE[B+220.4]", ""))
            # Dense extraction is a diagnostic oracle only; production selects cells.
            sample = load_position(source, 1, full_domain=True)
            preview = self.infer(source, [1000000])
            self.assertEqual(preview.returncode, 0, preview.stderr)
            selected = json.loads(preview.stdout)["cells"]
            self.assertTrue(selected)
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
            for cell in selected:
                coordinates = cell_actions([cell], sample.resolution)
                legal = coordinates[sample.legal[coordinates]]
                self.assertGreater(legal.numel(), 0)
                actions += legal[[0, legal.numel() // 2, -1]].tolist()
            with torch.inference_mode():
                logits, value = self.model(*sample.network_inputs())
                spatial, pass_logit, _ = self.model.encode(sample.coarse[None])
                coarse_logp = self.model.coarse_logits(spatial, pass_logit).log_softmax(dim=1)
                # Native inference preserves the global placement/pass strata,
                # then conditions only the placement distribution on legal cells.
                placement_shift = coarse_logp[0, :-1].logsumexp(0) - coarse_logp[
                    0, selected
                ].logsumexp(0)
            native = self.infer(source, actions)
            self.assertEqual(native.returncode, 0, native.stderr)
            result = json.loads(native.stdout)
            self.assertEqual(result["cells"], selected)
            self.assertAlmostEqual(result["policy_mass"], 1.0, places=5)
            self.assertAlmostEqual(
                result["logits"][actions.index(1000000)], coarse_logp[0, -1].item(), places=5
            )
            self.assertAlmostEqual(result["value"], value.item(), places=5)
            value_only = self.infer(source, actions, value_only=True)
            self.assertEqual(value_only.returncode, 0, value_only.stderr)
            self.assertAlmostEqual(
                json.loads(value_only.stdout)["value"], result["value"], places=7
            )
            self.assertIn("calls: 0, value_calls: 1", value_only.stderr)
            for action, actual in zip(actions, result["logits"], strict=True):
                expected = logits[0, action].item()
                if action < 1000000:
                    expected += placement_shift.item()
                cell = (
                    cell_ids(torch.tensor(action), sample.resolution).item()
                    if action < 1000000
                    else None
                )
                if not np.isfinite(expected) or (action < 1000000 and cell not in selected):
                    self.assertIsNone(actual)
                else:
                    self.assertIsNotNone(actual)
                    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
            self.assertLess(int(re.search(r"exact_coordinates: (\d+)", native.stderr)[1]), 50000)

    def test_pass_mass_does_not_depend_on_cell_budget_and_all_illegal_falls_back(self):
        source = self.root / "pass-mass.sgf"
        source.write_text(SGF.replace("SZ[13]", "SZ[9]").replace("RE[B+220.4]", ""))
        for top, exploration in [(1, 1), (32, 8)]:
            model = PolicyValueNet(
                NetworkConfig(
                    channels=4,
                    blocks=1,
                    policy_hidden=4,
                    value_hidden=4,
                    top_cells=top,
                    exploration_cells=exploration,
                )
            ).eval()
            for parameter in model.parameters():
                parameter.data.zero_()
            # Uniform 1444-cell opening, with exactly ten percent global pass.
            model.pass_head.bias.data.fill_(np.log(1444 / 9))
            checkpoint = self.root / f"mass-{top}.pt"
            torch.save(
                {
                    "config": asdict(model.config),
                    "model": model.state_dict(),
                    "feature_spec": feature_spec(4),
                    "epoch": 1,
                },
                checkpoint,
            )
            bundle = self.root / f"mass-{top}"
            export_checkpoint(checkpoint, bundle)
            result = self.infer(source, [1000000], model=bundle)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertTrue(report["cells"])
            self.assertAlmostEqual(np.exp(report["logits"][0]), 0.1, places=6)
            self.assertAlmostEqual(report["policy_mass"], 1.0, places=6)
            # Size one has no legal placements, even though the coarse head
            # assigns some mass to placements. Fallback must be exactly pass.
            impossible = self.root / "no-placement.sgf"
            impossible.write_text("(;FF[9071]GM[9071]SZ[1];B[])")
            result = self.infer(impossible, [1000000], model=bundle)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["cells"], [])
            self.assertEqual(report["logits"], [0.0])
            self.assertEqual(report["policy_mass"], 1.0)

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
        self.assertIn("requests: 0", result.stderr)  # One actual game bypasses the queue.
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
        self.assertIn('"model":"' + identity + '"', source.read_text())
        saved = torch.load(self.checkpoint, weights_only=True)
        from continugo_train.progress import network_name

        self.assertIn(f"PB[neural-{network_name(saved)}]", source.read_text())
        self.assertEqual(len(list(output.iterdir())), 1)  # SGF only, no sidecar.

    def test_native_search_generates_replayable_sparse_visit_targets(self):
        self.check_search()

    def check_batches(self, device="cpu", library=None, environment=None):
        environment = dict(environment or os.environ)
        environment.update(
            CONTINUGO_TEST_MODEL=str(self.bundle),
            CONTINUGO_TEST_DEVICE=device,
            ORT_DYLIB_PATH=str(library or self.library),
        )
        result = subprocess.run(
            [
                "cargo",
                "test",
                "--locked",
                "--release",
                "--test",
                "batching",
                "--",
                "--ignored",
                "--nocapture",
            ],
            check=False,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_native_batched_outputs_and_worker_lifecycle(self):
        self.check_batches()

    @unittest.skipUnless(GPU_READY, "requires setup-gpu and an NVIDIA GPU")
    def test_cuda_batched_outputs_and_worker_lifecycle(self):
        library, environment = inference_environment("cuda", gpu_library())
        self.check_batches("cuda", library, environment)

    def test_concurrent_games_keep_stable_seeds_and_replayable_records(self):
        def run(name, concurrency, batch):
            output = self.root / name
            result = subprocess.run(
                [
                    str(ROOT / "target/release/continugo-train"),
                    "--engine",
                    "neural",
                    "--model",
                    str(self.bundle),
                    "--ort-library",
                    str(self.library),
                    "--device",
                    "cpu",
                    "--games",
                    "7",
                    "--sizes",
                    "9,13,19",
                    "--seed",
                    "300",
                    "--simulations",
                    "4",
                    "--max-moves",
                    "3",
                    *(["--concurrent-games", str(concurrency)] if concurrency is not None else []),
                    *(["--inference-batch-size", str(batch)] if batch is not None else []),
                    "--inference-batch-wait-ms",
                    "20",
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            files = {path.name: path.read_text() for path in output.glob("*.sgf")}
            self.assertEqual(len(files), 7)
            self.assertEqual(len(list(output.iterdir())), 7)
            return output, files, result.stderr

        _, sequential, _ = run("sequential", 1, 1)
        _, concurrent, _ = run("concurrent-single", 6, 1)
        self.assertEqual(sequential, concurrent)
        output, batched, stats = run("concurrent-batched", 6, 6)
        self.assertEqual(sequential.keys(), batched.keys())
        self.assertGreater(int(re.search(r"scorer_max_batch: (\d+)", stats)[1]), 1)
        default_output, default_records, default_stats = run("concurrent-defaults", None, None)
        self.assertEqual(sequential.keys(), default_records.keys())
        self.assertIn("concurrent_games=12 batch_size=12", default_stats)
        self.assertGreater(int(re.search(r"scorer_max_batch: (\d+)", default_stats)[1]), 1)
        for folder, records in ((output, batched), (default_output, default_records)):
            for name in records:
                sample = load_position(folder / name, 0)
                self.assertEqual(sample.visits.sum().item(), 4)
        # Output failure cancels new games but drains in-flight queries and joins.
        failure = subprocess.run(
            [
                str(ROOT / "target/release/continugo-train"),
                "--engine",
                "neural",
                "--model",
                str(self.bundle),
                "--ort-library",
                str(self.library),
                "--device",
                "cpu",
                "--games",
                "7",
                "--sizes",
                "9,13,19",
                "--seed",
                "300",
                "--simulations",
                "1",
                "--max-moves",
                "1",
                "--concurrent-games",
                "4",
                "--output",
                str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertNotEqual(failure.returncode, 0)
        self.assertIn("exists", failure.stderr.lower())

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
        # Old dense bundles have no cell-policy version: never silently load them.
        metadata["feature_spec"]["moves"][0] = "legal"
        del metadata["feature_spec"]["policy"]
        for name in ["encoder.onnx", "policy.onnx"]:
            graph = onnx.load(bad / name)
            graph.metadata_props[0].value = json.dumps(metadata)
            onnx.save(graph, bad / name)
        result = self.infer(source, [500500], bad)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("retrain/re-export", result.stderr)
        old = torch.load(self.checkpoint, weights_only=True)
        del old["feature_spec"]["policy"]
        checkpoint = self.root / "old-dense.pt"
        torch.save(old, checkpoint)
        with self.assertRaisesRegex(ValueError, "incompatible features"):
            export_checkpoint(checkpoint, self.root / "old-export")


if __name__ == "__main__":
    unittest.main()
