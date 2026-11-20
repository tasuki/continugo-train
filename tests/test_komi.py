"""Komi is game metadata, a scalar network input, and part of net scoring."""

import importlib.util
import json
import subprocess
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from continugo_train.augmentation import augment_sample
from continugo_train.data import TrainingSample, collate
from continugo_train.features import FeatureClient, inspect_game
from continugo_train.model import NetworkConfig, PolicyValueNet

ROOT = Path(__file__).resolve().parents[1]


class KomiTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_replay_scalars_net_scores_and_unchanged_ownership(self):
        with tempfile.TemporaryDirectory() as directory, FeatureClient() as client:
            path = Path(directory) / "game.sgf"
            path.write_text("(;FF[9071]GM[9071]SZ[9]KM[7]RE[B+74.0];B[jGjG];W[];B[])")
            info = inspect_game(path)
            self.assertEqual(info["komi"], 7)
            black = client.load(path, 0, cells=[])
            white = client.load(path, 1, cells=[])
            self.assertAlmostEqual(black.komi, -7 / 81)
            self.assertAlmostEqual(white.komi, 7 / 81)
            self.assertAlmostEqual(black.score, 74 / 81)
            self.assertAlmostEqual(white.score, -74 / 81)
            self.assertEqual(black.outcome, 1)
            self.assertTrue((black.ownership == 1).all())
            self.assertEqual(black.coarse.shape[0], 17)
            self.assertEqual(augment_sample(white, 5).komi, white.komi)
            path = Path(directory) / "empty.sgf"
            path.write_text("(;FF[9071]GM[9071]SZ[9]KM[7]RE[W+7.0];B[];W[])")
            empty = client.load(path, 0, cells=[])
            self.assertEqual(empty.outcome, -1)
            self.assertAlmostEqual(empty.score, -7 / 81)
            self.assertEqual(empty.ownership.abs().sum(), 0)
            path = Path(directory) / "legacy.sgf"
            path.write_text("(;FF[9071]GM[9071]SZ[9]RE[0];B[];W[])")
            legacy = client.load(path, 0, cells=[])
            self.assertEqual(legacy.komi, 0)
            self.assertEqual(legacy.score, 0)
            self.assertEqual(legacy.outcome, 0)

    def test_non_half_point_komi_replays_and_reaches_the_network(self):
        from continugo_train.loop import LoopConfig

        with tempfile.TemporaryDirectory() as directory, FeatureClient() as client:
            for komi in (-100.0, -6.3, 6.3, 25.0, 100.0):
                LoopConfig(komi=komi, komi_stddev=0).validate()
                path = Path(directory) / f"komi-{komi}.sgf"
                margin = 81 - komi
                winner = "B" if margin > 0 else "W"
                path.write_text(
                    f"(;FF[9071]GM[9071]SZ[9]KM[{komi}]RE[{winner}+{abs(margin):.1f}]"
                    ";B[jGjG];W[];B[])"
                )
                self.assertEqual(inspect_game(path)["komi"], komi)
                for ply, sign in ((0, 1), (1, -1)):
                    sample = client.load(path, ply, cells=[])
                    self.assertAlmostEqual(sample.komi, -sign * komi / 81, places=6)
                    self.assertAlmostEqual(sample.score, sign * margin / 81)

    def test_cached_training_samples_keep_the_scalar(self):
        from continugo_train.feature_cache import SampleReader, build_cache

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "game.sgf"
            comment = json.dumps({"policy": [["", 1]]}).replace("]", r"\]")
            path.write_text(
                "(;FF[9071]GM[9071]SZ[9]KM[7]RE[W+7.0]"
                f";B[]C[{comment}];W[]C[{comment}])"
            )
            build_cache(path)
            with SampleReader(cache="require") as reader:
                black = reader.load(path, 0)
                white = reader.load(path, 1)
            self.assertAlmostEqual(black.komi, -7 / 81)
            self.assertAlmostEqual(white.komi, 7 / 81)
            self.assertAlmostEqual(black.score, -7 / 81)
            self.assertAlmostEqual(white.score, 7 / 81)

    def test_collation_and_komi_projection_gradient(self):
        model = PolicyValueNet(NetworkConfig(channels=4, blocks=1, gpool_every=0)).eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.global_projection.weight.fill_(1)
            model.pass_head.weight.fill_(1)
        coarse = torch.zeros(2, 17, 38, 38)
        komi = torch.tensor([-7 / 81, 7 / 81])
        globals_ = model.global_features(coarse, komi)
        torch.testing.assert_close(globals_[:, 1], komi)
        outputs = model.encode_training(coarse, komi)
        self.assertFalse(torch.equal(outputs[1][0], outputs[1][1]))
        outputs[1][0].sum().backward()
        self.assertGreater(model.global_projection.weight.grad[:, 1].abs().sum(), 0)
        samples = [
            TrainingSample(
                coarse[i],
                torch.empty(0, 9),
                torch.empty(0, dtype=torch.bool),
                torch.tensor([0]),
                torch.tensor([1]),
                1,
                "a",
                0,
                torch.empty(0, dtype=torch.long),
                9,
                0,
                torch.zeros(38, 38),
                komi=float(komi[i]),
            )
            for i in range(2)
        ]
        batch = collate(samples).to("cpu")
        torch.testing.assert_close(batch.komi, komi)

    @unittest.skipUnless(
        importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime"),
        "install inference extra",
    )
    def test_native_onnx_matches_pytorch_with_variable_komi(self):
        from continugo_train.export import export_checkpoint, runtime_path
        from continugo_train.train import feature_spec

        torch.manual_seed(13)
        model = PolicyValueNet(NetworkConfig(channels=4, blocks=1, gpool_every=0)).eval()
        with torch.no_grad():
            model.score_head[0].bias.fill_(3)
            model.score_uncertainty.bias.fill_(2)
        with tempfile.TemporaryDirectory() as directory, FeatureClient() as client:
            root = Path(directory)
            checkpoint = root / "model.pt"
            torch.save(
                {
                    "config": asdict(model.config),
                    "model": model.state_dict(),
                    "feature_spec": feature_spec(4),
                    "epoch": 1,
                },
                checkpoint,
            )
            export_checkpoint(checkpoint, root / "onnx")
            for komi in (-6.3, 0, 7, 14, 25):
                path = root / f"komi-{komi}.sgf"
                path.write_text(f"(;FF[9071]GM[9071]SZ[9]KM[{komi}])")
                sample = client.load(path, 0, cells=[])
                with torch.inference_mode():
                    _, _, value, score, stdev = model.encode_search(
                        sample.coarse[None], torch.tensor([sample.komi])
                    )
                result = subprocess.run(
                    [
                        str(ROOT / "target/release/infer"),
                        "--model",
                        str(root / "onnx"),
                        "--ort-library",
                        str(runtime_path()),
                        "--device",
                        "cpu",
                        "--sgf",
                        str(path),
                        "--value-only",
                        "--iterations",
                        "3",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                actual = json.loads(result.stdout)
                for key, expected in (("value", value), ("score", score), ("score_stdev", stdev)):
                    self.assertAlmostEqual(actual[key], expected.item(), places=5)

    def test_arena_uses_fixed_seven_for_both_colours(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "games"
            subprocess.run(
                [
                    str(ROOT / "target/release/arena"),
                    "--unguided",
                    "--reference-random",
                    "--output",
                    str(output),
                    "--games",
                    "2",
                    "--sizes",
                    "9",
                    "--simulations",
                    "1",
                    "--max-moves",
                    "1",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            games = list(output.glob("*.sgf"))
            self.assertEqual(len(games), 2)
            for game in games:
                self.assertEqual(inspect_game(game)["komi"], 7)

    def test_selfplay_sampling_is_seeded_bounded_and_can_be_disabled(self):
        binary = ROOT / "target/release/continugo-train"
        command = [
            str(binary),
            "--engine",
            "random",
            "--games",
            "24",
            "--sizes",
            "9",
            "--size-weights",
            "uniform",
            "--max-moves",
            "1",
            "--seed",
            "123",
        ]

        def values(extra):
            result = subprocess.run(command + extra, capture_output=True, text=True, check=True)
            import re

            return [float(k) for k in re.findall(r"KM\[([^\]]+)\]", result.stdout)]

        sampled = values([])
        self.assertEqual(len(sampled), 24)
        self.assertEqual(sampled, values([]))
        self.assertGreater(len(set(sampled)), 1)
        self.assertTrue(all(0 <= k <= 14 and k * 2 == round(k * 2) for k in sampled))
        self.assertEqual(values(["--komi-stddev", "0"]), [7.0] * 24)
        for komi in (6.3, -6.3, 25.0):
            self.assertEqual(values(["--komi", str(komi), "--komi-stddev", "0"]), [komi] * 24)
        capped = values(["--komi-stddev", "100"])
        self.assertIn(0, capped)
        self.assertIn(14, capped)
