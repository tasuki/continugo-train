import copy
import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from continugo_train.export import export_checkpoint, runtime_path
from continugo_train.features import load_position
from continugo_train.model import NetworkConfig, PolicyValueNet
from continugo_train.rate_saved import validate_evaluation_model
from continugo_train.train import feature_spec


class SchemaCompatibilityTests(unittest.TestCase):
    def test_legacy_history_and_approved_schema6_precision(self):
        fixture = json.loads(
            (Path(__file__).parent / "fixtures/feature-schema-history.json").read_text()
        )
        precision = json.loads((Path(__file__).parent / "fixtures/liberty-area8.json").read_text())
        self.assertEqual((precision["schema"], precision["coarsening"]), (6, 8))
        approved = {(c["game"], c["ply"]): c for c in precision["coarse_cases"]}
        expected_cases = {
            (game["name"], case["ply"])
            for game in fixture["games"]
            for case in game["cases"]
            if case["schema"] == 6
        }
        self.assertEqual(set(approved), expected_cases)
        self.assertTrue(fixture["schema5_git_revision"].startswith("71f8daf"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "game.sgf"
            for game in fixture["games"]:
                path.write_text(game["sgf"])
                for case in game["cases"]:
                    with self.subTest(game=game["name"], schema=case["schema"], ply=case["ply"]):
                        sample = load_position(
                            path,
                            case["ply"],
                            actions=fixture["actions"],
                            feature_schema=case["schema"],
                        )
                        for field, expected in case["hashes"].items():
                            if case["schema"] == 6 and field == "coarse":
                                record = approved[game["name"], case["ply"]]
                                self.assertEqual(record["precise_coarse_sha256"], expected)
                                expected = record["coarse_sha256"]
                                keep = [i for i in range(14) if i not in (4, 5)]
                                unchanged = hashlib.sha256(
                                    sample.coarse[keep].numpy().tobytes()
                                ).hexdigest()
                                self.assertEqual(unchanged, record["unchanged_planes_sha256"])
                            actual = hashlib.sha256(
                                getattr(sample, field).numpy().tobytes()
                            ).hexdigest()
                            self.assertEqual(actual, expected, field)

    def test_models_select_their_own_schema_and_widths(self):
        for schema in (4, 5, 6, 7):
            spec = feature_spec(4, schema=schema)
            saved = {
                "feature_spec": spec,
                "config": asdict(
                    NetworkConfig(
                        input_planes=len(spec["coarse"]),
                        move_features=len(spec["moves"]),
                        global_inputs=len(spec.get("globals", [])),
                        score_distribution="linear-softplus" if schema == 7 else "tanh-sigmoid",
                    )
                ),
            }
            for reference_schema in (4, 5, 6, 7):
                validate_evaluation_model(saved, feature_spec(4, schema=reference_schema))
            broken = copy.deepcopy(saved)
            broken["feature_spec"]["coarse"][0] = "wrong_channel"
            with self.assertRaisesRegex(ValueError, "semantics"):
                validate_evaluation_model(broken, feature_spec(4))
            if schema == 7:
                old_markers = copy.deepcopy(saved)
                old_markers["feature_spec"]["coarse"][2:5] = [
                    "enemy_move_1_ago",
                    "friendly_move_2_ago",
                    "enemy_move_3_ago",
                ]
                with self.assertRaisesRegex(ValueError, "semantics"):
                    validate_evaluation_model(old_markers, feature_spec(4))
            broken = copy.deepcopy(saved)
            broken["config"]["move_features"] += 1
            with self.assertRaisesRegex(ValueError, "architecture"):
                validate_evaluation_model(broken, feature_spec(4))
            with self.assertRaisesRegex(ValueError, "density"):
                validate_evaluation_model(saved, feature_spec(8))

    @unittest.skipUnless(
        importlib.util.find_spec("onnxruntime") and importlib.util.find_spec("onnx"),
        "install inference extra",
    )
    def test_native_arena_uses_legacy_and_history_schemas_on_alternating_turns(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for schema in (5, 6, 7):
                spec = feature_spec(4, schema=schema)
                model = PolicyValueNet(
                    NetworkConfig(
                        input_planes=len(spec["coarse"]),
                        move_features=len(spec["moves"]),
                        global_inputs=len(spec.get("globals", [])),
                        score_distribution="linear-softplus" if schema == 7 else "tanh-sigmoid",
                        channels=4,
                        blocks=1,
                        policy_hidden=4,
                        value_hidden=4,
                    )
                )
                checkpoint = root / f"schema{schema}.pt"
                torch.save(
                    {
                        "config": asdict(model.config),
                        "model": model.state_dict(),
                        "feature_spec": spec,
                        "epoch": 1,
                    },
                    checkpoint,
                )
                export_checkpoint(checkpoint, root / f"schema{schema}")
            for legacy_schema in (5, 6):
                output = root / f"games-vs-{legacy_schema}"
                result = subprocess.run(
                    [
                        str(Path(__file__).resolve().parents[1] / "target/release/arena"),
                        "--candidate",
                        str(root / "schema7"),
                        "--incumbent",
                        str(root / f"schema{legacy_schema}"),
                        "--output",
                        str(output),
                        "--games",
                        "2",
                        "--sizes",
                        "9",
                        "--size-weights",
                        "uniform",
                        "--simulations",
                        "2",
                        "--max-moves",
                        "4",
                        "--device",
                        "cpu",
                        "--ort-library",
                        str(runtime_path()),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(list(output.glob("*.sgf"))), 2)

    def test_schema5_splitting_is_not_the_replaced_footprint_column(self):
        spec = feature_spec(4, schema=5)
        self.assertEqual(spec["moves"][6], "splits_friendly_region")
        spec["moves"][6] = "local_liberty_footprint"
        saved = {
            "feature_spec": spec,
            "config": asdict(NetworkConfig(input_planes=12, move_features=11, global_inputs=0)),
        }
        with self.assertRaisesRegex(ValueError, "semantics"):
            validate_evaluation_model(saved, feature_spec(4))


if __name__ == "__main__":
    unittest.main()
