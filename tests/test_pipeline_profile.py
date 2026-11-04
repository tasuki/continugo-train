import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from test_train import fixtures

from continugo_train.data import scan_corpus
from continugo_train.loop import LoopConfig
from continugo_train.loop import parse_args as parse_loop_args
from continugo_train.model import NetworkConfig, PolicyValueNet
from continugo_train.pipeline_profile import main, representative_plies, select_positions
from continugo_train.pipeline_profile import parse_args as parse_profile_args
from continugo_train.train import Options, run_phase
from continugo_train.train import parse_args as parse_train_args


class PipelineProfileTests(unittest.TestCase):
    def test_six_worker_defaults_allow_explicit_and_saved_settings(self):
        self.assertEqual(Options().workers, 6)
        self.assertEqual(LoopConfig().workers, 6)
        for count in (0, 2, 6):
            args = ["--workers", str(count)]
            self.assertEqual(parse_train_args(["--data", "unused", *args]).workers, count)
            self.assertEqual(parse_loop_args(args).workers, count)
        # Omitted CLI settings must not override saved checkpoint/loop options.
        self.assertIsNone(parse_train_args(["--data", "unused"]).workers)
        self.assertIsNone(parse_loop_args([]).workers)

    def test_twelve_game_defaults_allow_explicit_and_saved_settings(self):
        required = ["--positions", "unused", "--model", "unused", "--checkpoint", "unused"]
        defaults = LoopConfig()
        profile = parse_profile_args(required)
        for name, expected in (
            ("concurrent_games", 12),
            ("inference_batch_size", 12),
            ("inference_batch_wait_ms", 2),
        ):
            self.assertEqual(getattr(defaults, name), expected)
            self.assertEqual(getattr(profile, name), expected)
            self.assertIsNone(getattr(parse_loop_args([]), name))
        overrides = [
            "--concurrent-games",
            "6",
            "--inference-batch-size",
            "8",
            "--inference-batch-wait-ms",
            "0",
        ]
        for args in (parse_profile_args([*required, *overrides]), parse_loop_args(overrides)):
            self.assertEqual(args.concurrent_games, 6)
            self.assertEqual(args.inference_batch_size, 8)
            self.assertEqual(args.inference_batch_wait_ms, 0)

    def test_training_window_counts_are_validated_before_measurement(self):
        for invalid in (
            ["--positions-per-game", "-1"],
            ["--warmup-steps", "0"],
            ["--concurrent-games", "0"],
            ["--concurrent-games", "65"],
            ["--inference-batch-size", "0"],
            ["--inference-batch-size", "33"],
            ["--inference-batch-wait-ms", "-1"],
            ["--inference-batch-wait-ms", "1001"],
        ):
            with self.subTest(invalid=invalid), patch("sys.stderr"):
                with self.assertRaises(SystemExit) as error:
                    main(
                        [
                            "--positions",
                            "unused",
                            "--model",
                            "unused",
                            "--checkpoint",
                            "unused",
                            *invalid,
                        ]
                    )
                self.assertEqual(error.exception.code, 2)

    def test_phase_proxies_are_nonterminal_and_reject_smoke_fixtures(self):
        self.assertEqual(representative_plies(102), {"early": 10, "middle": 50, "late": 100})
        with self.assertRaisesRegex(ValueError, "at least 10"):
            representative_plies(3)

    def test_selection_is_deterministic_complete_and_requires_all_sizes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("9", "13", "19", "short", "truncated"):
                (root / f"{name}.sgf").write_text(name)

            def info(path):
                name = path.stem
                return {
                    "board_size": int(name) if name.isdigit() else 9,
                    "completed": name != "truncated",
                    "moves": 3 if name == "short" else 999 if name == "truncated" else 100,
                    "game_id": name,
                }

            with patch("continugo_train.pipeline_profile.inspect_game", side_effect=info):
                selected = select_positions([root])
                self.assertEqual(len(selected), 9)
                self.assertEqual({p["size"] for p in selected}, {9, 13, 19})
                self.assertEqual({p["moves"] for p in selected}, {100})
                self.assertEqual(select_positions([root, root]), selected)
                (root / "19.sgf").unlink()
                with self.assertRaisesRegex(ValueError, "size-19"):
                    select_positions([root])

    def test_detailed_training_matches_normal_forward_and_optimizer(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)
        with tempfile.TemporaryDirectory() as directory:
            paths = fixtures(Path(directory))
            corpus = scan_corpus([str(p) for p in paths], 4, 0, 0.25)
            initial = PolicyValueNet(
                NetworkConfig(channels=4, blocks=1, policy_hidden=4, value_hidden=4)
            )
            states, metrics = [], []
            for detailed in (False, True):
                model = copy.deepcopy(initial)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
                metrics.append(
                    run_phase(
                        model,
                        optimizer,
                        corpus.train,
                        Options(batch_size=2, workers=0, steps_per_epoch=1),
                        0,
                        torch.device("cpu"),
                        None,
                        True,
                        profile_stages=detailed,
                    )
                )
                states.append((model.state_dict(), optimizer.state_dict()))
            self.assertNotIn("stage_seconds", metrics[0])
            self.assertNotIn("feature_profile", metrics[0])
            profile = metrics[1]["feature_profile"]
            self.assertGreaterEqual(profile["requests"], metrics[1]["samples"])
            self.assertGreater(profile["totals"]["server_coarse_seconds"], 0)
            self.assertGreater(profile["totals"]["decode_seconds"], 0)
            self.assertGreater(profile["totals"]["packet_bytes"], 0)
            self.assertEqual(
                set(metrics[1]["stage_seconds"]),
                {"transfer", "encoder", "scorer", "loss_backward_optimizer"},
            )
            self.assertTrue(all(v >= 0 for v in metrics[1]["stage_seconds"].values()))
            self.assertLessEqual(
                sum(metrics[1]["stage_seconds"].values()), metrics[1]["compute_seconds"]
            )
            for key in ("loss", "policy_ce", "value_mse", "samples", "steps"):
                self.assertEqual(metrics[0][key], metrics[1][key])
            for key, value in states[0][0].items():
                self.assertTrue(torch.equal(value, states[1][0][key]), key)
            for key, state in states[0][1]["state"].items():
                self.assertTrue(
                    torch.equal(
                        state["momentum_buffer"], states[1][1]["state"][key]["momentum_buffer"]
                    )
                )
