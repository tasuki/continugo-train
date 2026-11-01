import importlib.util
import io
import json
import shutil
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

import test_loop as helpers
import torch
from test_features import SGF

from continugo_train import archive, data, gate, loop, ratings
from continugo_train import train as trainer
from continugo_train.features import inspect_game
from continugo_train.sampling import DEFAULT_SIZE_WEIGHTS, position_quotas, weighted_sizes


class ArenaJobs(helpers.FakeJobs):
    def __init__(self):
        super().__init__()
        self.arena_mode = "normal"
        self.reference_mode = "normal"

    def __call__(self, command, logfile, lock_fd, environment=None):
        if "--candidate" not in command:
            return super().__call__(command, logfile, lock_fd, environment)
        self.calls.append(command)
        logfile.write_text(repr(command))

        def value(name):
            return command[command.index("--" + name.replace("_", "-")) + 1]

        candidate = (Path(value("candidate")) / "encoder.onnx").read_text()
        if "--reference-random" in command:
            return self.reference(command, value, candidate)
        incumbent = (Path(value("incumbent")) / "encoder.onnx").read_text()
        search = {
            name: type(default)(value(name)) for name, default in loop.LoopConfig().search().items()
        }
        count = search["simulations"]
        cycle = weighted_sizes(tuple(map(int, value("sizes").split(","))), value("size_weights"))
        for index in range(int(value("games"))):
            ordinal = int(value("game_offset")) + index
            size, seed, candidate_black = (
                cycle[(ordinal // 2) % len(cycle)],
                int(value("seed")) + ordinal // 2,
                ordinal % 2 == 0,
            )
            black, white = (candidate, incumbent) if candidate_black else (incumbent, candidate)
            root = {
                "model": black,
                "opponent": white,
                "seed": seed,
                "search": search,
                "evaluation": archive.digest(command[0]),
            }
            header = (
                f"SZ[{size}]PB[neural-{self.names[black]}]PW[neural-{self.names[white]}]"
                f"C[{json.dumps(root)}]"
            )
            text = (
                SGF.replace("SZ[13]", header)
                .replace('"action":200200,"visits":7', f'"action":200200,"visits":{count - 1}')
                .replace('"action":1000000,"visits":3', f'"action":1000000,"visits":{count}')
            )
            text = text.replace(
                ";B[]", f';B[]C[{{"policy":[{{"action":1000000,"visits":{count}}}\\]}}]'
            )
            if self.arena_mode == "draw":
                comment = f'C[{{"policy":[{{"action":1000000,"visits":{count}}}\\]}}]'
                text = f"(;FF[9071]GM[9071]{header}RE[0];B[]{comment};W[]{comment})"
            elif self.arena_mode == "truncate":
                text = text.replace("RE[B+220.4]", "")
                text = text[: text.index(";W[]")] + ")"
            path = Path(value("output")) / gate.filename(ordinal, size, seed, candidate_black)
            if self.arena_mode == "partial" and index == 1:
                path.write_text("(;incomplete")
                self.arena_mode = "normal"
                raise KeyboardInterrupt
            path.write_text(text)

    def reference(self, command, value, candidate):
        search = {
            name: type(default)(value(name)) for name, default in loop.LoopConfig().search().items()
        }
        count = search["simulations"]
        cycle = weighted_sizes(tuple(map(int, value("sizes").split(","))), value("size_weights"))
        for index in range(int(value("games"))):
            ordinal = int(value("game_offset")) + index
            size, seed = cycle[(ordinal // 2) % len(cycle)], int(value("seed")) + ordinal // 2
            candidate_black = ordinal % 2 == 0
            black, white = (
                (candidate, ratings.REFERENCE_ID)
                if candidate_black
                else (ratings.REFERENCE_ID, candidate)
            )
            root = {
                "model": black,
                "opponent": white,
                "seed": seed,
                "search": search,
                "evaluation": archive.digest(command[0]),
            }
            players = [
                ratings.REFERENCE_ID
                if network == ratings.REFERENCE_ID
                else f"neural-{self.names[network]}"
                for network in [black, white]
            ]
            header = f"SZ[{size}]PB[{players[0]}]PW[{players[1]}]C[{json.dumps(root)}]"

            def node(color, coordinate, action, candidate_black=candidate_black):
                neural = (color == "B") == candidate_black
                comment = (
                    (
                        "C["
                        + json.dumps({"policy": [{"action": action, "visits": count}]}).replace(
                            "]", "\\]"
                        )
                        + "]"
                    )
                    if neural
                    else ""
                )
                return f";{color}[{coordinate}]{comment}"

            if self.reference_mode == "draw":
                text = (
                    f"(;FF[9071]GM[9071]{header}RE[0]"
                    + node("B", "", 1000000)
                    + node("W", "", 1000000)
                    + ")"
                )
            elif self.reference_mode == "truncate":
                text = f"(;FF[9071]GM[9071]{header}" + node("B", "dSdS", 200200) + ")"
            else:
                text = (
                    f"(;FF[9071]GM[9071]{header}RE[B+220.4]"
                    + node("B", "dSdS", 200200)
                    + node("W", "", 1000000)
                    + node("B", "", 1000000)
                    + ")"
                )
            path = Path(value("output")) / gate.filename(ordinal, size, seed, candidate_black)
            if self.reference_mode == "partial" and index == 1:
                path.write_text("(;incomplete")
                self.reference_mode = "normal"
                raise KeyboardInterrupt
            path.write_text(text)


class SamplingTests(unittest.TestCase):
    def test_default_schedule_and_evaluation_pairs_are_40_40_20(self):
        self.assertEqual(weighted_sizes((9, 13, 19)), [9, 13, 19, 9, 13])
        self.assertEqual(Counter(weighted_sizes((9, 13, 19))), {9: 2, 13: 2, 19: 1})
        config = loop.LoopConfig(gate_games=20)
        self.assertEqual(
            Counter(gate.plan(config, 1, i)[1] for i in range(20)), {9: 8, 13: 8, 19: 4}
        )
        for index in range(0, 20, 2):
            first, second = gate.plan(config, 1, index), gate.plan(config, 1, index + 1)
            self.assertEqual(first[1:3], second[1:3])
            self.assertNotEqual(first[3], second[3])

    def test_surplus_long_board_positions_are_not_extracted(self):
        self.assertEqual(
            position_quotas({9: 100, 13: 100, 19: 900}, DEFAULT_SIZE_WEIGHTS),
            {9: 100, 13: 100, 19: 50},
        )
        games = [
            data.GameEntry(str(size), f"{size:064x}", "hash", size, 1, size, tuple(range(count)))
            for size, count in [(9, 100), (13, 100), (19, 900)]
        ]
        plans = data._plans(games, 0, 0, True, 0, DEFAULT_SIZE_WEIGHTS, 2)
        self.assertEqual(Counter(game.board_size for game, _ in plans), {9: 100, 13: 100, 19: 50})
        self.assertEqual(Counter(game.board_size for game, _ in plans[:20]), {9: 8, 13: 8, 19: 4})
        self.assertEqual(len({(game.game_id, plies[0]) for game, plies in plans}), 250)
        # Old fixed-corpus ordering stays exhaustive in legacy/uniform mode.
        self.assertEqual(sum(len(plies) for _, plies in data._plans(games, 0, 0, True, 0)), 1100)

    def test_truncated_matches_use_conservative_bounds_not_fake_draws(self):
        def records(wins, losses, truncated):
            return [
                {"board_size": 9, "candidate_black": True, "result": result}
                for result in [*["B+1.0"] * wins, *["W+1.0"] * losses, *[None] * truncated]
            ]

        accepted = gate.summarize(records(6, 0, 4), 0.5)
        self.assertEqual(accepted["decision"], "accepted")
        self.assertEqual(accepted["planned_score_bounds"], (0.6, 1.0))
        rejected = gate.summarize(records(0, 7, 3), 0.5)
        self.assertEqual(rejected["decision"], "rejected")
        inconclusive = gate.summarize(records(2, 4, 4), 0.5)
        self.assertEqual(inconclusive["decision"], "inconclusive")
        self.assertEqual(inconclusive["draws"], 0)

    def test_zero_weights_and_unavailable_sizes_are_renormalized(self):
        self.assertEqual(weighted_sizes((9, 13, 19), "9:2,13:2,19:0"), [9, 13])
        self.assertEqual(position_quotas({13: 17}, DEFAULT_SIZE_WEIGHTS), {13: 17})
        for text in ["9:-1", "9:2,9:3", "broken"]:
            with self.assertRaises(ValueError):
                weighted_sizes((9, 13, 19), text)


class AcceptanceTests(helpers.LoopHarness):
    def setUp(self):
        super().setUp()
        self.jobs = ArenaJobs()

    def test_store_refuses_foreign_data_and_archive_detects_corruption(self):
        foreign = self.root / "foreign"
        foreign.mkdir()
        existing = foreign / "keep.txt"
        existing.write_text("do not overwrite")
        with self.assertRaisesRegex(ValueError, "without a STORE.pt marker"):
            archive.publish(foreign, "training", self.source)
        self.assertEqual(existing.read_text(), "do not overwrite")
        reference = archive.publish(self.root / "private-store", "training", self.source)
        checkpoint = Path(reference["directory"]) / "checkpoint.pt"
        original = checkpoint.read_bytes()
        checkpoint.chmod(0o644)
        checkpoint.write_bytes(original + b"corrupt")
        with self.assertRaisesRegex(ValueError, "artifact missing or changed"):
            archive.verify_bundle(reference)
        checkpoint.write_bytes(original)
        checkpoint.chmod(0o444)
        archive.verify_bundle(reference)

    def test_model_store_cannot_live_in_disposable_loop_output(self):
        output = self.root / "bad-location"
        with self.assertRaisesRegex(ValueError, "separate from the loop output"):
            self.run_loop(output, extra=["--model-store", str(output / "models")])

    def test_rejection_keeps_latest_training_progress_and_incumbent_selfplay(self):
        output = self.root / "rejected"
        state = self.run_loop(
            output, extra=["--iterations", "2", "--gate-games", "10", "--acceptance-rate", "0.6"]
        )
        first, second = state["iterations"]
        self.assertEqual(first["decision"], "rejected")
        self.assertEqual(second["decision"], "rejected")
        self.assertEqual(second["parent_checkpoint"], first["checkpoint"])
        self.assertEqual(second["parent_model"], state["initial_model"])
        self.assertEqual(state["accepted"]["model"], state["initial_model"])
        latest = Path(state["config"]["model_store"]) / "LATEST-TRAINING"
        self.assertEqual(
            latest.read_text().strip(),
            str(Path(second["training_archive"]["directory"]).relative_to(latest.parent)),
        )
        self.assertTrue(second["training_archive"])
        self.assertIsNone(second["accepted_archive"])
        for record in second["games"]:
            self.assertEqual(
                inspect_game(output / record["path"])["root_comment"]["model"],
                state["initial_checkpoint"]["sha256"],
            )

    def test_unexpected_draws_are_visible_and_accepted_archives_are_complete(self):
        output = self.root / "draws"
        self.jobs.arena_mode = "draw"
        state = self.run_loop(output, extra=["--gate-games", "10"])
        iteration = state["iterations"][0]
        report = iteration["gate"]["report"]
        self.assertEqual(report["draws"], 10)
        self.assertEqual(report["score"], 0.5)
        self.assertEqual(iteration["decision"], "accepted")
        self.assertIn("WARNING DRAW", (output / "loop.log").read_text())
        reference = state["accepted"]["archive"]
        metadata = archive.verify_bundle(reference)
        bundle = Path(reference["directory"])
        self.assertEqual(metadata["checkpoint_sha256"], iteration["checkpoint"]["sha256"])
        self.assertEqual(
            archive.digest(bundle / "checkpoint.pt"), iteration["checkpoint"]["sha256"]
        )
        self.assertTrue((bundle / "encoder.onnx").is_file())
        self.assertTrue((bundle / "policy.onnx").is_file())
        saved = loop.load_training(bundle / "checkpoint.pt")
        self.assertEqual(len(list((bundle / "replay").glob("*.sgf"))), len(saved["manifest"]))
        self.assertEqual(len(list((bundle / "evaluation").glob("*.sgf"))), 10)
        self.assertEqual(bundle.stat().st_mode & 0o222, 0)
        before = {path: path.read_bytes() for path in bundle.rglob("*") if path.is_file()}
        calls = len(self.jobs.calls)
        self.run_loop(output, resume=True)
        self.assertEqual(len(self.jobs.calls), calls)
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_truncation_is_inconclusive_not_a_draw_or_a_promotion(self):
        output = self.root / "truncated"
        self.jobs.arena_mode = "truncate"
        state = self.run_loop(output, extra=["--gate-games", "10", "--gate-max-moves", "1"])
        iteration = state["iterations"][0]
        report = iteration["gate"]["report"]
        self.assertEqual(report["truncated"], 10)
        self.assertEqual(report["draws"], 0)
        self.assertIsNone(report["score"])
        self.assertEqual(iteration["decision"], "inconclusive")
        self.assertEqual(state["accepted"]["model"], state["initial_model"])
        self.assertIn("WARNING TRUNCATED", (output / "loop.log").read_text())
        self.assertTrue(iteration["training_archive"])

    def test_gate_interrupt_reuses_completed_games_without_retraining(self):
        output = self.root / "gate-gap"
        self.jobs.arena_mode = "partial"
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop(output, extra=["--gate-games", "10"])
        before = {p: p.read_bytes() for p in output.glob("iteration-0001/gate/*/*.sgf")}
        training_calls = len(self.stage_calls("continugo_train.train"))
        state = self.run_loop(output, resume=True)
        self.assertEqual(len(self.stage_calls("continugo_train.train")), training_calls)
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.assertEqual(
            state["iterations"][0]["gate"]["attempts"][1]["indices"], list(range(1, 10))
        )

    def test_recovery_after_losing_runs_keeps_rejected_training_and_accepted_model_separate(self):
        output = self.root / "lost-run"
        state = self.run_loop(output, extra=["--gate-games", "10", "--acceptance-rate", "0.6"])
        latest = Path(state["iterations"][0]["training_archive"]["directory"])
        accepted = Path(state["accepted"]["archive"]["directory"])
        training_hash = archive.digest(latest / "checkpoint.pt")
        accepted_hash = archive.digest(accepted / "checkpoint.pt")
        shutil.rmtree(output)
        shutil.rmtree(self.corpus)
        self.source.unlink()
        archive.verify_bundle(state["accepted"]["archive"])
        resumed = self.run_loop(
            self.root / "restored",
            extra=[
                "--checkpoint",
                str(latest / "checkpoint.pt"),
                "--incumbent-checkpoint",
                str(accepted / "checkpoint.pt"),
                "--bootstrap-data",
                str(latest / "replay"),
                "--seed",
                "300",
                "--gate-games",
                "10",
                "--acceptance-rate",
                "0.6",
            ],
        )
        self.assertEqual(resumed["initial_checkpoint"]["sha256"], training_hash)
        self.assertEqual(resumed["initial_model"]["checkpoint_sha256"], accepted_hash)
        self.assertNotEqual(training_hash, accepted_hash)
        checkpoint = loop.load_training(
            self.root / "restored" / resumed["iterations"][0]["checkpoint"]["path"]
        )
        self.assertEqual(checkpoint["initialization"]["source"]["sha256"], training_hash)

    @unittest.skipUnless(
        importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime"),
        "install inference extra",
    )
    def test_real_native_paired_gate_uses_weighted_sizes_and_full_recovery_bundle(self):
        output = self.root / "real-gate"
        arguments = self.args(
            output,
            extra=[
                "--sizes",
                "9,13,19",
                "--games-per-iteration",
                "5",
                "--gate-games",
                "10",
                "--simulations",
                "4",
                "--gate-simulations",
                "4",
                "--widening",
                "0.01",
                "--gate-max-moves",
                "8",
                "--max-moves",
                "24",
            ],
        )
        with redirect_stdout(io.StringIO()):
            state = loop.run_loop(arguments)
        iteration = state["iterations"][0]
        self.assertEqual(Counter(r["board_size"] for r in iteration["games"]), {9: 2, 13: 2, 19: 1})
        self.assertEqual(
            Counter(r["board_size"] for r in iteration["gate"]["games"]), {9: 4, 13: 4, 19: 2}
        )
        self.assertEqual(iteration["gate"]["report"]["draws"], 10)
        self.assertEqual(iteration["decision"], "accepted")
        archive.verify_bundle(state["accepted"]["archive"])
        bundle = Path(state["accepted"]["archive"]["directory"])
        # Exact checkpoint plus relocated SGFs can feed the ordinary strict loader.
        saved = loop.load_training(bundle / "checkpoint.pt")
        corpus = data.scan_corpus(
            [str(bundle / "replay")],
            4,
            saved["options"]["seed"],
            saved["options"]["validation_fraction"],
            split_history=saved["split_history"],
        )
        self.assertEqual(corpus.fingerprint, saved["corpus_fingerprint"])

        def continuation(name, inputs):
            directory = self.root / name
            directory.mkdir()
            copied = directory / "checkpoint.pt"
            shutil.copyfile(bundle / "checkpoint.pt", copied)
            with redirect_stdout(io.StringIO()):
                return trainer.train(
                    trainer.parse_args(
                        [
                            "--data",
                            *inputs,
                            "--resume",
                            str(copied),
                            "--epochs",
                            "2",
                            "--device",
                            "cpu",
                            "--threads",
                            "1",
                        ]
                    )
                )

        expected = continuation(
            "original-continuation", [str(output / r["path"]) for r in iteration["replay"]]
        )
        shutil.rmtree(output)
        actual = continuation("archive-continuation", [str(bundle / "replay")])
        expected, actual = (
            torch.load(expected, weights_only=True),
            torch.load(actual, weights_only=True),
        )
        self.assertEqual(expected["global_step"], actual["global_step"])
        for name, value in expected["model"].items():
            torch.testing.assert_close(value, actual["model"][name], atol=0, rtol=0)
        for key, value in expected["optimizer"]["state"].items():
            torch.testing.assert_close(
                value["momentum_buffer"],
                actual["optimizer"]["state"][key]["momentum_buffer"],
                atol=0,
                rtol=0,
            )


if __name__ == "__main__":
    unittest.main()
