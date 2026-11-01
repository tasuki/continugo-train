import copy
import importlib.util
import io
import json
import random
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from contextlib import redirect_stdout
from dataclasses import asdict, fields
from pathlib import Path
from unittest.mock import patch

import torch
from test_features import SGF
from test_train import fixtures

from continugo_train import loop
from continugo_train import train as trainer
from continugo_train.data import scan_corpus
from continugo_train.features import inspect_game
from continugo_train.model import NetworkConfig, PolicyValueNet
from continugo_train.train import Options, atomic_checkpoint, feature_spec


class FakeJobs:
    """Fast fault injection around real SGF inspection/corpus/checkpoint metadata."""

    def __init__(self):
        self.calls = []
        self.fault = None

    @staticmethod
    def bundle(directory, checkpoint, _saved):
        for name in ["encoder.onnx", "policy.onnx"]:
            if (directory / name).read_text() != loop.digest(checkpoint):
                raise ValueError("wrong fake export checkpoint")

    def __call__(self, command, logfile, _lock_fd, _environment=None):
        self.calls.append(command)
        logfile.parent.mkdir(parents=True, exist_ok=True)
        logfile.write_text(repr(command))
        if "continugo_train.export" in command:
            args = command[command.index("--checkpoint") + 1]
            output = Path(command[command.index("--output") + 1])
            output.mkdir()
            for name in ["encoder.onnx", "policy.onnx"]:
                (output / name).write_text(loop.digest(args))
        elif "continugo_train.train" in command:
            self.train(command[3:])
        else:
            self.selfplay(command)

    def selfplay(self, command):
        def value(name):
            return command[command.index("--" + name.replace("_", "-")) + 1]

        output = Path(value("output"))
        size = int(value("sizes"))  # Fake jobs deliberately use only size 13.
        seed = int(value("seed"))
        search = {
            name: type(default)(value(name)) for name, default in loop.LoopConfig().search().items()
        }
        model = (Path(value("model")) / "encoder.onnx").read_text()
        for index in range(int(value("games"))):
            offset = (seed + index) % 8
            coordinate, letter = 200 + offset, "STUVWXYZ"[offset]
            text = SGF.replace("B[dSdS]", f"B[d{letter}d{letter}]").replace(
                "200200", str(1001 * coordinate)
            )
            text = text.replace('"visits":3', '"visits":8')
            text = text.replace(";B[]", ';B[]C[{"policy":[{"action":1000000,"visits":8}\\]}]')
            root = {"seed": seed + index, "model": model, "search": search}
            if self.fault == "wrong_model":
                root["model"] = "0" * 64
            text = text.replace("SZ[13]", "SZ[13]C[" + json.dumps(root) + "]")
            if self.fault == "unlabeled":
                text = text.replace("RE[B+220.4]", "")
                text = text[: text.rindex(";B[]")] + ")"
            path = output / f"game-{index + 1:06d}-size{size}-seed{seed + index}.sgf"
            if self.fault == "partial_selfplay" and index == 1:
                path.write_text("(;broken")
                self.fault = None
                raise KeyboardInterrupt
            path.write_text(text)

    def train(self, argv):
        args = trainer.parse_args(argv)
        output = args.output
        output.mkdir(exist_ok=True)
        if self.fault == "no_epoch":
            (output / "training.log").write_text("interrupted first epoch")
            self.fault = None
            raise KeyboardInterrupt
        parent = loop.load_training(args.resume or args.warm_start)
        state = copy.deepcopy(parent)
        options = Options(**parent["options"])
        for field in fields(Options):
            value = getattr(args, field.name)
            if value is not None:
                setattr(options, field.name, value)
        corpus = scan_corpus(
            args.data,
            options.samples_per_diameter,
            options.seed,
            options.validation_fraction,
            split_history=trainer.checkpoint_splits(parent),
        )
        completed = state["epoch"] if args.resume else 0
        if not args.resume:
            state.update(
                run_id=uuid.uuid4().hex,
                global_step=0,
                initialization={
                    "mode": "warm_start",
                    "optimizer": args.warm_start_optimizer,
                    "source": {"sha256": loop.digest(args.warm_start)},
                },
            )
        state.update(
            options=asdict(options),
            corpus_fingerprint=corpus.fingerprint,
            manifest=corpus.manifest(),
            split_history=corpus.split_history,
        )
        for epoch in range(completed + 1, args.epochs + 1):
            state.update(epoch=epoch, global_step=state["global_step"] + 1)
            atomic_checkpoint(output / f"epoch-{epoch:04d}.pt", state)
            if self.fault == "epoch_gap":
                # Epoch publication succeeded but aliases and supervisor commit did not.
                (output / ".train.lock").write_text("2000000000")
                self.fault = None
                raise KeyboardInterrupt
            atomic_checkpoint(output / "latest.pt", state)


class LoopHarness(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.corpus = self.root / "sgf"
        self.corpus.mkdir()
        fixtures(self.corpus)
        corpus = scan_corpus([str(self.corpus)], 4, 0, 0.25)
        self.model = PolicyValueNet(
            NetworkConfig(channels=4, blocks=1, policy_hidden=4, value_hidden=4)
        )
        # Pass-biased plumbing model for the real CPU integration test, not strength.
        with torch.no_grad():
            self.model.pass_head.bias.fill_(30)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.source = self.root / "source.pt"
        atomic_checkpoint(
            self.source,
            {
                "options": asdict(
                    Options(
                        batch_size=1,
                        workers=0,
                        shuffle_buffer=0,
                        steps_per_epoch=1,
                        positions_per_game=1,
                        validation_positions_per_game=1,
                    )
                ),
                "config": asdict(self.model.config),
                "model": self.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": 1,
                "global_step": 1,
                "run_id": "source",
                "feature_spec": feature_spec(4),
                "corpus_fingerprint": corpus.fingerprint,
                "manifest": corpus.manifest(),
                "best_validation": 1.0,
                "rng_cpu": torch.get_rng_state(),
                "rng_python": random.getstate(),
                "rng_cuda": [],
            },
        )
        self.jobs = FakeJobs()

    def args(self, output, resume=False, extra=()):
        base = ["--output", str(output)]
        if resume:
            base += ["--resume"]
        else:
            base += [
                "--checkpoint",
                str(self.source),
                "--bootstrap-data",
                str(self.corpus),
                "--model-store",
                str(self.root / "models"),
                "--gate-games",
                "0",
                "--sizes",
                "13",
                "--games-per-iteration",
                "4",
                "--seed",
                "204",
                "--simulations",
                "8",
                "--max-moves",
                "32",
                "--device",
                "cpu",
                "--train-epochs",
                "1",
                "--steps-per-epoch",
                "1",
                "--batch-size",
                "1",
                "--workers",
                "0",
                "--shuffle-buffer",
                "0",
                "--positions-per-game",
                "1",
                "--validation-positions-per-game",
                "1",
                "--replay-games",
                "16",
                "--replay-iterations",
                "1",
                "--threads",
                "1",
            ]
        return loop.parse_args([*base, *extra])

    def run_loop(self, output, resume=False, extra=()):
        with (
            patch.object(loop, "run_process", self.jobs),
            patch.object(loop, "verify_export", self.jobs.bundle),
            redirect_stdout(io.StringIO()),
        ):
            return loop.run_loop(self.args(output, resume, extra))

    def stage_calls(self, name):
        return [command for command in self.jobs.calls if name in command]


class LoopTests(LoopHarness):
    def test_two_iterations_handoff_replay_window_and_idempotent_resume(self):
        output = self.root / "loop"
        state = self.run_loop(output)
        first = state["iterations"][0]
        committed = [
            first["checkpoint"],
            *first["games"],
            first["model"]["encoder"],
            first["model"]["policy"],
        ]
        before = {record["path"]: loop.digest(output / record["path"]) for record in committed}
        state = self.run_loop(output, resume=True, extra=["--iterations", "2"])
        second = state["iterations"][1]
        self.assertEqual(second["parent_checkpoint"], first["checkpoint"])
        self.assertEqual(second["parent_model"], first["model"])
        for record in second["games"]:
            info = inspect_game(output / record["path"])
            self.assertEqual(info["root_comment"]["model"], first["checkpoint"]["sha256"])
        self.assertTrue(
            all(r["path"].startswith(("iteration-0002/", "bootstrap/")) for r in second["replay"])
        )
        self.assertLessEqual(len(second["replay"]), 16)
        self.assertEqual(before, {path: loop.digest(output / path) for path in before})
        calls = len(self.jobs.calls)
        self.run_loop(output, resume=True)
        self.assertEqual(len(self.jobs.calls), calls)

    def test_partial_selfplay_only_regenerates_missing_or_invalid_uncommitted_games(self):
        output = self.root / "partial"
        self.jobs.fault = "partial_selfplay"
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop(output)
        paths = sorted(output.rglob("*.sgf"))
        before = {str(path): path.read_bytes() for path in paths}
        state = self.run_loop(output, resume=True)
        attempts = state["iterations"][0]["selfplay_attempts"]
        self.assertEqual(attempts[1]["indices"], [1, 2, 3])
        self.assertEqual(before, {str(path): path.read_bytes() for path in paths})
        self.assertTrue(all(state["iterations"][0]["games"]))

    def test_epoch_boundary_resume_preserves_snapshot_despite_alias_publication_gap(self):
        output = self.root / "epoch-gap"
        self.jobs.fault = "epoch_gap"
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop(output, extra=["--train-epochs", "2"])
        epoch = next(output.rglob("epoch-0001.pt"))
        before = epoch.read_bytes()
        self.assertFalse((epoch.parent / "latest.pt").exists())
        state = self.run_loop(output, resume=True)
        calls = self.stage_calls("continugo_train.train")
        self.assertEqual(len(calls), 2)
        self.assertIn("--resume", calls[-1])
        self.assertEqual(epoch.read_bytes(), before)
        self.assertFalse((epoch.parent / ".train.lock").exists())
        saved = loop.load_training(output / state["iterations"][0]["checkpoint"]["path"])
        self.assertEqual(saved["epoch"], 2)

    def test_resumed_epoch_cannot_drop_ancestral_splits_absent_from_its_window(self):
        output = self.root / "lost-history"
        self.jobs.fault = "epoch_gap"
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop(output, extra=["--train-epochs", "2", "--replay-games", "4"])
        path = next(output.rglob("epoch-0001.pt"))
        saved = loop.load_training(path)
        current_ids = {g["game_id"] for g in saved["manifest"]}
        absent = next(
            identity for identity in saved["split_history"] if identity not in current_ids
        )
        del saved["split_history"][absent]
        atomic_checkpoint(path, saved)
        with self.assertRaisesRegex(ValueError, "does not match its frozen job"):
            self.run_loop(output, resume=True)

    def test_failed_first_epoch_gets_a_new_attempt_without_overwriting_old_files(self):
        output = self.root / "no-epoch"
        self.jobs.fault = "no_epoch"
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop(output)
        log = next(output.rglob("training.log"))
        before = log.read_bytes()
        state = self.run_loop(output, resume=True)
        self.assertEqual(len(state["iterations"][0]["train_attempts"]), 2)
        self.assertEqual(log.read_bytes(), before)
        self.assertIn("--warm-start", self.stage_calls("continugo_train.train")[-1])

    def test_atomic_exports_are_adopted_after_supervisor_commit_gaps(self):
        for initial in [True, False]:
            with self.subTest(initial=initial):
                output = self.root / f"export-gap-{initial}"
                save = loop.LoopRunner.save
                failed = False

                def gap(runner, initial=initial, save=save):
                    nonlocal failed
                    ready = (
                        runner.state["initial_model"]
                        if initial
                        else (runner.state["iterations"] and runner.state["iterations"][0]["model"])
                    )
                    if ready and not failed:
                        failed = True
                        raise KeyboardInterrupt
                    save(runner)

                with (
                    patch.object(loop.LoopRunner, "save", gap),
                    self.assertRaises(KeyboardInterrupt),
                ):
                    self.run_loop(output)
                count = len(self.stage_calls("continugo_train.export"))
                self.run_loop(output, resume=True)
                self.assertEqual(
                    len(self.stage_calls("continugo_train.export")), count + int(initial)
                )

    def test_snapshots_survive_missing_original_source_and_bootstrap_files(self):
        output = self.root / "snapshots"
        save = loop.LoopRunner.save

        def gap(runner):
            if runner.state["snapshots_ready"]:
                raise KeyboardInterrupt
            save(runner)

        with patch.object(loop.LoopRunner, "save", gap), self.assertRaises(KeyboardInterrupt):
            self.run_loop(output)
        self.source.unlink()
        for path in self.corpus.glob("*.sgf"):
            path.unlink()
        state = self.run_loop(output, resume=True)
        self.assertTrue(state["iterations"][0]["model"])

    def test_committed_artifact_changes_and_configuration_overrides_fail(self):
        output = self.root / "immutable"
        state = self.run_loop(output)
        with self.assertRaisesRegex(ValueError, "resume restores --seed"):
            self.run_loop(output, resume=True, extra=["--seed", "999"])
        calls = len(self.jobs.calls)
        for record in [state["iterations"][0]["games"][0], state["initial_model"]["encoder"]]:
            path = output / record["path"]
            before = path.read_bytes()
            path.write_bytes(before + b"\n")
            with self.assertRaisesRegex(ValueError, "artifact missing or changed"):
                self.run_loop(output, resume=True)
            path.write_bytes(before)
        self.assertEqual(len(self.jobs.calls), calls)

    def test_wrong_model_metadata_and_no_new_completed_labels_do_not_train(self):
        for fault, extra, message in [
            ("wrong_model", [], "planned job/model"),
            ("unlabeled", ["--max-moves", "2"], "no completed policy-labeled games"),
        ]:
            with self.subTest(fault=fault):
                self.jobs = FakeJobs()
                self.jobs.fault = fault
                with self.assertRaisesRegex(ValueError, message):
                    self.run_loop(self.root / fault, extra=extra)
                self.assertFalse(self.stage_calls("continugo_train.train"))

    def test_configuration_and_live_training_lock_guards(self):
        for index, extra in enumerate(
            [
                ["--sizes", "8"],
                ["--iterations", "0"],
                ["--games-per-iteration", "0"],
                ["--inference-batch-size", "33"],
                ["--widening-exponent", "1"],
            ]
        ):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.run_loop(self.root / f"invalid-{index}", extra=extra)
        output = self.root / "live-trainer"
        self.jobs.fault = "epoch_gap"
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop(output, extra=["--train-epochs", "2"])
        lock = next(output.rglob(".train.lock"))
        lock.write_text(str(loop.os.getpid()))
        with self.assertRaisesRegex(ValueError, "may still be alive"):
            self.run_loop(output, resume=True)
        self.assertTrue(lock.exists())

    def test_replay_selection_is_bounded_balanced_and_prefers_newest_duplicate(self):
        records = [
            {"game_id": "a", "board_size": 9, "eligible": True, "version": "new"},
            {"game_id": "a", "board_size": 9, "eligible": True, "version": "old"},
            {"game_id": "b", "board_size": 9, "eligible": True},
            {"game_id": "c", "board_size": 13, "eligible": True},
            {"game_id": "d", "board_size": 19, "eligible": True},
            {"game_id": "e", "board_size": 19, "eligible": False},
        ]
        selected = loop.select_replay(records, 4)
        self.assertEqual([r["game_id"] for r in selected], ["a", "c", "d", "b"])
        self.assertEqual(selected[0]["version"], "new")

    def test_child_inherits_lock_and_interrupt_handler_reaps_process_group(self):
        output = self.root / "locking"
        output.mkdir()
        with loop.loop_lock(output) as descriptor:
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(30)"], pass_fds=(descriptor,)
            )
        try:
            with self.assertRaisesRegex(ValueError, "locked"), loop.loop_lock(output):
                pass
        finally:
            child.terminate()
            child.wait(timeout=5)
        with (
            loop.loop_lock(output) as descriptor,
            patch.object(loop.subprocess, "Popen") as popen,
            patch.object(loop.os, "killpg") as kill,
        ):
            popen.return_value.pid = 1234
            popen.return_value.wait.side_effect = [KeyboardInterrupt, 0]
            with self.assertRaises(KeyboardInterrupt):
                loop.run_process(["fake"], output / "child.log", descriptor)
            kill.assert_called_once_with(1234, signal.SIGINT)

    @unittest.skipUnless(
        importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime"),
        "install inference extra",
    )
    def test_real_cpu_two_iteration_loop_across_sizes_and_restart(self):
        output = self.root / "real"
        arguments = [
            "--output",
            str(output),
            "--checkpoint",
            str(self.source),
            "--bootstrap-data",
            str(self.corpus),
            "--device",
            "cpu",
            "--model-store",
            str(self.root / "real-models"),
            "--gate-games",
            "0",
            "--sizes",
            "9,13,19",
            "--games-per-iteration",
            "6",
            "--seed",
            "600",
            "--simulations",
            "4",
            "--widening",
            "0.01",
            "--max-moves",
            "24",
            "--concurrent-games",
            "6",
            "--inference-batch-size",
            "6",
            "--train-epochs",
            "1",
            "--steps-per-epoch",
            "1",
            "--positions-per-game",
            "1",
            "--validation-positions-per-game",
            "1",
            "--batch-size",
            "1",
            "--workers",
            "0",
            "--shuffle-buffer",
            "0",
            "--threads",
            "1",
            "--replay-iterations",
            "1",
            "--replay-games",
            "64",
        ]
        # Interrupt a real supervisor after native SGFs start appearing. Its
        # handler must reap the child group and release the inherited loop lock.
        with (self.root / "interrupted-process.log").open("w") as logfile:
            process = subprocess.Popen(
                [sys.executable, "-m", "continugo_train.loop", *arguments],
                stdout=logfile,
                stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + 60
            try:
                while process.poll() is None and not list(
                    output.glob("iteration-0001/selfplay/*/*.sgf")
                ):
                    if time.monotonic() >= deadline:
                        self.fail("native self-play did not produce SGFs before timeout")
                    time.sleep(0.02)
                self.assertIsNone(process.poll(), "loop ended before interruption injection")
                process.send_signal(signal.SIGINT)
                self.assertEqual(process.wait(timeout=30), 130)
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    process.wait(timeout=30)
        partial = {
            path: path.read_bytes() for path in output.glob("iteration-0001/selfplay/*/*.sgf")
        }
        # Real subprocesses, exports, SGFs and million-coordinate SGD steps.
        with redirect_stdout(io.StringIO()):
            first = loop.run_loop(loop.parse_args(["--output", str(output), "--resume"]))
            self.assertEqual(partial, {path: path.read_bytes() for path in partial})
            before = loop.digest(output / first["iterations"][0]["checkpoint"]["path"])
            state = loop.run_loop(
                loop.parse_args(["--output", str(output), "--resume", "--iterations", "2"])
            )
            loop.run_loop(loop.parse_args(["--output", str(output), "--resume"]))
        self.assertEqual(before, loop.digest(output / state["iterations"][0]["checkpoint"]["path"]))
        for number, iteration in enumerate(state["iterations"], 1):
            self.assertEqual({g["board_size"] for g in iteration["games"]}, {9, 13, 19})
            saved = loop.load_training(output / iteration["checkpoint"]["path"])
            self.assertEqual(
                saved["initialization"]["source"]["sha256"],
                iteration["parent_checkpoint"]["sha256"],
            )
            self.assertEqual(saved["epoch"], 1)
            self.assertTrue(iteration["model"])
            self.assertTrue(
                all(
                    r["path"].startswith((f"iteration-{number:04d}/", "bootstrap/"))
                    for r in iteration["replay"]
                )
            )


if __name__ == "__main__":
    unittest.main()
