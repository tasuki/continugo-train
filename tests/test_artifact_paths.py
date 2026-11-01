import io
from contextlib import redirect_stdout
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import test_loop as helpers

from continugo_train import archive, export, loop, paths, runtime
from continugo_train import train as trainer
from continugo_train.train import atomic_checkpoint


class ArtifactPathsTests(helpers.LoopHarness):
    def test_utc_prefix_and_atomic_collision_suffix(self):
        moment = datetime(2026, 11, 1, 12, 34, 56, tzinfo=UTC)
        base = self.root / "outputs"
        with patch.object(paths, "utc_now", return_value=moment):
            first = paths.new_run_directory("train", base)
            (first / "keep").write_text("old output")
            second = paths.new_run_directory("train", base)
        self.assertEqual(first.name, "20261101-123456-train")
        self.assertEqual(second.name, "20261101-123456-train-0002")
        self.assertEqual((first / "keep").read_text(), "old output")
        local = moment.astimezone(timezone(timedelta(hours=3)))
        self.assertEqual(paths.timestamp(local), "20261101-123456")

    def test_model_bundle_dates_sequence_and_identity_survive_republication(self):
        store = self.root / "dated-store"
        moment = datetime(2026, 11, 1, 12, 34, 56, tzinfo=UTC)
        with patch.object(archive, "utc_now", return_value=moment):
            first = archive.publish(store, "training", self.source)
            other = self.root / "other.pt"
            saved = loop.load_training(self.source)
            saved["run_id"] = "different checkpoint"
            atomic_checkpoint(other, saved)
            second = archive.publish(store, "training", other)
        self.assertRegex(Path(first["directory"]).name, r"^20261101-123456-0001-[0-9a-f]{12}$")
        self.assertRegex(Path(second["directory"]).name, r"^20261101-123456-0002-[0-9a-f]{12}$")
        self.assertEqual(archive.verify_bundle(first)["created_at"], moment.isoformat())
        with patch.object(archive, "utc_now", return_value=moment + timedelta(days=1)):
            reused = archive.publish(store, "training", self.source)
        self.assertEqual(reused, first)
        self.assertEqual(len(list((store / "training").iterdir())), 2)
        self.assertEqual(
            (store / "LATEST-TRAINING").read_text().strip(),
            str(Path(first["directory"]).relative_to(store)),
        )

    def test_short_hash_prefix_is_not_the_identity(self):
        parent = self.root / "collisions"
        parent.mkdir()
        first_hash = "a" * 12 + "1" * 52
        second_hash = "a" * 12 + "2" * 52
        directories = []
        for number, checksum in enumerate([first_hash, second_hash], 1):
            directory = parent / f"20261101-123456-{number:04d}-{checksum[:12]}"
            directory.mkdir()
            atomic_checkpoint(
                directory / "recovery.pt", {"kind": "accepted", "checkpoint_sha256": checksum}
            )
            directories.append(directory)
        self.assertEqual(archive.find_bundle(parent, "accepted", first_hash), directories[0])
        self.assertEqual(archive.find_bundle(parent, "accepted", second_hash), directories[1])

    def test_new_loop_defaults_inside_repo_and_auto_dated_output(self):
        self.assertEqual(Path(loop.LoopConfig().model_store), loop.ROOT / "models")
        args = self.args(self.root / "unused")
        args.output = None

        def allocate(label):
            return paths.new_run_directory(label, self.root / "auto")

        with (
            patch.object(loop, "new_run_directory", side_effect=allocate),
            patch.object(loop, "run_process", self.jobs),
            patch.object(loop, "verify_export", self.jobs.bundle),
            redirect_stdout(io.StringIO()),
        ):
            state = loop.run_loop(args)
        output = Path(state["root"])
        self.assertEqual(output.parent, self.root / "auto")
        self.assertRegex(output.name, r"^\d{8}-\d{6}-loop$")
        self.assertTrue((output / "loop.pt").is_file())
        with patch.object(loop, "new_run_directory") as allocate:
            with self.assertRaisesRegex(ValueError, "resume requires --output"):
                loop.run_loop(loop.parse_args(["--resume"]))
            allocate.assert_not_called()

    def test_warm_start_auto_allocates_a_fresh_dated_training_run(self):
        def allocate(label):
            return paths.new_run_directory(label, self.root / "auto")

        with (
            patch.object(trainer, "new_run_directory", side_effect=allocate),
            redirect_stdout(io.StringIO()),
        ):
            latest = trainer.train(
                trainer.parse_args(
                    [
                        "--data",
                        str(self.corpus),
                        "--warm-start",
                        str(self.source),
                        "--device",
                        "cpu",
                        "--threads",
                        "1",
                        "--epochs",
                        "1",
                    ]
                )
            )
        self.assertEqual(latest.parent.parent, self.root / "auto")
        self.assertRegex(latest.parent.name, r"^\d{8}-\d{6}-train$")
        self.assertTrue(latest.is_file())

    def test_export_and_neural_launcher_default_to_dated_outputs(self):
        def allocate(label):
            return paths.new_run_directory(label, self.root / "auto")

        with (
            patch.object(export, "new_run_directory", side_effect=allocate),
            patch.object(export, "export_checkpoint", return_value=0.0) as exporter,
            redirect_stdout(io.StringIO()),
        ):
            export.main(["--checkpoint", str(self.source)])
        output = exporter.call_args.args[1]
        self.assertRegex(output.parent.name, r"^\d{8}-\d{6}-export$")
        self.assertEqual(output.name, "model")
        with (
            patch.object(runtime, "new_run_directory", side_effect=allocate),
            patch.object(runtime, "inference_environment", return_value=(Path("/fake.so"), {})),
            patch.object(runtime.os, "execvpe") as execute,
            redirect_stdout(io.StringIO()),
        ):
            runtime.main(["selfplay", "--device", "cpu", "--model", "some-model"])
        command = execute.call_args.args[1]
        output = Path(command[command.index("--output") + 1])
        self.assertEqual(output.parent, self.root / "auto")
        self.assertRegex(output.name, r"^\d{8}-\d{6}-selfplay$")
