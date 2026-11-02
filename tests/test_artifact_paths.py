import io
import shutil
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
        self.assertRegex(
            Path(first["directory"]).name, r"^20261101-123456-0001-test1-b1c4-s1-d8-[0-9a-f]{12}$"
        )
        self.assertRegex(
            Path(second["directory"]).name, r"^20261101-123456-0002-test1-b1c4-s1-d8-[0-9a-f]{12}$"
        )
        self.assertEqual(archive.verify_bundle(first)["created_at"], moment.isoformat())
        with patch.object(archive, "utc_now", return_value=moment + timedelta(days=1)):
            reused = archive.publish(store, "training", self.source)
        self.assertEqual(reused, first)
        self.assertEqual(len(list((store / "training").iterdir())), 2)
        self.assertEqual(
            (store / "LATEST-TRAINING").read_text().strip(),
            str(Path(first["directory"]).relative_to(store)),
        )

    def test_sgf_days_use_first_retention_and_deduplicate_across_midnight(self):
        store = self.root / "daily-store"
        first_day = datetime(2026, 11, 1, 23, 30, tzinfo=UTC)
        with patch.object(archive, "utc_now", return_value=first_day):
            reference = archive.publish(store, "training", self.source)
        bundle = Path(reference["directory"])
        replay = next((bundle / "replay").glob("*.sgf"))
        blobs = store / "sgf-blobs"
        original = blobs / "2026-11-01" / replay.name
        self.assertEqual(original.stat().st_ino, replay.stat().st_ino)
        self.assertEqual(list(blobs.glob("*.sgf")), [])
        evidence = store / "ratings" / "games"
        with patch.object(archive, "utc_now", return_value=first_day + timedelta(days=1)):
            archive.retain_evidence(store, [(replay, replay.stem)], evidence)
            self.assertFalse((blobs / "2026-11-02").exists())
            self.assertEqual((evidence / replay.name).stat().st_ino, original.stat().st_ino)
            # Different raw bytes (same valid game) are a genuinely new blob.
            source = self.root / "new-import.sgf"
            source.write_bytes(replay.read_bytes() + b"\n")
            checksum = archive.digest(source)
            archive.retain_evidence(store, [(source, checksum)], evidence)
        self.assertTrue((blobs / "2026-11-02" / f"{checksum}.sgf").is_file())
        self.assertEqual(len(list(blobs.rglob(replay.name))), 1)
        shutil.rmtree(self.corpus)
        self.source.unlink()
        archive.verify_bundle(reference)

    def test_sgf_retention_day_is_utc_not_local_calendar_day(self):
        store = self.root / "utc-blob-store"
        local = datetime(2026, 11, 2, 1, tzinfo=timezone(timedelta(hours=3)))
        with patch.object(archive, "utc_now", return_value=local):
            archive.publish(store, "training", self.source)
        self.assertEqual([path.name for path in (store / "sgf-blobs").iterdir()], ["2026-11-01"])

    def test_flat_legacy_blobs_are_reused_without_new_daily_duplicates(self):
        store = self.root / "legacy-blob-store"
        reference = archive.publish(store, "training", self.source)
        blobs = store / "sgf-blobs"
        for path in list(blobs.glob("????-??-??/*.sgf")):
            path.rename(blobs / path.name)
        for path in list(blobs.iterdir()):
            if path.is_dir():
                path.rmdir()
        replay = next((Path(reference["directory"]) / "replay").glob("*.sgf"))
        with patch.object(archive, "utc_now", return_value=datetime(2030, 1, 1, tzinfo=UTC)):
            archive.retain_evidence(store, [(replay, replay.stem)], store / "ratings/games")
        self.assertFalse((blobs / "2030-01-01").exists())
        self.assertEqual((blobs / replay.name).stat().st_ino, replay.stat().st_ino)
        archive.verify_bundle(reference)

    def test_duplicate_corrupt_and_symlinked_blob_paths_fail_closed(self):
        store = self.root / "bad-blob-store"
        reference = archive.publish(store, "training", self.source)
        blobs = store / "sgf-blobs"
        original = next(blobs.glob("????-??-??/*.sgf"))
        source = self.root / "source.sgf"
        source.write_bytes(original.read_bytes())
        checksum = archive.digest(source)
        legacy = blobs / original.name
        shutil.copyfile(original, legacy)
        with self.assertRaisesRegex(ValueError, "duplicate SGF blob"):
            archive.retain_evidence(store, [(source, checksum)], store / "ratings/games")
        legacy.unlink()
        original.chmod(0o644)
        original.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "blob corrupted"):
            archive.retain_evidence(store, [(source, checksum)], store / "ratings/games")
        original.write_bytes(source.read_bytes())
        original.chmod(0o444)
        outside = self.root / "outside"
        outside.mkdir()
        (blobs / "2030-01-01").symlink_to(outside, target_is_directory=True)
        source.write_bytes(source.read_bytes() + b"\n")
        with (
            patch.object(archive, "utc_now", return_value=datetime(2030, 1, 1, tzinfo=UTC)),
            self.assertRaisesRegex(ValueError, "retention day cannot be a symlink"),
        ):
            archive.retain_evidence(
                store, [(source, archive.digest(source))], store / "ratings/games"
            )
        self.assertEqual(list(outside.iterdir()), [])
        archive.verify_bundle(reference)

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

    def test_unguided_arena_launcher_does_not_require_inference_dependencies(self):
        with (
            patch.object(runtime, "inference_environment") as inference,
            patch.object(runtime.os, "execvpe") as execute,
        ):
            runtime.main(
                [
                    "arena",
                    "--unguided",
                    "--reference-random",
                    "--output",
                    str(self.root / "matches"),
                ]
            )
        inference.assert_not_called()
        command = execute.call_args.args[1]
        self.assertIn("--unguided", command)
        self.assertNotIn("--ort-library", command)

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
