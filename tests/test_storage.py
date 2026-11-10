import io
import shutil
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from rating_helpers import EvaluationHarness
from test_acceptance import ArenaJobs

from continugo_train import archive, loop, plot_ratings, promote, ratings, retain, round_robin


class RunOwnedStorageTests(EvaluationHarness):
    def test_no_automatic_model_copies_and_resume_without_external_inputs(self):
        output = self.root / "campaign"
        with patch.object(archive, "publish", side_effect=AssertionError("automatic backup")):
            state = self.run_loop(output)
            shutil.rmtree(self.corpus)
            self.source.unlink()
            resumed = self.run_loop(output, resume=True, extra=["--iterations", "2"])
        self.assertEqual(len(resumed["iterations"]), 2)
        self.assertEqual(
            resumed["iterations"][1]["parent_checkpoint"], state["training"]["checkpoint"]
        )
        store = Path(state["config"]["model_store"])
        for name in ["training", "experimental", "LATEST-TRAINING", "LATEST-EXPERIMENTAL"]:
            self.assertFalse((store / name).exists())

    def test_legacy_migration_needs_no_global_backups_and_launches_no_jobs(self):
        output = self.root / "legacy-campaign"
        state = self.run_loop(output)
        missing = {"directory": str(self.root / "deleted-backup"), "sha256": "0" * 64}
        state.pop("storage_version")
        for key in [
            "initial_training_archive",
            "initial_actor_training_archive",
            "initial_experimental_archive",
        ]:
            state[key] = missing
        for job in state["iterations"]:
            job.update(training_archive=missing, experimental_archive=missing)
        for role in ["training", "actor"]:
            state[role]["archive"] = missing
        loop.atomic_checkpoint(output / "loop.pt", state)
        before = (output / "loop.pt").read_bytes()
        calls = len(self.jobs.calls)
        migrated = self.run_loop(output, resume=True, extra=["--migrate-storage"])
        self.assertEqual(len(self.jobs.calls), calls)
        self.assertEqual((output / "before-run-owned-storage.pt").read_bytes(), before)
        self.assertEqual(migrated["storage_version"], 2)
        self.assertNotIn("archive", migrated["actor"])
        self.assertNotIn("experimental_archive", migrated["iterations"][0])
        self.assertEqual(migrated["config"], state["config"])
        self.assertEqual(migrated["training"]["checkpoint"], state["training"]["checkpoint"])
        self.run_loop(output, resume=True, extra=["--migrate-storage"])
        self.assertEqual((output / "before-run-owned-storage.pt").read_bytes(), before)
        self.assertEqual(len(self.jobs.calls), calls)
        self.run_loop(output, resume=True, extra=["--iterations", "2"])

    def test_storage_only_migration_does_not_register_models_into_a_replacement_store(self):
        self.jobs = ArenaJobs()
        output = self.root / "old-store-campaign"
        state = self.run_loop(output, extra=["--reference-games", "2"])
        replacement = self.root / "replacement-store"
        archive.initialize_store(replacement)
        state["config"]["model_store"] = str(replacement)
        loop.atomic_checkpoint(output / "loop.pt", state)
        calls = len(self.jobs.calls)
        self.run_loop(output, resume=True, extra=["--migrate-storage"])
        self.assertEqual(len(self.jobs.calls), calls)
        with ratings.transaction(replacement) as (tables, _):
            self.assertEqual(set(tables.networks), {ratings.REFERENCE_ID})
        self.assertFalse((replacement / "ratings/inference").exists())

    def test_migration_refuses_corrupt_run_owned_files(self):
        output = self.root / "corrupt"
        state = self.run_loop(output)
        state.pop("storage_version")
        loop.atomic_checkpoint(output / "loop.pt", state)
        checkpoint = output / state["training"]["checkpoint"]["path"]
        checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
        before = (output / "loop.pt").read_bytes()
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            self.run_loop(output, resume=True, extra=["--migrate-storage"])
        self.assertEqual((output / "loop.pt").read_bytes(), before)

    def test_run_owned_opponents_disappear_but_rating_history_and_plot_survive(self):
        self.jobs = ArenaJobs()
        output = self.root / "rated-campaign"
        state = self.run_loop(output, extra=["--reference-games", "2"])
        store = state["config"]["model_store"]
        available = round_robin.available_models(store)
        self.assertEqual(len(available), 2)
        self.assertTrue(all(isinstance(source, dict) for _, source in available))
        before = plot_ratings.collect_points(store)
        shutil.rmtree(output)
        self.assertEqual(round_robin.available_models(store), [])
        self.assertEqual(plot_ratings.collect_points(store), before)
        with ratings.transaction(store) as (tables, _):
            self.assertEqual(len(tables.networks), 3)
            self.assertEqual(len(tables.matches), 4)

    def test_changed_registered_graph_is_not_silently_used(self):
        self.jobs = ArenaJobs()
        output = self.root / "changed-graph"
        state = self.run_loop(output, extra=["--reference-games", "2"])
        graph = output / state["actor"]["model"]["encoder"]["path"]
        graph.write_text("changed")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            round_robin.available_models(state["config"]["model_store"])

    def test_deliberate_retention_survives_run_loss_without_acceptance(self):
        self.jobs = ArenaJobs()
        output = self.root / "retain-campaign"
        state = self.run_loop(output, extra=["--reference-games", "2"])
        before = (output / "loop.pt").read_bytes()
        with redirect_stdout(io.StringIO()):
            reference = retain.retain(output)
            self.assertEqual(retain.retain(output), reference)
        self.assertEqual((output / "loop.pt").read_bytes(), before)
        self.assertEqual(Path(reference["directory"]).parent.name, "retained")
        shutil.rmtree(output)
        shutil.rmtree(self.corpus)
        self.source.unlink()
        metadata = archive.verify_bundle(reference)
        self.assertEqual(metadata["kind"], "retained")
        self.assertEqual(metadata["checkpoint_sha256"], state["training"]["checkpoint"]["sha256"])
        available = round_robin.available_models(state["config"]["model_store"])
        self.assertEqual(len(available), 1)
        self.assertEqual(available[0][1], Path(reference["directory"]))
        self.assertIsNone(promote.accepted_head(state["config"]["model_store"]))

    def test_model_and_approval_stores_cannot_live_in_scratch_or_runs(self):
        for name in ["runs", "scratch"]:
            with self.subTest(name=name):
                store = self.root / name / "not-durable"
                with (
                    patch.object(archive, "ROOT", self.root),
                    self.assertRaisesRegex(ValueError, "outside disposable"),
                ):
                    archive.initialize_store(store)
                with (
                    patch.object(promote, "ROOT", self.root),
                    self.assertRaisesRegex(ValueError, "outside disposable"),
                ):
                    promote.recover(store)
                self.assertFalse(store.exists())

    def test_retain_rejects_incomplete_or_unknown_round(self):
        output = self.root / "incomplete"
        self.jobs.fault = "epoch_gap"
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop(output)
        for number in [None, 1, 999, -1]:
            with (
                self.subTest(number=number),
                self.assertRaisesRegex(ValueError, "completed iteration"),
            ):
                retain.retain(output, number)
