import importlib.util
import io
import os
import shutil
import subprocess
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import test_loop as helpers
import torch

from continugo_train import archive, export, loop, promote, ratings, retain, status


class PromotionTests(helpers.LoopHarness):
    def make_candidates(self, count=1):
        output = self.root / "experiment"
        state = self.run_loop(output, extra=["--iterations", str(count)])
        self.store = Path(state["config"]["model_store"])
        # Publication fixtures explicitly retain candidates; loops do not.
        state["initial_experimental_archive"] = retain.retain(output, 0)
        for iteration in state["iterations"]:
            iteration["experimental_archive"] = retain.retain(output, iteration["number"])
        return output, state

    def approve(self, state, index=0):
        iteration = state["iterations"][index]
        with patch.object(promote, "verify_model"):
            return promote.approve(
                self.store,
                iteration["experimental_archive"]["directory"],
                iteration["checkpoint"]["sha256"],
                approved_by="fixture-user",
                reason="Explicit approval fixture; not a real strength assertion",
                reference_strength_confirmed=True,
            )

    def test_no_publication_or_rating_acceptance_without_approval_including_startup(self):
        _, state = self.make_candidates(2)
        self.assertIsNone(state["accepted"])
        self.assertFalse((self.store / "accepted").exists())
        self.assertFalse((self.store / "LATEST-ACCEPTED").exists())
        for operation in [archive.publish, archive._publish]:
            with self.assertRaisesRegex(ValueError, "explicit user approval"):
                operation(self.store, "accepted", self.source)
        with ratings.transaction(self.store) as (tables, _):
            self.assertTrue(all(row["accepted"] == "0" for row in tables.networks.values()))
        self.assertEqual(state["actor"]["checkpoint"], state["iterations"][-1]["checkpoint"])
        self.assertEqual(state["training"]["checkpoint"], state["iterations"][-1]["checkpoint"])

    def test_approval_pins_one_checkpoint_without_switching_training_or_actor(self):
        output, state = self.make_candidates(2)
        for name in ["LATEST-TRAINING", "LATEST-EXPERIMENTAL"]:
            self.assertFalse((self.store / name).exists())
        control = (output / "loop.pt").read_bytes()
        result = self.approve(state, 0)
        metadata = archive.verify_bundle(result)
        self.assertEqual(
            metadata["checkpoint_sha256"], state["iterations"][0]["checkpoint"]["sha256"]
        )
        self.assertTrue((Path(result["directory"]) / "approval.pt").is_file())
        self.assertEqual(metadata["version"], 2)
        self.assertEqual((output / "loop.pt").read_bytes(), control)
        for name in ["LATEST-TRAINING", "LATEST-EXPERIMENTAL"]:
            self.assertFalse((self.store / name).exists())
        resumed = self.run_loop(output, resume=True)
        self.assertEqual(resumed["actor"], state["actor"])
        self.assertEqual(resumed["training"], state["training"])
        self.assertEqual(resumed["accepted"]["archive"], result)
        with ratings.transaction(self.store) as (tables, _):
            self.assertEqual(sum(row["accepted"] == "1" for row in tables.networks.values()), 1)

    def test_gate_passing_cannot_be_used_as_an_approval_or_blanket_authorization(self):
        _, state = self.make_candidates()
        iteration = state["iterations"][0]
        candidate = iteration["experimental_archive"]["directory"]
        checksum = iteration["checkpoint"]["sha256"]
        for supplied_hash, confirmation in [("0" * 64, True), (checksum, False)]:
            with self.assertRaises(ValueError):
                promote.approve(
                    self.store,
                    candidate,
                    supplied_hash,
                    approved_by="user",
                    reason="review",
                    reference_strength_confirmed=confirmation,
                )
        self.assertFalse((self.store / "LATEST-ACCEPTED").exists())
        self.assertEqual(promote.recover(self.store), {})
        self.assertFalse((self.store / "LATEST-ACCEPTED").exists())

    def test_duplicate_older_approval_never_rolls_back_newer_accepted_head(self):
        _, state = self.make_candidates(2)
        first = self.approve(state, 0)
        second = self.approve(state, 1)
        head = (self.store / "LATEST-ACCEPTED").read_bytes()
        self.assertEqual(self.approve(state, 0), first)
        self.assertEqual((self.store / "LATEST-ACCEPTED").read_bytes(), head)
        self.assertEqual(promote.accepted_head(self.store)["archive"], second)
        with self.assertRaisesRegex(ValueError, "latest committed approval"):
            archive.latest_pointer(self.store, "accepted", Path(first["directory"]))
        self.assertEqual(len(list((self.store / "accepted").iterdir())), 2)
        self.assertEqual(len(list((self.store / "approvals").glob("*.receipt.pt"))), 2)

    def test_concurrent_duplicate_requests_publish_once(self):
        _, state = self.make_candidates()
        iteration = state["iterations"][0]

        def request(_):
            return promote.approve(
                self.store,
                iteration["experimental_archive"]["directory"],
                iteration["checkpoint"]["sha256"],
                approved_by="fixture-user",
                reason="Explicit concurrent test approval",
                reference_strength_confirmed=True,
            )

        with patch.object(promote, "verify_model"), ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(request, range(3)))
        self.assertTrue(all(result == results[0] for result in results))
        self.assertEqual(len(list((self.store / "accepted").iterdir())), 1)
        self.assertEqual(len(list((self.store / "approvals").glob("*.receipt.pt"))), 1)

    def test_recover_authorized_intent_after_publication_gap_and_losing_runs(self):
        output, state = self.make_candidates()
        atomic = promote.atomic_checkpoint

        def gap(path, value):
            if str(path).endswith(".receipt.pt"):
                raise KeyboardInterrupt
            atomic(path, value)

        with (
            patch.object(promote, "atomic_checkpoint", side_effect=gap),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.approve(state)
        self.assertFalse((self.store / "LATEST-ACCEPTED").exists())
        self.assertEqual(len(list((self.store / "accepted").iterdir())), 1)
        shutil.rmtree(output)
        shutil.rmtree(self.corpus)
        self.source.unlink()
        recovered = promote.recover(self.store)
        checksum = state["iterations"][0]["checkpoint"]["sha256"]
        result = recovered[checksum]
        archive.verify_bundle(result)
        self.assertEqual(promote.accepted_head(self.store)["archive"], result)
        self.assertEqual(len(list((self.store / "accepted").iterdir())), 1)

    def test_receipt_before_head_gap_is_repaired_without_reapproving(self):
        _, state = self.make_candidates()
        pointer = archive.latest_pointer

        def gap(store, kind, destination):
            if kind == "accepted":
                raise KeyboardInterrupt
            pointer(store, kind, destination)

        with (
            patch.object(archive, "latest_pointer", side_effect=gap),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.approve(state)
        self.assertFalse((self.store / "LATEST-ACCEPTED").exists())
        self.assertEqual(len(list((self.store / "approvals").glob("*.receipt.pt"))), 1)
        result = promote.recover(self.store)
        self.assertTrue(result)
        self.assertIsNotNone(promote.accepted_head(self.store))

    def test_changed_authorization_and_model_are_not_reused(self):
        _, state = self.make_candidates()
        publish = archive.publish

        def gap(store, kind, checkpoint, **kwargs):
            if kind == "accepted":
                raise KeyboardInterrupt
            return publish(store, kind, checkpoint, **kwargs)

        with (
            patch.object(archive, "publish", side_effect=gap),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.approve(state)
        path = next((self.store / "approvals").glob("*.pt"))
        saved = torch.load(path, weights_only=True)
        saved["approved_by"] = "changed user"
        path.chmod(0o644)
        loop.atomic_checkpoint(path, saved)
        with self.assertRaisesRegex(ValueError, "changed approval intent"):
            promote.recover(self.store)
        self.assertFalse((self.store / "LATEST-ACCEPTED").exists())

    def test_default_actor_advances_but_accepted_never_does(self):
        output = self.root / "actor-advancement"
        state = self.run_loop(output, extra=["--iterations", "2"])
        first, second = state["iterations"]
        self.assertEqual(second["parent_model"], first["model"])
        self.assertEqual(second["parent_checkpoint"], first["checkpoint"])
        self.assertEqual(state["actor"]["model"], second["model"])
        self.assertIsNone(state["accepted"])
        self.assertFalse((Path(state["config"]["model_store"]) / "LATEST-ACCEPTED").exists())

    def test_training_and_actor_updates_do_not_launch_evaluation(self):
        state = self.run_loop(self.root / "periodic", extra=["--iterations", "3"])
        self.assertTrue(all(job["actor_updated"] for job in state["iterations"]))
        self.assertFalse(any("--candidate" in c for c in self.jobs.calls))
        self.assertTrue(all("gate" not in job for job in state["iterations"]))
        self.assertIsNone(state["accepted"])

    def test_actor_selection_checkpoint_gap_is_repaired_without_retraining(self):
        output = self.root / "actor-gap"
        save = loop.LoopRunner.save

        def gap(runner):
            save(runner)
            if runner.state["iterations"] and runner.state["iterations"][0]["completed"]:
                raise KeyboardInterrupt

        with patch.object(loop.LoopRunner, "save", gap), self.assertRaises(KeyboardInterrupt):
            self.run_loop(output)
        calls = len(self.stage_calls("continugo_train.train"))
        state = self.run_loop(output, resume=True)
        self.assertEqual(len(self.stage_calls("continugo_train.train")), calls)
        self.assertTrue(state["iterations"][0]["completed"])
        self.assertEqual(state["actor"]["checkpoint"], state["iterations"][0]["checkpoint"])
        self.assertFalse((Path(state["config"]["model_store"]) / "LATEST-EXPERIMENTAL").exists())
        self.assertIsNone(state["accepted"])

    def test_approval_for_one_hash_cannot_publish_a_different_candidate(self):
        _, state = self.make_candidates(2)
        accepted = self.approve(state, 0)
        metadata = archive.verify_bundle(accepted)
        other = Path(state["iterations"][1]["experimental_archive"]["directory"])
        before = (self.store / "LATEST-ACCEPTED").read_bytes()
        with self.assertRaisesRegex(ValueError, "does not authorize"):
            archive.publish(
                self.store,
                "accepted",
                other / "checkpoint.pt",
                model=other,
                approval=metadata["approval"],
            )
        self.assertEqual((self.store / "LATEST-ACCEPTED").read_bytes(), before)
        self.assertEqual(len(list((self.store / "accepted").iterdir())), 1)

    def test_changed_model_after_approval_intent_cannot_be_published(self):
        _, state = self.make_candidates()
        publish = archive.publish

        def gap(store, kind, checkpoint, **kwargs):
            if kind == "accepted":
                raise KeyboardInterrupt
            return publish(store, kind, checkpoint, **kwargs)

        with (
            patch.object(archive, "publish", side_effect=gap),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.approve(state)
        model = Path(state["iterations"][0]["experimental_archive"]["directory"]) / "encoder.onnx"
        model.chmod(0o644)
        model.write_text("changed graph")
        with self.assertRaisesRegex(ValueError, "artifact missing or changed"):
            promote.recover(self.store)
        self.assertFalse((self.store / "LATEST-ACCEPTED").exists())

    def test_starting_new_loop_does_not_accept_its_initial_checkpoint_or_change_approved_head(self):
        _, state = self.make_candidates()
        accepted = self.approve(state)
        head = (self.store / "LATEST-ACCEPTED").read_bytes()
        other = self.run_loop(self.root / "another-experiment")
        self.assertEqual(other["accepted"]["archive"], accepted)
        self.assertEqual((self.store / "LATEST-ACCEPTED").read_bytes(), head)
        self.assertNotEqual(
            other["initial_checkpoint"]["sha256"], other["accepted"]["checkpoint_sha256"]
        )
        self.assertEqual(len(list((self.store / "accepted").iterdir())), 1)

    def test_status_is_read_only_and_displays_the_three_independent_roles(self):
        output, state = self.make_candidates(2)
        accepted = self.approve(state, 0)
        before = (output / "loop.pt").read_bytes()
        info = status.inspect_loop(output)
        self.assertEqual(info["accepted"]["archive"], accepted)
        self.assertEqual(info["actor"], state["actor"])
        self.assertEqual(info["training"], state["training"])
        with redirect_stdout(io.StringIO()) as result:
            status.main(["--output", str(output)])
        self.assertIn("Experimental actor", result.getvalue())
        self.assertIn("Accepted:", result.getvalue())
        self.assertEqual((output / "loop.pt").read_bytes(), before)

    def test_historical_accepted_bundle_is_preserved_without_authorizing_new_models(self):
        _, state = self.make_candidates()
        original = Path(state["initial_experimental_archive"]["directory"])
        legacy = self.store / "accepted" / "historical-pre-manual-approval"
        legacy.parent.mkdir()
        shutil.copytree(original, legacy)
        legacy.chmod(0o755)
        path = legacy / "recovery.pt"
        saved = torch.load(path, weights_only=True)
        saved.update(version=1, kind="accepted")
        path.chmod(0o644)
        loop.atomic_checkpoint(path, saved)
        archive.seal_file(path)
        archive.seal_directories(legacy)
        # Fixture emulates an existing v4 publication; production code cannot
        # create this historical format through archive.publish anymore.
        (self.store / "LATEST-ACCEPTED").write_text(str(legacy.relative_to(self.store)) + "\n")
        before = (self.store / "LATEST-ACCEPTED").read_bytes()
        continued = self.run_loop(self.root / "legacy-read-only")
        self.assertIsNone(continued["accepted"]["approval"])
        self.assertEqual((self.store / "LATEST-ACCEPTED").read_bytes(), before)
        self.assertEqual(len(list((self.store / "accepted").iterdir())), 1)
        self.assertEqual(continued["accepted"]["checkpoint_sha256"], saved["checkpoint_sha256"])

    def test_native_generation_changes_are_refused_even_when_evaluation_is_disabled(self):
        binary = self.root / "frozen-selfplay"
        shutil.copyfile(loop.LoopConfig().selfplay_binary, binary)
        binary.chmod(0o755)
        output = self.root / "frozen-code"
        state = self.run_loop(output, extra=["--selfplay-binary", str(binary)])
        before = (output / "loop.pt").read_bytes()
        with binary.open("ab") as stream:
            stream.write(b"changed executable")
        with self.assertRaisesRegex(ValueError, "generation/feature executable changed"):
            self.run_loop(output, resume=True)
        self.assertEqual((output / "loop.pt").read_bytes(), before)
        self.assertIsNone(state["accepted"])

    def test_old_auto_promotion_loop_is_not_silently_resumed(self):
        output, _ = self.make_candidates()
        path = output / "loop.pt"
        state = torch.load(path, weights_only=True)
        state["version"] = 4
        loop.atomic_checkpoint(path, state)
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "new v6 loop"):
            self.run_loop(output, resume=True)
        self.assertEqual(path.read_bytes(), before)

    @unittest.skipUnless(
        importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime"),
        "install inference extra",
    )
    def test_real_export_is_verified_by_the_explicit_approval_cli(self):
        model = self.root / "real-model"
        export.export_checkpoint(self.source, model)
        store = self.root / "real-approval-store"
        candidate = archive.publish(store, "experimental", self.source, model=model)
        arguments = [
            "--model-store",
            str(store),
            "--candidate",
            candidate["directory"],
            "--checkpoint-sha256",
            archive.digest(self.source),
            "--approved-by",
            "fixture-user",
            "--reason",
            "Real-export approval test in an isolated temporary store",
            "--confirm-much-stronger-than-reference",
        ]
        with (
            patch.object(promote, "verify_model", wraps=promote.verify_model) as verification,
            redirect_stdout(io.StringIO()),
        ):
            promote.main(arguments)
            promote.main(arguments)
        self.assertEqual(verification.call_count, 1)
        head = promote.accepted_head(store)
        self.assertEqual(head["checkpoint_sha256"], archive.digest(self.source))
        archive.verify_bundle(head["archive"])

    @unittest.skipUnless(
        importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime"),
        "install inference extra",
    )
    def test_two_approval_processes_serialize_one_real_publication(self):
        model = self.root / "concurrent-real-model"
        export.export_checkpoint(self.source, model)
        store = self.root / "concurrent-real-store"
        candidate = archive.publish(store, "experimental", self.source, model=model)
        command = [
            sys.executable,
            "-m",
            "continugo_train.promote",
            "--model-store",
            str(store),
            "--candidate",
            candidate["directory"],
            "--checkpoint-sha256",
            archive.digest(self.source),
            "--approved-by",
            "fixture-user",
            "--reason",
            "Concurrent real-model test approval in temporary store",
            "--confirm-much-stronger-than-reference",
        ]
        children = [
            subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            )
            for _ in range(2)
        ]
        try:
            for child in children:
                stdout, stderr = child.communicate(timeout=45)
                self.assertEqual(child.returncode, 0, stdout + stderr)
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.wait()
        self.assertEqual(len(list((store / "accepted").iterdir())), 1)
        self.assertEqual(len(list((store / "approvals").glob("*.receipt.pt"))), 1)
        self.assertIsNotNone(promote.accepted_head(store))

    def test_approval_journal_cannot_be_put_in_disposable_runs(self):
        with (
            patch.object(promote, "ROOT", self.root),
            self.assertRaisesRegex(ValueError, "outside disposable runs"),
        ):
            promote.recover(self.root / "runs" / "not-durable")
        self.assertFalse((self.root / "runs").exists())

    def test_cli_refuses_missing_approval_and_recovery_cannot_issue_new_permission(self):
        self.make_candidates()
        for arguments in [
            ["--model-store", str(self.store)],
            ["--model-store", str(self.store), "--recover", "--reason", "new permission"],
        ]:
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as error:
                promote.main(arguments)
            self.assertEqual(error.exception.code, 1)
        self.assertFalse((self.store / "LATEST-ACCEPTED").exists())


if __name__ == "__main__":
    unittest.main()
