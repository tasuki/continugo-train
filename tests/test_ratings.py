import csv
import importlib.util
import io
import shutil
import subprocess
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import UTC, datetime
from unittest.mock import patch

import test_acceptance as arena_helpers
import test_loop as helpers

from continugo_train import archive, export, loop, ratings
from continugo_train.features import inspect_game
from continugo_train.runtime import inference_environment


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


class RatingsTests(helpers.LoopHarness):
    def setUp(self):
        super().setUp()
        self.jobs = arena_helpers.ArenaJobs()
        self.store = self.root / "models"
        self.directory = self.store / "ratings"

    def rated_loop(self, output, extra=()):
        return self.run_loop(output, extra=["--gate-games", "2", "--reference-games", "2", *extra])

    def rows(self):
        return read_csv(self.directory / "networks.csv")

    def matches(self):
        return read_csv(self.directory / "matches.csv")

    def test_two_csvs_all_networks_fixed_reference_and_durable_evidence(self):
        output = self.root / "loop"
        state = self.rated_loop(output)
        rows, matches = self.rows(), self.matches()
        self.assertEqual(
            {p.name for p in self.directory.iterdir()},
            {".lock", "networks.csv", "matches.csv", "games"},
        )
        self.assertEqual(
            len(matches), 6
        )  # Initial reference, experimental match, candidate reference.
        self.assertEqual(len(rows), 3)
        candidate = next(
            r for r in rows if r["network_id"] == state["iterations"][0]["checkpoint"]["sha256"]
        )
        self.assertEqual((candidate["samples_seen"], candidate["data_positions"]), ("2", "20"))
        self.assertEqual(sum(r["accepted"] == "1" for r in rows), 0)
        reference = next(r for r in rows if r["network_id"] == ratings.REFERENCE_ID)
        self.assertEqual(
            (reference["rating"], reference["deviation"], reference["status"]),
            ("100.0", "0.0", "fixed"),
        )
        for row in rows:
            if row["network_id"] != ratings.REFERENCE_ID:
                self.assertEqual(row["status"], "rated")
                self.assertEqual(float(row["rating"]), 100)
                self.assertLess(float(row["deviation"]), 350)
                self.assertEqual(row["games"], "4")
        for row in matches:
            path = self.directory / "games" / f"{row['sgf_sha256']}.sgf"
            self.assertEqual(archive.digest(path), row["sgf_sha256"])
            blobs = list((self.store / "sgf-blobs").glob(f"????-??-??/{path.name}"))
            self.assertEqual(len(blobs), 1)
            self.assertEqual(path.stat().st_ino, blobs[0].stat().st_ino)
        bundle = state["actor"]["archive"]
        self.assertIsNone(state["accepted"])
        self.assertEqual(
            archive.verify_bundle(bundle)["network_name"],
            state["actor"]["model"]["network_name"],
        )
        # Exact SGFs and CSVs do not depend on disposable runs or the source corpus.
        before = (self.directory / "networks.csv").read_bytes()
        shutil.rmtree(output)
        shutil.rmtree(self.corpus)
        self.source.unlink()
        ratings.refresh(self.store)
        self.assertEqual((self.directory / "networks.csv").read_bytes(), before)
        self.assertEqual(len(list((self.directory / "games").glob("*.sgf"))), 6)
        with redirect_stdout(io.StringIO()) as result:
            ratings.main(["--model-store", str(self.store), "--accepted-only"])
        accepted = list(csv.DictReader(io.StringIO(result.getvalue())))
        self.assertEqual(accepted, [])

    def test_resume_reimport_and_future_clock_do_not_change_ratings(self):
        output = self.root / "loop"
        state = self.rated_loop(output)
        before = {
            name: (self.directory / name).read_bytes() for name in ["networks.csv", "matches.csv"]
        }
        calls = len(self.jobs.calls)
        self.run_loop(output, resume=True)
        self.assertEqual(len(self.jobs.calls), calls)
        with patch.object(ratings, "utc_now", return_value=datetime(2050, 1, 1, tzinfo=UTC)):
            ratings.refresh(self.store)
            paths = [
                self.directory / "games" / f"{row['sgf_sha256']}.sgf" for row in self.matches()
            ]
            result = ratings.ingest(
                self.store, state["rating_protocol"], list(reversed(paths)) + paths[:1]
            )
        self.assertEqual(result["new_games"], 0)
        for name, content in before.items():
            self.assertEqual((self.directory / name).read_bytes(), content)

    def test_pools_are_separate_and_wrong_budget_evidence_is_rejected(self):
        state = self.rated_loop(self.root / "loop")
        settings = state["rating_protocol"]
        gate_rows = [row for row in self.matches() if row["kind"] == "gate"]
        paths = [self.directory / "games" / f"{r['sgf_sha256']}.sgf" for r in gate_rows]
        original = {row["network_id"]: row for row in self.rows()}
        changed = dict(settings, neural_max_moves=settings["neural_max_moves"] + 1)
        new = ratings.ingest(self.store, changed, paths)
        self.assertEqual(new["new_games"], 2)
        for row in self.rows():
            if row["pool"] == ratings.identity(settings):
                self.assertEqual(row, original[row["network_id"]])
            elif row["network_id"] != ratings.REFERENCE_ID:
                self.assertEqual(row["status"], "unrated")  # No reference connection in this pool.
        before = (self.directory / "matches.csv").read_bytes()
        wrong = dict(settings, search={**settings["search"], "simulations": 33})
        with self.assertRaisesRegex(ValueError, "evaluation protocol"):
            ratings.ingest(self.store, wrong, paths)
        self.assertEqual((self.directory / "matches.csv").read_bytes(), before)

    def test_rejected_candidates_are_kept_and_rated_but_not_marked_accepted(self):
        state = self.rated_loop(self.root / "loop", ["--acceptance-rate", "0.6"])
        iteration = state["iterations"][0]
        self.assertEqual(iteration["decision"], "failed")
        self.assertEqual(iteration["parent_model"], state["initial_model"])
        self.assertIsNone(state["accepted"])
        candidate = next(
            r for r in self.rows() if r["network_id"] == iteration["checkpoint"]["sha256"]
        )
        self.assertEqual(
            (candidate["accepted"], candidate["games"], candidate["status"]), ("0", "4", "rated")
        )
        self.assertEqual(len(self.matches()), 6)
        self.assertIsNotNone(iteration.get("reference"))

    def test_truncations_are_unrated_not_draws_and_real_draws_are_half_points(self):
        self.jobs.reference_mode = self.jobs.arena_mode = "truncate"
        state = self.rated_loop(
            self.root / "truncated", ["--reference-max-moves", "1", "--gate-max-moves", "1"]
        )
        self.assertEqual(state["iterations"][0]["decision"], "inconclusive")
        for row in self.rows():
            self.assertEqual((row["games"], row["draws"]), ("0", "0"))
            if row["network_id"] != ratings.REFERENCE_ID:
                self.assertEqual(row["status"], "unrated")
                self.assertEqual(row["rating"], "")
        self.assertTrue(all(row["result"] == "" for row in self.matches()))
        self.assertTrue(
            all(float(row["black_deviation_after_batch"]) in (0, 350) for row in self.matches())
        )
        self.jobs.reference_mode = self.jobs.arena_mode = "draw"
        other = self.root / "draw-models"
        state = self.rated_loop(self.root / "draws", ["--model-store", str(other)])
        self.assertEqual(state["iterations"][0]["decision"], "passed")
        for row in read_csv(other / "ratings/networks.csv"):
            self.assertEqual((row["wins"], row["losses"], row["draws"]), ("0", "0", "4"))
            self.assertEqual(float(row["rating"]), 100)

    def test_crash_before_match_commit_rolls_back_cached_ratings_on_refresh(self):
        write = ratings._atomic_csv

        def fail_match_commit(path, columns, rows):
            if path.name == "matches.csv" and rows:
                raise OSError("injected CSV commit failure")
            return write(path, columns, rows)

        output = self.root / "before-commit"
        with (
            patch.object(ratings, "_atomic_csv", side_effect=fail_match_commit),
            self.assertRaisesRegex(OSError, "commit failure"),
        ):
            self.rated_loop(output)
        self.assertEqual(self.matches(), [])
        self.assertTrue(
            any(row["games"] == "2" for row in self.rows())
        )  # Ahead cache is not committed evidence.
        ratings.refresh(self.store)
        self.assertTrue(all(row["games"] == "0" for row in self.rows()))
        self.run_loop(output, resume=True)
        self.assertEqual(len(self.matches()), 6)
        self.assertEqual(len({(r["pool"], r["sgf_sha256"]) for r in self.matches()}), 6)
        self.assertTrue(all(row["games"] == "4" for row in self.rows()))

    def test_crash_after_match_commit_reimport_does_not_rate_again(self):
        save = loop.LoopRunner.save
        failed = False

        def gap(runner):
            nonlocal failed
            initial = runner.state.get("initial_reference")
            if not failed and initial and (initial.get("reference") or {}).get("rating_event"):
                failed = True
                raise KeyboardInterrupt
            return save(runner)

        output = self.root / "after-commit"
        with patch.object(loop.LoopRunner, "save", gap), self.assertRaises(KeyboardInterrupt):
            self.rated_loop(output)
        committed = self.matches()
        self.assertEqual(len(committed), 2)
        self.run_loop(output, resume=True)
        self.assertEqual(self.matches()[:2], committed)
        self.assertEqual(len(self.matches()), 6)
        self.assertTrue(all(row["games"] == "4" for row in self.rows()))

    def test_numeric_cache_is_rebuilt_and_missing_match_ledger_is_not_reset(self):
        self.rated_loop(self.root / "loop")
        path = self.directory / "networks.csv"
        before = path.read_bytes()
        rows = self.rows()
        rows[0].update(rating="999999", deviation="1", games="999999")
        ratings._atomic_csv(path, ratings.NETWORK_COLUMNS, rows)
        ratings.refresh(self.store)
        self.assertEqual(path.read_bytes(), before)
        (self.directory / "matches.csv").unlink()
        with self.assertRaisesRegex(ValueError, "matches.csv missing"):
            ratings.refresh(self.store)
        self.assertEqual(path.read_bytes(), before)

    def test_partial_reference_batch_reuses_games_and_does_not_reexport(self):
        self.jobs.reference_mode = "partial"
        output = self.root / "partial"
        with self.assertRaises(KeyboardInterrupt):
            self.rated_loop(output)
        first = next(output.glob("initial-reference/*/*candidate-black.sgf"))
        before = first.read_bytes()
        exports = len(self.stage_calls("continugo_train.export"))
        state = self.run_loop(output, resume=True)
        self.assertEqual(first.read_bytes(), before)
        self.assertEqual(
            len(self.stage_calls("continugo_train.export")), exports + 1
        )  # Candidate only.
        self.assertEqual(state["initial_reference"]["reference"]["attempts"][1]["indices"], [1])
        self.assertEqual(len(self.matches()), 6)

    @unittest.skipUnless(
        importlib.util.find_spec("onnxruntime") and importlib.util.find_spec("onnx"),
        "install inference extra",
    )
    def test_real_cpu_reference_matches_both_colors_with_no_random_policy_targets(self):
        bundle = self.root / "onnx"
        export.export_checkpoint(self.source, bundle)
        library, environment = inference_environment("cpu", None)
        output = self.root / "native-reference"
        result = subprocess.run(
            [
                str(loop.ROOT / "target/release/arena"),
                "--candidate",
                str(bundle),
                "--reference-random",
                "--output",
                str(output),
                "--games",
                "2",
                "--sizes",
                "9",
                "--size-weights",
                "uniform",
                "--simulations",
                "1",
                "--max-moves",
                "2",
                "--device",
                "cpu",
                "--ort-library",
                str(library),
                "--feature-threads",
                "1",
            ],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("truncated=2", result.stdout)
        settings = ratings.protocol(
            replace(
                loop.LoopConfig(),
                sizes=(9,),
                size_weights="uniform",
                gate_simulations=1,
                reference_max_moves=2,
            )
        )
        archive.publish(self.store, "training", self.source, model=bundle)
        paths = sorted(output.glob("*.sgf"))
        for index, path in enumerate(paths):
            info = inspect_game(path)
            self.assertIsNone(info["result"])
            self.assertEqual(info["policy_visit_totals"], [1, None] if index == 0 else [None, 1])
            self.assertEqual(info["root_comment"]["evaluation"], settings["arena_sha256"])
        ratings.ingest(self.store, settings, paths)
        network = next(r for r in self.rows() if r["network_id"] == archive.digest(self.source))
        self.assertEqual(network["accepted"], "0")
        self.assertEqual(
            (network["status"], network["truncated"], network["draws"]), ("unrated", "2", "0")
        )
