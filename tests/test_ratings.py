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


class NetworkOrderTests(unittest.TestCase):
    def test_evidence_names_are_bounded_safe_and_use_full_sgf_hash(self):
        network = "a" * 64
        tables = ratings.Tables(
            networks={
                network: {"network": "A/../very long network " * 50},
                ratings.REFERENCE_ID: {"network": ratings.REFERENCE_ID},
            }
        )
        row = {
            "recorded_at": "2026-11-02T01:02:03+03:00",
            "black": network,
            "white": ratings.REFERENCE_ID,
            "board_size": "9",
            "game_number": "1",
            "sgf_sha256": "b" * 12 + "1" * 52,
        }
        first = ratings._sgf_name(tables, row)
        second = ratings._sgf_name(tables, {**row, "sgf_sha256": "b" * 12 + "2" * 52})
        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("20261101-220203-"))
        self.assertIn(f"-vs-{ratings.REFERENCE_ID}-size9-game000001-", first)
        self.assertTrue(first.endswith(row["sgf_sha256"] + ".sgf"))
        self.assertNotIn("/", first)
        self.assertLess(len(first.encode()), 255)

    def test_newest_first_id_ties_grouped_pools_and_reference_last(self):
        tables = ratings.Tables(pools={"z": {}, "a": {}})
        dates = {
            "a" * 64: "2026-11-01T00:00:00+00:00",
            "b" * 64: "2026-11-02T00:00:00+00:00",
            "c" * 64: "2026-11-02T01:00:00+01:00",  # Same instant, different spelling.
            "d" * 64: "",  # Unknown legacy registration.
            ratings.REFERENCE_ID: "2050-01-01T00:00:00+00:00",
        }
        for network, date in reversed(dates.items()):
            tables.networks[network] = {
                **dict.fromkeys(ratings.META, ""),
                "network_id": network,
                "registered_at": date,
            }
        rows = list(tables.network_rows())
        expected = ["b" * 64, "c" * 64, "a" * 64, "d" * 64, ratings.REFERENCE_ID]
        self.assertEqual(
            [(row["network_id"], row["pool"]) for row in rows],
            [(network, pool) for network in expected for pool in ["a", "z"]],
        )
        self.assertTrue(all(row["status"] == "fixed" for row in rows[-2:]))
        tables.pools.clear()
        self.assertEqual([r["network_id"] for r in tables.network_rows()], expected)
        self.assertTrue(all(r["pool"] == "" for r in tables.network_rows()))


class RatingsTests(helpers.LoopHarness):
    def setUp(self):
        super().setUp()
        self.jobs = arena_helpers.ArenaJobs()
        self.store = self.root / "models"
        self.directory = self.store / "ratings"

    def rated_loop(self, output, extra=()):
        return self.run_loop(output, extra=["--gate-games", "2", "--reference-games", "2", *extra])

    def legacy_ledger(self):
        rows = sorted(self.matches(), key=lambda r: (int(r["batch_number"]), int(r["game_number"])))
        for row in rows:
            path = ratings.evidence_path(self.store, row)
            path.rename(self.directory / "games" / f"{row['sgf_sha256']}.sgf")
        legacy = [{key: row[key] for key in ratings.LEGACY_MATCH_COLUMNS} for row in rows]
        ratings._atomic_csv(self.directory / "matches.csv", ratings.LEGACY_MATCH_COLUMNS, legacy)
        return rows, legacy

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
            path = ratings.evidence_path(self.store, row)
            self.assertEqual(archive.digest(path), row["sgf_sha256"])
            self.assertIn(f"-size{row['board_size']}-game{int(row['game_number']):06}-", path.name)
            self.assertTrue(path.name.endswith(f"-{row['sgf_sha256']}.sgf"))
            self.assertIn("-vs-", path.name)
            blobs = list((self.store / "sgf-blobs").glob(f"????-??-??/{row['sgf_sha256']}.sgf"))
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

    def test_registration_time_is_persisted_and_cli_uses_the_same_order(self):
        registered = datetime(2026, 11, 1, tzinfo=UTC)
        with patch.object(ratings, "utc_now", return_value=registered):
            self.rated_loop(self.root / "loop")
        rows = self.rows()
        self.assertEqual([row["registered_at"] for row in rows[:-1]], [registered.isoformat()] * 2)
        self.assertEqual(
            [row["network_id"] for row in rows[:-1]],
            sorted(row["network_id"] for row in rows[:-1]),
        )
        self.assertEqual(rows[-1]["network_id"], ratings.REFERENCE_ID)
        self.assertEqual(rows[-1]["status"], "fixed")
        before = (self.directory / "networks.csv").read_bytes()
        with patch.object(ratings, "utc_now", return_value=datetime(2050, 1, 1, tzinfo=UTC)):
            ratings.register_checkpoint(self.store, self.source)
            with redirect_stdout(io.StringIO()) as output:
                ratings.main(["--model-store", str(self.store)])
        self.assertEqual(list(csv.DictReader(io.StringIO(output.getvalue()))), rows)
        self.assertEqual((self.directory / "networks.csv").read_bytes(), before)

    def test_legacy_network_table_migrates_without_changing_history_or_ratings(self):
        output = self.root / "loop"
        self.rated_loop(output)
        rows = self.rows()
        ledger = (self.directory / "matches.csv").read_bytes()
        evidence = {p.name: p.read_bytes() for p in (self.directory / "games").glob("*.sgf")}
        legacy = [
            {key: row[key] for key in ratings.LEGACY_NETWORK_COLUMNS} for row in reversed(rows)
        ]
        path = self.directory / "networks.csv"
        ratings._atomic_csv(path, ratings.LEGACY_NETWORK_COLUMNS, legacy)
        ratings.refresh(self.store)
        migrated = self.rows()
        expected = sorted(
            legacy, key=lambda row: (row["network_id"] == ratings.REFERENCE_ID, row["network_id"])
        )
        self.assertEqual(migrated, [{**row, "registered_at": ""} for row in expected])
        self.assertEqual((self.directory / "matches.csv").read_bytes(), ledger)
        self.assertEqual(
            {p.name: p.read_bytes() for p in (self.directory / "games").glob("*.sgf")}, evidence
        )
        before = path.read_bytes()
        ratings.refresh(self.store)
        self.assertEqual(path.read_bytes(), before)
        # A genuinely new registration sorts ahead of unknown legacy dates.
        registered = datetime(2050, 1, 1, tzinfo=UTC)
        with patch.object(ratings, "utc_now", return_value=registered):
            state = self.run_loop(output, resume=True, extra=["--iterations", "2"])
        newest = self.rows()[0]
        self.assertEqual(newest["network_id"], state["iterations"][-1]["checkpoint"]["sha256"])
        self.assertEqual(newest["registered_at"], registered.isoformat())
        self.assertTrue(all(row["registered_at"] == "" for row in self.rows()[1:]))

    def test_matches_use_commit_order_not_display_order_or_wall_clock(self):
        output = self.root / "loop"
        with patch.object(ratings, "utc_now", return_value=datetime(2026, 11, 4, tzinfo=UTC)):
            self.rated_loop(output)
        rows = self.matches()
        self.assertEqual([r["batch_number"] for r in rows], ["3", "3", "2", "2", "1", "1"])
        self.assertEqual([r["game_number"] for r in rows], ["1", "2"] * 3)
        for start in range(0, len(rows), 2):
            hashes = [r["sgf_sha256"] for r in rows[start : start + 2]]
            self.assertEqual(hashes, sorted(hashes))
        before = {
            name: (self.directory / name).read_bytes() for name in ["networks.csv", "matches.csv"]
        }
        scrambled = rows[::2] + list(reversed(rows[1::2]))
        ratings._atomic_csv(self.directory / "matches.csv", ratings.MATCH_COLUMNS, scrambled)
        ratings.refresh(self.store)
        for name, content in before.items():
            self.assertEqual((self.directory / name).read_bytes(), content)
        # Even a backwards clock cannot rewind ratings or mix old/new batches.
        with patch.object(ratings, "utc_now", return_value=datetime(2026, 11, 1, tzinfo=UTC)):
            self.run_loop(output, resume=True, extra=["--iterations", "2"])
        newer = self.matches()
        self.assertEqual([r["batch_number"] for r in newer[:4]], ["5", "5", "4", "4"])
        self.assertLess(newer[0]["recorded_at"], rows[0]["recorded_at"])
        self.assertEqual(newer[4:], rows)
        with redirect_stdout(io.StringIO()) as result:
            ratings.main(["--model-store", str(self.store), "--matches"])
        self.assertEqual(list(csv.DictReader(io.StringIO(result.getvalue()))), newer)

    def test_legacy_match_migration_preserves_snapshots_evidence_and_deduplication(self):
        state = self.rated_loop(self.root / "loop")
        networks = (self.directory / "networks.csv").read_bytes()
        original, legacy = self.legacy_ledger()
        # Simultaneously support the original network-table schema too.
        ratings._atomic_csv(
            self.directory / "networks.csv",
            ratings.LEGACY_NETWORK_COLUMNS,
            [{k: row[k] for k in ratings.LEGACY_NETWORK_COLUMNS} for row in self.rows()],
        )
        with patch.object(ratings.subprocess, "run", side_effect=AssertionError("no math replay")):
            ratings.refresh(self.store)
        migrated = self.matches()
        self.assertEqual(
            migrated,
            sorted(original, key=lambda r: (-int(r["batch_number"]), int(r["game_number"]))),
        )
        chronological = sorted(
            migrated, key=lambda r: (int(r["batch_number"]), int(r["game_number"]))
        )
        self.assertEqual(
            [{key: row[key] for key in ratings.LEGACY_MATCH_COLUMNS} for row in chronological],
            legacy,
        )
        # Only unknown legacy registration dates change in the numeric cache.
        old_networks = list(csv.DictReader(io.StringIO(networks.decode())))
        for row in old_networks:
            row["registered_at"] = ""
        self.assertEqual(
            self.rows(),
            sorted(
                old_networks,
                key=lambda r: (r["network_id"] == ratings.REFERENCE_ID, r["network_id"]),
            ),
        )
        sources = []
        for row in migrated:
            old = self.directory / "games" / f"{row['sgf_sha256']}.sgf"
            new = ratings.evidence_path(self.store, row)
            self.assertEqual(old.stat().st_ino, new.stat().st_ino)
            sources.append(old)
        before = (self.directory / "matches.csv").read_bytes()
        result = ratings.ingest(self.store, state["rating_protocol"], sources)
        self.assertEqual(result["new_games"], 0)
        ratings.refresh(self.store)
        self.assertEqual((self.directory / "matches.csv").read_bytes(), before)

    def test_interrupted_legacy_migration_can_be_retried_without_losing_history(self):
        self.rated_loop(self.root / "loop")
        original, _ = self.legacy_ledger()
        path = self.directory / "matches.csv"
        before = path.read_bytes()
        write = ratings._atomic_csv

        def fail_commit(target, columns, rows):
            if target.name == "matches.csv":
                raise OSError("migration commit interrupted")
            return write(target, columns, rows)

        with (
            patch.object(ratings, "_atomic_csv", side_effect=fail_commit),
            self.assertRaisesRegex(OSError, "migration commit interrupted"),
        ):
            ratings.refresh(self.store)
        self.assertEqual(path.read_bytes(), before)
        for row in original:
            self.assertTrue((self.directory / "games" / f"{row['sgf_sha256']}.sgf").is_file())
            ratings.evidence_path(self.store, row)
        ratings.refresh(self.store)
        self.assertEqual(
            self.matches(),
            sorted(original, key=lambda r: (-int(r["batch_number"]), int(r["game_number"]))),
        )

    def test_evidence_lookup_pool_filter_and_invalid_ledger_metadata(self):
        self.rated_loop(self.root / "loop")
        rows = self.matches()
        row = rows[0]
        with redirect_stdout(io.StringIO()) as result:
            ratings.main(
                [
                    "--model-store",
                    str(self.store),
                    "--sgf",
                    row["sgf_sha256"],
                    "--pool",
                    row["pool"],
                ]
            )
        self.assertEqual(result.getvalue().strip(), str(ratings.evidence_path(self.store, row)))
        with self.assertRaisesRegex(ValueError, "not in the match ledger"):
            ratings.main(["--model-store", str(self.store), "--sgf", "0" * 64])
        path = self.directory / "matches.csv"
        for changes in [
            {"sgf_path": "../outside.sgf"},
            {"batch_number": ""},
            {"batch_number": "1"},
            {"game_number": "2"},
        ]:
            with self.subTest(changes=changes):
                corrupted = [{**rows[0], **changes}, *rows[1:]]
                ratings._atomic_csv(path, ratings.MATCH_COLUMNS, corrupted)
                before = path.read_bytes()
                with self.assertRaises(ValueError):
                    ratings.refresh(self.store)
                self.assertEqual(path.read_bytes(), before)
        ratings._atomic_csv(path, ratings.MATCH_COLUMNS, rows)
        target = ratings.evidence_path(self.store, row)
        target.unlink()
        target.symlink_to(self.source)
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            ratings.evidence_path(self.store, row)

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
            paths = [ratings.evidence_path(self.store, row) for row in self.matches()]
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
        paths = [ratings.evidence_path(self.store, row) for row in gate_rows]
        original = {row["network_id"]: row for row in self.rows()}
        changed = dict(settings, neural_max_moves=settings["neural_max_moves"] + 1)
        new = ratings.ingest(self.store, changed, paths)
        self.assertEqual(new["new_games"], 2)
        for row in self.matches():
            if row["pool"] == new["pool"]:
                previous = next(r for r in gate_rows if r["sgf_sha256"] == row["sgf_sha256"])
                self.assertEqual(
                    ratings.evidence_path(self.store, row).stat().st_ino,
                    ratings.evidence_path(self.store, previous).stat().st_ino,
                )
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
        self.assertEqual(self.matches()[-2:], committed)
        self.assertEqual(len(self.matches()), 6)
        self.assertTrue(all(row["games"] == "4" for row in self.rows()))

    def test_numeric_cache_is_rebuilt_and_missing_match_ledger_is_not_reset(self):
        self.rated_loop(self.root / "loop")
        path = self.directory / "networks.csv"
        before = path.read_bytes()
        rows = self.rows()
        rows[0].update(rating="999999", deviation="1", games="999999")
        ratings._atomic_csv(path, ratings.NETWORK_COLUMNS, reversed(rows))
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
