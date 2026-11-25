import argparse
import csv
import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch

from continugo_train import champion_wins, joint_ratings, lineage_ratings, ratings
from continugo_train.compare_lineages import (
    ComparisonRunner,
    historical_inspectors,
    latest_champions,
    main,
    prepare,
)
from continugo_train.evaluation import EvaluationConfig


class LineageRatingsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.binary = self.root / "features"
        self.binary.write_text("frozen inspector")
        self.catalog = {
            name: {
                "id": name,
                "lineage": lineage,
                "network_name": name,
                "checkpoint_sha256": name * 64,
            }
            for name, lineage in [("a", "11"), ("b", "22")]
        }
        self.search = {"simulations": 128}
        self.settings = {
            "search": self.search,
            "arena_sha256": "comparison-runtime",
            "board_mix": {"9": 1},
            "komi": 0.0,
            "neural_max_moves": 1024,
            "reference_max_moves": 2048,
        }
        for lineage, name in [("11", "a"), ("22", "b")]:
            store = self.root / lineage
            (store / "sgfs/day").mkdir(parents=True)
            (store / "champions.json").write_text(
                json.dumps(
                    {
                        "history": [{"model": "random"}, {"model": name}],
                    }
                )
            )
            (store / "networks.csv").write_text(f"network\nrandom\n{name}\n")
            self.game(
                store / f"sgfs/day/{name}-vs-random-seed1.sgf",
                name,
                "random",
                1,
                "B+1",
                7.0,
                "historical-runtime",
            )
        (self.root / "ratings/comparisons").mkdir(parents=True)
        self.state = {
            "version": "lineage-comparison-v1",
            "config": {"sizes": [9]},
            "rating_protocol": self.settings,
            "jobs": [{"candidate_name": "a", "opponent_name": "b", "games": 40, "seed": 100}],
        }
        torch.save(self.state, self.root / "ratings/comparisons/test.pt")
        self.shared = self.root / "sgfs/day"
        self.shared.mkdir(parents=True)

    def game(self, path, black, white, seed, result, komi=0.0, runtime="comparison-runtime"):
        def checksum(name):
            return ratings.REFERENCE_ID if name == "random" else name * 64

        metadata = {
            "model": checksum(black),
            "opponent": checksum(white),
            "search": self.search,
            "evaluation": runtime,
            "seed": seed,
        }
        comment = json.dumps(metadata).replace("]", "\\]")
        pb = ratings.REFERENCE_ID if black == "random" else f"neural-{black}"
        pw = ratings.REFERENCE_ID if white == "random" else f"neural-{white}"
        path.write_text(f"(;SZ[9]KM[{komi}]PB[{pb}]PW[{pw}]RE[{result}]C[{comment}];B[];W[])")
        return path

    def inspect(self, path, **_kwargs):
        props = champion_wins.root_properties(path)
        metadata = json.loads(props["C"][0])
        return {
            "root_comment": metadata,
            "komi": float(props["KM"][0]),
            "board_size": 9,
            "moves": 2,
            "completed": True,
            "labeled": True,
            "policy_visit_totals": [
                None if metadata[key] == ratings.REFERENCE_ID else 128
                for key in ("model", "opponent")
            ],
            "black_player": props["PB"][0],
            "white_player": props["PW"][0],
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "result": props["RE"][0],
        }

    def refresh(self):
        with (
            patch.object(lineage_ratings, "sources", return_value=self.catalog),
            patch.object(ratings, "inspect_game", side_effect=self.inspect) as inspect,
        ):
            result = lineage_ratings.refresh(self.root, self.binary)
        return result, inspect.call_count

    def test_network_order_is_numeric_by_lineage_generation_and_revision(self):
        expected = [
            ("9", "cg9-r9"),
            ("9", "cg9-r10"),
            ("9", "cg9-r99"),
            ("9", "cg9-r100"),
            ("9", "cg10-r1"),
            ("9", "cg99-r1"),
            ("9", "cg100-r1"),
            ("10", "cg1-r1"),
            ("99", "cg1-r2"),
            ("100", "cg1-r3"),
        ]
        catalog = {name: {"lineage": lineage} for lineage, name in reversed(expected)}
        self.assertEqual(
            lineage_ratings.network_order(catalog),
            [ratings.REFERENCE_ID, *(name for _, name in expected)],
        )

    def test_joint_fit_updates_both_lineages_and_is_cached_and_order_independent(self):
        self.catalog = dict(reversed(list(self.catalog.items())))
        self.game(self.shared / "a-vs-b-seed100.sgf", "a", "b", 100, "B+1")
        self.game(self.shared / "b-vs-a-seed100.sgf", "b", "a", 100, "W+1")
        states, calls = self.refresh()
        self.assertEqual(calls, 4)
        self.assertGreater(states["a"]["rating"], states["b"]["rating"])
        self.assertEqual(states["a"]["games"], 3)
        self.assertEqual(states["b"]["games"], 3)
        self.assertEqual(states[ratings.REFERENCE_ID]["rating"], 100)
        self.assertEqual(self.refresh(), (states, 0))
        rows = lineage_ratings.overlay(self.root)
        self.assertEqual(list(rows), [("", "random"), ("11", "a"), ("22", "b")])
        self.assertEqual(float(rows["11", "a"]["rating"]), states["a"]["rating"])
        champion_wins.update(self.root)
        self.assertIn("a,1,,2", (self.root / "champion-wins.csv").read_text())
        records = [
            item["record"]
            for item in torch.load(self.root / "ratings/joint-cache.pt", weights_only=True)[
                "files"
            ].values()
        ]
        reversed_fit = joint_ratings.fit(list(reversed(records)))
        self.assertEqual(states, reversed_fit)

    def test_lineage_games_are_fitted_without_any_shared_games(self):
        states, calls = self.refresh()
        self.assertEqual(calls, 2)
        self.assertEqual(states["a"]["games"], 1)
        self.assertEqual(states["b"]["games"], 1)
        self.assertTrue((self.root / "networks.csv").exists())

    def test_local_tables_are_views_of_the_single_fit(self):
        self.game(self.shared / "a-vs-b-seed100.sgf", "a", "b", 100, "B+1")
        with patch.object(joint_ratings, "fit", wraps=joint_ratings.fit) as fit:
            states, _ = self.refresh()
        self.assertEqual(fit.call_count, 1)
        for lineage, name in [("11", "a"), ("22", "b")]:
            with (self.root / lineage / "networks.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["network"] for row in rows], ["random", name])
            self.assertEqual(float(rows[1]["rating"]), states[name]["rating"])
            self.assertEqual(int(rows[1]["games"]), 2)
        # Publishing another game's evidence changes both views, not only its caller.
        before = states["b"]["rating"]
        self.game(self.shared / "b-vs-a-seed100.sgf", "b", "a", 100, "W+1")
        states, _ = self.refresh()
        self.assertNotEqual(states["b"]["rating"], before)
        with (self.root / "22/networks.csv").open(newline="") as stream:
            row = list(csv.DictReader(stream))[1]
        self.assertEqual(float(row["rating"]), states["b"]["rating"])

    def test_wrong_komi_or_runtime_or_seed_is_rejected(self):
        path = self.shared / "a-vs-b-seed100.sgf"
        for komi, runtime, seed in [
            (7, "comparison-runtime", 100),
            (0, "wrong-runtime", 100),
            (0, "comparison-runtime", 101),
        ]:
            with self.subTest(komi=komi, runtime=runtime, seed=seed):
                self.game(path, "a", "b", seed, "B+1", komi, runtime)
                with self.assertRaises(ValueError):
                    self.refresh()
                self.assertFalse((self.root / "networks.csv").exists())

    def test_duplicate_seed_color_evidence_is_rejected(self):
        self.game(self.shared / "a-vs-b-seed100.sgf", "a", "b", 100, "B+1")
        another = self.shared.parent / "another-day"
        another.mkdir()
        self.game(another / "a-vs-b-seed100.sgf", "a", "b", 100, "W+1")
        with self.assertRaisesRegex(ValueError, "duplicate seed/color"):
            self.refresh()

    def test_unreserved_shared_games_are_rejected(self):
        self.game(self.shared / "a-vs-b-seed999.sgf", "a", "b", 999, "B+1")
        with self.assertRaisesRegex(ValueError, "no frozen comparison reservation"):
            self.refresh()

    def test_recover_archived_only_identities_without_model_files(self):
        with patch.object(lineage_ratings, "saved_sources", return_value={}):
            sources = lineage_ratings.sources(self.root)
        self.assertEqual(set(sources), {"a", "b"})
        self.assertTrue(sources["a"]["pruned"])
        self.assertEqual(sources["a"]["checkpoint_sha256"], "a" * 64)

    def test_prepare_freezes_nine_pairings_without_starting_games(self):
        native = self.root / "bin"
        native.mkdir()
        for binary in ("arena", "features", "rating", "continugo-train"):
            path = native / binary
            path.write_text("#!/bin/sh\necho --komi --check-models\n")
            path.chmod(0o755)
        selected = [[f"a{i}" for i in range(1, 4)], [f"b{i}" for i in range(1, 4)]]
        catalog, local = {}, {}
        for lineage, names in zip(("11", "22"), selected, strict=True):
            store = self.root / lineage
            (store / "champions.json").write_text(
                json.dumps(
                    {
                        "history": [{"model": n} for n in ["random", *names]],
                    }
                )
            )
            local[lineage] = {}
            for name in names:
                directory = store / "models" / name
                directory.mkdir(parents=True)
                record = {
                    "id": name,
                    "network_name": name,
                    "checkpoint_sha256": hashlib.sha256(name.encode()).hexdigest(),
                    "lineage": lineage,
                }
                for key, filename in [
                    ("checkpoint", "checkpoint.pt"),
                    ("encoder", "encoder.onnx"),
                    ("policy", "policy.onnx"),
                ]:
                    path = directory / filename
                    path.write_text(name)
                    record[key] = {"path": str(path), "sha256": record["checkpoint_sha256"]}
                catalog[name] = record
                local[lineage][record["checkpoint_sha256"]] = record
        (self.root / "11/ratings").mkdir()
        torch.save(
            {
                "config": asdict(EvaluationConfig(device="cpu", sizes=[9], size_weights="uniform")),
                "feature_spec": {},
            },
            self.root / "11/ratings/evaluator.pt",
        )
        args = argparse.Namespace(
            name="prepared",
            lineage_a="11",
            lineage_b="22",
            latest=3,
            games=40,
            komi=0.0,
            runtime=native,
            device="cpu",
            historical_inspector=[],
        )
        state_path = self.root / "ratings/comparisons/prepared.pt"
        with (
            patch.object(lineage_ratings, "sources", return_value=catalog),
            patch(
                "continugo_train.compare_lineages.saved_sources",
                side_effect=lambda store: local[store.name],
            ),
            patch(
                "continugo_train.compare_lineages.inference_environment",
                return_value=(native / "ort.so", {}),
            ),
            patch("continugo_train.nn_view.load_checkpoint"),
            patch(
                "continugo_train.compare_lineages.historical_inspectors",
                return_value={"11": native / "features", "22": native / "features"},
            ),
            patch("continugo_train.loop.run_process") as execute,
        ):
            state = prepare(self.root, args, state_path)
            execute.assert_not_called()
            self.assertEqual(state["selected"], selected)
            self.assertEqual(len(state["jobs"]), 9)
            self.assertEqual(sum(job["games"] for job in state["jobs"]), 360)
            self.assertEqual(
                [j["seed"] for j in state["jobs"]], list(range(98000000, 98000180, 20))
            )
            self.assertTrue(all(j["gate"] is None for j in state["jobs"]))
            self.assertTrue(all(j["komi"] == 0 for j in state["jobs"]))
            self.assertTrue(Path(state["config"]["arena_binary"]).stat().st_mode & 0o111)
            self.assertFalse(state["complete"])
            self.assertEqual(state["rating_protocol"]["komi"], 0)
            runner = ComparisonRunner(self.root, state, 0, state_path)
            runner.verify_runtime()
            for name in state["models"]:
                runner.check(state["checkpoints"][name])
                runner.check_model(state["models"][name])
            with self.assertRaisesRegex(ValueError, "already exists"):
                prepare(self.root, args, state_path)
            args.name = "second"
            another = prepare(self.root, args, state_path.with_name("second.pt"))
            self.assertEqual(another["jobs"][0]["seed"], 98000180)

    def test_historical_scorer_is_selected_and_preserved(self):
        old = self.root / "old-features"
        old.write_text("historical scorer")
        self.catalog["a"]["source_context"] = {"config": {"feature_binary": str(old)}}

        def inspect(path, *, binary):
            if path.is_relative_to(self.root / "11") and Path(binary) == self.binary:
                raise ValueError("historical score units differ")
            return self.inspect(path)

        with patch.object(ratings, "inspect_game", side_effect=inspect):
            chosen = historical_inspectors(self.root, self.catalog, self.binary)
        self.assertEqual(chosen["11"], old)
        self.assertEqual(chosen["22"], self.binary)
        self.state["historical_inspectors"] = {
            "11": {"path": str(old), "sha256": hashlib.sha256(old.read_bytes()).hexdigest()},
        }
        torch.save(self.state, self.root / "ratings/comparisons/test.pt")
        self.game(self.shared / "a-vs-b-seed100.sgf", "a", "b", 100, "B+1")
        with (
            patch.object(lineage_ratings, "sources", return_value=self.catalog),
            patch.object(ratings, "inspect_game", side_effect=inspect),
        ):
            states = lineage_ratings.refresh(self.root, self.binary)
        self.assertGreater(states["a"]["rating"], states["b"]["rating"])
        old.write_text("changed scorer")
        with self.assertRaisesRegex(ValueError, "inspector changed"):
            self.refresh()

    def test_latest_champions_and_frozen_run_arguments(self):
        self.assertEqual(latest_champions(self.root / "11", 1), ["a"])
        with self.assertRaises(ValueError):
            latest_champions(self.root / "11", 3)
        with self.assertRaises(SystemExit), patch("continugo_train.compare_lineages.run") as run:
            main(["run", "--name", "test", "--komi", "7"])
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
