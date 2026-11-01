import copy
import hashlib
import io
import random
import tempfile
import time
import unittest
from contextlib import closing, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
from test_features import SGF

from continugo_train import data
from continugo_train import train as trainer
from continugo_train.features import FeatureClient, inspect_game
from continugo_train.train import atomic_checkpoint, parse_args, train


def fixtures(directory, letters="STUV", start=200):
    paths = []
    for offset, letter in enumerate(letters):
        coordinate = start + offset
        text = SGF.replace("B[dSdS]", f"B[d{letter}d{letter}]").replace(
            "200200", str(1001 * coordinate)
        )
        path = directory / f"game-{offset}.sgf"
        path.write_text(text)
        paths.append(path)
    return paths


class OfflineDataTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.paths = fixtures(self.root)

    def test_game_holdout_deduplicates_moves_ignores_unlabeled_and_no_policy(self):
        (self.root / "copy.sgf").write_text(SGF.replace("SZ[13]", "SZ[13]PB[copy]"))
        (self.root / "unlabeled.sgf").write_text(SGF.replace("RE[B+220.4]", ""))
        (self.root / "random.sgf").write_text("(;FF[9071]GM[9071]SZ[13]RE[0];B[];W[])")
        corpus = data.scan_corpus([str(self.root)], 4, 7, 0.25)
        self.assertEqual(len(corpus.train) + len(corpus.validation), 4)
        self.assertEqual(corpus.skipped, {"duplicate_game": 1, "unlabeled": 1, "no_policy": 1})
        self.assertFalse({g.game_id for g in corpus.train} & {g.game_id for g in corpus.validation})
        repeated = data.scan_corpus([str(self.root)], 4, 7, 0.25)
        self.assertEqual(corpus.fingerprint, repeated.fingerprint)
        self.assertTrue(all(g.plies == (0, 1) for g in corpus.train + corpus.validation))

    def test_split_history_survives_changing_windows_seeds_and_copied_games(self):
        original = data.scan_corpus([str(p) for p in self.paths], 4, 0, 0.25)
        window = self.root / "window"
        window.mkdir()
        new_paths = fixtures(window, "WXYZ", 204)
        changed = data.scan_corpus(
            [str(p) for p in new_paths],
            4,
            7,
            0.5,
            split_history=original.split_history,
        )
        self.assertEqual(len(changed.split_history), 8)
        copied = self.root / "copied.sgf"
        old_validation = Path(original.validation[0].path)
        copied.write_text(old_validation.read_text().replace("SZ[13]", "SZ[13]PB[copy]"))
        replay = data.scan_corpus(
            [str(p) for p in [*self.paths, *new_paths, copied]],
            4,
            42,
            0.75,
            split_history=changed.split_history,
        )
        self.assertEqual(replay.skipped["duplicate_game"], 1)
        self.assertEqual(replay.split_history, changed.split_history)
        self.assertEqual(
            {g.game_id for g in replay.validation},
            {
                identity
                for identity, split in changed.split_history.items()
                if split == "validation"
            },
        )
        # A window containing only previously trained games cannot invent a holdout.
        with self.assertRaisesRegex(ValueError, "inherited split"):
            data.scan_corpus(
                [g.path for g in original.train],
                4,
                42,
                0.75,
                split_history=original.split_history,
            )

    def test_metadata_server_reuses_game_and_rejects_changed_input_on_reopen(self):
        path = self.paths[0]
        info = inspect_game(path)
        self.assertEqual(info["eligible_plies"], [0, 1])
        self.assertEqual(info["moves"], 3)
        with FeatureClient() as client:
            first = client.load(path, 0, file_sha256=info["file_sha256"])
            self.assertEqual(first.outcome, 1)
            del first
            with self.assertRaises(ValueError):
                client.load(path, 99, file_sha256=info["file_sha256"])
            # Still serves the same validated immutable snapshot, not another replay.
            path.write_text("not a game")
            second = client.load(path, 1, file_sha256=info["file_sha256"])
            self.assertEqual(second.outcome, -1)
        self.assertEqual(client.process.poll(), 0)
        with FeatureClient() as fresh, self.assertRaisesRegex(ValueError, "changed"):
            fresh.load(path, 0, file_sha256=info["file_sha256"])

    def test_collate_pads_sparse_targets_and_rejects_mixed_resolutions(self):
        def sample(resolution, indices, visits):
            return data.TrainingSample(
                torch.zeros(12, resolution, resolution),
                torch.zeros(1, 10).expand(1_000_000, -1),
                torch.ones(1_000_000, dtype=torch.bool),
                torch.tensor(indices),
                torch.tensor(visits),
                1.0,
                "fixture",
                0,
            )

        first = sample(4, [0, 1_000_000], [7.0, 1.0])
        second = sample(4, [1_000_000], [3.0])
        batch = data.collate([first, second])
        self.assertEqual(batch.coarse.shape, (2, 12, 4, 4))
        self.assertEqual(batch.exact.shape, (2, 1_000_000, 10))
        self.assertEqual(batch.indices.tolist(), [[0, 1_000_000], [1_000_000, 1_000_000]])
        self.assertEqual(batch.visits.tolist(), [[7.0, 1.0], [3.0, 0.0]])
        with self.assertRaises(ValueError):
            data.collate([first, sample(7, [1_000_000], [1.0])])

    def test_parallel_order_is_stable_and_early_close_and_errors_clean_up(self):
        closed = []

        def serial(shard, *_):
            try:
                for game, plies in shard:
                    for ply in plies:
                        time.sleep(0.001 if game % 2 else 0.003)
                        yield (game, ply)
            finally:
                closed.append(True)

        plans = [(i, [0, 1, 2]) for i in range(4)]
        with patch.object(data, "_serial_samples", serial):
            first = list(data._parallel_samples(plans, 4, 2, None))
            second = list(data._parallel_samples(plans, 4, 2, None))
            self.assertEqual(first, second)
            with closing(data._parallel_samples(plans, 4, 2, None)) as stream:
                next(stream)
        self.assertEqual(len(closed), 6)

        def failing(*_):
            raise ValueError("worker failed")
            yield  # Make this a generator.

        with (
            patch.object(data, "_serial_samples", failing),
            self.assertRaisesRegex(ValueError, "worker failed"),
        ):
            list(data._parallel_samples(plans, 4, 2, None))

    def test_buckets_flush_leftovers_without_losing_samples(self):
        game = data.GameEntry("unused", "0" * 64, "hash", 13, 76, 4, (0, 1, 2))

        def serial(*_):
            for i, resolution in enumerate([4, 7, 4, 7, 4]):
                yield data.TrainingSample(
                    torch.full((12, resolution, resolution), float(i)),
                    torch.zeros(1, 10).expand(1_000_000, -1),
                    torch.ones(1_000_000, dtype=torch.bool),
                    torch.tensor([1_000_000]),
                    torch.ones(1),
                    0.0,
                    "fixture",
                    i,
                )

        with patch.object(data, "_serial_samples", serial):
            stream = data.batches(
                [game],
                batch_size=2,
                seed=0,
                epoch=0,
                training=True,
                density=4,
                workers=0,
                shuffle_buffer=2,
            )
            seen = []
            for batch in stream:
                seen.extend(batch.coarse[:, 0, 0, 0].tolist())
                self.assertIn(batch.coarse.shape[-1], [4, 7])
            self.assertEqual(sorted(seen), [0.0, 1.0, 2.0, 3.0, 4.0])


class OfflineTrainerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.corpus = self.root / "sgf"
        self.corpus.mkdir()
        fixtures(self.corpus)

    def run_training(self, output, epochs=1, resume=None, warm_start=None, extra=()):
        arguments = [
            "--data",
            str(self.corpus),
            "--output",
            str(output),
            "--epochs",
            str(epochs),
            "--device",
            "cpu",
            "--threads",
            "1",
        ]
        if resume:
            arguments += ["--resume", str(resume)]
        elif warm_start:
            arguments += ["--warm-start", str(warm_start)]
        else:
            arguments += [
                "--channels",
                "4",
                "--blocks",
                "1",
                "--policy-hidden",
                "4",
                "--value-hidden",
                "4",
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
                "--steps-per-epoch",
                "1",
            ]
        with redirect_stdout(io.StringIO()):
            return train(parse_args([*arguments, *extra]))

    def test_checkpoints_resume_cpu_training_exactly_and_keep_optimizer_and_split(self):
        continuous = self.run_training(self.root / "continuous", epochs=2)
        resumed = self.run_training(self.root / "resumed")
        self.assertTrue((resumed.parent / "best.pt").is_file())
        self.run_training(resumed.parent, epochs=2, resume=resumed)
        expected = torch.load(continuous, weights_only=True)
        actual = torch.load(resumed, weights_only=True)
        self.assertEqual(actual["epoch"], 2)
        self.assertEqual(actual["global_step"], 2)
        self.assertEqual(actual["progress"], expected["progress"])
        self.assertEqual(
            actual["progress"], {"family": "cg1", "samples_seen": 2, "data_positions": 8}
        )
        self.assertEqual(actual["corpus_fingerprint"], expected["corpus_fingerprint"])
        self.assertEqual(actual["optimizer"]["param_groups"], expected["optimizer"]["param_groups"])
        for name in expected["model"]:
            torch.testing.assert_close(
                actual["model"][name], expected["model"][name], atol=0, rtol=0
            )
        for key, value in expected["optimizer"]["state"].items():
            torch.testing.assert_close(
                actual["optimizer"]["state"][key]["momentum_buffer"],
                value["momentum_buffer"],
                atol=0,
                rtol=0,
            )
        train_ids = {g["game_id"] for g in actual["manifest"] if g["split"] == "train"}
        val_ids = {g["game_id"] for g in actual["manifest"] if g["split"] == "validation"}
        self.assertFalse(train_ids & val_ids)
        self.run_training(
            resumed.parent,
            epochs=3,
            resume=resumed,
            extra=[
                "--lr",
                "0.005",
                "--batch-size",
                "2",
                "--chunk-size",
                "32768",
                "--steps-per-epoch",
                "2",
                "--positions-per-game",
                "0",
            ],
        )
        adjusted = torch.load(resumed, weights_only=True)
        self.assertEqual(adjusted["global_step"], 4)
        self.assertEqual(adjusted["progress"]["samples_seen"], 6)  # Two two-row minibatches.
        self.assertEqual(adjusted["progress"]["data_positions"], 8)
        self.assertEqual(adjusted["optimizer"]["param_groups"][0]["lr"], 0.005)
        self.assertEqual(adjusted["options"]["batch_size"], 2)
        self.assertEqual(adjusted["config"]["chunk_size"], 32768)

    def test_legacy_resume_keeps_uniform_sampling_and_rejects_mix_changes(self):
        checkpoint = self.run_training(self.root / "legacy")
        state = torch.load(checkpoint, weights_only=True)
        del state["options"]["size_weights"]
        legacy = checkpoint.parent / "legacy.pt"
        atomic_checkpoint(legacy, state)
        with self.assertRaisesRegex(ValueError, "resume restores --size-weights"):
            self.run_training(
                checkpoint.parent,
                epochs=2,
                resume=legacy,
                extra=["--size-weights", "9:2,13:2,19:1"],
            )
        continued = self.run_training(checkpoint.parent, epochs=2, resume=legacy)
        self.assertEqual(
            torch.load(continued, weights_only=True)["options"]["size_weights"], "uniform"
        )

    def test_resume_rejects_corpus_and_feature_changes_and_fresh_run_overwrite(self):
        checkpoint = self.run_training(self.root / "run")
        before = checkpoint.read_bytes()
        with self.assertRaisesRegex(ValueError, "not empty"):
            self.run_training(checkpoint.parent)
        state = torch.load(checkpoint, weights_only=True)
        state["feature_spec"]["coarse"][0] = "wrong-feature"
        incompatible = checkpoint.parent / "incompatible.pt"
        atomic_checkpoint(incompatible, state)
        with self.assertRaisesRegex(ValueError, "incompatible feature"):
            self.run_training(checkpoint.parent, epochs=2, resume=incompatible)
        path = self.corpus / "game-0.sgf"
        path.write_text(path.read_text().replace("SZ[13]", "SZ[13]PB[changed]"))
        with self.assertRaisesRegex(ValueError, "corpus/targets changed"):
            self.run_training(checkpoint.parent, epochs=2, resume=checkpoint)
        self.assertEqual(checkpoint.read_bytes(), before)

    def test_warm_start_new_corpus_resets_state_keeps_weights_and_resumes_exactly(self):
        checkpoint = self.run_training(self.root / "parent", epochs=2)
        parent = torch.load(checkpoint, weights_only=True)
        # Pre-warm-start checkpoints remain usable; derive their split ledger.
        parent.pop("split_history")
        parent.pop("initialization")
        parent["best_validation"] = -1.0
        source = checkpoint.parent / "legacy.pt"
        atomic_checkpoint(source, parent)
        before = source.read_bytes()
        window = self.root / "new-games"
        window.mkdir()
        fixtures(window, "WXYZ", 204)
        extra = [
            "--data",
            str(self.corpus),
            str(window),
            "--seed",
            "7",
            "--validation-fraction",
            "0.5",
            "--validation-positions-per-game",
            "2",
            "--lr",
            "0.005",
            "--momentum",
            "0.7",
            "--weight-decay",
            "0.001",
            "--chunk-size",
            "32768",
        ]
        captured = {}
        original_phase = trainer.run_phase

        def phase(model, optimizer, *args):
            if not captured:
                captured["model"] = copy.deepcopy(model.state_dict())
                captured["optimizer"] = copy.deepcopy(optimizer.state_dict())
            return original_phase(model, optimizer, *args)

        with patch.object(trainer, "run_phase", side_effect=phase):
            warmed = self.run_training(self.root / "warm", warm_start=source, extra=extra)
        actual = torch.load(warmed, weights_only=True)
        self.assertEqual(captured["optimizer"]["state"], {})
        for name, value in parent["model"].items():
            torch.testing.assert_close(captured["model"][name], value, atol=0, rtol=0)
        self.assertNotEqual(actual["run_id"], parent["run_id"])
        self.assertEqual(actual["epoch"], 1)
        self.assertEqual(actual["global_step"], 1)
        self.assertEqual(
            actual["progress"]["samples_seen"], 3
        )  # Includes the parent, despite SGD reset.
        self.assertEqual(actual["progress"]["data_positions"], 16)
        self.assertEqual(actual["config"]["channels"], parent["config"]["channels"])
        self.assertEqual(actual["config"]["chunk_size"], 32768)
        self.assertNotEqual(actual["corpus_fingerprint"], parent["corpus_fingerprint"])
        self.assertEqual(actual["best_validation"], actual["metrics"]["validation"]["loss"])
        self.assertTrue((warmed.parent / "best.pt").is_file())
        origin = actual["initialization"]
        self.assertEqual(origin["mode"], "warm_start")
        self.assertEqual(origin["optimizer"], "reset")
        self.assertEqual(origin["source"]["sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual(origin["source"]["run_id"], parent["run_id"])
        self.assertEqual(origin["source"]["epoch"], 2)
        self.assertEqual(origin["source"]["global_step"], 2)
        self.assertEqual(source.read_bytes(), before)
        for game in parent["manifest"]:
            self.assertEqual(actual["split_history"][game["game_id"]], game["split"])
        groups = actual["optimizer"]["param_groups"][0]
        self.assertEqual(
            (groups["lr"], groups["momentum"], groups["weight_decay"]), (0.005, 0.7, 0.001)
        )
        self.assertTrue(
            any(not torch.equal(actual["model"][k], v) for k, v in parent["model"].items())
        )

        # A warm-start run's own strict resume must match uninterrupted training.
        continuous = self.run_training(
            self.root / "warm-continuous", epochs=2, warm_start=source, extra=extra
        )
        self.run_training(
            warmed.parent, epochs=2, resume=warmed, extra=["--data", str(self.corpus), str(window)]
        )
        resumed = torch.load(warmed, weights_only=True)
        continuous = torch.load(continuous, weights_only=True)
        self.assertEqual(resumed["global_step"], 2)
        self.assertEqual(resumed["progress"], continuous["progress"])
        self.assertEqual(resumed["progress"]["samples_seen"], 4)
        self.assertEqual(resumed["progress"]["data_positions"], 16)
        self.assertEqual(resumed["initialization"], origin)
        self.assertEqual(resumed["split_history"], actual["split_history"])
        for name in continuous["model"]:
            torch.testing.assert_close(
                resumed["model"][name], continuous["model"][name], atol=0, rtol=0
            )
        for key, value in continuous["optimizer"]["state"].items():
            torch.testing.assert_close(
                resumed["optimizer"]["state"][key]["momentum_buffer"],
                value["momentum_buffer"],
                atol=0,
                rtol=0,
            )
        with self.assertRaisesRegex(ValueError, "corpus/targets changed"):
            self.run_training(
                checkpoint.parent,
                epochs=3,
                resume=checkpoint,
                extra=["--data", str(self.corpus), str(window)],
            )

    def test_warm_start_can_keep_momentum_but_reseeds_rng_and_resets_counters(self):
        checkpoint = self.run_training(self.root / "parent", epochs=2)
        parent = torch.load(checkpoint, weights_only=True)
        original_phase = trainer.run_phase
        captured = {}

        def phase(model, optimizer, *args):
            if not captured:
                captured["optimizer"] = copy.deepcopy(optimizer.state_dict())
                captured["python_rng"] = random.getstate()
                captured["torch_rng"] = torch.get_rng_state()
            return original_phase(model, optimizer, *args)

        with patch.object(trainer, "run_phase", side_effect=phase):
            warmed = self.run_training(
                self.root / "keep",
                warm_start=checkpoint,
                extra=[
                    "--warm-start-optimizer",
                    "keep",
                    "--seed",
                    "123",
                    "--lr",
                    "0.003",
                    "--momentum",
                    "0.8",
                    "--weight-decay",
                    "0.002",
                ],
            )
        actual = torch.load(warmed, weights_only=True)
        self.assertEqual(actual["epoch"], 1)
        self.assertEqual(actual["global_step"], 1)
        self.assertEqual(actual["initialization"]["optimizer"], "keep")
        for key, value in parent["optimizer"]["state"].items():
            torch.testing.assert_close(
                captured["optimizer"]["state"][key]["momentum_buffer"],
                value["momentum_buffer"],
                atol=0,
                rtol=0,
            )
        group = captured["optimizer"]["param_groups"][0]
        self.assertEqual(
            (group["lr"], group["momentum"], group["weight_decay"]), (0.003, 0.8, 0.002)
        )
        self.assertEqual(captured["python_rng"], random.Random(123).getstate())
        self.assertFalse(torch.equal(captured["torch_rng"], parent["rng_cpu"]))

    def test_warm_start_persists_splits_for_games_absent_from_a_replay_window(self):
        checkpoint = self.run_training(self.root / "parent")
        parent = torch.load(checkpoint, weights_only=True)
        window = self.root / "window"
        window.mkdir()
        fixtures(window, "WXYZ", 204)
        first = self.run_training(
            self.root / "first-window",
            warm_start=checkpoint,
            extra=["--data", str(window), "--seed", "17", "--validation-fraction", "0.5"],
        )
        first_state = torch.load(first, weights_only=True)
        self.assertEqual(len(first_state["manifest"]), 4)
        self.assertEqual(len(first_state["split_history"]), 8)
        second = self.run_training(
            self.root / "second-window",
            warm_start=first,
            extra=["--seed", "13", "--validation-fraction", "0.75"],
        )
        second_state = torch.load(second, weights_only=True)
        self.assertEqual(second_state["split_history"], first_state["split_history"])
        self.assertEqual(
            {g["game_id"]: g["split"] for g in second_state["manifest"]},
            {g["game_id"]: g["split"] for g in parent["manifest"]},
        )

    def test_warm_start_guards_sources_features_architecture_and_output(self):
        with redirect_stderr(io.StringIO()):
            for extra in [
                ["--resume", "a.pt", "--warm-start", "b.pt"],
                ["--warm-start-optimizer", "keep"],
            ]:
                with self.assertRaises(SystemExit):
                    parse_args(["--data", str(self.corpus), *extra])
        checkpoint = self.run_training(self.root / "parent")
        before = checkpoint.read_bytes()
        with self.assertRaisesRegex(ValueError, "not empty"):
            self.run_training(checkpoint.parent, warm_start=checkpoint)
        for extra, message in [
            (["--samples-per-diameter", "8"], "incompatible feature"),
            (["--channels", "8"], "architecture conflicts"),
        ]:
            with self.assertRaisesRegex(ValueError, message):
                self.run_training(self.root / "invalid", warm_start=checkpoint, extra=extra)
        state = torch.load(checkpoint, weights_only=True)
        state["feature_spec"]["coarse"][0] = "wrong-feature"
        bad = checkpoint.parent / "incompatible.pt"
        atomic_checkpoint(bad, state)
        with self.assertRaisesRegex(ValueError, "incompatible feature"):
            self.run_training(self.root / "invalid", warm_start=bad)
        state = torch.load(checkpoint, weights_only=True)
        identity = state["manifest"][0]["game_id"]
        state["split_history"][identity] = (
            "validation" if state["manifest"][0]["split"] == "train" else "train"
        )
        atomic_checkpoint(bad, state)
        with self.assertRaisesRegex(ValueError, "conflicting split history"):
            self.run_training(self.root / "invalid", warm_start=bad)
        atomic_checkpoint(bad, {"model": {}})
        with self.assertRaisesRegex(ValueError, "not an offline-training checkpoint"):
            self.run_training(self.root / "invalid", warm_start=bad)
        self.assertEqual(checkpoint.read_bytes(), before)
        self.assertFalse((self.root / "invalid" / ".train.lock").exists())


if __name__ == "__main__":
    unittest.main()
