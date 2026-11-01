import io
import tempfile
import time
import unittest
from contextlib import closing, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
from test_features import SGF

from continugo_train import data
from continugo_train.features import FeatureClient, inspect_game
from continugo_train.train import atomic_checkpoint, parse_args, train


def fixtures(directory):
    paths = []
    for offset, letter in enumerate("STUV"):
        coordinate = 200 + offset
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

    def run_training(self, output, epochs=1, resume=None, extra=()):
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
        self.assertEqual(adjusted["optimizer"]["param_groups"][0]["lr"], 0.005)
        self.assertEqual(adjusted["options"]["batch_size"], 2)
        self.assertEqual(adjusted["config"]["chunk_size"], 32768)

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


if __name__ == "__main__":
    unittest.main()
