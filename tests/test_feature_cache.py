import copy
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from test_features import SGF

from continugo_train import data
from continugo_train import feature_cache as cache
from continugo_train import train as trainer
from continugo_train.features import FeatureClient, feature_binary


class FeatureCacheTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sgf = self.root / "game.sgf"
        self.sgf.write_text(SGF)

    def build(self):
        return cache.build_cache(self.sgf)

    def test_packets_and_all_tensors_round_trip_exactly_and_existing_is_verified(self):
        before = self.sgf.read_bytes()
        result = self.build()
        self.assertEqual(result["positions"], 2)
        self.assertEqual(cache.sidecar_path(self.sgf), self.root / "features/game.cfeat")
        with cache.CacheReader(self.sgf) as reader, FeatureClient(retain_packet=True) as live:
            for ply in [1, 0, 1]:
                actual = live.load(self.sgf, ply)
                restored = reader.load(ply)
                self.assertEqual(reader.packet(ply), live.last_packet)
                for name, value in vars(actual).items():
                    if isinstance(value, torch.Tensor):
                        self.assertTrue(torch.equal(value, getattr(restored, name)), name)
                    else:
                        self.assertEqual(value, getattr(restored, name), name)
            with self.assertRaisesRegex(ValueError, "missing"):
                reader.load(100)
        self.assertEqual(cache.build_cache(self.sgf)["status"], "verified-existing")
        self.assertEqual(self.sgf.read_bytes(), before)

    def test_missing_stale_incompatible_and_corrupt_caches_fail_closed(self):
        with cache.SampleReader(cache="require") as reader, self.assertRaises(FileNotFoundError):
            reader.load(self.sgf, 0)
        with cache.SampleReader(cache="prefer") as reader:
            self.assertEqual(reader.load(self.sgf, 0).outcome, 1)
        self.build()
        with self.assertRaisesRegex(ValueError, "incompatible"):
            cache.CacheReader(self.sgf, density=5)
        other_binary = self.root / "features-bin"
        other_binary.write_bytes(feature_binary().read_bytes() + b"different extractor")
        with self.assertRaisesRegex(ValueError, "incompatible"):
            cache.CacheReader(self.sgf, binary=other_binary)
        original = self.sgf.read_text()
        self.sgf.write_text(original + "\n")
        with (
            cache.SampleReader(cache="prefer") as reader,
            self.assertRaisesRegex(ValueError, "stale"),
        ):
            reader.load(self.sgf, 0)
        self.sgf.write_text(original)
        path = cache.sidecar_path(self.sgf)
        content = bytearray(path.read_bytes())
        saved = bytes(content)
        content[-1] ^= 1
        path.write_bytes(content)
        with self.assertRaisesRegex(ValueError, "checksum"):
            cache.CacheReader(self.sgf)
        path.write_bytes(saved)
        with cache.CacheReader(self.sgf) as reader:
            offset = reader.records[0][1]
        content = bytearray(saved)
        content[offset + 2] ^= 128
        path.write_bytes(content)
        with (
            cache.CacheReader(self.sgf) as reader,
            self.assertRaisesRegex(ValueError, "corrupt|checksum"),
        ):
            reader.load(0)
        path.write_bytes(saved[:10])
        with self.assertRaisesRegex(ValueError, "truncated"):
            cache.CacheReader(self.sgf)

    def test_failed_build_is_not_published_and_cannot_remove_another_writers_lock(self):
        with (
            patch.object(cache.zlib, "compress", side_effect=RuntimeError("interrupted")),
            self.assertRaisesRegex(RuntimeError, "interrupted"),
        ):
            self.build()
        destination = cache.sidecar_path(self.sgf)
        self.assertFalse(destination.exists())
        self.assertEqual(list(destination.parent.iterdir()), [])
        lock = destination.with_suffix(".cfeat.lock")
        lock.write_text("another writer")
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(lock.read_text(), "another writer")

    def test_cached_batches_keep_order_policy_mask_and_sgd_trajectory(self):
        self.sgf.write_text(SGF.replace('{"policy":', '{"train_policy":false,"policy":', 1))
        self.build()
        games, _ = data.inspect_corpus([str(self.sgf)], 4)
        kwargs = {
            "batch_size": 2,
            "seed": 23,
            "epoch": 2,
            "training": True,
            "density": 4,
            "workers": 2,
            "shuffle_buffer": 0,
            "size_weights": "9:1,13:1,19:1",
        }
        live = list(data.batches(games, **kwargs, feature_cache="off"))
        with patch.object(cache, "FeatureClient", side_effect=AssertionError("used Rust")):
            cached = list(data.batches(games, **kwargs, feature_cache="require"))
        self.assertEqual(len(live), len(cached))
        for first, second in zip(live, cached):
            for key, value in vars(first).items():
                self.assertTrue(torch.equal(value, getattr(second, key)), key)
        self.assertEqual(sorted(cached[0].policy_weights.tolist()), [0, 1])
        config = trainer.NetworkConfig(channels=4, blocks=1, policy_hidden=4, value_hidden=4)
        torch.manual_seed(7)
        first = trainer.PolicyValueNet(config)
        second = copy.deepcopy(first)
        results = []
        for mode, model in [("off", first), ("require", second)]:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            options = trainer.Options(seed=3, workers=0, batch_size=2, feature_cache=mode)
            for epoch in range(2):
                random.seed(7)
                results.append(
                    trainer.run_phase(
                        model, optimizer, games, options, epoch, torch.device("cpu"), None, True
                    )
                )
        for key, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[key]), key)
        self.assertEqual(results[0]["loss"], results[2]["loss"])
        self.assertEqual(results[1]["loss"], results[3]["loss"])

    def test_auto_is_default_and_builds_once_without_live_fallback(self):
        self.assertEqual(trainer.Options().feature_cache, "auto")
        self.assertEqual(trainer.parse_args(["--data", str(self.sgf)]).feature_cache, None)
        games, _ = data.inspect_corpus([str(self.sgf)], 4)
        self.assertEqual(cache.ensure_caches(games, workers=2), 1)
        self.assertEqual(cache.ensure_caches(games, workers=2), 0)
        self.assertTrue(cache.sidecar_path(self.sgf).is_file())
        with patch.object(cache, "FeatureClient", side_effect=AssertionError("live fallback")):
            with cache.SampleReader() as reader:
                self.assertEqual(reader.load(self.sgf, 0).outcome, 1)
            list(
                data.batches(
                    games,
                    batch_size=2,
                    seed=0,
                    epoch=0,
                    training=True,
                    density=4,
                    workers=2,
                    shuffle_buffer=0,
                )
            )
        # Concurrent first access builds once, even when plans share a game.
        cache.sidecar_path(self.sgf).unlink()
        samples = list(data._parallel_samples([(games[0], [0]), (games[0], [1])], 4, 2, None))
        self.assertEqual([s.ply for s in samples], [0, 1])
        self.assertTrue(cache.sidecar_path(self.sgf).is_file())
        self.sgf.write_text(self.sgf.read_text() + "\n")
        with cache.SampleReader() as reader, self.assertRaisesRegex(ValueError, "stale"):
            reader.load(self.sgf, 0)

    def test_ineligible_generation_is_skipped_without_publishing_a_cache(self):
        self.sgf.write_text("(;FF[9071]GM[9071]SZ[13];B[])")
        result = cache.build_cache(self.sgf, skip_ineligible=True)
        self.assertEqual(result["status"], "skipped-ineligible")
        self.assertFalse(cache.sidecar_path(self.sgf).exists())

    def test_cache_mode_validation(self):
        with self.assertRaises(ValueError):
            trainer.Options(feature_cache="typo").validate()
        with self.assertRaises(ValueError):
            cache.SampleReader(cache="typo")
        with self.assertRaises(SystemExit):
            cache.main(["--data", str(self.root / "missing.sgf")])


if __name__ == "__main__":
    unittest.main()
