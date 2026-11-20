"""Recent-move transport, pre-move alignment and symmetry integration."""

import struct
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from test_features import SGF

from continugo_train import feature_cache
from continugo_train.augmentation import augment_sample
from continugo_train.features import HEADER, FeatureClient, coarse_planes, decode_sample


def fixture(plays, size, symmetry=0):
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    nodes = []
    for ply, spot in enumerate(plays):
        coordinates = ""
        if spot is not None:
            x, y = spot
            if symmetry // 4:
                x = 999 - x
            for _ in range(symmetry % 4):
                x, y = y, 999 - x
            coordinates = "".join(alphabet[c // 52] + alphabet[c % 52] for c in (x, y))
        nodes.append(f";{'B' if ply % 2 == 0 else 'W'}[{coordinates}]")
    return f"(;FF[9071]GM[9071]SZ[{size}]" + "".join(nodes) + ")"


class HistoryFeatureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_all_eight_symmetries_match_authoritative_history_extraction(self):
        for size in (9, 13, 19):
            # Even-sized grids have exactly mirrored rounded sample locations;
            # odd-grid rounding is shared with occupancy, checked below.
            for density in (2, 4):
                for plays in (
                    [(221, 312), (760, 820), (510, 402), (130, 800)],
                    [(221, 312), None, (510, 402), (130, 800)],
                ):
                    source = self.root / "original.sgf"
                    source.write_text(fixture(plays, size))
                    with FeatureClient() as client:
                        original = client.load(source, 4, cells=[], density=density)
                        original = replace(original, ownership=torch.zeros_like(original.coarse[0]))
                        for symmetry in range(8):
                            transformed = self.root / f"symmetry-{symmetry}.sgf"
                            transformed.write_text(fixture(plays, size, symmetry))
                            actual = client.load(transformed, 4, cells=[], density=density)
                            expected = augment_sample(original, symmetry)
                            torch.testing.assert_close(
                                actual.coarse[2:5], expected.coarse[2:5], rtol=0, atol=0
                            )
                            self.assertEqual(actual.feature_schema, 7)

    def test_augmentation_matches_occupancy_even_on_odd_rounded_grids(self):
        source = self.root / "game.sgf"
        source.write_text(SGF)
        with FeatureClient() as client:
            for density in (1, 4):
                original = client.load(source, 1, cells=[], density=density)
                original = replace(original, ownership=torch.zeros_like(original.coarse[0]))
                for symmetry in range(8):
                    augmented = augment_sample(original, symmetry)
                    torch.testing.assert_close(
                        augmented.coarse[2], augmented.coarse[1], rtol=0, atol=0
                    )

    def test_history_is_pre_move_not_the_target_or_future_moves(self):
        source = self.root / "game.sgf"
        source.write_text(SGF)
        with FeatureClient() as client:
            first = client.load(source, 0, cells=[])
            white = client.load(source, 1, cells=[])
            black = client.load(source, 2, cells=[])
        self.assertEqual(first.coarse[2:5].count_nonzero(), 0)
        torch.testing.assert_close(white.coarse[2], white.coarse[1], rtol=0, atol=0)
        self.assertGreater(white.coarse[2].sum().item(), 1)
        self.assertTrue(((white.coarse[2] > 0) & (white.coarse[2] < 1)).any())
        self.assertEqual(white.coarse[3:5].count_nonzero(), 0)
        self.assertEqual(black.coarse[2].count_nonzero(), 0)  # White's pass occupies lag 1.
        torch.testing.assert_close(black.coarse[3], white.coarse[2], rtol=0, atol=0)
        self.assertEqual(black.coarse[4].count_nonzero(), 0)

    def test_history_packet_validation_and_previous_schema_rejection(self):
        source = self.root / "game.sgf"
        source.write_text(SGF)
        with FeatureClient(retain_packet=True) as client:
            sample = client.load(source, 1, cells=[])
            packet = client.last_packet
            legacy = client.load(source, 1, cells=[], feature_schema=6)
            old_packet = client.last_packet
        with self.assertRaisesRegex(ValueError, "schema"):
            decode_sample(old_packet)
        self.assertEqual(decode_sample(old_packet, feature_schema=6).coarse.shape[0], 14)
        torch.testing.assert_close(sample.coarse[5:], legacy.coarse[2:], rtol=0, atol=0)
        area = sample.resolution**2
        offset = HEADER.size + 2 * area * 4
        for value in (-1.0, 2.0, float("nan"), float("inf")):
            invalid = bytearray(packet)
            struct.pack_into("<f", invalid, offset, value)
            with self.assertRaises(ValueError):
                decode_sample(invalid)
        restored = decode_sample(packet)
        torch.testing.assert_close(restored.coarse[2:5], sample.coarse[2:5], rtol=0, atol=0)

    def test_sidecar_history_round_trip_and_old_semantics_are_incompatible(self):
        source = self.root / "game.sgf"
        source.write_text(SGF)
        feature_cache.build_cache(source)
        with feature_cache.CacheReader(source) as cached, FeatureClient() as live:
            for ply in (0, 1):
                torch.testing.assert_close(
                    cached.load(ply).coarse, live.load(source, ply).coarse, rtol=0, atol=0
                )
        old = feature_cache.semantics(4)
        old.update(schema=6, coarse=list(coarse_planes(6)))
        with (
            patch.object(feature_cache, "semantics", return_value=old),
            self.assertRaisesRegex(ValueError, "incompatible"),
        ):
            feature_cache.CacheReader(source)
