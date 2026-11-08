import os
import struct
import subprocess
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from continugo_train.features import (
    COARSE_PLANES,
    HEADER,
    LEGACY_MOVE_FEATURES,
    MOVE_FEATURES,
    SCHEMA_VERSION,
    FeatureClient,
    FeatureProfile,
    decode_sample,
    load_position,
    ownership_weights,
)
from continugo_train.losses import joint_loss
from continugo_train.model import NetworkConfig, PolicyValueNet, cell_actions, cell_ids

# Escaped SGF comments, native coordinates, and a checked one-stone area result.
SGF = r"""(;FF[9071]GM[9071]SZ[13]RE[B+220.4]
;B[dSdS]C[{"policy":[["dSdS",7\],["",1\]\]}]
;W[]C[{"policy":[["",3\]\]}]
;B[]
)"""


class RealFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.binary = Path(
            os.environ.get("CONTINUGO_FEATURE_BIN", "target/release/features")
        ).resolve()
        if not cls.binary.is_file():
            raise RuntimeError("run `mise run build-features` before Python integration tests")

    def setUp(self):
        torch.manual_seed(0)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.sgf = Path(self.directory.name) / "game.sgf"
        self.sgf.write_text(SGF)

    def load(self, ply=0, **kwargs):
        return load_position(self.sgf, ply, binary=self.binary, **kwargs)

    def test_real_subset_side_to_move_targets_and_value_alignment(self):
        first = self.load(actions=[200200, 0, 276200])
        self.assertEqual(first.coarse.shape, (len(COARSE_PLANES), 54, 54))
        self.assertEqual(first.exact.shape, (3, len(MOVE_FEATURES)))
        self.assertEqual(first.actions.tolist(), [200200, 0, 276200])
        self.assertEqual(first.legal.tolist(), [True, False, True])
        self.assertEqual(first.coarse[:8].count_nonzero(), 0)
        self.assertEqual(first.policy_targets()[0].tolist(), [0, 3])
        self.assertEqual(first.visits.tolist(), [7, 1])
        self.assertEqual(first.outcome, 1)
        coarser = self.load(actions=[200200], samples_per_diameter=2)
        self.assertEqual(coarser.coarse.shape, (len(COARSE_PLANES), 28, 28))
        white = self.load(1, actions=[200200, 276200])
        self.assertEqual(white.player, 1)
        self.assertEqual(white.outcome, -1)
        self.assertEqual(white.legal.tolist(), [False, True])
        self.assertEqual(white.coarse[0].count_nonzero(), 0)
        self.assertGreater(white.coarse[1].count_nonzero(), 0)
        self.assertGreater(white.coarse[5].sum().item(), 0)  # Enemy liberty-center coverage.
        self.assertTrue(((white.coarse >= 0) & (white.coarse <= 1)).all())
        self.assertGreater(white.coarse[7].sum().item(), white.coarse[5].sum().item())
        self.assertEqual(white.policy_targets()[0].tolist(), [2])
        black = self.load(2, actions=[276200])
        self.assertEqual(black.outcome, 1)
        self.assertGreater(black.coarse[0].count_nonzero(), 0)
        torch.testing.assert_close(black.coarse[4], white.coarse[5])
        with self.assertRaises(ValueError):
            black.policy_targets()  # No fabricated targets for an unannotated node.

    def test_legacy_capture_flags_are_preserved_only_for_old_models(self):
        self.sgf.write_text(SGF.replace("dSdS", "aMaM"))
        sample = self.load(1, actions=[38038, 92092], feature_schema=4)
        self.assertEqual(sample.coarse.dtype, torch.float32)
        self.assertEqual(sample.exact.dtype, torch.float32)
        self.assertEqual(sample.legal.dtype, torch.bool)
        self.assertEqual(sample.actions.dtype, torch.int64)
        self.assertEqual(sample.exact.shape, (2, len(LEGACY_MOVE_FEATURES)))
        flags = sample.coarse[6:8]
        self.assertTrue(((flags == 0) | (flags == 1)).all())
        self.assertEqual(sample.coarse[6].count_nonzero(), 0)
        self.assertGreater(sample.coarse[7].count_nonzero(), 0)
        hard = self.load(1, actions=[38038], feature_schema=4)
        torch.testing.assert_close(sample.coarse[7], hard.coarse[1])
        self.assertEqual(sample.legal.tolist(), [False, True])
        self.assertGreater(sample.exact[1, 3], 0)  # The concrete placement captures.
        defended = self.load(2, actions=[38038], feature_schema=4)
        hard = self.load(2, actions=[38038], feature_schema=4)
        torch.testing.assert_close(defended.coarse[6], hard.coarse[0])

    def test_post_move_footprint_is_versioned_and_not_clipped(self):
        actions = [200200, 0, 276200]
        current = self.load(actions=actions)
        legacy = self.load(actions=actions, feature_schema=4)
        self.assertEqual(MOVE_FEATURES[6], "local_liberty_footprint_stones_cap3")
        self.assertEqual(MOVE_FEATURES[7:], LEGACY_MOVE_FEATURES[6:])
        self.assertEqual(current.exact.shape, (3, 11))
        without_footprint = torch.cat((current.exact[:, :6], current.exact[:, 7:]), dim=1)
        torch.testing.assert_close(without_footprint, legacy.exact, rtol=0, atol=0)
        self.assertEqual(current.exact[0, 6], 3)
        self.assertEqual(current.exact[1, 6], 0)  # Illegal placement.
        self.assertEqual(current.exact[2, 6], 3)
        with FeatureClient(self.binary) as client:
            empty = client.load(self.sgf, 0, cells=[])
        self.assertEqual(empty.exact.shape, (0, 11))
        model = PolicyValueNet()
        self.assertEqual(model.policy_exact.in_features, 11)

    def test_area_antialiasing_is_versioned_and_preserves_other_planes(self):
        with FeatureClient(self.binary) as client:
            aa = client.load(self.sgf, 1, cells=[])
            hard = client.load(self.sgf, 1, cells=[], feature_schema=4)
            self.assertEqual(aa.feature_schema, 5)
            self.assertEqual(len(COARSE_PLANES), 12)
            self.assertFalse(
                any("capturable" in name or "liberty_area" in name for name in COARSE_PLANES)
            )
            self.assertEqual(
                COARSE_PLANES[4:8],
                (
                    "friendly_liberty_centers",
                    "enemy_liberty_centers",
                    "friendly_liberty_footprints",
                    "enemy_liberty_footprints",
                ),
            )
            self.assertEqual(hard.feature_schema, 4)
            self.assertTrue(((aa.coarse[1] > 0) & (aa.coarse[1] < 1)).any())
            self.assertTrue(((hard.coarse[1] == 0) | (hard.coarse[1] == 1)).all())
            torch.testing.assert_close(aa.coarse[8:], hard.coarse[8:], rtol=0, atol=0)
            explicit = client.load(self.sgf, 1, cells=[], feature_schema=5)
            self.assertEqual(explicit.coarse.shape[0], 12)
            torch.testing.assert_close(aa.coarse, explicit.coarse, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "schema"):
                client.load(self.sgf, 1, cells=[], feature_schema=6)
            import math

            covered = (aa.coarse[1] * ownership_weights(aa.resolution)).sum().item()
            self.assertAlmostEqual(covered, math.pi * (aa.diameter / 2) ** 2 / 1_000_000, places=8)
            with self.assertRaisesRegex(ValueError, "schema"):
                client.load(self.sgf, 1, cells=[], feature_schema=999)
            again = client.load(self.sgf, 1, cells=[])
            torch.testing.assert_close(aa.coarse, again.coarse)

    def test_antialiasing_encodes_motion_that_binary_occupancy_loses(self):
        hard_before = self.load(1, actions=[500500], feature_schema=4)
        aa_before = self.load(1, actions=[500500])
        self.sgf.write_text(SGF.replace("dSdS", "dTdS"))  # x=200 -> x=201, same coarse cell.
        hard_after = self.load(1, actions=[500500], feature_schema=4)
        aa_after = self.load(1, actions=[500500])
        torch.testing.assert_close(hard_before.coarse[:2], hard_after.coarse[:2], rtol=0, atol=0)
        self.assertFalse(torch.equal(aa_before.coarse[:2], aa_after.coarse[:2]))

    def test_complete_target_cell_sample_drives_a_hierarchical_training_step(self):
        sample = self.load()
        self.assertFalse(sample.full_domain)
        cells = cell_ids(torch.tensor([200200]), sample.resolution).tolist()
        torch.testing.assert_close(sample.actions, cell_actions(cells, sample.resolution))
        self.assertLess(sample.actions.numel(), 1000)
        self.assertTrue(sample.legal.all())
        model = PolicyValueNet(NetworkConfig(channels=4, blocks=1, policy_hidden=4, value_hidden=8))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        logits, values, scores, ownership = model.forward_all(*sample.network_inputs())
        indices, counts = sample.policy_targets()
        loss = joint_loss(
            logits,
            values,
            scores,
            ownership,
            indices[None],
            counts[None],
            torch.tensor([sample.outcome]),
            torch.tensor([sample.score]),
            sample.ownership[None],
        )
        loss.total.backward()
        for parameter in model.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
        before = model.pass_head.weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.pass_head.weight))

    def test_missing_targets_truncation_invalid_records_and_terminal_positions(self):
        sample = self.load(actions=[500500])
        with self.assertRaisesRegex(ValueError, "omits positive"):
            sample.policy_targets()
        self.sgf.write_text(SGF.replace("RE[B+220.4]", "").split(";W[]")[0] + ")")
        self.assertIsNone(self.load(actions=[200200]).outcome)
        self.sgf.write_text(SGF.replace("B+220.4", "W+1.0"))
        with self.assertRaisesRegex(ValueError, "disagrees"):
            self.load(actions=[200200])
        self.sgf.write_text(SGF)
        with self.assertRaisesRegex(ValueError, "terminal"):
            self.load(3, actions=[200200])
        with self.assertRaises(ValueError):
            self.load(actions=[1_000_000])
        with self.assertRaises(ValueError):
            self.load(actions=[200200, 200200])

    def test_profile_trailer_preserves_tensors_and_recovers_after_errors(self):
        with FeatureClient(self.binary, profile=True) as client:
            for ply in (0, 1, 2):  # Placement targets, pass-only, then no targets.
                expected = self.load(ply)
                sample = client.load(self.sgf, ply)
                for name, value in vars(expected).items():
                    actual = getattr(sample, name)
                    if isinstance(value, torch.Tensor):
                        self.assertTrue(torch.equal(value, actual), name)
                    else:
                        self.assertEqual(value, actual, name)
                timings = client.last_profile
                self.assertTrue(
                    all(
                        v >= 0
                        for k, v in timings.items()
                        if k.endswith("seconds") and k != "ipc_residual_seconds"
                    )
                )
                self.assertGreater(timings["packet_bytes"], HEADER.size)
                self.assertLessEqual(
                    sum(
                        timings[f"decode_{part}_seconds"]
                        for part in ("arrays", "validation", "tensors")
                    ),
                    timings["decode_seconds"],
                )
            with self.assertRaisesRegex(ValueError, "ply"):
                client.load(self.sgf, 99)
            self.assertIsNone(client.last_profile)
            client.load(self.sgf, 0)
            # Mix ordinary and diagnostic requests on one connection: no unread
            # timing bytes after errors, nor unexpected trailers on ordinary loads.
            client.profile = False
            client.load(self.sgf, 1)
            self.assertIsNone(client.last_profile)
            client.profile = True
            client.load(self.sgf, 0)
            totals = FeatureProfile()
            self.assertIsNone(totals.summary()["sum_child_peak_rss_mib"])
            totals.add(client.process.pid, client.last_profile)
            totals.add(client.process.pid, client.last_profile)
            summary = totals.summary()
            self.assertEqual(summary["requests"], 2)
            self.assertEqual(
                summary["totals"]["packet_bytes"], 2 * client.last_profile["packet_bytes"]
            )
            if client.last_profile["server_peak_rss_mib"] is not None:
                self.assertEqual(len(summary["child_peak_rss_mib"]), 1)
                self.assertEqual(
                    summary["sum_child_peak_rss_mib"], client.last_profile["server_peak_rss_mib"]
                )
        self.assertEqual(client.process.poll(), 0)
        concurrent = FeatureProfile()
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(
                pool.map(
                    lambda _: concurrent.add(123, {"packet_bytes": 64, "server_peak_rss_mib": 2.0}),
                    range(100),
                )
            )
        self.assertEqual(concurrent.summary()["requests"], 100)
        self.assertEqual(concurrent.summary()["totals"]["packet_bytes"], 6400)
        self.assertEqual(concurrent.summary()["sum_child_peak_rss_mib"], 2.0)

    def test_binary_reader_rejects_malformed_or_wrong_schema_packets(self):
        result = subprocess.run(
            [str(self.binary), "--sgf", str(self.sgf), "--actions", "200200"],
            check=True,
            capture_output=True,
        )
        packet = result.stdout
        self.assertEqual(HEADER.size, 68)
        mutations = [packet[:16], packet[:-1], packet + b"x", b"BADMAGIC" + packet[8:]]
        nonfinite = bytearray(packet)
        struct.pack_into("<f", nonfinite, HEADER.size, float("nan"))
        mutations.append(bytes(nonfinite))
        invalid_flag = bytearray(packet)
        resolution = struct.unpack_from("<I", packet, 28)[0]
        struct.pack_into("<f", invalid_flag, HEADER.size, -0.5)
        mutations.append(bytes(invalid_flag))
        for value in (-1.0, 2.0, float("nan"), float("inf")):
            flag = bytearray(packet)
            struct.pack_into("<f", flag, HEADER.size, value)
            mutations.append(bytes(flag))
        illegal_byte = bytearray(packet)
        struct.pack_into(
            "B",
            illegal_byte,
            HEADER.size + len(COARSE_PLANES) * resolution**2 * 4 + len(MOVE_FEATURES) * 4,
            255,
        )
        mutations.append(bytes(illegal_byte))
        footprint_offset = HEADER.size + len(COARSE_PLANES) * resolution**2 * 4 + 6 * 4
        for value in (-1.0, 3.01, float("nan"), float("inf")):
            bad_footprint = bytearray(packet)
            struct.pack_into("<f", bad_footprint, footprint_offset, value)
            mutations.append(bytes(bad_footprint))
        old_width = bytearray(packet)
        struct.pack_into("<I", old_width, 16, 10)
        mutations.append(bytes(old_width))
        old_schema = bytearray(packet)
        struct.pack_into("<I", old_schema, 8, SCHEMA_VERSION - 1)
        mutations.append(bytes(old_schema))
        for invalid in mutations:
            with self.assertRaises(ValueError):
                decode_sample(invalid)


if __name__ == "__main__":
    unittest.main()
