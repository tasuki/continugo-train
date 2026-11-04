import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

from continugo_train.features import (
    COARSE_PLANES,
    HEADER,
    MOVE_FEATURES,
    SCHEMA_VERSION,
    decode_sample,
    load_position,
)
from continugo_train.losses import hierarchical_policy_value_loss
from continugo_train.model import NetworkConfig, PolicyValueNet, cell_actions, cell_ids

# Escaped SGF comments, native coordinates, and a checked one-stone area result.
SGF = r"""(;FF[9071]GM[9071]SZ[13]RE[B+220.4]
;B[dSdS]C[{"policy":[{"action":200200,"visits":7},{"action":1000000,"visits":1}\]}]
;W[]C[{"policy":[{"action":1000000,"visits":3}\]}]
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
        self.assertGreater(white.coarse[5].max().item(), 1.0)  # Raw stone-disc area, not clipped.
        self.assertEqual(white.policy_targets()[0].tolist(), [2])
        black = self.load(2, actions=[276200])
        self.assertEqual(black.outcome, 1)
        self.assertGreater(black.coarse[0].count_nonzero(), 0)
        torch.testing.assert_close(black.coarse[4], white.coarse[5])
        with self.assertRaises(ValueError):
            black.policy_targets()  # No fabricated targets for an unannotated node.

    def test_capture_flags_are_float32_and_painted_on_group_stones(self):
        self.sgf.write_text(SGF.replace("B[dSdS]", "B[aMaM]").replace("200200", "38038"))
        sample = self.load(1, actions=[38038, 92092])
        self.assertEqual(sample.coarse.dtype, torch.float32)
        self.assertEqual(sample.exact.dtype, torch.float32)
        self.assertEqual(sample.legal.dtype, torch.bool)
        self.assertEqual(sample.actions.dtype, torch.int64)
        flags = sample.coarse[6:8]
        self.assertTrue(((flags == 0) | (flags == 1)).all())
        self.assertEqual(sample.coarse[6].count_nonzero(), 0)
        self.assertGreater(sample.coarse[7].count_nonzero(), 0)
        torch.testing.assert_close(sample.coarse[7], sample.coarse[1])
        self.assertEqual(sample.legal.tolist(), [False, True])
        self.assertGreater(sample.exact[1, 3], 0)  # The concrete placement captures.
        defended = self.load(2, actions=[38038])
        torch.testing.assert_close(defended.coarse[6], defended.coarse[0])

    def test_complete_target_cell_sample_drives_a_hierarchical_training_step(self):
        sample = self.load()
        self.assertFalse(sample.full_domain)
        cells = cell_ids(torch.tensor([200200]), sample.resolution).tolist()
        torch.testing.assert_close(sample.actions, cell_actions(cells, sample.resolution))
        self.assertLess(sample.actions.numel(), 1000)
        self.assertTrue(sample.legal.all())
        model = PolicyValueNet(NetworkConfig(channels=4, blocks=1, policy_hidden=4, value_hidden=8))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        logits, values = model(*sample.network_inputs())
        indices, counts = sample.policy_targets()
        loss = hierarchical_policy_value_loss(
            logits, values, indices[None], counts[None], torch.tensor([sample.outcome])
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

    def test_binary_reader_rejects_malformed_or_wrong_schema_packets(self):
        result = subprocess.run(
            [str(self.binary), "--sgf", str(self.sgf), "--actions", "200200"],
            check=True,
            capture_output=True,
        )
        packet = result.stdout
        self.assertEqual(HEADER.size, 64)
        mutations = [packet[:16], packet[:-1], packet + b"x", b"BADMAGIC" + packet[8:]]
        nonfinite = bytearray(packet)
        struct.pack_into("<f", nonfinite, HEADER.size, float("nan"))
        mutations.append(bytes(nonfinite))
        invalid_flag = bytearray(packet)
        resolution = struct.unpack_from("<I", packet, 28)[0]
        struct.pack_into("<f", invalid_flag, HEADER.size + 6 * resolution**2 * 4, 0.5)
        mutations.append(bytes(invalid_flag))
        old_schema = bytearray(packet)
        struct.pack_into("<I", old_schema, 8, SCHEMA_VERSION - 1)
        mutations.append(bytes(old_schema))
        for invalid in mutations:
            with self.assertRaises(ValueError):
                decode_sample(invalid)


if __name__ == "__main__":
    unittest.main()
