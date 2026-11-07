import math
import unittest

import torch

from continugo_train.data import TrainingSample, collate
from continugo_train.features import COARSE_PLANES
from continugo_train.losses import hierarchical_policy_value_loss, joint_loss
from continugo_train.model import (
    NetworkConfig,
    PolicyValueNet,
    cell_actions,
    cell_ids,
    local_log_probs,
)


class CellPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(23)
        self.model = PolicyValueNet(
            NetworkConfig(channels=4, blocks=1, policy_hidden=4, value_hidden=4, chunk_size=17)
        ).eval()

    def test_integer_partition_edges_and_no_overlap(self):
        for resolution in [38, 54, 78]:
            ids = torch.arange(1000000)
            owners = cell_ids(ids, resolution)
            self.assertEqual(owners[0], 0)
            self.assertEqual(owners[-1], resolution**2 - 1)
            for cell in [0, 1, resolution + 1, resolution**2 - 1]:
                actions = cell_actions([cell, cell], resolution)
                torch.testing.assert_close(actions.sort().values, ids[owners == cell])
                self.assertEqual(actions.unique().numel(), actions.numel())

    def test_stable_segment_softmax_and_empty_cells(self):
        actions = cell_actions([0, 39, 40], 38)[None].expand(2, -1)
        cells = cell_ids(actions, 38)
        scores = (torch.randn(actions.shape) * 1000).requires_grad_()
        legal = cells != 39  # An entirely illegal selected cell.
        legal[1] = False  # An all-illegal position must not produce NaNs.
        log_probs = local_log_probs(scores, actions, legal, 38)
        self.assertFalse(torch.isnan(log_probs).any())
        for cell in [0, 40]:
            torch.testing.assert_close(
                log_probs[0, cells[0] == cell].exp().sum(), torch.tensor(1.0)
            )
        self.assertTrue(torch.isneginf(log_probs[~legal]).all())
        log_probs[legal].sum().backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertTrue((scores.grad[~legal] == 0).all())

    def test_each_cell_keeps_its_coarse_mass_regardless_of_cell_area(self):
        model = self.model
        for parameter in model.parameters():
            parameter.data.zero_()
        actions = cell_actions([0, 39], 38)
        exact = torch.zeros(1, actions.numel(), 10)
        legal = torch.ones(1, actions.numel(), dtype=torch.bool)
        logits, _ = model(torch.zeros(1, len(COARSE_PLANES), 38, 38), exact, actions, legal)
        owners = cell_ids(actions, 38)
        for cell in [0, 39]:
            torch.testing.assert_close(
                logits[0, :-1][owners == cell].exp().sum(), torch.tensor(1 / 1445)
            )
        torch.testing.assert_close(logits[0, -1].exp(), torch.tensor(1 / 1445))
        # A subset does NOT steal omitted cells' probability, or boost pass.
        torch.testing.assert_close(logits.exp().sum(), torch.tensor(3 / 1445))
        target = torch.tensor([[0, actions.numel()]])
        visits = torch.tensor([[3.0, 1.0]])
        loss = hierarchical_policy_value_loss(
            logits, torch.zeros(1), target, visits, torch.zeros(1)
        )
        expected = math.log(1445) + 0.75 * math.log((owners == 0).sum().item())
        self.assertAlmostEqual(loss.policy.item(), expected, places=5)

    def test_omitting_zero_target_cells_does_not_change_loss_or_gradients(self):
        coarse = torch.randn(1, len(COARSE_PLANES), 38, 38)
        all_actions = cell_actions([39, 40], 38)
        all_exact = torch.randn(1, all_actions.numel(), 10)
        count = cell_actions([39], 38).numel()
        gradients, losses = [], []
        for n in [count, all_actions.numel()]:
            self.model.zero_grad(set_to_none=True)
            logits, values = self.model(
                coarse, all_exact[:, :n], all_actions[:n], torch.ones(1, n, dtype=torch.bool)
            )
            loss = hierarchical_policy_value_loss(
                logits, values, torch.tensor([[0, n]]), torch.tensor([[7.0, 1.0]]), torch.ones(1)
            )
            loss.total.backward()
            losses.append(loss.total.detach())
            gradients.append(
                [p.grad.clone() if p.grad is not None else None for p in self.model.parameters()]
            )
        torch.testing.assert_close(*losses)
        for first, second in zip(*gradients, strict=True):
            if first is None:
                self.assertIsNone(second)  # Auxiliary heads are not in this isolated policy loss.
            else:
                torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-5)

    def test_ragged_padding_and_pass_only_training_match_unbatched(self):
        actions = cell_actions([39], 38)
        coarse = torch.randn(len(COARSE_PLANES), 38, 38)
        first = TrainingSample(
            coarse,
            torch.randn(actions.numel(), 10),
            torch.ones(actions.numel(), dtype=torch.bool),
            torch.tensor([0, actions.numel()]),
            torch.tensor([7.0, 1.0]),
            1.0,
            "first",
            0,
            actions,
            13,
            1.0,
            torch.ones(38, 38),
        )
        second = TrainingSample(
            coarse,
            torch.zeros(0, 10),
            torch.zeros(0, dtype=torch.bool),
            torch.tensor([0]),
            torch.tensor([3.0]),
            -1.0,
            "second",
            0,
            torch.empty(0, dtype=torch.int64),
            13,
            -1.0,
            -torch.ones(38, 38),
        )

        def loss(samples):
            batch = collate(samples)
            logits, values, scores, ownership = self.model.forward_all(
                batch.coarse, batch.exact, batch.actions, batch.legal
            )
            return joint_loss(
                logits,
                values,
                scores,
                ownership,
                batch.indices,
                batch.visits,
                batch.outcomes,
                batch.scores,
                batch.ownership,
            )

        combined = loss([first, second])
        singles = [loss([sample]) for sample in [first, second]]
        torch.testing.assert_close(combined.total, (singles[0].total + singles[1].total) / 2)
        combined.total.backward()
        for parameter in self.model.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
