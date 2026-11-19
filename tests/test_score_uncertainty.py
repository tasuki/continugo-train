"""Score moments are supervised only by the supplied training targets."""

import math
import unittest

import numpy as np
import torch

from continugo_train.losses import joint_loss
from continugo_train.model import SCORE_STDEV_FLOOR, NetworkConfig, PolicyValueNet
from continugo_train.utility import score_utility


def score_loss(means, stdev, targets):
    n = len(targets)
    return joint_loss(
        torch.zeros(n, 1),
        torch.zeros(n),
        means,
        torch.zeros(n, 2, 2),
        torch.zeros(n, 1, dtype=torch.int64),
        torch.ones(n, 1),
        torch.ones(n),
        targets,
        torch.zeros(n, 2, 2),
        score_stdev=stdev,
    ).score


class ScoreUncertaintyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)

    def test_gaussian_loss_has_correct_moment_optimum_and_validates_sigma(self):
        means = torch.tensor([0.2, 0.2], requires_grad=True)
        stdev = torch.tensor([0.3, 0.3], requires_grad=True)
        targets = torch.tensor([-0.1, 0.5])
        loss = score_loss(means, stdev, targets)
        self.assertAlmostEqual(loss.item(), math.log(0.3) + 0.5, places=6)
        loss.backward()
        self.assertAlmostEqual(means.grad.sum().item(), 0, places=6)
        torch.testing.assert_close(stdev.grad, torch.zeros(2), atol=1e-6, rtol=0)
        self.assertLess(loss, score_loss(means, torch.full((2,), 0.1), targets))
        self.assertLess(loss, score_loss(means, torch.full((2,), 0.7), targets))
        for invalid in (torch.zeros(2), torch.full((2,), float("nan")), torch.ones(3)):
            with self.assertRaisesRegex(ValueError, "stdev"):
                score_loss(means, invalid, targets)

    def test_shared_model_learns_mean_and_noise_on_repeated_training_input(self):
        model = PolicyValueNet(
            NetworkConfig(channels=4, blocks=1, value_channels=4, value_hidden=8)
        ).eval()
        coarse = torch.randn(1, model.config.input_planes, 6, 6).expand(4, -1, -1, -1)
        targets = torch.tensor([-0.1, 0.5, -0.1, 0.5])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(160):
            optimizer.zero_grad(set_to_none=True)
            _, _, _, means, stdev = model.encode_search(coarse)
            loss = score_loss(means, stdev, targets)
            loss.backward()
            optimizer.step()
        torch.testing.assert_close(means, torch.full((4,), 0.2), atol=0.015, rtol=0)
        torch.testing.assert_close(stdev, torch.full((4,), 0.3), atol=0.015, rtol=0)
        self.assertGreater(model.value_shared[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.value_spatial[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.score_uncertainty.weight.grad.abs().sum().item(), 0)

    def test_sigma_floor_keeps_extreme_target_gradients_finite(self):
        mean = torch.tensor([-1.0, 1.0], requires_grad=True)
        sigma = torch.full((2,), SCORE_STDEV_FLOOR, requires_grad=True)
        loss = score_loss(mean, sigma, -mean.detach())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(mean.grad).all())
        self.assertTrue(torch.isfinite(sigma.grad).all())

    def test_curve_integrates_uncertainty_and_matches_fine_reference(self):
        z = np.linspace(-8, 8, 32769)
        density = np.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
        for size in (9, 13, 19):
            for mean in (-0.9, -0.1, 0, 0.025, 0.8):
                for sigma in (0, 0.001, 0.1, 0.5, 1):
                    actual = score_utility(mean, sigma, size)
                    reference = np.trapezoid(
                        2 / math.pi * np.arctan((mean + sigma * z) / (2 / size)) * density, z
                    )
                    self.assertAlmostEqual(actual, reference, delta=0.0002)
                    self.assertAlmostEqual(actual, -score_utility(-mean, sigma, size), places=14)
                    self.assertLessEqual(abs(actual), 1)
            self.assertLess(score_utility(0.2, 0.5, size), score_utility(0.2, 0, size))


if __name__ == "__main__":
    unittest.main()
