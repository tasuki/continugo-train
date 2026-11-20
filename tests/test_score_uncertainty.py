"""Score moments are supervised only by the supplied training targets."""

import math
import unittest

import numpy as np
import torch

from continugo_train.losses import joint_loss
from continugo_train.model import (
    SCORE_STDEV_FLOOR,
    NetworkConfig,
    PolicyValueNet,
    validate_feature_config,
)
from continugo_train.train import feature_spec
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

    def test_linear_mean_and_softplus_uncertainty_are_unbounded(self):
        model = PolicyValueNet(NetworkConfig(channels=4, blocks=1)).eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.score_head[0].bias.fill_(3)
            model.score_uncertainty.bias.fill_(2)
        coarse = torch.zeros(2, model.config.input_planes, 6, 6)
        _, _, values, means, ownership, stdev = model.encode_training(coarse)
        torch.testing.assert_close(means, torch.full((2,), 3.0))
        self.assertTrue((stdev > 2).all())
        self.assertTrue((values.abs() <= 1).all())
        self.assertTrue((ownership.abs() <= 1).all())
        means.sum().backward()
        self.assertEqual(model.score_head[0].bias.grad.item(), 2)
        with torch.no_grad():
            model.score_uncertainty.bias.fill_(-1000)
        self.assertTrue((model.encode_search(coarse)[-1] >= SCORE_STDEV_FLOOR).all())

    def test_unbounded_targets_and_predictions_have_finite_gaussian_loss(self):
        mean = torch.tensor([3.0, -2.0], requires_grad=True)
        sigma = torch.tensor([2.0, 4.0], requires_grad=True)
        loss = score_loss(mean, sigma, torch.tensor([-3.0, 5.0]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(mean.grad).all())
        self.assertTrue(torch.isfinite(sigma.grad).all())
        for invalid in (float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite"):
                score_loss(mean, sigma, torch.full((2,), invalid))

    def test_score_contract_rejects_bounded_heads_for_fresh_schema7(self):
        spec = feature_spec(4)
        self.assertEqual(spec["value"], "shared-gaussian-linear-v2")
        self.assertEqual(spec["targets"], "final-area-komi-v2")
        validate_feature_config(NetworkConfig(), spec)
        with self.assertRaisesRegex(ValueError, "score distribution"):
            validate_feature_config(NetworkConfig(score_distribution="tanh-sigmoid"), spec)

    def test_legacy_config_retains_bounded_heads(self):
        settings = {"channels": 4, "blocks": 1, "global_inputs": 0}
        config = NetworkConfig.from_checkpoint(settings)
        model = PolicyValueNet(config).eval()
        with torch.no_grad():
            model.score_head[0].bias.fill_(3)
            model.score_uncertainty.bias.fill_(2)
        _, _, _, mean, stdev = model.encode_search(torch.zeros(2, config.input_planes, 6, 6))
        self.assertTrue((mean.abs() <= 1).all())
        self.assertTrue((stdev <= 1).all())

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
            for mean in (-3, -0.9, -0.1, 0, 0.025, 0.8, 3):
                for sigma in (0, 0.001, 0.1, 0.5, 1, 2, 10):
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
