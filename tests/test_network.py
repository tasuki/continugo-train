import io
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from unittest.mock import patch

import torch
from torch.nn import functional as F

from continugo_train.benchmark import main as benchmark_main
from continugo_train.losses import policy_value_loss
from continugo_train.model import NetworkConfig, PolicyValueNet, resolution_for_diameter


def small_config(**kwargs):
    return NetworkConfig(
        input_planes=3,
        move_features=2,
        channels=8,
        blocks=1,
        policy_hidden=4,
        value_hidden=8,
        **kwargs,
    )


def inputs(config, batch=2, count=17, resolution=7):
    return (
        torch.randn(batch, config.input_planes, resolution, resolution),
        torch.randn(batch, count, config.move_features),
        torch.arange(count, dtype=torch.int64),
        torch.ones(batch, count, dtype=torch.bool),
    )


class NetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)

    def test_single_model_handles_resolution_buckets_and_masks_plus_pass(self):
        config = small_config(chunk_size=7)
        model = PolicyValueNet(config).eval()
        for diameter, resolution in [(110, 38), (76, 54), (52, 78)]:
            self.assertEqual(resolution_for_diameter(diameter), resolution)
            coarse, exact, actions, legal = inputs(config, resolution=resolution)
            legal[:, ::2] = False
            logits, value = model(coarse, exact, actions, legal)
            self.assertEqual(logits.shape, (2, 18))
            self.assertEqual(value.shape, (2,))
            self.assertTrue((value.abs() <= 1).all())
            self.assertTrue(torch.isneginf(logits[:, :-1][~legal]).all())
            self.assertTrue(torch.isfinite(logits[:, -1]).all())
            self.assertTrue(torch.isfinite(logits.softmax(dim=1)).all())
        legal[:] = False
        logits, _ = model(coarse, exact, actions, legal)
        torch.testing.assert_close(logits.softmax(dim=1)[:, -1], torch.ones(2))

    def test_chunked_and_unchunked_logits_and_gradients_match(self):
        config = small_config(chunk_size=7)
        chunked = PolicyValueNet(config).eval()
        whole = PolicyValueNet(replace(config, chunk_size=100)).eval()
        whole.load_state_dict(chunked.state_dict())
        args = inputs(config)
        a, av = chunked(*args)
        b, bv = whole(*args)
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(av, bv)
        (a.square().mean() + av.square().mean()).backward()
        (b.square().mean() + bv.square().mean()).backward()
        for first, second in zip(chunked.parameters(), whole.parameters(), strict=True):
            torch.testing.assert_close(first.grad, second.grad, atol=1e-6, rtol=1e-5)

    def test_sampling_axes_endpoints_and_exact_neighbor_distinctions(self):
        model = PolicyValueNet(small_config()).eval()
        with torch.no_grad():
            model.policy_exact.weight.zero_()
            model.policy_exact.weight[0, 0] = 1
            model.policy_score.weight.zero_()
            model.policy_score.weight[0, 0] = 1
            model.policy_score.bias.zero_()
        spatial = torch.zeros(1, 4, 2, 2)
        spatial[0, 0] = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        actions = torch.tensor([0, 999, 999000, 999999, 500500])
        exact = torch.zeros(1, 5, 2)
        logits = model.score_chunk(spatial, exact, actions)
        torch.testing.assert_close(logits, torch.tensor([[0.0, 1.0, 2.0, 3.0, 1500 / 999]]))
        # Identical coarse context, neighboring coordinates, different exact feature.
        spatial.zero_()
        exact = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
        logits = model.score_chunk(spatial, exact, torch.tensor([500500, 500501]))
        torch.testing.assert_close(logits, torch.tensor([[0.0, 1.0]]))

    def test_full_million_coordinate_domain(self):
        config = small_config()
        model = PolicyValueNet(config).eval()
        with torch.inference_mode():
            logits, _ = model(*inputs(config, batch=1, count=1000000))
        self.assertEqual(logits.shape, (1, 1000001))
        self.assertTrue(torch.isfinite(logits).all())

    def test_empty_subset_is_pass_only(self):
        model = PolicyValueNet(small_config()).eval()
        logits, value = model(*inputs(model.config, count=0))
        self.assertEqual(logits.shape, (2, 1))
        self.assertEqual(value.shape, (2,))
        torch.testing.assert_close(logits.softmax(dim=1), torch.ones(2, 1))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_forward_matches_cpu_and_backward_is_finite(self):
        model = PolicyValueNet(small_config(chunk_size=7)).eval()
        gpu = PolicyValueNet(model.config).cuda().eval()
        gpu.load_state_dict(model.state_dict())
        args = inputs(model.config)
        args[-1][:, 0] = False
        expected = model(*args)
        actual = gpu(*(tensor.cuda() for tensor in args))
        for cpu, cuda in zip(expected, actual, strict=True):
            torch.testing.assert_close(cpu, cuda.cpu(), atol=1e-5, rtol=1e-4)
        indices = torch.tensor([[1, 17], [1, 17]], device="cuda")
        visits = torch.ones(2, 2, device="cuda")
        loss = policy_value_loss(*actual, indices, visits, torch.zeros(2, device="cuda"))
        loss.total.backward()
        for parameter in gpu.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_save_load_includes_configuration_and_batchnorm_state(self):
        config = small_config()
        model = PolicyValueNet(config)
        args = inputs(config)
        model(*args)  # Populate running statistics.
        model.eval()
        buffer = io.BytesIO()
        torch.save({"config": asdict(config), "state_dict": model.state_dict()}, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        restored = PolicyValueNet(NetworkConfig(**saved["config"]))
        restored.load_state_dict(saved["state_dict"])
        restored.eval()
        for expected, actual in zip(model(*args), restored(*args), strict=True):
            torch.testing.assert_close(expected, actual)

    def test_tiny_batch_can_be_overfit_and_all_heads_receive_gradients(self):
        config = small_config()
        model = PolicyValueNet(config).train()
        args = inputs(config, count=8)
        indices = torch.tensor([[0, 8], [3, 8]])
        visits = torch.tensor([[8.0, 1.0], [8.0, 1.0]])
        outcomes = torch.tensor([-1.0, 1.0])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        initial = None
        for step in range(200):
            optimizer.zero_grad(set_to_none=True)
            logits, values = model(*args)
            losses = policy_value_loss(logits, values, indices, visits, outcomes)
            if initial is None:
                initial = losses.total.item()
            losses.total.backward()
            if step == 0:
                for name in [
                    "trunk.0.weight",
                    "policy_exact.weight",
                    "policy_spatial.weight",
                    "policy_global.weight",
                    "policy_score.weight",
                    "pass_head.weight",
                    "value_head.2.weight",
                ]:
                    grad = dict(model.named_parameters())[name].grad
                    self.assertTrue(torch.isfinite(grad).all(), name)
                    self.assertGreater(grad.abs().sum().item(), 0, name)
            optimizer.step()
        self.assertLess(losses.total.item(), initial * 0.3)

    def test_leaky_value_activation_restores_gradients_without_changing_policy_or_parameters(self):
        config = small_config()
        relu = PolicyValueNet(config).eval()
        with torch.no_grad():
            relu.value_head[0].weight.fill_(0.01)
            relu.value_head[0].bias.fill_(-1)
            relu.value_head[2].weight.fill_(0.1)
            relu.value_head[2].bias.zero_()
        leaky = PolicyValueNet(replace(config, value_activation="leaky_relu")).eval()
        leaky.load_state_dict(relu.state_dict())
        self.assertEqual(leaky.value_head[1].negative_slope, 0.1)
        self.assertEqual(list(relu.state_dict()), list(leaky.state_dict()))
        args = inputs(config)
        torch.testing.assert_close(relu(*args)[0], leaky(*args)[0], atol=0, rtol=0)
        for model, alive in [(relu, False), (leaky, True)]:
            pooled = torch.full((2, config.channels), 0.5, requires_grad=True)
            self.assertTrue((model.value_head[0](pooled) < 0).all())
            model.value_head(pooled).sum().backward()
            for grad in [
                pooled.grad,
                model.value_head[0].weight.grad,
                model.value_head[2].weight.grad,
            ]:
                self.assertEqual(grad.abs().sum().item() > 0, alive)
        legacy = asdict(config)
        legacy.pop("value_activation")
        self.assertEqual(NetworkConfig(**legacy).value_activation, "relu")
        restored = PolicyValueNet(NetworkConfig(**asdict(leaky.config))).eval()
        restored.load_state_dict(leaky.state_dict())
        torch.testing.assert_close(restored(*args)[1], leaky(*args)[1], atol=0, rtol=0)

    def test_configuration_and_bad_action_inputs_are_rejected(self):
        for kwargs in [
            {"channels": 0},
            {"blocks": -1},
            {"chunk_size": True},
            {"value_activation": "bad"},
        ]:
            with self.assertRaises(ValueError):
                NetworkConfig(**kwargs)
        model = PolicyValueNet(small_config()).eval()
        coarse, exact, actions, legal = inputs(model.config)
        for invalid in [actions.float(), actions + 1000000, actions - 1000000]:
            with self.assertRaises(ValueError):
                model(coarse, exact, invalid, legal)
        with self.assertRaises(ValueError):
            model(coarse, exact, actions, legal.float())
        with self.assertRaises(ValueError):
            resolution_for_diameter(0)


class LossTests(unittest.TestCase):
    def test_sparse_visits_equal_dense_cross_entropy_including_zero_padding(self):
        logits = torch.tensor([[0.5, -torch.inf, -0.3, 0.1]], requires_grad=True)
        values = torch.tensor([0.3], requires_grad=True)
        indices = torch.tensor([[0, 2, 3, 1]])
        visits = torch.tensor([[2.0, 3.0, 1.0, 0.0]])
        loss = policy_value_loss(logits, values, indices, visits, torch.tensor([-1.0]))
        dense = torch.tensor([[2 / 6, 3 / 6, 1 / 6]])
        expected_policy = -(F.log_softmax(logits[:, [0, 2, 3]], dim=1) * dense).sum()
        torch.testing.assert_close(loss.policy, expected_policy)
        torch.testing.assert_close(loss.value, torch.tensor(1.3**2))
        loss.total.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertEqual(logits.grad[0, 1].item(), 0)

    def test_pass_only_target_and_duplicate_counts(self):
        logits = torch.tensor([[-torch.inf, 0.2]])
        losses = policy_value_loss(
            logits,
            torch.zeros(1),
            torch.tensor([[1, 1]]),
            torch.tensor([[2.0, 3.0]]),
            torch.zeros(1),
        )
        torch.testing.assert_close(losses.total, torch.tensor(0.0))

    def test_invalid_or_illegal_targets_fail_instead_of_producing_nan(self):
        logits = torch.tensor([[0.0, -torch.inf, 0.0]])
        for indices, visits, outcomes in [
            ([[1]], [[1.0]], [0.0]),
            ([[0]], [[0.0]], [0.0]),
            ([[0]], [[-1.0]], [0.0]),
            ([[0]], [[float("nan")]], [0.0]),
            ([[0]], [[1.0]], [2.0]),
            ([[0]], [[1.0]], [float("nan")]),
            ([[3]], [[1.0]], [0.0]),
        ]:
            with self.assertRaises(ValueError):
                policy_value_loss(
                    logits,
                    torch.zeros(1),
                    torch.tensor(indices),
                    torch.tensor(visits),
                    torch.tensor(outcomes),
                )


class BenchmarkTests(unittest.TestCase):
    def test_small_cpu_benchmark_runs_all_stages(self):
        arguments = [
            "benchmark",
            "--device",
            "cpu",
            "--diameters",
            "110",
            "--batch-sizes",
            "1",
            "--actions",
            "17",
            "--channels",
            "4",
            "--blocks",
            "1",
            "--threads",
            "1",
            "--warmup",
            "0",
            "--iterations",
            "1",
        ]
        output = io.StringIO()
        with patch("sys.argv", arguments), redirect_stdout(output):
            benchmark_main()
        for stage in ["coarse_infer", "policy_infer", "full_infer", "coarse_step", "full_step"]:
            self.assertIn(stage, output.getvalue())


if __name__ == "__main__":
    unittest.main()
