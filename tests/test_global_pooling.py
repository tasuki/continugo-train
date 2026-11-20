import unittest
from dataclasses import replace

import torch

from continugo_train.model import (
    GlobalPoolingBlock,
    NetworkConfig,
    PolicyValueNet,
    ResidualBlock,
)
from continugo_train.train import feature_spec, parse_args


class GlobalPoolingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(47)

    def test_default_schedule_and_disabled_ablation(self):
        model = PolicyValueNet()
        blocks = list(model.trunk.children())[3:]
        self.assertEqual(
            [i + 1 for i, block in enumerate(blocks) if isinstance(block, GlobalPoolingBlock)],
            [3, 6],
        )
        self.assertEqual(feature_spec(4)["trunk"], "global-pooling-scalars-v1")
        disabled = PolicyValueNet(replace(model.config, gpool_every=0))
        self.assertFalse(any(isinstance(block, GlobalPoolingBlock) for block in disabled.modules()))
        args = parse_args(["--data", "unused", "--gpool-every", "2", "--gpool-channels", "8"])
        self.assertEqual((args.gpool_every, args.gpool_channels), (2, 8))
        for settings in ({"gpool_every": -1}, {"gpool_every": True}, {"gpool_channels": 0}):
            with self.assertRaises(ValueError):
                NetworkConfig(**settings)

    def test_distant_input_influences_local_output_and_gets_gradient(self):
        block = GlobalPoolingBlock(1, 1).eval()
        with torch.no_grad():
            block.conv1.weight.zero_()
            block.conv2.weight.zero_()
            block.conv2.weight[0, 0, 1, 1] = 1
            block.gpool_conv.weight.zero_()
            block.gpool_conv.weight[0, 0, 1, 1] = 1
            block.gpool_linear.weight.copy_(torch.tensor([[1.0, 0.0]]))
        x = torch.ones(1, 1, 17, 17, requires_grad=True)
        baseline = block(x)[0, 0, 8, 8]
        changed = x.detach().clone()
        changed[0, 0, 0, 0] += 1
        self.assertGreater(block(changed)[0, 0, 8, 8].item(), baseline.item())
        baseline.backward()
        self.assertGreater(x.grad[0, 0, 0, 0].item(), 0)

    def test_zero_context_recovers_ordinary_residual_block(self):
        pooled = GlobalPoolingBlock(4, 2).eval()
        local = ResidualBlock(4).eval()
        local.load_state_dict(
            {k: v for k, v in pooled.state_dict().items() if not k.startswith("gpool_")}
        )
        with torch.no_grad():
            pooled.gpool_linear.weight.zero_()
            for height, width in ((5, 5), (9, 13), (17, 17)):
                x = torch.randn(2, 4, height, width)
                torch.testing.assert_close(pooled(x), local(x), rtol=0, atol=0)

    def test_mean_and_max_branches_broadcast_their_own_context(self):
        block = GlobalPoolingBlock(2, 3).eval()
        x = torch.randn(2, 2, 7, 9)
        actual = []
        hook = block.gpool_linear.register_forward_pre_hook(lambda _, args: actual.append(args[0]))
        block(x)
        hook.remove()
        features = torch.relu(block.gpool_bn(block.gpool_conv(x)))
        expected = torch.cat((features.mean((-2, -1)), features.amax((-2, -1))), dim=1)
        torch.testing.assert_close(actual[0], expected)

    def test_all_global_parameters_receive_finite_training_gradients(self):
        model = PolicyValueNet(NetworkConfig(channels=4, blocks=3, gpool_channels=2)).train()
        x = torch.randn(2, model.config.input_planes, 9, 9)
        spatial, pass_logit, value, score, ownership, stdev = model.encode_training(x)
        loss = sum(
            tensor.square().mean()
            for tensor in (spatial, pass_logit, value, score, ownership, stdev)
        )
        loss.backward()
        for name, parameter in model.named_parameters():
            if "gpool_" in name:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(parameter.grad.abs().sum().item(), 0, name)


if __name__ == "__main__":
    unittest.main()
