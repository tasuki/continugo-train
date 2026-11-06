import json
import math
import struct
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch
from test_features import SGF

from continugo_train.data import scan_corpus
from continugo_train.diagnose import diagnose
from continugo_train.features import HEADER, FeatureClient, decode_sample, ownership_weights
from continugo_train.losses import joint_loss
from continugo_train.model import NetworkConfig, PolicyValueNet
from continugo_train.train import Options, feature_spec


class AuxiliaryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "game.sgf"
        self.path.write_text(SGF)

    def model(self):
        return PolicyValueNet(
            NetworkConfig(
                channels=4,
                blocks=1,
                policy_hidden=4,
                value_hidden=4,
                score_hidden=4,
                ownership_hidden=4,
            )
        )

    def test_final_labels_flip_perspective_and_are_absent_without_validated_result(self):
        with FeatureClient() as client:
            black = client.load(self.path, 0)
            white = client.load(self.path, 1)
            late_black = client.load(self.path, 2)
            self.assertEqual(black.score, 1)
            self.assertEqual(white.score, -1)
            torch.testing.assert_close(black.ownership, torch.ones(54, 54))
            torch.testing.assert_close(white.ownership, -black.ownership)
            torch.testing.assert_close(late_black.ownership, black.ownership)
            for sample in (black, white, late_black):
                self.assertAlmostEqual(
                    (sample.ownership * ownership_weights(54)).sum().item(), sample.score
                )
        self.path.write_text(SGF.replace("RE[B+220.4]", ""))
        with FeatureClient() as client:
            unknown = client.load(self.path, 0)
            self.assertIsNone(unknown.score)
            self.assertIsNone(unknown.ownership)
        self.path.write_text(r'(;FF[9071]GM[9071]SZ[13]RE[0];B[]C[{"policy":[["",1\]\]}];W[])')
        with FeatureClient() as client:
            draw = client.load(self.path, 0)
            self.assertEqual(draw.score, 0)
            self.assertEqual(draw.ownership.count_nonzero(), 0)

    def test_ownership_loss_is_area_weighted_and_new_heads_receive_gradients(self):
        n = 8
        model = self.model().train()
        coarse = torch.randn(2, 12, n, n)
        exact = torch.empty(2, 0, 10)
        actions = torch.empty(2, 0, dtype=torch.int64)
        legal = torch.empty(2, 0, dtype=torch.bool)
        outputs = model.forward_all(coarse, exact, actions, legal)
        self.assertEqual(outputs[2].shape, (2,))
        self.assertEqual(outputs[3].shape, (2, n, n))
        outcome = torch.tensor([1.0, -1.0])
        score = torch.tensor([0.8, -0.5])
        owned = torch.stack((torch.ones(n, n), -torch.ones(n, n)))
        loss = joint_loss(
            *outputs, torch.zeros(2, 1, dtype=torch.int64), torch.ones(2, 1), outcome, score, owned
        )
        torch.testing.assert_close(
            loss.total, loss.policy + loss.value + 0.25 * loss.score + 0.25 * loss.ownership
        )
        expected = ((outputs[3] - owned).square() * ownership_weights(n)).sum((-2, -1)).mean()
        torch.testing.assert_close(loss.ownership, expected)
        loss.total.backward()
        for name, parameter in model.named_parameters():
            # No placements in this batch: the exact scorer is unused.
            if name.startswith(("policy_exact.", "policy_score.")):
                continue
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for name in (
            "score_head.0.weight",
            "score_head.2.weight",
            "ownership_head.0.weight",
            "ownership_head.2.weight",
            "trunk.0.weight",
        ):
            self.assertGreater(
                dict(model.named_parameters())[name].grad.abs().sum().item(), 0, name
            )
        torch.testing.assert_close(ownership_weights(n).sum(), torch.tensor(1.0))
        self.assertLess(ownership_weights(n)[0, 0], ownership_weights(n)[1, 1])

    def test_policy_mask_excludes_cheap_positions_without_masking_other_heads(self):
        logits = torch.randn(2, 3, requires_grad=True)
        values = torch.zeros(2, requires_grad=True)
        scores = torch.zeros(2, requires_grad=True)
        ownership = torch.zeros(2, 4, 4, requires_grad=True)
        targets = torch.tensor([[0], [1]])
        visits = torch.tensor([[32.0], [256.0]])
        outcomes = torch.tensor([1.0, -1.0])
        owned = torch.ones(2, 4, 4)

        def loss(mask):
            return joint_loss(
                logits.log_softmax(1),
                values,
                scores,
                ownership,
                targets,
                visits,
                outcomes,
                outcomes,
                owned,
                policy_weights=mask,
            )

        mixed = loss(torch.tensor([0.0, 1.0]))
        torch.testing.assert_close(mixed.policy, -logits.log_softmax(1)[1, 1])
        mixed.total.backward()
        self.assertEqual(logits.grad[0].count_nonzero(), 0)
        self.assertGreater(logits.grad[1].abs().sum(), 0)
        for tensor in (values, scores, ownership):
            self.assertTrue((tensor.grad.reshape(2, -1).abs().sum(1) > 0).all())
        logits.grad = None
        cheap = loss(torch.zeros(2))
        self.assertEqual(cheap.policy.item(), 0)
        cheap.total.backward()
        self.assertEqual(logits.grad.count_nonzero(), 0)
        self.assertTrue(torch.isfinite(cheap.total))
        for mask in (torch.ones(3), torch.tensor([0.0, float("nan")]), torch.tensor([0.0, 0.5])):
            with self.assertRaisesRegex(ValueError, "policy weights"):
                loss(mask)

    def test_tiny_auxiliary_problem_overfits_and_search_omits_ownership(self):
        model = self.model().eval()
        coarse = torch.randn(2, 12, 8, 8)
        score_target = torch.tensor([0.7, -0.7])
        ownership_target = torch.stack((torch.ones(8, 8), -torch.ones(8, 8)))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        initial = None
        for _ in range(65):
            optimizer.zero_grad(set_to_none=True)
            _, _, _, scores, ownership = model.encode_training(coarse)
            loss = (scores - score_target).square().mean() + (
                ownership - ownership_target
            ).square().mean()
            initial = loss.item() if initial is None else initial
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), initial * 0.1)
        called = []
        hook = model.ownership_head.register_forward_hook(lambda *_: called.append(True))
        expected = model.encode_training(coarse)
        called.clear()
        actual = model.encode_search(coarse)
        hook.remove()
        self.assertFalse(called)
        for a, b in zip(expected[:4], actual, strict=True):
            torch.testing.assert_close(a, b)

    def test_new_target_contract_and_loss_options_are_validated(self):
        self.assertEqual(feature_spec(4)["targets"], "final-area-v1")
        self.assertEqual(Options().score_loss_weight, 0.25)
        self.assertEqual(Options().ownership_loss_weight, 0.25)
        for weight in (-1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                Options(score_loss_weight=weight).validate()
        # Header-only old-format packets fail rather than receiving invented labels.
        old = struct.pack("<8s11If2I", b"CGFEAT01", 4, 12, 10, 13, 76, 54, 0, 0, 0, 0, 0, 0.0, 4, 0)
        with self.assertRaises(ValueError):
            decode_sample(old)
        self.assertEqual(HEADER.size, 68)

    def test_diagnostics_only_use_pinned_validation_games(self):
        other = self.root / "other.sgf"
        other.write_text(SGF.replace("dSdS", "dTdT"))
        corpus = scan_corpus([str(self.root)], 4, 7, 0.5)
        model = self.model().eval()
        checkpoint = self.root / "model.pt"
        torch.save(
            {
                "config": asdict(model.config),
                "model": model.state_dict(),
                "feature_spec": feature_spec(4),
                "manifest": corpus.manifest(),
                "split_history": corpus.split_history,
            },
            checkpoint,
        )
        report = diagnose(checkpoint, [str(self.root)])
        self.assertEqual(report["summary"]["positions"], 2)
        self.assertTrue(
            all(
                row["game_id"] in {g.game_id for g in corpus.validation}
                for row in report["positions"]
            )
        )
        self.assertEqual(len(report["checkpoint_sha256"]), 64)
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
