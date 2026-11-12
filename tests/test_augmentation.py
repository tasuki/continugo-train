"""Square symmetries preserve candidate identity and rotate directional fields."""
from dataclasses import replace
from unittest.mock import patch

import unittest
import torch

from continugo_train.augmentation import augment_sample
from continugo_train.data import TrainingSample, batches


def sample(fields=11):
    axis = torch.linspace(0, 1, 5)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    coarse = torch.stack([x + 3*y + i for i in range(8)] + [x, 1-x, y, 1-y])
    actions = torch.tensor([0, 999, 999000, 999999, 237 + 1000*641])
    ax, ay = actions % 1000 / 999, actions // 1000 / 999
    exact = torch.arange(5*fields).reshape(5, fields).float()
    exact[:, -4:] = torch.stack([ax, 1-ax, ay, 1-ay], dim=1)
    return TrainingSample(coarse, exact, torch.tensor([1, 0, 1, 1, 1]).bool(),
                          torch.tensor([0, 4, 5]), torch.tensor([3., 7., 2.]),
                          -1., "game", 7, actions, 9, -.2, x + 3*y, .5)


def check_all_eight(fields):
    original = sample(fields)
    seen = set()
    for symmetry in range(8):
        result = augment_sample(original, symmetry)
        seen.add(tuple(result.actions.tolist()))
        ax, ay = result.actions % 1000 / 999, result.actions // 1000 / 999
        expected = torch.stack([ax, 1-ax, ay, 1-ay], dim=1)
        torch.testing.assert_close(result.exact[:, -4:], expected)
        torch.testing.assert_close(result.coarse[-4:], original.coarse[-4:])
        torch.testing.assert_close(result.coarse[0], result.ownership)
        for name in ("indices", "visits", "legal", "outcome", "score", "policy_weight"):
            assert getattr(result, name) is getattr(original, name)
        torch.testing.assert_close(result.exact[:, :-4], original.exact[:, :-4])
        # Every symmetry has an inverse, including its border-channel action.
        assert any(torch.equal(augment_sample(result, inverse).actions, original.actions)
                   and torch.equal(augment_sample(result, inverse).coarse, original.coarse)
                   and torch.equal(augment_sample(result, inverse).exact, original.exact)
                   for inverse in range(8))
    assert len(seen) == 8
    torch.testing.assert_close(original.coarse, sample(fields).coarse)
    torch.testing.assert_close(original.exact, sample(fields).exact)


def check_empty_candidates_and_invalid_symmetry():
    original = sample()
    empty = replace(original, actions=original.actions[:0], exact=original.exact[:0],
                    legal=original.legal[:0], indices=torch.tensor([0]))
    for symmetry in range(8):
        assert augment_sample(empty, symmetry).actions.numel() == 0
    with unittest.TestCase().assertRaises(ValueError):
        augment_sample(original, 8)


def check_stream_seed_validation_and_no_extra_exposures():
    def collect(training, epoch, augmentation="d4"):
        with patch("continugo_train.data._plans", return_value=[]), patch(
            "continugo_train.data._parallel_samples", side_effect=lambda *a: (sample() for _ in range(16))):
            return list(batches([], batch_size=4, seed=42, epoch=epoch, training=training,
                                density=4, workers=0, shuffle_buffer=0, augmentation=augmentation))
    first, repeated, second = collect(True, 0), collect(True, 0), collect(True, 1)
    assert len(first) == 4
    assert all(torch.equal(a.actions, b.actions) for a, b in zip(first, repeated))
    assert any(not torch.equal(a.actions, b.actions) for a, b in zip(first, second))
    for a, b in zip(collect(False, 0), collect(False, 12)):
        assert torch.equal(a.actions, b.actions)
        assert torch.equal(a.actions[0], sample().actions)
    assert all(torch.equal(b.actions[0], sample().actions) for b in collect(True, 0, "none"))


class AugmentationTests(unittest.TestCase):
    def test_all_eight(self):
        for fields in (10, 11):
            with self.subTest(fields=fields):
                check_all_eight(fields)

    def test_empty(self):
        check_empty_candidates_and_invalid_symmetry()

    def test_stream(self):
        check_stream_seed_validation_and_no_extra_exposures()
