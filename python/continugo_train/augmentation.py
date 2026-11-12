"""D4 tensor augmentation: reflect horizontally, then rotate counterclockwise.

Candidate rows are never reordered, so sparse policy indices (including pass),
visits and legality retain their meaning. No game rules are implemented here.
"""
from dataclasses import replace

import torch


def augment_sample(sample, symmetry: int):
    """Return a non-mutating view/copy under one of eight square symmetries."""
    if not 0 <= symmetry < 8:
        raise ValueError("symmetry must be in 0..7")
    if symmetry == 0:
        return sample
    rotations, reflection = symmetry % 4, symmetry // 4
    borders = [0, 1, 2, 3]  # left, right, top, bottom
    x = sample.actions.remainder(1000)
    y = sample.actions.div(1000, rounding_mode="floor")
    if reflection:
        x = 999 - x
        borders = [borders[i] for i in (1, 0, 2, 3)]
    for _ in range(rotations):
        x, y = y, 999 - x
        borders = [borders[i] for i in (2, 3, 1, 0)]

    def spatial(tensor):
        if reflection:
            tensor = tensor.flip(-1)
        return torch.rot90(tensor, rotations, (-2, -1))

    coarse = spatial(sample.coarse).clone()
    coarse[-4:] = coarse[-4:][borders]
    exact = sample.exact.clone()
    exact[:, -4:] = sample.exact[:, -4:][:, borders]
    return replace(sample, coarse=coarse, exact=exact, actions=x + 1000 * y,
                   ownership=spatial(sample.ownership))
