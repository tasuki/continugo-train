"""Sparse visit-count cross-entropy and expected-outcome regression.

No dense target distribution is needed, but the policy normalization must still
include ALL legal logits plus pass. A candidate-only batch instead trains a
conditional policy and must not be confused with full-action training.
Targets/outcomes are supplied by the SGF/Rust pipeline in a later stage. Exclude
truncated games from outcome training, and flip RE to the pre-move player's view.
"""

from typing import NamedTuple

import torch
from torch import Tensor
from torch.nn import functional as F


class Losses(NamedTuple):
    total: Tensor
    policy: Tensor
    value: Tensor


def policy_value_loss(
    logits: Tensor, values: Tensor, target_indices: Tensor, visits: Tensor, outcomes: Tensor
) -> Losses:
    """Targets are output-column indices [B,K], counts [B,K], outcomes [B].

    Zero-count padding is allowed (with valid indices), including illegal moves;
    any positive count on a masked/illegal move is an error. Duplicate target
    indices are equivalent to summing their counts.
    """
    if logits.ndim != 2:
        raise ValueError("policy logits must have shape [batch, actions]")
    batch = logits.shape[0]
    if (
        values.shape != (batch,)
        or outcomes.shape != (batch,)
        or visits.ndim != 2
        or visits.shape != target_indices.shape
        or visits.shape[0] != batch
        or target_indices.dtype != torch.int64
    ):
        raise ValueError("incompatible policy/value target shapes or index dtype")
    if (
        not torch.isfinite(visits).all()
        or (visits < 0).any()
        or not torch.isfinite(outcomes).all()
        or (outcomes.abs() > 1).any()
    ):
        raise ValueError("counts must be finite/nonnegative and outcomes in [-1,1]")
    totals = visits.sum(dim=1)
    if not torch.isfinite(totals).all() or (totals <= 0).any():
        raise ValueError("every policy target needs positive visit mass")
    if target_indices.numel() and (
        target_indices.min() < 0 or target_indices.max() >= logits.shape[1]
    ):
        raise ValueError("target index outside policy output")
    selected = logits.gather(1, target_indices)
    positive = visits > 0
    if not torch.isfinite(selected[positive]).all():
        raise ValueError("positive policy target on a masked or nonfinite logit")
    log_normalizer = torch.logsumexp(logits, dim=1)
    if not torch.isfinite(log_normalizer).all():
        raise ValueError("policy needs a finite legal normalization")
    selected = torch.where(positive, selected, 0.0)  # Avoid 0 * -inf for padding.
    policy = (log_normalizer - (selected * (visits / totals[:, None])).sum(dim=1)).mean()
    value = F.mse_loss(values, outcomes)
    return Losses(policy + value, policy, value)
