"""Sparse visit-count policy losses and expected-outcome regression.

Training uses hierarchical_policy_value_loss on globally coarse-normalized,
complete-cell log probabilities; it must not renormalize the candidate subset.
policy_value_loss is the generic logit-CE helper used by diagnostic tests.
Targets/outcomes come from validated SGF/Rust data; exclude truncated outcomes
and flip RE to the pre-move player's view.
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
    logits: Tensor,
    values: Tensor,
    target_indices: Tensor,
    visits: Tensor,
    outcomes: Tensor,
    *,
    normalize: bool = True,
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
    log_normalizer = torch.logsumexp(logits, dim=1) if normalize else logits.new_zeros(batch)
    if not torch.isfinite(log_normalizer).all():
        raise ValueError("policy needs a finite legal normalization")
    selected = torch.where(positive, selected, 0.0)  # Avoid 0 * -inf for padding.
    policy = (log_normalizer - (selected * (visits / totals[:, None])).sum(dim=1)).mean()
    value = F.mse_loss(values, outcomes)
    return Losses(policy + value, policy, value)


def hierarchical_policy_value_loss(log_probs, values, target_indices, visits, outcomes):
    """Exact hierarchical CE from full-coarse / complete-target-cell log probs.

    Every positive target's cell must be supplied in full. There is no sampled
    coordinate denominator or importance correction: omitted cells have zero
    target mass, but still participate in the global coarse softmax.
    """
    return policy_value_loss(log_probs, values, target_indices, visits, outcomes, normalize=False)
