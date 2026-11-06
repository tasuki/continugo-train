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

from .features import ownership_weights


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
    policy_weights: Tensor | None = None,
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
    if policy_weights is None:
        policy_weights = values.new_ones(batch)
    if (
        policy_weights.shape != (batch,)
        or not torch.isfinite(policy_weights).all()
        or ((policy_weights != 0) & (policy_weights != 1)).any()
    ):
        raise ValueError("policy weights must be a binary mask of shape [batch]")
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
    per_position = log_normalizer - (selected * (visits / totals[:, None])).sum(dim=1)
    # Normalize over enabled positions, not all value samples. All-cheap batches
    # retain a differentiable zero policy loss and full value/auxiliary losses.
    policy = (per_position * policy_weights).sum() / policy_weights.sum().clamp_min(1)
    value = F.mse_loss(values, outcomes)
    return Losses(policy + value, policy, value)


class AuxiliaryLosses(NamedTuple):
    total: Tensor
    policy: Tensor
    value: Tensor
    score: Tensor
    ownership: Tensor


def joint_loss(
    log_probs,
    values,
    scores,
    ownership,
    target_indices,
    visits,
    outcomes,
    score_targets,
    ownership_targets,
    *,
    score_weight=0.25,
    ownership_weight=0.25,
    policy_weights=None,
):
    base = hierarchical_policy_value_loss(
        log_probs, values, target_indices, visits, outcomes, policy_weights=policy_weights
    )
    if (
        scores.shape != outcomes.shape
        or ownership.shape != ownership_targets.shape
        or (
            ownership.ndim != 3
            or ownership.shape[0] != outcomes.shape[0]
            or ownership.shape[1] != ownership.shape[2]
        )
    ):
        raise ValueError("incompatible auxiliary target shapes")
    for target in (score_targets, ownership_targets):
        if not torch.isfinite(target).all() or (target.abs() > 1).any():
            raise ValueError("auxiliary targets must be finite and in [-1,1]")
    if score_targets.shape != scores.shape:
        raise ValueError("incompatible score target shape")
    score = F.mse_loss(scores, score_targets)
    weights = ownership_weights(ownership.shape[-1], device=ownership.device)
    owned = ((ownership - ownership_targets).square() * weights).sum(dim=(-2, -1)).mean()
    return AuxiliaryLosses(
        base.total + score_weight * score + ownership_weight * owned,
        base.policy,
        base.value,
        score,
        owned,
    )


def hierarchical_policy_value_loss(
    log_probs, values, target_indices, visits, outcomes, *, policy_weights=None
):
    """Exact hierarchical CE from full-coarse / complete-target-cell log probs.

    Every positive target's cell must be supplied in full. There is no sampled
    coordinate denominator or importance correction: omitted cells have zero
    target mass, but still participate in the global coarse softmax.
    """
    return policy_value_loss(
        log_probs,
        values,
        target_indices,
        visits,
        outcomes,
        normalize=False,
        policy_weights=policy_weights,
    )
