"""Coarse cell policy, exact within-cell policy, and side-to-move value.

Each raster sample owns the integer coordinates nearest to it (ties round up).
Cells do not overlap. The policy factorizes as P(cell) * P(move | cell), with
pass as a separate coarse action. Complete cells, not individual sampled moves,
are required for exact local normalization. Training only needs target-bearing
cells; the coarse softmax still covers the whole board and pass.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .features import COARSE_PLANES, MOVE_FEATURES

COORD_RANGE = 1000
COORDINATE_ACTIONS = COORD_RANGE**2
POLICY_VERSION = "cells-v1"


def cell_ids(actions: Tensor, resolution: int) -> Tensor:
    """Same integer partition as Rust cells::cell; actions exclude pass."""
    x = (actions.remainder(1000) * (resolution - 1) + 499).div(999, rounding_mode="floor")
    y = (actions.div(1000, rounding_mode="floor") * (resolution - 1) + 499).div(
        999, rounding_mode="floor"
    )
    return x + resolution * y


def cell_actions(cells: list[int], resolution: int) -> Tensor:
    """Enumerate complete cells for synthetic benchmarks/geometry checks, not rules."""
    axis = torch.arange(1000)
    owners = (axis * (resolution - 1) + 499).div(999, rounding_mode="floor")
    result = []
    for cell in sorted(set(cells)):
        if not 0 <= cell < resolution**2:
            raise ValueError("cell outside coarse grid")
        xs, ys = axis[owners == cell % resolution], axis[owners == cell // resolution]
        result.append((xs[None] + 1000 * ys[:, None]).flatten())
    return torch.cat(result) if result else torch.empty(0, dtype=torch.int64)


def local_log_probs(scores: Tensor, actions: Tensor, legal: Tensor, resolution: int) -> Tensor:
    """Segmented legal softmax. Empty/all-illegal cells remain -inf, never NaN."""
    cells = cell_ids(actions, resolution).expand(scores.shape[0], -1)
    masked = scores.masked_fill(~legal, -torch.inf)
    maxima = scores.new_full((scores.shape[0], resolution**2), -torch.inf)
    maxima.scatter_reduce_(1, cells, masked.detach(), reduce="amax", include_self=True)
    maxima = maxima.masked_fill(~torch.isfinite(maxima), 0)
    shifted = masked - maxima.gather(1, cells)
    sums = scores.new_zeros(maxima.shape).scatter_add(1, cells, shifted.exp())
    normalizers = sums.clamp_min(torch.finfo(scores.dtype).tiny).log()
    return shifted - normalizers.gather(1, cells)


@dataclass(frozen=True)
class NetworkConfig:
    input_planes: int = len(COARSE_PLANES)
    move_features: int = len(MOVE_FEATURES)
    channels: int = 32
    blocks: int = 4
    policy_hidden: int = 16
    value_hidden: int = 32
    chunk_size: int = 65536
    top_cells: int = 32
    exploration_cells: int = 8

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


def resolution_for_diameter(diameter: int, samples_per_diameter: int = 4) -> int:
    if not 1 <= diameter <= COORD_RANGE or samples_per_diameter <= 0:
        raise ValueError("expected a positive diameter <= 1000 and sample density")
    return math.ceil((COORD_RANGE - 1) * samples_per_diameter / diameter) + 1


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = F.relu(self.bn1(self.conv1(x)))
        return F.relu(x + self.bn2(self.conv2(residual)))


class PolicyValueNet(nn.Module):
    def __init__(self, config: NetworkConfig | None = None) -> None:
        super().__init__()
        self.config = config = config or NetworkConfig()
        self.trunk = nn.Sequential(
            nn.Conv2d(config.input_planes, config.channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(config.channels),
            nn.ReLU(),
            *(ResidualBlock(config.channels) for _ in range(config.blocks)),
        )
        self.policy_spatial = nn.Conv2d(config.channels, config.policy_hidden, 1)
        self.policy_global = nn.Linear(config.channels, config.policy_hidden, bias=False)
        self.cell_head = nn.Conv2d(config.policy_hidden, 1, 1)
        self.policy_exact = nn.Linear(config.move_features, config.policy_hidden, bias=False)
        self.policy_score = nn.Linear(config.policy_hidden, 1)
        self.pass_head = nn.Linear(config.channels, 1)
        self.value_head = nn.Sequential(
            nn.Linear(config.channels, config.value_hidden),
            nn.ReLU(),
            nn.Linear(config.value_hidden, 1),
            nn.Tanh(),
        )

    def encode(self, coarse: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        trunk = self.trunk(coarse)
        pooled = trunk.mean(dim=(-2, -1))
        spatial = self.policy_spatial(trunk) + self.policy_global(pooled)[:, :, None, None]
        return spatial, self.pass_head(pooled), self.value_head(pooled).squeeze(-1)

    def coarse_logits(self, spatial: Tensor, pass_logit: Tensor) -> Tensor:
        return torch.cat((self.cell_head(spatial).flatten(1), pass_logit), dim=1)

    def score_chunk(self, spatial: Tensor, exact: Tensor, actions: Tensor) -> Tensor:
        # Shared IDs [N] or different cells per position [B,N]. Export uses [B,N].
        if actions.ndim == 1:
            actions = actions[None].expand(spatial.shape[0], -1)
        x = actions.remainder(COORD_RANGE).to(spatial.dtype)
        y = actions.div(COORD_RANGE, rounding_mode="floor").to(spatial.dtype)
        grid = torch.stack((x, y), dim=-1) * (2.0 / (COORD_RANGE - 1)) - 1.0
        sampled = F.grid_sample(spatial, grid[:, None], mode="bilinear", align_corners=True)
        hidden = sampled.squeeze(2).transpose(1, 2) + self.policy_exact(exact)
        return self.policy_score(F.relu(hidden)).squeeze(-1)

    def policy(
        self, spatial: Tensor, pass_logit: Tensor, exact: Tensor, actions: Tensor, legal: Tensor
    ) -> Tensor:
        """Full hierarchical log probabilities at supplied complete cells + pass.

        Omitted cells retain coarse probability mass. These training outputs need
        not sum to one. Native inference separately redistributes full placement
        mass over selected nonempty cells, preserving global coarse P(pass).
        Training uses hierarchical_policy_value_loss without a candidate-only softmax.
        """
        batch, count, features = exact.shape
        if (
            features != self.config.move_features
            or legal.shape != (batch, count)
            or spatial.shape[0] != batch
            or actions.shape not in ((count,), (batch, count))
            or actions.dtype != torch.int64
            or legal.dtype != torch.bool
        ):
            raise ValueError("incompatible coarse context, move features, actions or legality")
        if count and (actions.min() < 0 or actions.max() >= COORDINATE_ACTIONS):
            raise ValueError("coordinate IDs must exclude pass and be in [0,1000000)")
        coarse = self.coarse_logits(spatial, pass_logit).log_softmax(dim=1)
        if not count:
            return coarse[:, -1:]
        chunks = [
            self.score_chunk(
                spatial,
                exact[:, start : start + self.config.chunk_size],
                actions[..., start : start + self.config.chunk_size],
            )
            for start in range(0, count, self.config.chunk_size)
        ]
        scores = torch.cat(chunks, dim=1)
        cells = cell_ids(actions, spatial.shape[-1]).expand(batch, -1)
        local = local_log_probs(scores, actions, legal, spatial.shape[-1])
        return torch.cat((local + coarse.gather(1, cells), coarse[:, -1:]), dim=1)

    def forward(
        self, coarse: Tensor, exact: Tensor, actions: Tensor, legal: Tensor
    ) -> tuple[Tensor, Tensor]:
        spatial, pass_logit, value = self.encode(coarse)
        return self.policy(spatial, pass_logit, exact, actions, legal), value
