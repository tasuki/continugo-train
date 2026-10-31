"""Small coarse ResNet and a shared, exact-feature-conditioned coordinate scorer.

Inputs are side-to-move-relative Rust features, not images of inferred legality:
  coarse:       [B, input_planes, H, W], one resolution bucket per batch
  move_features:[B, N, move_features], exact features for each candidate
  actions:      [N] int64 coordinate IDs, x + 1000*y (pass is separate)
  legal:        [B, N] bool, authoritative legality, NOT the bootstrap eye filter

Full policy evaluation supplies IDs 0..999999 in order. Subsets use the very same
scorer; their softmax is conditional unless every legal placement is included.
Output columns follow the supplied IDs, with pass appended at column N. Values
are expected outcomes in [-1,1] from the current player's perspective.

Defaults match the versioned real Rust feature schema in FEATURES.md. Alternate
feature widths remain configurable for experiments. No game rules live here.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .features import COARSE_PLANES, MOVE_FEATURES

COORD_RANGE = 1000
COORDINATE_ACTIONS = COORD_RANGE**2


@dataclass(frozen=True)
class NetworkConfig:
    input_planes: int = len(COARSE_PLANES)
    move_features: int = len(MOVE_FEATURES)
    channels: int = 32
    blocks: int = 4
    policy_hidden: int = 16
    value_hidden: int = 32
    chunk_size: int = 65536

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


def resolution_for_diameter(diameter: int, samples_per_diameter: int = 4) -> int:
    """Use the diameter supplied by Rust, without duplicating board-size rules.

    Coarse grid endpoints correspond to coordinates 0 and 999, inclusively.
    Four intervals per diameter give 38/54/78 samples for diameters 110/76/52.
    """
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
        # Project BEFORE sampling: no channels-wide million-coordinate tensor.
        self.policy_spatial = nn.Conv2d(config.channels, config.policy_hidden, 1)
        self.policy_global = nn.Linear(config.channels, config.policy_hidden, bias=False)
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
        """Return projected coarse policy features, pass logit, and value."""
        trunk = self.trunk(coarse)
        pooled = trunk.mean(dim=(-2, -1))
        spatial = self.policy_spatial(trunk) + self.policy_global(pooled)[:, :, None, None]
        return spatial, self.pass_head(pooled), self.value_head(pooled).squeeze(-1)

    def score_chunk(self, spatial: Tensor, exact: Tensor, actions: Tensor) -> Tensor:
        """Shared coordinate kernel, also usable independently for sparse proposals.

        Bilinear interpolation only smooths the coarse context; the exact rule
        features remain independent at every neighboring integer coordinate.
        """
        x = actions.remainder(COORD_RANGE).to(spatial.dtype)
        y = actions.div(COORD_RANGE, rounding_mode="floor").to(spatial.dtype)
        grid = torch.stack((x, y), dim=-1) * (2.0 / (COORD_RANGE - 1)) - 1.0
        grid = grid[None, None].expand(spatial.shape[0], -1, -1, -1)
        sampled = F.grid_sample(spatial, grid, mode="bilinear", align_corners=True)
        hidden = sampled.squeeze(2).transpose(1, 2) + self.policy_exact(exact)
        return self.policy_score(F.relu(hidden)).squeeze(-1)

    def policy(
        self, spatial: Tensor, pass_logit: Tensor, exact: Tensor, actions: Tensor, legal: Tensor
    ) -> Tensor:
        """Chunk hidden activations, then append the always-legal pass logit.

        Chunking bounds inference intermediates, not the whole training graph:
        autograd retains each chunk until backward. Tune batch size accordingly.
        """
        batch, count, features = exact.shape
        if (
            features != self.config.move_features
            or legal.shape != (batch, count)
            or spatial.shape[:2] != (batch, self.config.policy_hidden)
            or pass_logit.shape != (batch, 1)
        ):
            raise ValueError("coarse context, move features and legality have incompatible shapes")
        if actions.shape != (count,) or actions.dtype != torch.int64:
            raise ValueError("actions must be a shared 1D int64 coordinate-ID tensor")
        if legal.dtype != torch.bool:
            raise ValueError("legal must be boolean")
        if count and (actions.min() < 0 or actions.max() >= COORDINATE_ACTIONS):
            raise ValueError("coordinate IDs must exclude pass and be in [0,1000000)")
        chunks = []
        for start in range(0, count, self.config.chunk_size):
            end = start + self.config.chunk_size
            logits = self.score_chunk(spatial, exact[:, start:end], actions[start:end])
            chunks.append(logits.masked_fill(~legal[:, start:end], -torch.inf))
        return torch.cat([*chunks, pass_logit], dim=1)

    def forward(
        self, coarse: Tensor, exact: Tensor, actions: Tensor, legal: Tensor
    ) -> tuple[Tensor, Tensor]:
        spatial, pass_logit, value = self.encode(coarse)
        return self.policy(spatial, pass_logit, exact, actions, legal), value
