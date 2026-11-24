"""Read-only original separate-head architecture for diagnostic replay.

No checkpoint migration, fabricated shared heads or learned uncertainty. Policy
operations are identical to the current cells-v1 model. Kept out of training.
"""

from dataclasses import dataclass

import torch
from torch import nn

from .model import PolicyValueNet, ResidualBlock


@dataclass(frozen=True)
class LegacyConfig:
    input_planes: int = 14
    move_features: int = 9
    channels: int = 64
    blocks: int = 8
    policy_hidden: int = 16
    value_hidden: int = 32
    score_hidden: int = 32
    ownership_hidden: int = 16
    chunk_size: int = 65536
    top_cells: int = 32
    exploration_cells: int = 8
    value_activation: str = "leaky_relu"

    @property
    def global_inputs(self):
        return 0


class LegacyPolicyValueNet(nn.Module):
    legacy_separate_heads = True
    coarse_logits = PolicyValueNet.coarse_logits
    score_chunk = PolicyValueNet.score_chunk

    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.value_activation not in {"relu", "leaky_relu"}:
            raise ValueError("unsupported legacy activation")
        if any(
            not isinstance(v, int) or isinstance(v, bool) or v <= 0
            for k, v in vars(config).items()
            if k != "value_activation"
        ):
            raise ValueError("legacy architecture dimensions must be positive integers")
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
            nn.LeakyReLU(0.1) if config.value_activation == "leaky_relu" else nn.ReLU(),
            nn.Linear(config.value_hidden, 1),
            nn.Tanh(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(config.channels, config.score_hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(config.score_hidden, 1),
            nn.Tanh(),
        )
        self.ownership_head = nn.Sequential(
            nn.Conv2d(config.channels, config.ownership_hidden, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(config.ownership_hidden, 1, 1),
            nn.Tanh(),
        )

    def _encode(self, coarse, komi=None):
        trunk = self.trunk(coarse)
        pooled = trunk.mean(dim=(-2, -1))
        spatial = self.policy_spatial(trunk) + self.policy_global(pooled)[:, :, None, None]
        return trunk, pooled, spatial, self.pass_head(pooled), self.value_head(pooled).squeeze(-1)

    def score_stdev(self, pooled):
        """Point estimate only, not a predicted confidence or learned deviation."""
        return torch.zeros(pooled.shape[0], device=pooled.device, dtype=pooled.dtype)
