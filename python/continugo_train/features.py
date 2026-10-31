"""Read real Rust features. Python only decodes tensors; it implements no rules.

SGF remains the source game record. CGFEAT01 is an ephemeral single-position
training/inference tensor packet, carried via stdout or an optional cache file.
See FEATURES.md for feature names, normalization and binary layout.
"""

import os
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

SCHEMA_VERSION = 4
COARSE_PLANES = (
    "friendly_stones",
    "enemy_stones",
    "friendly_group_size",
    "enemy_group_size",
    "friendly_liberty_area_stones",
    "enemy_liberty_area_stones",
    "friendly_capturable_in_one",
    "enemy_capturable_in_one",
    "left_border",
    "right_border",
    "top_border",
    "bottom_border",
)
MOVE_FEATURES = (
    "legal",
    "connected_groups",
    "connected_stones",
    "captured_groups",
    "captured_stones",
    "resulting_group_size",
    "left_border",
    "right_border",
    "top_border",
    "bottom_border",
)
PASS = 1_000_000
HEADER = struct.Struct("<8s11If2I")


@dataclass
class PositionFeatures:
    coarse: Tensor
    exact: Tensor
    actions: Tensor
    legal: Tensor
    target_actions: Tensor
    visits: Tensor
    outcome: float | None
    board_size: int
    diameter: int
    resolution: int
    ply: int
    player: int  # 0 Black, 1 White; arrays are already side-to-move relative.
    samples_per_diameter: int
    full_domain: bool

    def network_inputs(self, device="cpu") -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return (
            self.coarse[None].to(device),
            self.exact[None].to(device),
            self.actions.to(device),
            self.legal[None].to(device),
        )

    def policy_targets(self) -> tuple[Tensor, Tensor]:
        """Map canonical SGF action IDs to output columns, with pass last.

        No fake policy is invented for random SGFs. Missing positive targets in
        a diagnostic subset are rejected, rather than silently renormalized.
        """
        if self.visits.numel() == 0:
            raise ValueError("SGF position has no MCTS visit-count policy target")
        if self.full_domain:
            columns = self.target_actions.clone()
        else:
            sorted_actions, order = self.actions.sort()
            columns = torch.full_like(self.target_actions, self.actions.numel())
            coordinate = self.target_actions != PASS
            targets = self.target_actions[coordinate]
            locations = torch.searchsorted(sorted_actions, targets)
            found = locations < sorted_actions.numel()
            if sorted_actions.numel():
                found &= sorted_actions[locations.clamp(max=sorted_actions.numel() - 1)] == targets
            if ((~found) & (self.visits[coordinate] > 0)).any():
                raise ValueError("coordinate subset omits positive MCTS targets")
            if sorted_actions.numel():
                columns[coordinate] = torch.where(
                    found, order[locations.clamp(max=order.numel() - 1)], self.actions.numel()
                )
        placements = (columns < self.actions.numel()) & (self.visits > 0)
        if not self.legal[columns[placements]].all():
            raise ValueError("positive policy target on an illegal coordinate")
        return columns, self.visits.to(torch.float32)


def decode_sample(data: bytes) -> PositionFeatures:
    if len(data) < HEADER.size:
        raise ValueError("truncated feature header")
    (
        magic,
        version,
        planes,
        fields,
        size,
        diameter,
        resolution,
        ply,
        player,
        count,
        targets,
        labeled,
        outcome,
        density,
        full,
    ) = HEADER.unpack_from(data)
    if (
        magic != b"CGFEAT01"
        or version != SCHEMA_VERSION
        or planes != len(COARSE_PLANES)
        or fields != len(MOVE_FEATURES)
    ):
        raise ValueError("unsupported feature schema")
    if (
        not 1 <= size <= 500
        or not 2 <= diameter <= 1000
        or not 2 <= resolution <= 2048
        or player not in (0, 1)
        or labeled not in (0, 1)
        or full not in (0, 1)
        or not density
        or count > PASS
        or targets > PASS + 1
    ):
        raise ValueError("invalid feature dimensions or flags")
    expected = (
        HEADER.size + 4 * planes * resolution**2 + 4 * count * fields + 5 * count + 8 * targets
    )
    if len(data) != expected:
        raise ValueError("feature packet length does not match header")
    offset = HEADER.size

    def array(dtype, shape):
        nonlocal offset
        amount = int(np.prod(shape))
        value = np.frombuffer(data, dtype=dtype, count=amount, offset=offset).reshape(shape)
        offset += amount * np.dtype(dtype).itemsize
        return value

    coarse = array("<f4", (planes, resolution, resolution)).copy()
    exact = array("<f4", (count, fields)).copy()
    raw_legal = array("u1", (count,))
    actions = array("<u4", (count,)).astype(np.int64)
    target_actions = array("<u4", (targets,)).astype(np.int64)
    visits = array("<u4", (targets,)).astype(np.int64)
    if (
        not np.isfinite(coarse).all()
        or not np.isin(coarse[6:8], [0, 1]).all()
        or not np.isfinite(exact).all()
        or not np.isin(raw_legal, [0, 1]).all()
        or (actions >= PASS).any()
        or not np.array_equal(exact[:, 0], raw_legal)
    ):
        raise ValueError(
            "nonfinite features, invalid capture flags/actions or inconsistent legality"
        )
    if full:
        if count != PASS or not np.array_equal(actions, np.arange(PASS)):
            raise ValueError("full-domain coordinates must be in canonical action order")
    elif np.unique(actions).size != count:
        raise ValueError("duplicate coordinate action")
    if targets and (
        (target_actions > PASS).any()
        or np.unique(target_actions).size != targets
        or visits.sum() <= 0
    ):
        raise ValueError("invalid MCTS policy targets")
    if labeled and (not np.isfinite(outcome) or abs(outcome) > 1):
        raise ValueError("invalid outcome label")
    return PositionFeatures(
        torch.from_numpy(coarse),
        torch.from_numpy(exact),
        torch.from_numpy(actions),
        torch.from_numpy(raw_legal.astype(np.bool_)),
        torch.from_numpy(target_actions),
        torch.from_numpy(visits),
        outcome if labeled else None,
        size,
        diameter,
        resolution,
        ply,
        player,
        density,
        bool(full),
    )


def load_position(
    sgf: str | Path,
    ply: int,
    *,
    actions: list[int] | None = None,
    samples_per_diameter: int = 4,
    binary: str | Path | None = None,
) -> PositionFeatures:
    """Replay/validate SGF with Rust, returning one pre-move feature sample.

    This simple subprocess bridge is for initial offline wiring. It replays the
    SGF per request, not a persistent high-throughput training data service.
    """
    default = Path(__file__).resolve().parents[2] / "target/release/features"
    binary = Path(binary or os.environ.get("CONTINUGO_FEATURE_BIN", default))
    if not binary.is_file():
        raise FileNotFoundError("feature binary missing; run `mise run build-features`")
    if ply < 0 or samples_per_diameter <= 0:
        raise ValueError("ply must be nonnegative and sample density positive")
    command = [
        str(binary.resolve()),
        "--sgf",
        str(sgf),
        "--ply",
        str(ply),
        "--samples-per-diameter",
        str(samples_per_diameter),
    ]
    if actions is not None:
        if not actions or any(not isinstance(a, int) or not 0 <= a < PASS for a in actions):
            raise ValueError("coordinate subset must be nonempty and exclude pass")
        command += ["--actions", ",".join(map(str, actions))]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode:
        raise ValueError(result.stderr.decode(errors="replace").strip())
    return decode_sample(result.stdout)
