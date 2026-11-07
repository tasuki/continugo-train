"""Read real Rust features. Python only decodes tensors; it implements no rules.

SGF remains the source game record. CGFEAT02 is an ephemeral single-position
training/inference tensor packet, carried via stdout or an optional cache file.
See docs/current/features.md for feature names, normalization and binary layout.
"""

import json
import math
import os
import struct
import subprocess
import threading
import time
from collections import Counter
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
TARGET_VERSION = "final-area-v1"
HEADER = struct.Struct("<8s11If2If")


def ownership_weights(resolution: int, *, device="cpu") -> Tensor:
    """Area fractions about Rust's rounded raster sample locations."""
    i = torch.arange(resolution, device=device, dtype=torch.int64)
    samples = ((i * 999 + (resolution - 1) // 2) // (resolution - 1)).float()
    bounds = torch.cat(
        (samples.new_zeros(1), (samples[:-1] + samples[1:]) / 2, samples.new_full((1,), 1000))
    )
    widths = bounds.diff() / 1000
    return widths[:, None] * widths[None, :]


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
    score: float | None
    ownership: Tensor | None

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


def decode_sample(data: bytes, *, timings: dict | None = None) -> PositionFeatures:
    tick = time.perf_counter() if timings is not None else 0
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
        score,
    ) = HEADER.unpack_from(data)
    if (
        magic != b"CGFEAT02"
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
        HEADER.size
        + 4 * (planes + 1) * resolution**2
        + 4 * count * fields
        + 5 * count
        + 8 * targets
    )
    if len(data) != expected:
        raise ValueError("feature packet length does not match header")
    offset = HEADER.size

    def array(dtype, shape):
        nonlocal offset
        amount = math.prod(shape)
        value = np.frombuffer(data, dtype=dtype, count=amount, offset=offset).reshape(shape)
        offset += amount * np.dtype(dtype).itemsize
        return value

    coarse = array("<f4", (planes, resolution, resolution)).copy()
    exact = array("<f4", (count, fields)).copy()
    raw_legal = array("u1", (count,))
    actions = array("<u4", (count,)).astype(np.int64)
    target_actions = array("<u4", (targets,)).astype(np.int64)
    visits = array("<u4", (targets,)).astype(np.int64)
    ownership = array("<f4", (resolution, resolution)).copy()
    if (
        not np.isfinite(score)
        or abs(score) > 1
        or not np.isfinite(ownership).all()
        or (np.abs(ownership) > 1).any()
    ):
        raise ValueError("invalid auxiliary targets")
    if not labeled and (score != 0 or ownership.any()):
        raise ValueError("unlabeled packet contains auxiliary targets")
    if timings is not None:
        now = time.perf_counter()
        timings["decode_arrays_seconds"] = now - tick
        tick = now
    if (
        not np.isfinite(coarse).all()
        or ((coarse[6:8] != 0) & (coarse[6:8] != 1)).any()
        or not np.isfinite(exact).all()
        or (raw_legal > 1).any()
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
    if timings is not None:
        now = time.perf_counter()
        timings["decode_validation_seconds"] = now - tick
        tick = now
    sample = PositionFeatures(
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
        score if labeled else None,
        torch.from_numpy(ownership) if labeled else None,
    )
    if timings is not None:
        timings["decode_tensors_seconds"] = time.perf_counter() - tick
    return sample


class FeatureProfile:
    """Thread-safe diagnostics; worker wall-time sums are not exposed loader wait."""

    def __init__(self):
        self.lock = threading.Lock()
        self.requests = 0
        self.totals = Counter()
        self.child_peaks = {}

    def add(self, pid, measurements):
        with self.lock:
            self.requests += 1
            for key, value in measurements.items():
                if key == "server_peak_rss_mib":
                    if value is not None:
                        self.child_peaks[pid] = max(self.child_peaks.get(pid, 0), value)
                else:
                    self.totals[key] += value

    def summary(self):
        with self.lock:
            return {
                "requests": self.requests,
                "totals": dict(self.totals),
                "child_peak_rss_mib": dict(self.child_peaks),
                # Not a simultaneous peak: children may peak at different times.
                "sum_child_peak_rss_mib": sum(self.child_peaks.values())
                if self.child_peaks
                else None,
            }


def feature_binary(binary: str | Path | None = None) -> Path:
    default = Path(__file__).resolve().parents[2] / "target/release/features"
    path = Path(binary or os.environ.get("CONTINUGO_FEATURE_BIN", default)).resolve()
    if not path.is_file():
        raise FileNotFoundError("feature binary missing; run `mise run build-features`")
    return path


def inspect_game(sgf: str | Path, density: int = 4, binary: str | Path | None = None) -> dict:
    """Ask Rust for validated metadata/eligible plies, never parse SGF in Python."""
    result = subprocess.run(
        [
            str(feature_binary(binary)),
            "--inspect",
            "--sgf",
            str(sgf),
            "--samples-per-diameter",
            str(density),
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ValueError(f"{sgf}: {result.stderr.decode(errors='replace').strip()}")
    return json.loads(result.stdout)


class FeatureClient:
    """One Rust process, one cached validated game, arbitrary pre-move requests.

    Each response is status u8 + length u64 + a CGFEAT02 packet or UTF-8 error.
    The feature schema is unchanged; JSON lines are IPC commands, not game records.
    """

    def __init__(self, binary: str | Path | None = None, *, profile: bool = False):
        self.profile = profile
        self.last_profile = None
        self.last_view = None
        self.process = subprocess.Popen(
            [str(feature_binary(binary)), "--serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process.stdout.close()
        self.process.stderr.close()

    def _read(self, length: int) -> bytes:
        chunks = []
        remaining = length
        while remaining:
            chunk = self.process.stdout.read(remaining)
            if not chunk:
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
                error = self.process.stderr.read().decode(errors="replace")
                raise RuntimeError(f"Rust feature server stopped: {error.strip()}")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def load(
        self,
        sgf: str | Path,
        ply: int,
        *,
        density: int = 4,
        file_sha256: str | None = None,
        cells: list[int] | None = None,
        view: bool = False,
    ) -> PositionFeatures:
        self.last_profile = None
        self.last_view = None
        started = time.perf_counter() if self.profile else 0
        request = {
            "sgf": str(Path(sgf).resolve()),
            "ply": ply,
            "samples_per_diameter": density,
            "file_sha256": file_sha256,
        }
        if cells is not None:
            request["cells"] = cells
        if view:
            request["view"] = True
        if self.profile:
            request["profile"] = True
        encoded = json.dumps(request).encode() + b"\n"
        tick = time.perf_counter() if self.profile else 0
        encode_seconds = tick - started
        try:
            self.process.stdin.write(encoded)
            self.process.stdin.flush()
        except (BrokenPipeError, ValueError) as error:
            raise RuntimeError(
                "feature server unavailable; rebuild with `mise run build-features`"
            ) from error
        received = time.perf_counter() if self.profile else 0
        write_seconds = received - tick
        status, length = struct.unpack("<BQ", self._read(9))
        if length > 1_000_000_000:
            raise RuntimeError("invalid feature-server response length")
        data = self._read(length)
        if status == 1:
            raise ValueError(data.decode(errors="replace"))
        if status != 0:
            raise RuntimeError("invalid feature-server status")
        if self.profile or view:
            packet_received = time.perf_counter()
            trailer_length = struct.unpack("<Q", self._read(8))[0]
            if trailer_length > 1_000_000:
                raise RuntimeError("invalid feature-server timing length")
            server = json.loads(self._read(trailer_length))
            self.last_view = server.pop("view", None)
        if self.profile:
            decode_started = time.perf_counter()
            measurements = {f"server_{key}": value for key, value in server.items()}
            sample = decode_sample(data, timings=measurements)
            measurements.update(
                request_encode_seconds=encode_seconds,
                request_write_seconds=write_seconds,
                response_read_seconds=packet_received - received,
                telemetry_read_seconds=decode_started - packet_received,
                decode_seconds=time.perf_counter() - decode_started,
                packet_bytes=length,
            )
            # Residual, not a directly clocked bandwidth measurement: includes
            # framing, scheduling and server dispatch. Write and read overlap.
            work = sum(
                server[key]
                for key in (
                    "replay_and_selection_seconds",
                    "coarse_seconds",
                    "exact_seconds",
                    "serialization_seconds",
                )
            )
            measurements["ipc_residual_seconds"] = write_seconds + packet_received - received - work
            measurements["roundtrip_seconds"] = time.perf_counter() - started
            self.last_profile = measurements
        else:
            sample = decode_sample(data)
        if sample.ply != ply or sample.samples_per_diameter != density or sample.full_domain:
            raise RuntimeError("feature-server response does not match the request")
        return sample


def load_position(
    sgf: str | Path,
    ply: int,
    *,
    actions: list[int] | None = None,
    full_domain: bool = False,
    samples_per_diameter: int = 4,
    binary: str | Path | None = None,
) -> PositionFeatures:
    """Replay/validate SGF with Rust, returning one pre-move feature sample.

    This simple subprocess bridge is for initial offline wiring. It replays the
    SGF per request, not a persistent high-throughput training data service.
    """
    binary = feature_binary(binary)
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
    if full_domain:
        if actions is not None:
            raise ValueError("full_domain conflicts with an explicit subset")
        command.append("--full-domain")
    if actions is not None:
        if not actions or any(not isinstance(a, int) or not 0 <= a < PASS for a in actions):
            raise ValueError("coordinate subset must be nonempty and exclude pass")
        command += ["--actions", ",".join(map(str, actions))]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode:
        raise ValueError(result.stderr.decode(errors="replace").strip())
    return decode_sample(result.stdout)
