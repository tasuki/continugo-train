"""Human-readable, UTC-dated artifact directories. Identities remain content hashes."""

import re
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def utc_now():
    return datetime.now(UTC)


def timestamp(moment=None):
    # Seconds help distinguish short jobs; counters handle same-second collisions.
    return (moment or utc_now()).astimezone(UTC).strftime("%Y%m%d-%H%M%S")


def new_run_directory(label, base=None):
    """Reserve dated output: training campaigns in runs/, standalone work in scratch/."""
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", label):
        raise ValueError("invalid run label")
    base = (
        Path(base)
        if base is not None
        else ROOT / ("runs" if label in {"train", "loop"} else "scratch")
    )
    base.mkdir(parents=True, exist_ok=True)
    stem = f"{timestamp()}-{label}"
    for number in range(1, 10000):
        path = base / (stem if number == 1 else f"{stem}-{number:04d}")
        try:
            path.mkdir()
        except FileExistsError:
            continue
        return path.resolve()
    raise ValueError("too many run-directory collisions in one second")


def bundle_directory(parent, checksum, moment, name=None):
    """Call under the store lock: sortable publication time + sequence + short ID."""
    stamp = timestamp(moment)
    pattern = re.compile(rf"^{re.escape(stamp)}-(\d{{4}})-")
    highest = max(
        (int(match[1]) for p in parent.iterdir() if (match := pattern.match(p.name))), default=0
    )
    if highest >= 9999:
        raise ValueError("too many model publications in one second")
    label = f"{name}-" if name else ""
    return parent / f"{stamp}-{highest + 1:04d}-{label}{checksum[:12]}"
