"""Plot global ratings against cumulative unique training games.

Counts ancestral checkpoint games assigned to training, excluding validation.
Repeated epochs/replay do not count games again. The random reference is at 0,
separate from the logarithmic training-game axis which starts at 100.
"""

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from . import archive
from .plot_metrics import pyplot, save_figure
from .ratings import REFERENCE_ID
from .train import checkpoint_splits

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Point:
    network: str
    games: int
    rating: float
    deviation: float
    reference: bool


def training_games(store, network_id):
    """Read the ancestral split ledger, not just the current replay window."""
    for kind in ("training", "experimental", "accepted"):
        bundle = archive.find_bundle(Path(store) / kind, kind, network_id)
        if bundle is None:
            continue
        checkpoint = bundle / "checkpoint.pt"
        if archive.digest(checkpoint) != network_id:
            raise ValueError(f"archived checkpoint identity mismatch: {checkpoint}")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        return sum(split == "train" for split in checkpoint_splits(saved).values())
    raise ValueError(f"no archived checkpoint for network {network_id} in {store}")


def collect_points(model_store):
    model_store = Path(model_store)
    path = model_store / "ratings" / "networks.csv"
    points = []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"network_id", "network", "status", "rating", "deviation"}
        if not required.issubset(reader.fieldnames or []) or "pool" in reader.fieldnames:
            raise ValueError(
                f"expected global rating columns; refresh/migrate ratings first: {path}"
            )
        seen = set()
        for number, row in enumerate(reader, 2):
            if row["network_id"] in seen:
                raise ValueError(f"{path}:{number}: duplicate network")
            seen.add(row["network_id"])
            if row["status"] not in {"rated", "fixed"}:
                continue
            reference = row["network_id"] == REFERENCE_ID
            try:
                rating = float(row["rating"])
                deviation = float(row["deviation"])
                if deviation < 0 or not all(math.isfinite(v) for v in (rating, deviation)):
                    raise ValueError("invalid rating")
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{number}: invalid rating") from error
            games = 0 if reference else training_games(model_store, row["network_id"])
            points.append(
                Point(row["network"] or row["network_id"][:12], games, rating, deviation, reference)
            )
    if not any(not point.reference for point in points):
        raise ValueError("no rated networks found")
    return points


def plot(points, output, title="Network ratings", labels=False):
    from matplotlib.ticker import FuncFormatter, LogLocator

    if any(p.games < 100 for p in points if not p.reference):
        raise ValueError(
            "a rated network has fewer than 100 training games; cannot use x minimum 100"
        )
    plt = pyplot(output)
    figure, (zero, axis) = plt.subplots(
        1,
        2,
        figsize=(11, 7),
        sharey=True,
        gridspec_kw={"width_ratios": [0.65, 10], "wspace": 0.04},
    )
    networks = [p for p in points if not p.reference]
    axis.errorbar(
        [p.games for p in networks],
        [p.rating for p in networks],
        yerr=[p.deviation for p in networks],
        fmt="o",
        linestyle="none",
        capsize=3,
        alpha=0.85,
        label="Networks",
    )
    if labels:
        for point in networks:
            axis.annotate(
                point.network,
                (point.games, point.rating),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
            )
    references = {p.rating for p in points if p.reference}
    for rating in references:
        zero.scatter(0, rating, marker="*", s=150, color="black", zorder=5)
    zero.set_xlim(-0.6, 0.6)  # Keep the entire star inside the zero panel.
    zero.set_xticks([0])
    zero.set_ylabel("Rating")
    axis.set_xscale("log")
    axis.set_xlim(100, max(120, max(p.games for p in networks) * 1.3))
    axis.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    zero.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.tick_params(axis="y", which="both", left=False)
    for panel in (zero, axis):
        panel.grid(True, which="major", alpha=0.25)
    # Diagonal marks make clear that 0 is separate from the genuine log axis.
    for panel, x in ((zero, 1), (axis, 0)):
        panel.plot(
            [x - 0.008, x + 0.008],
            [-0.012, 0.012],
            transform=panel.transAxes,
            color="gray",
            clip_on=False,
        )
    from matplotlib.lines import Line2D

    axis.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", color="C0", label="Networks"),
            Line2D(
                [],
                [],
                marker="*",
                linestyle="none",
                color="black",
                markersize=12,
                label="Random bot (0 games)",
            ),
        ]
    )
    figure.suptitle(title)
    figure.supxlabel("Cumulative unique training games (log scale from 100)", y=0.065)
    figure.text(
        0.5,
        0.02,
        "Error bars: ± rating deviation (not a 95% confidence interval). "
        "Validation excluded; replay counted once.",
        ha="center",
        fontsize=8,
    )
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.16, top=0.9)
    save_figure(plt, figure, output)


def update(model_store):
    """Best-effort refresh after a committed arena batch; never abort training."""
    try:
        points = collect_points(model_store)
        plot(points, Path(model_store) / "ratings.png")
    except Exception as error:  # noqa: BLE001 — diagnostics must never abort training
        print(f"WARNING: ratings plot not updated: {error}", file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-store", type=Path, default=ROOT / "models")
    parser.add_argument("--output", type=Path, help="PNG/SVG/PDF; default MODEL_STORE/ratings.png")
    parser.add_argument("--title", default="Network ratings")
    parser.add_argument("--labels", action="store_true", help="label points with network names")
    args = parser.parse_args(argv)
    try:
        points = collect_points(args.model_store)
        output = args.output or args.model_store / "ratings.png"
        plot(points, output, args.title, args.labels)
    except (OSError, ValueError, ImportError) as error:
        print(f"plot error: {error}", file=sys.stderr)
        return 1
    print(f"Plotted {sum(not p.reference for p in points)} rated networks: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
