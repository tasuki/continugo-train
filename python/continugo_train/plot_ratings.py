"""Plot global ratings against cumulative unique training games.

Counts ancestral checkpoint games assigned to training, excluding validation.
Repeated epochs/replay do not count games again. The random reference is at 0
on the same linear training-game axis as the networks.
"""

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
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
    campaign: str = "Unknown campaign"
    round: int | None = None
    color: int | None = None


def training_games(store, network_id):
    """Read the ancestral split ledger, not just the current replay window."""
    metadata = Path(store) / "ratings/inference" / f"{network_id}.pt"
    if metadata.is_file():
        source = torch.load(metadata, map_location="cpu", weights_only=True)
        if source.get("checkpoint_sha256") != network_id:
            raise ValueError("inference source identity mismatch")
        count = source.get("training_games")
        if isinstance(count, int) and count >= 0:
            return count
    for path in Path(store).glob("*-r*/metadata.json"):
        import json

        source = json.loads(path.read_text())
        if source.get("checkpoint_sha256") == network_id:
            return source["training_games"]
    for kind in ("retained", "accepted", "training", "experimental"):
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
    from .saved_models import saved_sources

    sources = {
        file.stem: torch.load(file, map_location="cpu", weights_only=True)
        for file in (model_store / "ratings/inference").glob("*.pt")
    }
    sources.update(saved_sources(model_store))
    campaigns = sorted(
        {source["campaign"] for source in sources.values() if source.get("campaign")}
    )
    colors = {name: index for index, name in enumerate(campaigns)}
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
            source = sources.get(row["network_id"], {})
            if source and source.get("checkpoint_sha256") != row["network_id"]:
                raise ValueError("inference source identity mismatch")
            games = 0 if reference else source.get("training_games")
            if games is None:
                games = training_games(model_store, row["network_id"])
            if not isinstance(games, int) or games < 0:
                raise ValueError("invalid cumulative training-game count")
            points.append(
                Point(
                    row["network"] or row["network_id"][:12],
                    games,
                    rating,
                    deviation,
                    reference,
                    source.get("campaign", "Unknown campaign"),
                    source.get("round"),
                    colors.get(source.get("campaign")),
                )
            )
    if not any(not point.reference for point in points):
        raise ValueError("no rated networks found")
    return points


def campaign_label(campaign, points):
    """Include the family prefixes used in network names and rated-game files."""
    families = {match[0] for point in points if (match := re.match(r"cg\d+(?=-|$)", point.network))}
    prefix = "/".join(sorted(families, key=lambda family: int(family[2:])))
    return f"{prefix} · {campaign}" if prefix else campaign


def plot(points, output, title="Network ratings", labels=False):
    from matplotlib.ticker import FuncFormatter

    plt = pyplot(output)
    figure, axis = plt.subplots(figsize=(13, 8))
    networks = [p for p in points if not p.reference]
    campaigns = defaultdict(list)
    for point in networks:
        campaigns[point.campaign].append(point)
    colors = plt.get_cmap("tab20")
    for index, (campaign, series) in enumerate(sorted(campaigns.items())):
        series.sort(key=lambda p: (p.round if p.round is not None else p.games, p.games))
        # Use the ten distinct saturated hues before their lighter variants.
        color_index = series[0].color if series[0].color is not None else index
        color_index = 2 * (color_index % 10) + (color_index // 10) % 2
        axis.errorbar(
            [p.games for p in series],
            [p.rating for p in series],
            yerr=[p.deviation for p in series],
            fmt="o-",
            capsize=3,
            alpha=0.85,
            color=colors(color_index),
            label=campaign_label(campaign, series),
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
        axis.scatter(0, rating, marker="*", s=150, color="black", zorder=5)
    axis.set_ylabel("Rating")
    # Leave room around zero so the random-reference star remains fully visible.
    maximum = max(1, max(p.games for p in networks))
    axis.set_xlim(-0.03 * maximum, 1.1 * maximum)
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axis.grid(True, which="major", alpha=0.25)
    from matplotlib.lines import Line2D

    handles, _ = axis.get_legend_handles_labels()
    axis.legend(
        handles=[
            *handles,
            Line2D(
                [],
                [],
                marker="*",
                linestyle="none",
                color="black",
                markersize=12,
                label="Random bot (0 games)",
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(0, -0.13),
        fontsize=8,
    )
    figure.suptitle(title)
    figure.supxlabel("Cumulative unique training games", y=0.27)
    figure.text(
        0.5,
        0.02,
        "Error bars: ± rating deviation (not a 95% confidence interval). "
        "Validation excluded; replay counted once. Lines connect rounds within each campaign.\n"
        "Different model schemas use separate compatible evaluation runtimes; compare across protocols cautiously.",
        ha="center",
        fontsize=8,
    )
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.34, top=0.9)
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
