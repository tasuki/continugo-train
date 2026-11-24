"""Plot ratings against d: cumulative generated positions in each training lineage.

Includes validation and truncated games, not repeated training exposures (s).
Unknown legacy counters are omitted rather than replaced with game counts.
"""

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch

from . import archive
from .paths import DEFAULT_MODEL_STORE
from .plot_metrics import pyplot, save_figure
from .progress import checkpoint_progress
from .ratings import REFERENCE_ID, transaction


@dataclass(frozen=True)
class Point:
    network: str
    positions: int
    rating: float
    deviation: float
    reference: bool
    campaign: str = "Unknown campaign"
    round: int | None = None
    color: int | None = None
    champion: bool | None = None
    estimated: bool = False


def data_positions(store, network_id):
    """Read d from stored progress, never estimate positions from game counts."""
    metadata = Path(store) / "ratings/inference" / f"{network_id}.pt"
    if metadata.is_file():
        source = torch.load(metadata, map_location="cpu", weights_only=True)
        if source.get("checkpoint_sha256") != network_id:
            raise ValueError("inference source identity mismatch")
        count = source.get("progress", {}).get("data_positions")
        if isinstance(count, int) and count >= 0:
            return count
    for path in Path(store).glob("models/*/r*/metadata.json"):
        source = json.loads(path.read_text())
        if source.get("checkpoint_sha256") == network_id:
            return source["progress"]["data_positions"]
    for kind in ("retained", "accepted", "training", "experimental"):
        bundle = archive.find_bundle(Path(store) / kind, kind, network_id)
        if bundle is None:
            continue
        checkpoint = bundle / "checkpoint.pt"
        if archive.digest(checkpoint) != network_id:
            raise ValueError(f"archived checkpoint identity mismatch: {checkpoint}")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        return checkpoint_progress(saved)["data_positions"]
    raise ValueError(f"no archived checkpoint for network {network_id} in {store}")


def collect_artifact_points(model_store):
    from .lineage_ratings import overlay
    from .saved_models import saved_sources

    joint = overlay(Path(model_store).parent)

    sources = {
        source["id"]: source for source in saved_sources(model_store, include_pruned=True).values()
    }
    champion_file = Path(model_store) / "champions.json"
    lineage = (
        {item["model"] for item in json.loads(champion_file.read_text())["history"]}
        if champion_file.exists()
        else {"random"}
    )
    points = []
    with (Path(model_store) / "networks.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = ("" if row["network"] == "random" else Path(model_store).name, row["network"])
            row = joint.get(key, row)
            reference = row["network"] == "random"
            source = {} if reference else sources[row["network"]]
            positions = 0 if reference else source["progress"]["data_positions"]
            if positions is None:
                print(f"WARNING: omitting {row['network']}: unknown lineage d", file=sys.stderr)
                continue
            if type(positions) is not int or positions < 0:
                raise ValueError("invalid cumulative generated-position count")
            rating, deviation = float(row["rating"]), float(row["deviation"])
            if deviation < 0 or not all(math.isfinite(value) for value in [rating, deviation]):
                raise ValueError("invalid rating summary")
            points.append(
                Point(
                    row["network"],
                    positions,
                    rating,
                    deviation,
                    reference,
                    source.get("family", "random"),
                    source.get("round"),
                    champion=row["network"] in lineage,
                    estimated=not reference and int(row["games"]) == 0,
                )
            )
    return points


def collect_points(model_store):
    model_store = Path(model_store)
    path = model_store / "networks.csv"
    if not path.is_file():
        path = model_store / "ratings/networks.csv"  # Legacy match-ledger stores only.
    with path.open(newline="") as stream:
        if "network_id" not in (csv.DictReader(stream).fieldnames or []):
            return collect_artifact_points(model_store)
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
    lineage = {}
    champion_file = model_store / "champions.json"
    if champion_file.exists():
        for cohort in json.loads(champion_file.read_text())["cohorts"].values():
            promoted = {item["checkpoint"] for item in cohort["history"]}
            lineage.update(
                {
                    network: network in promoted
                    for network in cohort.get("candidates", cohort["evaluated"])
                }
            )
    estimates = None
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
            estimated = row["status"] == "unrated" and row["network_id"] in lineage
            if row["status"] not in {"rated", "fixed"} and not estimated:
                continue
            reference = row["network_id"] == REFERENCE_ID
            if estimated and estimates is None:
                # Show an explicitly uncertain prior, never invent game outcomes.
                with transaction(model_store) as (tables, _):
                    estimates = {
                        network: dict(tables.state(network)) for network in tables.networks
                    }
            try:
                rating = (
                    estimates[row["network_id"]]["rating"] if estimated else float(row["rating"])
                )
                deviation = (
                    estimates[row["network_id"]]["deviation"]
                    if estimated
                    else float(row["deviation"])
                )
                if deviation < 0 or not all(math.isfinite(v) for v in (rating, deviation)):
                    raise ValueError("invalid rating")
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{number}: invalid rating") from error
            source = sources.get(row["network_id"], {})
            if source and source.get("checkpoint_sha256") != row["network_id"]:
                raise ValueError("inference source identity mismatch")
            positions = 0 if reference else source.get("progress", {}).get("data_positions")
            if positions is None and row.get("data_positions", "") != "":
                # Registration preserves lineage counters even after run-owned
                # checkpoints disappear; plotting must not require those files.
                positions = int(row["data_positions"])
            if positions is None:
                positions = data_positions(model_store, row["network_id"])
            if positions is None:
                print(f"WARNING: omitting {row['network']}: unknown lineage d", file=sys.stderr)
                continue
            if type(positions) is not int or positions < 0:
                raise ValueError("invalid cumulative generated-position count")
            points.append(
                Point(
                    row["network"] or row["network_id"][:12],
                    positions,
                    rating,
                    deviation,
                    reference,
                    source.get("campaign", "Unknown campaign"),
                    source.get("round"),
                    colors.get(source.get("campaign")),
                    lineage.get(row["network_id"]),
                    estimated,
                )
            )
    if not any(not point.reference for point in points):
        raise ValueError("no rated networks found")
    return points


def collect_lineage_champions(artifacts_root):
    """Read portable champion metadata only; no checkpoints or training runs."""
    from .lineage_ratings import overlay

    joint = overlay(artifacts_root)
    points = []
    for registry_path in sorted(Path(artifacts_root).glob("*/champions.json")):
        store = registry_path.parent
        history = json.loads(registry_path.read_text())["history"]
        if not any(entry["model"] != "random" for entry in history):
            continue  # A new lineage has no rated neural champion yet.
        with (store / "networks.csv").open(newline="") as stream:
            ratings = {row["network"]: row for row in csv.DictReader(stream)}
        for order, entry in enumerate(history):
            name = entry["model"]
            reference = name == "random"
            if reference:
                positions = 0
            else:
                family, round_text = name.rsplit("-r", 1)
                metadata = json.loads(
                    (
                        store / "models" / family / f"r{int(round_text):04d}" / "metadata.json"
                    ).read_text()
                )
                positions = metadata["progress"]["data_positions"]
            row = joint.get(("" if reference else store.name, name), ratings[name])
            rating, deviation = float(row["rating"]), float(row["deviation"])
            if (
                type(positions) is not int
                or positions < 0
                or deviation < 0
                or not all(math.isfinite(value) for value in (rating, deviation))
            ):
                raise ValueError(f"invalid champion rating/progress: {store.name}/{name}")
            points.append(
                Point(
                    name, positions, rating, deviation, reference, store.name, order, champion=True
                )
            )
    if not any(not point.reference for point in points):
        raise ValueError("no lineage champions found")
    return points


def campaign_label(campaign, points):
    """Include the family prefixes used in network names and rated-game files."""
    families = {match[0] for point in points if (match := re.match(r"cg\d+(?=-|$)", point.network))}
    prefix = "/".join(sorted(families, key=lambda family: int(family[2:])))
    return f"{prefix} · {campaign}" if prefix and prefix != campaign else campaign


def plot(points, output, title="Network ratings", labels=False, *, champions_only=False):
    from matplotlib.ticker import FuncFormatter

    plt = pyplot(output)
    figure, axis = plt.subplots(figsize=(13, 8))
    networks = [p for p in points if not p.reference]
    campaigns = defaultdict(list)
    for point in networks:
        campaigns[point.campaign].append(point)
    colors = plt.get_cmap("tab20")
    for index, (campaign, series) in enumerate(sorted(campaigns.items())):
        series.sort(key=lambda p: (p.round if p.round is not None else p.positions, p.positions))
        # Use the ten distinct saturated hues before their lighter variants.
        color_index = series[0].color if series[0].color is not None else index
        color_index = 2 * (color_index % 10) + (color_index // 10) % 2
        axis.errorbar(
            [p.positions for p in series],
            [p.rating for p in series],
            fmt="-",
            alpha=0.85,
            color=colors(color_index),
            label=campaign if champions_only else campaign_label(campaign, series),
        )
        champions = [point for point in series if point.champion is True]
        unknown = [point for point in series if point.champion is None]
        muted = [point for point in series if point.champion is False]
        hue = colors(color_index)
        gray = tuple(0.7 * 0.6 + 0.3 * component for component in hue[:3])
        for group, color, alpha, marker, size in [
            (champions, hue, 1.0, "o", 32),
            (unknown, hue, 1.0, "o", 32),
            (muted, gray, 0.4, "o", 32),
        ]:
            if group:
                axis.errorbar(
                    [p.positions for p in group],
                    [p.rating for p in group],
                    yerr=[p.deviation for p in group],
                    fmt="none",
                    capsize=3,
                    color=color,
                    alpha=alpha,
                )
                axis.scatter(
                    [p.positions for p in group],
                    [p.rating for p in group],
                    marker=marker,
                    s=size,
                    color=color,
                    alpha=alpha,
                    zorder=4,
                )
    if champions_only:
        axis.legend(title="Lineage")
    if labels:
        for point in networks:
            axis.annotate(
                point.network,
                (point.positions, point.rating),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
            )
    references = {p.rating for p in points if p.reference}
    for rating in references:
        axis.scatter(0, rating, marker="*", s=150, color="black", zorder=5)
    axis.set_ylabel("Rating")
    # Leave room around zero so the random-reference star remains fully visible.
    maximum = max(1, max((p.positions for p in networks), default=1))
    axis.set_xlim(-0.03 * maximum, 1.1 * maximum)
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axis.grid(True, which="major", alpha=0.25)
    figure.suptitle(title)
    figure.supxlabel("d · Cumulative generated positions in training lineage", y=0.10)
    figure.text(
        0.5,
        0.02,
        "Error bars: ± rating deviation (not a 95% confidence interval). "
        "Validation and truncated games included; repeated training exposures excluded.\n"
        + (
            "Lines connect successive champions within each lineage.\n"
            if champions_only
            else "Lines connect evaluated rounds; solid dots are champion lineage, muted dots are non-champions.\n"
        )
        + "Different model schemas use separate compatible evaluation runtimes; compare across protocols cautiously.",
        ha="center",
        fontsize=8,
    )
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.9)
    save_figure(plt, figure, output)


def update(model_store):
    """Best-effort refresh after a committed arena batch; never abort training."""
    try:
        points = collect_points(model_store)
        plot(points, Path(model_store) / "ratings.png")
        artifacts_root = Path(model_store).parent
        if (Path(model_store) / "champions.json").is_file():
            from .champion_wins import update as update_wins

            update_wins(artifacts_root)
            plot(
                collect_lineage_champions(artifacts_root),
                artifacts_root / "ratings.png",
                "Champion ratings by lineage",
                champions_only=True,
            )
    except Exception as error:  # noqa: BLE001 — diagnostics must never abort training
        print(f"WARNING: ratings plot not updated: {error}", file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model-store", type=Path, help="plot all evaluated rounds of one store")
    source.add_argument(
        "--artifacts-root",
        type=Path,
        default=DEFAULT_MODEL_STORE.parent,
        help="plot champions across lineages (default %(default)s)",
    )
    parser.add_argument(
        "--output", type=Path, help="PNG/SVG/PDF; default ARTIFACTS_ROOT/ratings.png"
    )
    parser.add_argument("--title", default="Network ratings")
    parser.add_argument("--labels", action="store_true", help="label points with network names")
    args = parser.parse_args(argv)
    try:
        if args.model_store is not None:
            points = collect_points(args.model_store)
            output = args.output or args.model_store / "ratings.png"
        else:
            points = collect_lineage_champions(args.artifacts_root)
            output = args.output or args.artifacts_root / "ratings.png"
        plot(points, output, args.title, args.labels, champions_only=args.model_store is None)
        from .champion_wins import update as update_wins

        root = args.model_store.parent if args.model_store is not None else args.artifacts_root
        update_wins(root)
    except (OSError, ValueError, ImportError) as error:
        print(f"plot error: {error}", file=sys.stderr)
        return 1
    print(f"Plotted {sum(not p.reference for p in points)} rated networks: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
