"""Rebuildable ratings from the date-grouped SGFs, without a second game ledger."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from . import nn_view, ratings
from .evaluation import EvaluationConfig
from .paths import ROOT
from .saved_models import saved_sources

COLUMNS = ["network", "rating", "deviation", "games", "wins", "losses", "draws", "truncated"]
GAME_NAME = re.compile(
    r"^(?P<black>[A-Za-z0-9_-]+)-vs-(?P<white>[A-Za-z0-9_-]+)-seed(?P<seed>\d+)\.sgf$"
)


def refresh(store, config, cache=None, *, visualize=True):
    """The checkpoint cache is disposable: delete it to replay the SGFs exactly.

    Color-swapped games with a shared seed form a simultaneous Glicko batch.
    File paths/stat signatures only accelerate the unchanged prefix; no results
    are duplicated into a CSV/JSON game ledger.
    """
    store = Path(store).resolve()
    sources = saved_sources(store)
    by_name = {source["id"]: source for source in sources.values()}
    names = {
        key: {f"neural-{source['network_name']}", f"neural-{key}"}
        for key, source in sources.items()
    }
    names[ratings.REFERENCE_ID] = {ratings.REFERENCE_ID}
    identities = {key: source["id"] for key, source in sources.items()}
    identities[ratings.REFERENCE_ID] = ratings.REFERENCE_ID
    checkpoints = {key: Path(source["checkpoint"]["path"]) for key, source in sources.items()}
    settings = ratings.protocol(config)
    grouped = defaultdict(list)
    for path in sorted((store / "sgfs").glob("*/*.sgf")):
        match = GAME_NAME.fullmatch(path.name)
        if match is None or path.is_symlink():
            raise ValueError(f"unexpected rated SGF path: {path}")
        black, white = match["black"], match["white"]
        if black == white or any(
            name != "random" and name not in by_name for name in [black, white]
        ):
            raise ValueError(f"unknown/self-matching model in {path.name}")
        key = (int(match["seed"]), *sorted((black, white)))
        grouped[key].append(path)
    groups = []
    for key, paths in sorted(grouped.items()):
        if len(paths) > 2:
            raise ValueError(f"duplicate seed/color evidence: {key}")
        groups.append(
            {
                "key": list(key),
                "files": [
                    [str(path.relative_to(store)), path.stat().st_size, path.stat().st_mtime_ns]
                    for path in paths
                ],
            }
        )
    cache = cache or {}
    previous = cache.get("groups", [])
    reusable = (
        cache.get("protocol") == settings
        and previous == groups[: len(previous)]
        and len(previous) <= len(groups)
    )
    tables = ratings.Tables()
    tables.ensure_reference()
    for name, source in by_name.items():
        parent = f"{source['family']}-r{source['round'] - 1:04d}"
        tables.networks[name] = {"parent": parent if parent in by_name else ""}
    if reusable:
        tables.states = cache["states"]
    else:
        previous = []
    # Validate already-cached files too when their missing microscope needs repair.
    infos = {}
    for group in groups[len(previous) :]:
        records = []
        for relative, _size, _mtime in group["files"]:
            path = store / relative
            record = ratings._inspect(path, settings, names, config.feature_binary)
            match = GAME_NAME.fullmatch(path.name)
            black, white = (identities[record[color]] for color in ["black", "white"])
            expected = [
                ratings.REFERENCE_ID if match[color] == "random" else match[color]
                for color in ["black", "white"]
            ]
            if [black, white] != expected or record["info"]["root_comment"]["seed"] != int(
                match["seed"]
            ):
                raise ValueError(f"SGF filename disagrees with its contents: {path}")
            infos[relative] = record["info"]
            records.append({**record, "black": black, "white": white})
        ratings._rate(tables, records, config.rating_binary)
    summary = {}
    for name, state in tables.states.items():
        public_name = "random" if name == ratings.REFERENCE_ID else name
        summary[public_name] = {key: state[key] for key in COLUMNS if key != "network"}
    directory = store / "ratings"
    directory.mkdir(parents=True, exist_ok=True)
    rows = [
        {"network": name, **value}
        for name, value in sorted(summary.items(), key=lambda pair: (pair[0] != "random", pair[0]))
    ]
    ratings._atomic_csv(directory / "networks.csv", COLUMNS, rows)
    from .plot_ratings import update

    update(store)  # Publish the chart before potentially slow neural reports.
    if visualize:
        for group in groups:
            for relative, _size, _mtime in group["files"]:
                path = store / relative
                output = path.with_suffix(".nn.html")
                if output.is_file():
                    continue
                info = (
                    infos.get(relative)
                    or ratings._inspect(path, settings, names, config.feature_binary)["info"]
                )
                nn_view.generate_rated(
                    path, output, info, checkpoints, binary=config.feature_binary
                )
    return {"protocol": settings, "groups": groups, "states": tables.states}, summary


def main(argv=None):
    import torch

    from .rate_saved import champion_lock
    from .train import atomic_checkpoint

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-store", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args(argv)
    store = args.model_store.resolve()
    state_path = store / "ratings/evaluator.pt"
    with champion_lock(store):
        if not state_path.is_file():
            parser.error("start rate-saved first to freeze the evaluation protocol")
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        state["rating_cache"], summary = refresh(
            store, EvaluationConfig.from_config(state["config"])
        )
        atomic_checkpoint(state_path, state)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
