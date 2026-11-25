"""Rebuildable ratings from the date-grouped SGFs, without a second game ledger."""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from . import joint_ratings, ratings, viewer_jobs
from .evaluation import EvaluationConfig
from .saved_models import saved_sources

COLUMNS = ["network", "rating", "deviation", "games", "wins", "losses", "draws", "truncated"]
GAME_NAME = re.compile(
    r"^(?P<black>[A-Za-z0-9_-]+)-vs-(?P<white>[A-Za-z0-9_-]+)-seed(?P<seed>\d+)\.sgf$"
)


def refresh(store, config, cache=None, *, visualize=True):
    """The checkpoint cache is disposable: delete it to replay the SGFs exactly.

    All games are fitted jointly on every refresh. File paths/stat signatures
    cache validated SGF parsing, never sequential rating states. The cache is
    disposable and is not a second authoritative game ledger.
    """
    store = Path(store).resolve()
    sources = saved_sources(store, include_pruned=True)
    by_name = {source["id"]: source for source in sources.values()}
    names = {
        key: {f"neural-{source['network_name']}", f"neural-{key}"}
        for key, source in sources.items()
    }
    names[ratings.REFERENCE_ID] = {ratings.REFERENCE_ID}
    identities = {key: source["id"] for key, source in sources.items()}
    identities[ratings.REFERENCE_ID] = ratings.REFERENCE_ID
    checkpoints = {
        key: Path(source["checkpoint"]["path"])
        for key, source in sources.items()
        if not source.get("pruned")
    }
    settings = ratings.protocol(config)
    champion_file = store / "champions.json"
    registry = json.loads(champion_file.read_text()) if champion_file.exists() else {}
    evaluator_path = store / "ratings/evaluator.pt"
    evaluator = {}
    if evaluator_path.is_file():
        import torch

        evaluator = torch.load(evaluator_path, map_location="cpu", weights_only=True)
    protocols = [settings]
    for change in evaluator.get("runtime_changes", []):
        for key in ("previous_protocol", "protocol"):
            protocol = change[key]
            if any(
                protocol.get(k) != value
                for k, value in settings.items()
                if k not in ("arena_sha256", "rating_sha256")
            ):
                raise ValueError("runtime history changes rating search/match settings")
            if protocol not in protocols:
                protocols.append(protocol)

    def inspect(path):
        info = ratings.inspect_game(path, binary=config.feature_binary)
        root = info.get("root_comment")
        if not isinstance(root, dict):
            raise ValueError("rating SGF has no frozen match metadata")  # noqa: TRY004 — invalid SGF data
        arena = root.get("evaluation")
        protocol = next((p for p in protocols if p["arena_sha256"] == arena), None)
        if protocol is None:
            raise ValueError(f"rating SGF has an unregistered evaluation runtime: {path}")
        return ratings._inspect(path, protocol, names, config.feature_binary, info=info)

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
        cache.get("method") == "joint-bt-v2"
        and cache.get("protocol") == settings
        and cache.get("protocols", [cache.get("protocol")]) == protocols
        and previous == groups[: len(previous)]
        and len(previous) <= len(groups)
    )
    records = list(cache["records"]) if reusable else []
    if not reusable:
        previous = []
    # Validate already-cached files too when their missing microscope needs repair.
    infos = {}
    for group in groups[len(previous) :]:
        for relative, _size, _mtime in group["files"]:
            path = store / relative
            record = inspect(path)
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
            records.append({"black": black, "white": white, "result": record["result"]})
    states = joint_ratings.fit(records)
    summary = {}
    for name, state in states.items():
        public_name = "random" if name == ratings.REFERENCE_ID else name
        summary[public_name] = {key: state[key] for key in COLUMNS if key != "network"}
    directory = store
    directory.mkdir(parents=True, exist_ok=True)
    # Updating old ratings must not reshuffle their rows. New networks follow
    # first evidence order (paired seed order), not lexical family-name order.
    order = ["random"]
    csv_path = directory / "networks.csv"
    if csv_path.exists():
        with csv_path.open(newline="") as stream:
            order.extend(row["network"] for row in csv.DictReader(stream))
    order.extend(
        "random" if record[color] == ratings.REFERENCE_ID else record[color]
        for record in records
        for color in ("black", "white")
    )
    rows = [{"network": name, **summary[name]} for name in dict.fromkeys(order) if name in summary]
    ratings._atomic_csv(directory / "networks.csv", COLUMNS, rows)
    from . import lineage_ratings
    from .plot_ratings import update

    lineage_ratings.refresh(store.parent, config.feature_binary)
    update(store)  # Publish the chart before potentially slow neural reports.
    if visualize:
        skipped = set(registry.get("skipped", {}))

        def abandoned(name):
            if name in skipped:
                return True
            if name == "random" or registry.get("evaluated", {}).get(name, {}).get("complete"):
                return False
            family, round_text = name.rsplit("-r", 1)
            cadence = evaluator.get("family_evaluation_every", {}).get(
                family, registry.get("policy", {}).get("evaluation_every", 1)
            )
            return int(round_text) % cadence != 0

        jobs = []
        for group in groups:
            for relative, _size, _mtime in group["files"]:
                path = store / relative
                output = path.with_suffix(".nn.html")
                game = GAME_NAME.fullmatch(path.name)
                if abandoned(game["black"]) or abandoned(game["white"]):
                    continue  # Preserve partial SGFs, but do no more work on off-cycle rounds.
                if output.is_file() or any(
                    name != "random" and by_name[name].get("pruned")
                    for name in (game["black"], game["white"])
                ):
                    continue
                info = infos.get(relative) or inspect(path)["info"]
                jobs.append((path, output, info, checkpoints, config.feature_binary))
        viewer_jobs.generate(jobs)
    return {
        "method": "joint-bt-v2",
        "protocol": settings,
        "protocols": protocols,
        "groups": groups,
        "records": records,
        "states": states,
    }, summary


def main(argv=None):
    import torch

    from .rate_saved import champion_lock
    from .train import atomic_checkpoint

    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--model-store", type=Path)
    selection.add_argument("--artifacts-root", type=Path,
                           help="rebuild the joint cross-lineage ratings")
    args = parser.parse_args(argv)
    if args.artifacts_root is not None:
        from . import lineage_ratings, plot_ratings
        from .compare_lineages import ComparisonRunner, comparison_lock

        root = args.artifacts_root.resolve()
        manifests = sorted((root / "ratings/comparisons").glob("*.pt"))
        if not manifests:
            parser.error("prepare a lineage comparison first")
        state = torch.load(manifests[-1], map_location="cpu", weights_only=True)
        with comparison_lock(root) as lock_fd:
            runner = ComparisonRunner(root, state, lock_fd, manifests[-1])
            runner.verify_runtime()
            summary = lineage_ratings.refresh(root, runner.config.feature_binary)
            if summary is not None:
                plot_ratings.update(root / state["lineages"][0])
        print(json.dumps(summary, indent=2))
        return
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
