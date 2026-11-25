"""Joint ratings across lineages and explicitly scheduled shared-root matches.

Lineage SGFs are trusted historical evidence, revalidated by a compatible frozen
rules inspector. Shared-root SGFs must additionally match a frozen comparison job.
The joint fit pools komi settings: it estimates one strength per checkpoint.
"""

import csv
import fcntl
import json
import re
from contextlib import contextmanager
from pathlib import Path

import torch

from . import archive, joint_ratings, ratings
from .rating_store import COLUMNS, GAME_NAME
from .sampling import weighted_sizes
from .saved_models import saved_sources
from .train import atomic_checkpoint


@contextmanager
def joint_lock(root):
    directory = Path(root) / "ratings"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".joint.lock").open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def sources(root):
    """Require globally unambiguous model names and checkpoint identities."""
    result = {}
    for registry in sorted(Path(root).glob("*/champions.json")):
        for checksum, source in saved_sources(registry.parent, include_pruned=True).items():
            name = source["id"]
            if name in result or any(s["checkpoint_sha256"] == checksum for s in result.values()):
                raise ValueError(f"duplicate cross-lineage model identity: {name}")
            result[name] = {**source, "lineage": registry.parent.name}
    # Older lineages deliberately retained only champions. Recover missing
    # historical player identities from their authoritative SGF metadata, not
    # from invented checkpoints or ratings. Require their existing registry row.
    from .champion_wins import root_properties

    for registry in sorted(Path(root).glob("*/champions.json")):
        store = registry.parent
        registered = set()
        if (store / "networks.csv").exists():
            with (store / "networks.csv").open(newline="") as stream:
                registered = {row["network"] for row in csv.DictReader(stream)}
        for path in sorted((store / "sgfs").glob("*/*.sgf")):
            match = GAME_NAME.fullmatch(path.name)
            if match is None:
                continue
            missing = [
                color
                for color in ("black", "white")
                if match[color] != "random" and match[color] not in result
            ]
            if not missing:
                continue
            properties = root_properties(path)
            metadata = json.loads(properties["C"][0])
            for color in missing:
                name = match[color]
                if name not in registered:
                    raise ValueError(f"unregistered historical player: {name}")
                checksum = metadata["model" if color == "black" else "opponent"]
                if any(s["checkpoint_sha256"] == checksum for s in result.values()):
                    raise ValueError(f"duplicate historical checkpoint identity: {name}")
                player = properties["PB" if color == "black" else "PW"][0]
                if not player.startswith("neural-"):
                    raise ValueError(f"invalid historical player: {path}")
                result[name] = {
                    "id": name,
                    "checkpoint_sha256": checksum,
                    "network_name": player.removeprefix("neural-"),
                    "lineage": store.name,
                    "pruned": True,
                }
    return result


def network_order(catalog):
    """Reference first, then lineage and checkpoint in natural numeric order."""
    def natural(value):
        return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value))

    return [
        ratings.REFERENCE_ID,
        *sorted(catalog, key=lambda name: (natural(catalog[name]["lineage"]), natural(name), name)),
    ]


def shared_jobs(root):
    """Only reserved, frozen job/color/seed combinations authorize shared SGFs."""
    jobs = {}
    for path in sorted((Path(root) / "ratings/comparisons").glob("*.pt")):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("version") != "lineage-comparison-v1":
            raise ValueError(f"unsupported comparison manifest: {path}")
        for job in state["jobs"]:
            for index in range(job["games"]):
                black, white = job["candidate_name"], job["opponent_name"]
                if index % 2:
                    black, white = white, black
                seed = job["seed"] + index // 2
                name = f"{black}-vs-{white}-seed{seed}.sgf"
                if name in jobs:
                    raise ValueError(f"duplicate comparison reservation: {name}")
                jobs[name] = (state, job)
    return jobs


def refresh(root, feature_binary):
    root = Path(root).resolve()
    with joint_lock(root):
        return _refresh(root, feature_binary)


def _refresh(root, feature_binary):
    catalog = sources(root)
    names = {
        s["checkpoint_sha256"]: {f"neural-{s['network_name']}", f"neural-{name}"}
        for name, s in catalog.items()
    }
    names[ratings.REFERENCE_ID] = {ratings.REFERENCE_ID}
    identities = {s["checkpoint_sha256"]: name for name, s in catalog.items()}
    identities[ratings.REFERENCE_ID] = ratings.REFERENCE_ID
    reservations = shared_jobs(root)
    historical = {}
    manifests = {id(state): state for state, _job in reservations.values()}
    for state in sorted(manifests.values(), key=lambda s: s.get("prepared_at", "")):
        historical.update(state.get("historical_inspectors", {}))
    # Shared games always use their own frozen rules inspector. Historical
    # inspectors may predate komi/score-unit changes and must not rescore archives.
    inspectors = {
        "default": {"path": str(feature_binary), "sha256": archive.digest(feature_binary)},
        **historical,
    }
    for name, (state, _job) in reservations.items():
        binary = state["config"].get("feature_binary", str(feature_binary))
        expected = state.get("runtime_fingerprints", {}).get("features")
        inspectors[f"shared:{name}"] = {
            "path": binary,
            "sha256": expected or archive.digest(binary),
        }
    checked_paths = {}
    for record in inspectors.values():
        path = record["path"]
        if path not in checked_paths:
            checked_paths[path] = archive.digest(path)
        if checked_paths[path] != record["sha256"]:
            raise ValueError(f"historical/comparison inspector changed: {path}")
    cache_path = root / "ratings/joint-cache.pt"
    previous = torch.load(cache_path, weights_only=True) if cache_path.exists() else {}
    context = {
        "inspectors": inspectors,
        "names": names,
        "protocols": {
            name: state["rating_protocol"] for name, (state, _job) in reservations.items()
        },
    }
    cached = previous.get("files", {}) if previous.get("context") == context else {}
    files, records, seen, evidence = {}, [], set(), set()
    directories = [
        (registry.parent / "sgfs", registry.parent.name)
        for registry in sorted(root.glob("*/champions.json"))
    ]
    directories.append((root / "sgfs", None))
    for directory, lineage in directories:
        for path in sorted(directory.glob("*/*.sgf")):
            match = GAME_NAME.fullmatch(path.name)
            if match is None and path.name.startswith("game-"):
                continue  # Unadopted, possibly still being written by an arena.
            if match is None or path.is_symlink():
                raise ValueError(f"unexpected rated SGF: {path}")
            if lineage is None and path.name not in reservations:
                raise ValueError(f"shared SGF has no frozen comparison reservation: {path}")
            black, white = match["black"], match["white"]
            for name in (black, white):
                if name != "random" and (
                    name not in catalog
                    or lineage is not None
                    and catalog[name]["lineage"] != lineage
                ):
                    raise ValueError(f"unknown/misplaced player in {path}")
            if black == white:
                raise ValueError(f"self-matching SGF: {path}")
            game_key = (black, white, int(match["seed"]))
            if game_key in evidence:
                raise ValueError(f"duplicate seed/color evidence: {path}")
            evidence.add(game_key)
            key = str(path.relative_to(root))
            signature = [path.stat().st_size, path.stat().st_mtime_ns]
            old = cached.get(key)
            if old and old["signature"] == signature:
                record, digest = old["record"], old["sha256"]
            else:
                inspector = (
                    inspectors.get(lineage, inspectors["default"])
                    if lineage
                    else inspectors[f"shared:{path.name}"]
                )
                info = ratings.inspect_game(path, binary=inspector["path"])
                metadata = info.get("root_comment")
                if not isinstance(metadata, dict):
                    raise ValueError(f"missing frozen match metadata: {path}")
                if lineage is None:
                    state, job = reservations[path.name]
                    settings = state["rating_protocol"]
                    cycle = weighted_sizes(
                        state["config"]["sizes"], state["config"].get("size_weights", "uniform")
                    )
                    pair_index = int(match["seed"]) - job["seed"]
                    if info["board_size"] != cycle[pair_index % len(cycle)]:
                        raise ValueError(f"unexpected comparison board size: {path}")
                else:
                    # Historical lineages predate the shared evaluator (some no
                    # longer have evaluator.pt). Validate their recorded runtime,
                    # search/visit counts and legal result using their preserved
                    # scorer, never the new comparison runtime's score units.
                    settings = {
                        "search": metadata.get("search"),
                        "arena_sha256": metadata.get("evaluation"),
                        "board_mix": {str(info["board_size"]): 1},
                        # Some legacy trials used smaller caps; their unfinished
                        # legal records remain unscored, never forced results.
                        "neural_max_moves": max(1024, info["moves"])
                        if info["completed"]
                        else info["moves"],
                        "reference_max_moves": max(2048, info["moves"])
                        if info["completed"]
                        else info["moves"],
                    }
                    if not isinstance(settings["search"], dict) or not settings["arena_sha256"]:
                        raise ValueError(f"missing historical protocol: {path}")
                checked = ratings._inspect(path, settings, names, inspector["path"], info=info)
                expected = [
                    ratings.REFERENCE_ID if name == "random" else name for name in (black, white)
                ]
                actual = [identities[checked[color]] for color in ("black", "white")]
                if actual != expected or metadata.get("seed") != int(match["seed"]):
                    raise ValueError(f"SGF filename disagrees with contents: {path}")
                record = dict(zip(("black", "white"), actual, strict=True))
                record["result"] = checked["result"]
                digest = checked["sha256"]
            if digest in seen:
                raise ValueError(f"duplicate game evidence: {path}")
            seen.add(digest)
            files[key] = {"signature": signature, "sha256": digest, "record": record}
            records.append(record)
    states = joint_ratings.fit(records)
    order = network_order(catalog)
    rows = [
        {
            "lineage": "" if name == ratings.REFERENCE_ID else catalog[name]["lineage"],
            "network": "random" if name == ratings.REFERENCE_ID else name,
            **{key: states[name][key] for key in COLUMNS if key != "network"},
        }
        for name in order
        if name in states
    ]
    ratings._atomic_csv(root / "networks.csv", ["lineage", *COLUMNS], rows)
    # Lineage tables are views of this same fit, never independent estimates.
    # Preserve their existing display order and include only local participants.
    reference = next((row for row in rows if row["network"] == "random"), None)
    for registry in sorted(root.glob("*/champions.json")):
        path = registry.parent / "networks.csv"
        local = {row["network"]: row for row in rows if row["lineage"] == registry.parent.name}
        if reference is not None:
            local["random"] = reference
        order = ["random"]
        if path.exists():
            with path.open(newline="") as stream:
                order.extend(row["network"] for row in csv.DictReader(stream))
        order.extend(local)
        ratings._atomic_csv(path, COLUMNS, [
            {key: local[name][key] for key in COLUMNS}
            for name in dict.fromkeys(order) if name in local
        ])
    atomic_checkpoint(cache_path, {"context": context, "files": files})
    return states


def overlay(root):
    path = Path(root) / "networks.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {(row["lineage"], row["network"]): row for row in rows}
