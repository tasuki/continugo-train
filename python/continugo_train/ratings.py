"""Static-Glicko ratings in two CSVs; games remain authoritative SGFs.

networks.csv holds scheduled/evaluated metadata and one global rating per network.
matches.csv is the commit ledger: one row per SGF, with a batch ID, protocol,
and both players' post-batch ratings. This preserves history without JSON/SQL.
Under one lock, metadata is saved BEFORE the atomic match-ledger replacement;
after a crash, current ratings are reconstructed from committed match rows,
never from a potentially ahead/behind networks.csv rating cache. No math replay
or time-based updates are needed. JSON is used only for hashing and Rust IPC.
"""

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import re
import subprocess
import tempfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import torch

from . import archive
from .features import inspect_game
from .paths import utc_now
from .progress import checkpoint_progress, network_name
from .sampling import weighted_sizes
from .train import atomic_checkpoint, checkpoint_splits

REFERENCE_ID = "random-no-eye-fill-v1"
ROOT = Path(__file__).resolve().parents[2]
META = [
    "network_id",
    "network",
    "registered_at",
    "parent",
    "samples_seen",
    "data_positions",
    "accepted",
    "accepted_at",
    "bundle",
]
COUNTS = ["games", "wins", "draws", "losses", "truncated"]
NETWORK_COLUMNS = META + ["status", "rating", "deviation", "rating ± RD"] + COUNTS
LEGACY_NETWORK_COLUMNS = [column for column in NETWORK_COLUMNS if column != "registered_at"]
POOLED_NETWORK_COLUMNS = META + ["pool", "status", "rating", "deviation", "rating ± RD"] + COUNTS
SEARCH_TYPES = {
    "simulations": int,
    "cpuct": float,
    "widening": float,
    "widening_exponent": float,
    "proposal_attempts": int,
    "independent_exploration": float,
    "score_weight": float,
}
PROTOCOL_COLUMNS = [
    "version",
    "algorithm",
    "reference",
    "board_mix",
    *SEARCH_TYPES,
    "neural_max_moves",
    "reference_max_moves",
    "selection",
    "arena_sha256",
    "rating_sha256",
]
SNAPSHOTS = [
    f"{color}_{key}_after_batch"
    for color in ["black", "white"]
    for key in ["rating", "deviation", "anchored"]
]
MATCH_COLUMNS = [
    "event",
    "recorded_at",
    "batch_number",
    "game_number",
    "sgf_path",
    "kind",
    "decision",
    "sgf_sha256",
    "black",
    "white",
    "result",
    "board_size",
    *PROTOCOL_COLUMNS,
    *SNAPSHOTS,
]
LEGACY_MATCH_COLUMNS = [
    column for column in MATCH_COLUMNS if column not in {"batch_number", "game_number", "sgf_path"}
]
POOLED_MATCH_COLUMNS = ["event", "pool", *MATCH_COLUMNS[1:]]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def identity(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def protocol(config, *, legacy=False):
    counts = Counter(weighted_sizes(config.sizes, config.size_weights))
    divisor = math.gcd(*counts.values())
    search = {**config.search(), "simulations": config.gate_simulations}
    if legacy:
        search.pop("score_weight", None)
    return {
        "version": 1 if legacy else 2,
        "algorithm": "static-glicko-v1",
        "reference": REFERENCE_ID,
        "board_mix": {str(size): count // divisor for size, count in sorted(counts.items())},
        "search": search,
        "neural_max_moves": config.gate_max_moves,
        "reference_max_moves": config.reference_max_moves,
        "selection": "maximum-visits-random-ties",
        "arena_sha256": archive.digest(config.arena_binary),
        "rating_sha256": archive.digest(config.rating_binary),
    }


def _protocol_columns(settings):
    return {
        **{k: settings[k] for k in PROTOCOL_COLUMNS if k not in SEARCH_TYPES},
        **({"score_weight": ""} if settings["version"] == 1 else {}),
        **settings["search"],
        "board_mix": ",".join(f"{k}:{v}" for k, v in settings["board_mix"].items()),
    }


def _protocol_from_row(row):
    settings = {k: row[k] for k in PROTOCOL_COLUMNS if k not in SEARCH_TYPES}
    for key in ["version", "neural_max_moves", "reference_max_moves"]:
        settings[key] = int(settings[key])
    settings["board_mix"] = dict(item.split(":") for item in settings["board_mix"].split(","))
    settings["board_mix"] = {k: int(v) for k, v in settings["board_mix"].items()}
    settings["search"] = {
        k: kind(row[k])
        for k, kind in SEARCH_TYPES.items()
        if k != "score_weight" or settings["version"] != 1
    }
    return settings


def _atomic_csv(path, columns, rows):
    descriptor, name = tempfile.mkstemp(prefix=".table-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        archive.sync_directory(path.parent)
    finally:
        Path(name).unlink(missing_ok=True)


def _read_csv(path, columns, *, legacy_columns=()):
    if path.is_symlink():
        raise ValueError(f"unexpected CSV symlink: {path}")
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        alternatives = (
            [legacy_columns]
            if not legacy_columns or isinstance(legacy_columns[0], str)
            else legacy_columns
        )
        if reader.fieldnames not in (columns, *alternatives):
            raise ValueError(f"incompatible rating CSV header: {path}")
        rows = list(reader)
        if any(set(row) != set(reader.fieldnames) or None in row.values() for row in rows):
            raise ValueError(f"incomplete rating CSV row: {path}")
        # None distinguishes fields absent in a legacy header from corrupt blank
        # values in a current-schema row (which must not trigger migration).
        return [{**dict.fromkeys(columns), **row} for row in rows]


def _valid_id(value):
    return value == REFERENCE_ID or isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)


def _registration_time(meta):
    value = meta["registered_at"]
    if not value:
        # Legacy registration times are unknown, not the time of migration.
        return datetime.min.replace(tzinfo=UTC)
    moment = datetime.fromisoformat(value)
    if moment.utcoffset() is None:
        raise ValueError("network registration time must include a timezone")
    return moment


@dataclass
class Tables:
    networks: dict = field(default_factory=dict)
    matches: list = field(default_factory=list)
    states: dict = field(default_factory=dict)
    seen: dict = field(default_factory=dict)

    def state(self, network):
        if network not in self.states:
            parent = self.states.get(self.networks[network]["parent"])
            self.states[network] = {
                "rating": parent["rating"]
                if network != REFERENCE_ID and parent and parent["anchored"] and parent["games"]
                else 100.0,
                "deviation": 0.0 if network == REFERENCE_ID else 350.0,
                "anchored": int(network == REFERENCE_ID),
                **dict.fromkeys(COUNTS, 0),
            }
        return self.states[network]

    def ensure_reference(self):
        # The fixed anchor exists even before any games or neural registrations.
        self.networks.setdefault(
            REFERENCE_ID,
            {
                **dict.fromkeys(META, ""),
                "network_id": REFERENCE_ID,
                "network": REFERENCE_ID,
                "accepted": "0",
            },
        )
        self.state(REFERENCE_ID)

    def match_rows(self):
        return sorted(
            self.matches, key=lambda row: (-int(row["batch_number"]), int(row["game_number"]))
        )

    def network_rows(self):
        self.ensure_reference()
        # Always lead with the anchor; stable ID order breaks neural timestamp ties.
        networks = sorted(self.networks.items())
        networks.sort(
            key=lambda item: (item[0] == REFERENCE_ID, _registration_time(item[1])),
            reverse=True,
        )
        for network, meta in networks:
            state = self.states.get(network)
            fixed = network == REFERENCE_ID
            if fixed and state is None:
                state = self.state(network)
            rated = state is not None and (fixed or state["anchored"] and state["games"])
            rating, deviation = (state[k] if rated else "" for k in ["rating", "deviation"])
            yield {
                **meta,
                "status": "fixed" if fixed else "rated" if rated else "unrated",
                "rating": rating,
                "deviation": deviation,
                "rating ± RD": f"{rating:.1f} ± {deviation:.1f}" if rated else "unrated",
                **{k: state[k] if state else 0 for k in COUNTS},
            }


def _counts(tables, black, white, result):
    if black == white:
        return
    for network, color in [(black, "B"), (white, "W")]:
        state = tables.state(network)
        if not result:
            state["truncated"] += 1
        else:
            state["games"] += 1
            state[
                "draws" if result == "0" else "wins" if result.startswith(color + "+") else "losses"
            ] += 1


def _set_rating(state, network, rating, deviation, anchored):
    if (
        not math.isfinite(rating)
        or not math.isfinite(deviation)
        or deviation < 0
        or deviation > state["deviation"] + 1e-10
        or anchored not in (0, 1)
        or state["anchored"]
        and not anchored
    ):
        raise ValueError("invalid/inflated rating uncertainty or anchor state")
    if network == REFERENCE_ID:
        if (rating, deviation, anchored) != (100.0, 0.0, 1):
            raise ValueError("fixed reference must remain 100 with zero RD")
    elif deviation == 0:
        raise ValueError("only the reference may have zero RD")
    state.update(rating=rating, deviation=min(deviation, state["deviation"]), anchored=anchored)


def _sgf_name(tables, row):
    moment = datetime.fromisoformat(row["recorded_at"])
    if moment.utcoffset() is None:
        raise ValueError("match recording time must include a timezone")
    stamp = moment.astimezone(UTC).strftime("%Y%m%d-%H%M%S")

    def player(network):
        name = re.sub(r"[^a-zA-Z0-9_-]+", "-", tables.networks[network]["network"]).strip("-")
        return REFERENCE_ID if network == REFERENCE_ID else f"{name[:36] or 'neural'}-{network[:8]}"

    return (
        f"{stamp}-{player(row['black'])}-vs-{player(row['white'])}"
        f"-size{row['board_size']}-game{int(row['game_number']):06}-{row['sgf_sha256']}.sgf"
    )


def _check_sgf_path(row):
    path = Path(row["sgf_path"])
    if (
        len(path.parts) != 2
        or path.parts[0] != "games"
        or not path.name.endswith(f"-{row['sgf_sha256']}.sgf")
        or str(path) != row["sgf_path"]
    ):
        raise ValueError("invalid rating SGF path in match ledger")
    return path


def evidence_path(store, row):
    """Resolve and verify the exact store-owned evidence indexed by a match row."""
    directory = Path(store).resolve() / "ratings"
    _check_sgf_path(row)
    if (directory / "games").is_symlink():
        raise ValueError("rating evidence directory cannot be a symlink")
    return archive.checked_file(directory, {"path": row["sgf_path"], "sha256": row["sgf_sha256"]})


def _retain_rows(store, directory, rows, sources):
    archive.retain_evidence(
        store,
        [(sources[row["sgf_sha256"]], row["sgf_sha256"]) for row in rows],
        directory / "games",
        filenames={row["sgf_sha256"]: _check_sgf_path(row).name for row in rows},
    )


def _load(directory):
    tables = Tables()
    rows = _read_csv(
        directory / "networks.csv",
        NETWORK_COLUMNS,
        legacy_columns=[
            LEGACY_NETWORK_COLUMNS,
            POOLED_NETWORK_COLUMNS,
            [c for c in POOLED_NETWORK_COLUMNS if c != "registered_at"],
        ],
    )
    for row in rows:
        meta = {k: row[k] or "" for k in META}
        network = meta["network_id"]
        if not _valid_id(network) or meta["accepted"] != str(int(bool(meta["accepted_at"]))):
            raise ValueError("invalid network metadata in networks.csv")
        _registration_time(meta)
        if network in tables.networks and tables.networks[network] != meta:
            raise ValueError("conflicting network metadata")
        tables.networks[network] = meta
    tables.ensure_reference()
    if (directory / "networks.csv").exists() and not (directory / "matches.csv").exists():
        raise ValueError(
            "matches.csv missing; restore the match ledger rather than resetting ratings"
        )
    tables.matches = _read_csv(
        directory / "matches.csv",
        MATCH_COLUMNS,
        legacy_columns=[
            LEGACY_MATCH_COLUMNS,
            POOLED_MATCH_COLUMNS,
            [
                c
                for c in POOLED_MATCH_COLUMNS
                if c not in {"batch_number", "game_number", "sgf_path"}
            ],
        ],
    )
    pooled = any("pool" in row for row in tables.matches)
    legacy = bool(tables.matches) and all(row["batch_number"] is None for row in tables.matches)
    batches, games = {}, Counter()
    for row in tables.matches:
        if legacy:
            if row["game_number"] is not None or row["sgf_path"] is not None:
                raise ValueError("incomplete match ordering metadata")
            key = row["event"]
            batches.setdefault(key, len(batches) + 1)
            games[key] += 1
            row.update(batch_number=str(batches[key]), game_number=str(games[key]))
        if not re.fullmatch(r"[1-9][0-9]*", row["batch_number"]) or not re.fullmatch(
            r"[1-9][0-9]*", row["game_number"]
        ):
            raise ValueError("invalid match batch/game number")
    # Sequence numbers, never CSV display order or wall-clock dates, drive recovery.
    chronological = sorted(
        tables.matches, key=lambda row: (int(row["batch_number"]), int(row["game_number"]))
    )
    batches, games = {}, Counter()
    events = {}
    for row in chronological:
        number = int(row["batch_number"])
        batch = tuple(row[key] for key in ["event", "recorded_at", "kind", "decision"])
        if number in batches and batches[number] != batch:
            raise ValueError("conflicting match batch metadata")
        if batch[0] in events and events[batch[0]] != number:
            raise ValueError("duplicate match batch identity")
        batches[number] = batch
        events[batch[0]] = number
        games[number] += 1
        if int(row["game_number"]) != games[number]:
            raise ValueError("duplicate or missing game number in match batch")
        black, white = (row[k] for k in ["black", "white"])
        settings = _protocol_from_row(row)
        key = row["sgf_sha256"]
        if (
            (pooled and identity(settings) != row["pool"])
            or not re.fullmatch(r"[0-9a-f]{64}", row["sgf_sha256"])
            or black not in tables.networks
            or white not in tables.networks
            or (not pooled and key in tables.seen)
            or row["result"]
            and not re.fullmatch(r"0|[BW]\+\d+\.\d", row["result"])
        ):
            raise ValueError("invalid or duplicate match ledger row")
        if legacy:
            row["sgf_path"] = str(Path("games") / _sgf_name(tables, row))
        _check_sgf_path(row)
        tables.seen[key] = row
        if not pooled:
            _counts(tables, black, white, row["result"])
            for color, network in [("black", black), ("white", white)]:
                _set_rating(
                    tables.state(network),
                    network,
                    float(row[f"{color}_rating_after_batch"]),
                    float(row[f"{color}_deviation_after_batch"]),
                    int(row[f"{color}_anchored_after_batch"]),
                )
    if legacy:
        # Publish new links before the ledger replacement; keep all old links so
        # interrupted migrations and older backups still have their evidence.
        for row in tables.matches:
            _retain_rows(
                directory.parent,
                directory,
                [row],
                {row["sgf_sha256"]: directory / "games" / f"{row['sgf_sha256']}.sgf"},
            )
    if pooled:
        _merge_legacy_ratings(directory, tables, chronological)
    tables.ensure_reference()
    return tables


def _merge_legacy_ratings(directory, tables, chronological):
    """One-time replay of old split ledgers into a single chronological scale."""
    # Preserve exact source CSVs before replacing either table. Retrying after a
    # crash is safe: committed matches still select the old or the new format.
    backup = (
        directory / "backups" / ("before-global-" + archive.digest(directory / "matches.csv")[:16])
    )
    backup.mkdir(parents=True, exist_ok=True)
    for name in ("networks.csv", "matches.csv"):
        target = backup / name
        if not target.exists():
            target.write_bytes((directory / name).read_bytes())
            archive.seal_file(target)
    archive.sync_directory(backup)
    tables.matches, tables.states, tables.seen = [], {}, {}
    groups = {}
    for old in chronological:
        evidence_path(directory.parent, old)
        previous = tables.seen.get(old["sgf_sha256"])
        if previous:
            if any(previous[k] != old[k] for k in ("black", "white", "result", "board_size")):
                raise ValueError("conflicting duplicate SGF in legacy ledger")
            continue
        row = {key: old[key] for key in MATCH_COLUMNS}
        tables.seen[row["sgf_sha256"]] = row
        groups.setdefault(row["batch_number"], []).append(row)
    for rows in groups.values():
        _rate(
            tables,
            [{**row, "result": row["result"] or None} for row in rows],
            ROOT / "target/release/rating",
        )
        for number, row in enumerate(rows, 1):
            row["game_number"] = str(number)
            for color in ("black", "white"):
                state = tables.state(row[color])
                row.update(
                    {
                        f"{color}_{key}_after_batch": str(state[key])
                        for key in ("rating", "deviation", "anchored")
                    }
                )
            tables.matches.append(row)
    # Historical archive publication registered every intermediate checkpoint.
    # Only evaluated networks (including truncations) and approved ones belong
    # in the registry. Future evaluations explicitly register their candidates.
    used = {row[color] for row in tables.matches for color in ("black", "white")}
    tables.networks = {
        key: meta for key, meta in tables.networks.items() if key in used or meta["accepted"] == "1"
    }


@contextmanager
def transaction(store):
    store = Path(store).resolve()
    marker = store / "STORE.pt"
    if not marker.is_file() or torch.load(marker, weights_only=True) != {
        "format": "continugo-recovery-store",
        "version": 1,
    }:
        raise ValueError("ratings need an initialized ContinuGo model store")
    directory = store / "ratings"
    directory.mkdir(exist_ok=True)
    with (directory / ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (directory / "ledger.sqlite3").exists() or (directory / "ratings.json").exists():
            raise ValueError("foreign/experimental rating state found; refusing to overwrite it")
        tables = _load(directory)
        # Establish an empty commit ledger before publishing any network cache.
        if not (directory / "matches.csv").exists():
            _atomic_csv(directory / "matches.csv", MATCH_COLUMNS, [])
        yield tables, directory
        _atomic_csv(directory / "networks.csv", NETWORK_COLUMNS, tables.network_rows())
        _atomic_csv(directory / "matches.csv", MATCH_COLUMNS, tables.match_rows())


def _register(tables, checkpoint):
    checksum = archive.digest(checkpoint)
    if checksum not in tables.networks:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        progress = checkpoint_progress(saved)
        tables.networks[checksum] = {
            "network_id": checksum,
            "network": network_name(saved),
            "registered_at": utc_now().isoformat(),
            "parent": saved.get("initialization", {}).get("source", {}).get("sha256") or "",
            "samples_seen": ""
            if progress["samples_seen"] is None
            else str(progress["samples_seen"]),
            "data_positions": ""
            if progress["data_positions"] is None
            else str(progress["data_positions"]),
            "accepted": "0",
            "accepted_at": "",
            "bundle": "",
        }
    return checksum


def _accept(tables, store, reference):
    metadata = archive.verify_bundle(reference)
    if metadata["kind"] != "accepted":
        raise ValueError("not an accepted recovery bundle")
    bundle = Path(reference["directory"]).resolve()
    checksum = _register(tables, bundle / "checkpoint.pt")
    if checksum != metadata["checkpoint_sha256"]:
        raise ValueError("accepted checkpoint identity mismatch")
    meta = tables.networks[checksum]
    if not meta["accepted_at"]:
        meta.update(
            accepted="1",
            accepted_at=metadata["created_at"],
            bundle=str(bundle.relative_to(Path(store).resolve())),
        )


def register_checkpoint(store, checkpoint, *, model=None):
    """Register evaluated metadata, optionally locating a run-owned frozen export."""
    with transaction(store) as (tables, directory):
        checksum = _register(tables, checkpoint)
        if model is not None:
            files = {"checkpoint": Path(checkpoint).resolve()}
            files.update(
                {name: Path(model).resolve() / f"{name}.onnx" for name in ["encoder", "policy"]}
            )
            records = {
                name: {"path": str(path), "sha256": archive.digest(path)}
                for name, path in files.items()
            }
            if records["checkpoint"]["sha256"] != checksum:
                raise ValueError("checkpoint changed during registration")
            origin = {}
            for parent in Path(checkpoint).resolve().parents:
                if (parent / "loop.pt").is_file():
                    origin["campaign"] = (
                        parent.parent.name if parent.name == "loop" else parent.name
                    )
                    match = re.search(r"iteration-(\d+)", str(Path(checkpoint).relative_to(parent)))
                    origin["round"] = int(match[1]) if match else 0
                    break
            sources = directory / "inference"
            sources.mkdir(exist_ok=True)
            atomic_checkpoint(
                sources / f"{checksum}.pt",
                {
                    "checkpoint_sha256": checksum,
                    **origin,
                    "network_name": tables.networks[checksum]["network"],
                    "training_games": sum(
                        split == "train"
                        for split in checkpoint_splits(
                            torch.load(checkpoint, map_location="cpu", weights_only=True)
                        ).values()
                    ),
                    **records,
                },
            )


def inference_source(store, checksum):
    """Return a verified locator, not a copy. Deleted runs remain in rating history."""
    path = Path(store) / "ratings/inference" / f"{checksum}.pt"
    if not path.is_file():
        return None
    source = torch.load(path, map_location="cpu", weights_only=True)
    if source.get("checkpoint_sha256") != checksum:
        raise ValueError("inference source identity mismatch")
    for name in ["checkpoint", "encoder", "policy"]:
        record = source[name]
        artifact = Path(record["path"])
        if not artifact.is_file():
            return None
        if artifact.is_symlink() or archive.digest(artifact) != record["sha256"]:
            raise ValueError(f"registered inference artifact missing or changed: {artifact}")
    if source["checkpoint"]["sha256"] != checksum:
        raise ValueError("inference checkpoint identity mismatch")
    return source


def register_accepted(store, reference):
    with transaction(store) as (tables, _):
        _accept(tables, store, reference)


def _inspect(path, settings, names, feature_binary):
    info = inspect_game(path, binary=feature_binary)
    root = info.get("root_comment")
    if not isinstance(root, dict) or not {"model", "opponent"}.issubset(root):
        raise ValueError("rating SGF has no frozen match metadata")
    black, white = root["model"], root["opponent"]
    if (
        not isinstance(black, str)
        or not isinstance(white, str)
        or black not in names
        or white not in names
    ):
        raise ValueError("rating SGF has an unknown player; register its checkpoint first")
    reference = REFERENCE_ID in (black, white)
    cap = settings["reference_max_moves" if reference else "neural_max_moves"]
    expected_visits = [
        None
        if (black if ply % 2 == 0 else white) == REFERENCE_ID
        else settings["search"]["simulations"]
        for ply in range(info["moves"])
    ]
    if (
        root.get("search") != settings["search"]
        or root.get("evaluation") != settings["arena_sha256"]
        or str(info["board_size"]) not in settings["board_mix"]
        or not 0 < info["moves"] <= cap
        or info["completed"] != info["labeled"]
        or not info["completed"]
        and info["moves"] != cap
        or info["policy_visit_totals"] != expected_visits
        or info.get("black_player") not in names[black]
        or info.get("white_player") not in names[white]
    ):
        raise ValueError(f"rating SGF does not match its evaluation protocol: {path}")
    return {
        "sha256": info["file_sha256"],
        "black": black,
        "white": white,
        "result": info["result"],
        "board_size": info["board_size"],
        "path": Path(path),
    }


def _rate(tables, records, rating_binary):
    participants = sorted({r[k] for r in records for k in ["black", "white"]})
    states = {network: tables.state(network) for network in participants}
    scored, anchored = [], set()
    for record in records:
        black, white, result = (record[k] for k in ["black", "white", "result"])
        _counts(tables, black, white, result)
        if black == white or result is None:
            continue
        score = 0.5 if result == "0" else float(result.startswith("B+"))
        scored.append({"black": black, "white": white, "score": score})
        if states[black]["anchored"] or states[white]["anchored"]:
            anchored.update([black, white])
    if not scored:
        return
    request = {
        "players": [
            {"id": n, "rating": s["rating"], "deviation": s["deviation"]} for n, s in states.items()
        ],
        "games": scored,
    }
    result = subprocess.run(
        [str(rating_binary)], input=canonical(request), capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise RuntimeError(f"static Glicko update failed: {result.stderr}")
    players = json.loads(result.stdout)["players"]
    if len(players) != len(states) or {p["id"] for p in players} != set(states):
        raise ValueError("rating engine returned unexpected players")
    for player in players:
        network = player["id"]
        _set_rating(
            states[network],
            network,
            player["rating"],
            player["deviation"],
            int(states[network]["anchored"] or network in anchored),
        )


def ingest(
    store,
    settings,
    paths,
    *,
    checkpoints=(),
    kind="manual",
    decision=None,
    rating_binary=ROOT / "target/release/rating",
    feature_binary=None,
):
    """Validate SGFs, retain store-owned evidence, then commit only unseen games."""
    if (
        settings.get("version") not in {1, 2}
        or settings.get("algorithm") != "static-glicko-v1"
        or settings.get("reference") != REFERENCE_ID
    ):
        raise ValueError("incompatible rating protocol")
    if identity(_protocol_from_row(_protocol_columns(settings))) != identity(settings):
        raise ValueError("rating protocol is not in canonical CSV form")
    if archive.digest(rating_binary) != settings["rating_sha256"]:
        raise ValueError("rating executable changed from the frozen protocol")
    with transaction(store) as (tables, directory):
        for checkpoint in checkpoints:
            _register(tables, checkpoint)
        names = {n: {f"neural-{m['network']}", f"neural-{n}"} for n, m in tables.networks.items()}
        names[REFERENCE_ID] = {REFERENCE_ID}
        records = {
            r["sha256"]: r for r in (_inspect(p, settings, names, feature_binary) for p in paths)
        }
        event = identity({"games": sorted(records)})
        for record in records.values():
            previous = tables.seen.get(record["sha256"])
            if previous and any(
                previous[k] != (str(record[k]) if record[k] is not None else "")
                for k in ["black", "white", "result", "board_size"]
            ):
                raise ValueError("match ledger disagrees with its authoritative SGF")
        existing = [tables.seen[sha] for sha in records if sha in tables.seen]
        if existing:
            _retain_rows(store, directory, existing, {sha: r["path"] for sha, r in records.items()})
        new = [r for sha, r in sorted(records.items()) if sha not in tables.seen]
        if new:
            _rate(tables, new, rating_binary)
            recorded = utc_now().isoformat()
            batch_number = max((int(row["batch_number"]) for row in tables.matches), default=0) + 1
            rows = []
            for game_number, record in enumerate(new, 1):
                row = {
                    "event": event,
                    "recorded_at": recorded,
                    "batch_number": str(batch_number),
                    "game_number": str(game_number),
                    "kind": kind,
                    "decision": decision or "",
                    "sgf_sha256": record["sha256"],
                    **{
                        k: record[k] if record[k] is not None else ""
                        for k in ["black", "white", "result", "board_size"]
                    },
                    **_protocol_columns(settings),
                }
                for color in ["black", "white"]:
                    state = tables.state(record[color])
                    row.update(
                        {
                            f"{color}_{key}_after_batch": state[key]
                            for key in ["rating", "deviation", "anchored"]
                        }
                    )
                row["sgf_path"] = str(Path("games") / _sgf_name(tables, row))
                rows.append(row)
            _retain_rows(store, directory, rows, {r["sha256"]: r["path"] for r in new})
            tables.matches.extend(rows)
        result = {"event": event, "new_games": len(new)}
    if new:
        from .plot_ratings import update

        update(store)
    return result


def refresh(store):
    store = Path(store).resolve()
    with transaction(store) as (tables, _):
        known = {m["bundle"] for m in tables.networks.values()}
        for directory in sorted((store / "accepted").glob("*")):
            if directory.name.startswith("."):
                continue
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError(f"unexpected accepted archive entry: {directory}")
            if str(directory.relative_to(store)) not in known:
                _accept(
                    tables,
                    store,
                    {
                        "directory": str(directory),
                        "sha256": archive.digest(directory / "recovery.pt"),
                    },
                )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-store", type=Path, default=ROOT / "models")
    parser.add_argument("--ingest", nargs="+", type=Path, help="manual arena SGFs/directories")
    parser.add_argument(
        "--protocol-from",
        metavar="SGF_SHA256",
        help="reuse evaluation settings from this match for --ingest (default: latest match)",
    )
    parser.add_argument("--checkpoints", nargs="*", type=Path, default=[])
    parser.add_argument("--rating-binary", type=Path, default=ROOT / "target/release/rating")
    parser.add_argument("--feature-binary", type=Path)
    parser.add_argument("--accepted-only", action="store_true", help="show only accepted networks")
    view = parser.add_mutually_exclusive_group()
    view.add_argument("--matches", action="store_true", help="show newest-first match ledger CSV")
    view.add_argument(
        "--sgf", metavar="SHA256", help="print verified evidence paths for a full SGF hash"
    )
    args = parser.parse_args(argv)
    if args.accepted_only and (args.matches or args.sgf):
        parser.error("--accepted-only applies only to the network view")
    if args.sgf and (args.ingest or not re.fullmatch(r"[0-9a-f]{64}", args.sgf)):
        parser.error("--sgf requires a full SGF hash and cannot be combined with --ingest")
    if not (args.model_store / "STORE.pt").is_file():
        print("No durable model store yet.")
        return
    refresh(args.model_store)
    if args.ingest:
        with transaction(args.model_store) as (tables, _):
            rows = [
                row
                for row in tables.match_rows()
                if not args.protocol_from or row["sgf_sha256"] == args.protocol_from
            ]
            if not rows:
                raise ValueError("no committed match to supply evaluation settings")
            settings = _protocol_from_row(rows[0])
        paths = [
            p
            for source in args.ingest
            for p in (sorted(source.rglob("*.sgf")) if source.is_dir() else [source])
        ]
        print(
            ingest(
                args.model_store,
                settings,
                paths,
                checkpoints=args.checkpoints,
                rating_binary=args.rating_binary,
                feature_binary=args.feature_binary,
            )
        )
    with transaction(args.model_store) as (tables, _):
        if args.sgf:
            rows = [row for row in tables.match_rows() if row["sgf_sha256"] == args.sgf]
            if not rows:
                raise ValueError("SGF hash is not in the match ledger")
            output = "".join(
                f"{path}\n" for path in sorted({evidence_path(args.model_store, r) for r in rows})
            )
        else:
            output = io.StringIO(newline="")
            writer = csv.DictWriter(
                output, fieldnames=MATCH_COLUMNS if args.matches else NETWORK_COLUMNS
            )
            writer.writeheader()
            writer.writerows(
                row
                for row in (tables.match_rows() if args.matches else tables.network_rows())
                if not args.accepted_only or row["accepted"] == "1"
            )
            output = output.getvalue()
    print(output, end="")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit(f"ratings error: {error}") from error
