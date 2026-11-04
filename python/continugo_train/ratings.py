"""Static-Glicko ratings in two CSVs; games remain authoritative SGFs.

networks.csv holds model metadata and the current per-pool rating view.
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
NETWORK_COLUMNS = META + ["pool", "status", "rating", "deviation", "rating ± RD"] + COUNTS
LEGACY_NETWORK_COLUMNS = [column for column in NETWORK_COLUMNS if column != "registered_at"]
SEARCH_TYPES = {
    "simulations": int,
    "cpuct": float,
    "widening": float,
    "widening_exponent": float,
    "proposal_attempts": int,
    "independent_exploration": float,
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
    "pool",
    "recorded_at",
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


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def identity(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def protocol(config):
    counts = Counter(weighted_sizes(config.sizes, config.size_weights))
    divisor = math.gcd(*counts.values())
    return {
        "version": 1,
        "algorithm": "static-glicko-v1",
        "reference": REFERENCE_ID,
        "board_mix": {str(size): count // divisor for size, count in sorted(counts.items())},
        "search": {**config.search(), "simulations": config.gate_simulations},
        "neural_max_moves": config.gate_max_moves,
        "reference_max_moves": config.reference_max_moves,
        "selection": "maximum-visits-random-ties",
        "arena_sha256": archive.digest(config.arena_binary),
        "rating_sha256": archive.digest(config.rating_binary),
    }


def _protocol_columns(settings):
    return {
        **{k: settings[k] for k in PROTOCOL_COLUMNS if k not in SEARCH_TYPES},
        **settings["search"],
        "board_mix": ",".join(f"{k}:{v}" for k, v in settings["board_mix"].items()),
    }


def _protocol_from_row(row):
    settings = {k: row[k] for k in PROTOCOL_COLUMNS if k not in SEARCH_TYPES}
    for key in ["version", "neural_max_moves", "reference_max_moves"]:
        settings[key] = int(settings[key])
    settings["board_mix"] = dict(item.split(":") for item in settings["board_mix"].split(","))
    settings["board_mix"] = {k: int(v) for k, v in settings["board_mix"].items()}
    settings["search"] = {k: kind(row[k]) for k, kind in SEARCH_TYPES.items()}
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
        if reader.fieldnames not in (columns, legacy_columns):
            raise ValueError(f"incompatible rating CSV header: {path}")
        rows = list(reader)
        if any(set(row) != set(reader.fieldnames) or None in row.values() for row in rows):
            raise ValueError(f"incomplete rating CSV row: {path}")
        return [{**dict.fromkeys(columns, ""), **row} for row in rows]


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
    pools: dict = field(default_factory=dict)
    states: dict = field(default_factory=dict)
    seen: dict = field(default_factory=dict)

    def state(self, pool, network):
        key = (pool, network)
        if key not in self.states:
            parent = self.states.get((pool, self.networks[network]["parent"]))
            self.states[key] = {
                "rating": parent["rating"]
                if parent and parent["anchored"] and parent["games"]
                else 100.0,
                "deviation": 0.0 if network == REFERENCE_ID else 350.0,
                "anchored": int(network == REFERENCE_ID),
                **dict.fromkeys(COUNTS, 0),
            }
        return self.states[key]

    def network_rows(self):
        # Stable ID order breaks timestamp ties; keep the fixed reference last.
        networks = sorted(self.networks.items())
        networks.sort(
            key=lambda item: (item[0] != REFERENCE_ID, _registration_time(item[1])),
            reverse=True,
        )
        for network, meta in networks:
            for pool in sorted(self.pools) or [""]:
                state = self.states.get((pool, network))
                fixed = network == REFERENCE_ID
                if fixed and state is None:
                    state = {
                        "rating": 100.0,
                        "deviation": 0.0,
                        "anchored": 1,
                        **dict.fromkeys(COUNTS, 0),
                    }
                rated = state is not None and (fixed or state["anchored"] and state["games"])
                rating, deviation = (state[k] if rated else "" for k in ["rating", "deviation"])
                yield {
                    **meta,
                    "pool": pool,
                    "status": "fixed" if fixed else "rated" if rated else "unrated",
                    "rating": rating,
                    "deviation": deviation,
                    "rating ± RD": f"{rating:.1f} ± {deviation:.1f}" if rated else "unrated",
                    **{k: state[k] if state else 0 for k in COUNTS},
                }


def _counts(tables, pool, black, white, result):
    if black == white:
        return
    for network, color in [(black, "B"), (white, "W")]:
        state = tables.state(pool, network)
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


def _load(directory):
    tables = Tables()
    rows = _read_csv(
        directory / "networks.csv", NETWORK_COLUMNS, legacy_columns=LEGACY_NETWORK_COLUMNS
    )
    for row in rows:
        meta = {k: row[k] for k in META}
        network = meta["network_id"]
        if not _valid_id(network) or meta["accepted"] != str(int(bool(meta["accepted_at"]))):
            raise ValueError("invalid network metadata in networks.csv")
        _registration_time(meta)
        if network in tables.networks and tables.networks[network] != meta:
            raise ValueError("conflicting network metadata across pools")
        tables.networks[network] = meta
    if (directory / "networks.csv").exists() and not (directory / "matches.csv").exists():
        raise ValueError(
            "matches.csv missing; restore the match ledger rather than resetting ratings"
        )
    tables.matches = _read_csv(directory / "matches.csv", MATCH_COLUMNS)
    for row in tables.matches:
        pool, black, white = (row[k] for k in ["pool", "black", "white"])
        settings = _protocol_from_row(row)
        key = (pool, row["sgf_sha256"])
        if (
            identity(settings) != pool
            or not re.fullmatch(r"[0-9a-f]{64}", row["sgf_sha256"])
            or black not in tables.networks
            or white not in tables.networks
            or key in tables.seen
            or row["result"]
            and not re.fullmatch(r"0|[BW]\+\d+\.\d", row["result"])
        ):
            raise ValueError("invalid or duplicate match ledger row")
        tables.seen[key] = row
        tables.pools[pool] = settings
        _counts(tables, pool, black, white, row["result"])
        for color, network in [("black", black), ("white", white)]:
            _set_rating(
                tables.state(pool, network),
                network,
                float(row[f"{color}_rating_after_batch"]),
                float(row[f"{color}_deviation_after_batch"]),
                int(row[f"{color}_anchored_after_batch"]),
            )
    return tables


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
        _atomic_csv(directory / "matches.csv", MATCH_COLUMNS, tables.matches)


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


def register_checkpoint(store, checkpoint):
    with transaction(store) as (tables, _):
        _register(tables, checkpoint)


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


def _rate(tables, pool, records, rating_binary):
    participants = sorted({r[k] for r in records for k in ["black", "white"]})
    states = {network: tables.state(pool, network) for network in participants}
    scored, anchored = [], set()
    for record in records:
        black, white, result = (record[k] for k in ["black", "white", "result"])
        _counts(tables, pool, black, white, result)
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
        settings.get("version") != 1
        or settings.get("algorithm") != "static-glicko-v1"
        or settings.get("reference") != REFERENCE_ID
    ):
        raise ValueError("incompatible rating protocol")
    if identity(_protocol_from_row(_protocol_columns(settings))) != identity(settings):
        raise ValueError("rating protocol is not in canonical CSV form")
    if archive.digest(rating_binary) != settings["rating_sha256"]:
        raise ValueError("rating executable changed from the frozen protocol")
    pool = identity(settings)
    with transaction(store) as (tables, directory):
        for checkpoint in checkpoints:
            _register(tables, checkpoint)
        tables.networks.setdefault(
            REFERENCE_ID,
            {
                **dict.fromkeys(META, ""),
                "network_id": REFERENCE_ID,
                "network": REFERENCE_ID,
                "accepted": "0",
            },
        )
        names = {n: {f"neural-{m['network']}", f"neural-{n}"} for n, m in tables.networks.items()}
        names[REFERENCE_ID] = {REFERENCE_ID}
        records = {
            r["sha256"]: r for r in (_inspect(p, settings, names, feature_binary) for p in paths)
        }
        event = identity({"pool": pool, "games": sorted(records)})
        for record in records.values():
            previous = tables.seen.get((pool, record["sha256"]))
            if previous and any(
                previous[k] != (str(record[k]) if record[k] is not None else "")
                for k in ["black", "white", "result", "board_size"]
            ):
                raise ValueError("match ledger disagrees with its authoritative SGF")
        archive.retain_evidence(
            store, [(r["path"], r["sha256"]) for r in records.values()], directory / "games"
        )
        new = [r for sha, r in sorted(records.items()) if (pool, sha) not in tables.seen]
        if new:
            tables.pools[pool] = settings
            _rate(tables, pool, new, rating_binary)
            recorded = utc_now().isoformat()
            for record in new:
                row = {
                    "event": event,
                    "pool": pool,
                    "recorded_at": recorded,
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
                    state = tables.state(pool, record[color])
                    row.update(
                        {
                            f"{color}_{key}_after_batch": state[key]
                            for key in ["rating", "deviation", "anchored"]
                        }
                    )
                tables.matches.append(row)
        return {"pool": pool, "event": event, "new_games": len(new)}


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
    parser.add_argument("--pool", help="existing full pool ID, required for --ingest")
    parser.add_argument("--checkpoints", nargs="*", type=Path, default=[])
    parser.add_argument("--rating-binary", type=Path, default=ROOT / "target/release/rating")
    parser.add_argument("--feature-binary", type=Path)
    parser.add_argument("--accepted-only", action="store_true", help="show only accepted networks")
    args = parser.parse_args(argv)
    if args.ingest and not args.pool:
        parser.error("--ingest requires --pool")
    if not (args.model_store / "STORE.pt").is_file():
        print("No durable model store yet.")
        return
    refresh(args.model_store)
    if args.ingest:
        with transaction(args.model_store) as (tables, _):
            if args.pool not in tables.pools:
                raise ValueError("unknown rating pool")
            settings = tables.pools[args.pool]
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
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=NETWORK_COLUMNS)
        writer.writeheader()
        writer.writerows(
            r for r in tables.network_rows() if not args.accepted_only or r["accepted"] == "1"
        )
    print(output.getvalue(), end="")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit(f"ratings error: {error}") from error
