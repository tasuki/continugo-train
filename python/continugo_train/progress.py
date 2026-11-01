"""Checkpoint-lineage counters, independent of optimizer and replay-window state.

s counts optimized training rows, including repeats, but not validation rows.
d counts eligible pre-move rows in completed policy-labeled self-play games,
BEFORE position/board sampling, including validation. Truncated games contribute
no eligible rows. Offline imports deduplicate ancestral game IDs; the loop can
supply an exact generation total, including generated games evicted from replay
and repeated games produced by distinct generation jobs. Recovery copies are
not new generation. Missing legacy totals remain unknown throughout continuation.
"""

import re


def validate(progress):
    if not isinstance(progress, dict) or not {"family", "samples_seen", "data_positions"}.issubset(
        progress
    ):
        raise ValueError("invalid cumulative progress metadata")
    if not isinstance(progress["family"], str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}", progress["family"]
    ):
        raise ValueError("network family must be 1..32 letters, digits, underscores or hyphens")
    for key in ["samples_seen", "data_positions"]:
        value = progress.get(key)
        if value is not None and (type(value) is not int or not 0 <= value < 2**63):
            raise ValueError(f"invalid cumulative {key}")
    return dict(progress)


def checkpoint_progress(saved):
    return validate(
        saved.get("progress", {"family": "cg1", "samples_seen": None, "data_positions": None})
    )


def data_history(saved, split_history):
    ids = saved.get("data_game_ids", []) if saved is not None else []
    if not isinstance(ids, list) or any(
        not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{64}", identity)
        for identity in ids
    ):
        raise ValueError("invalid generated/imported data game history")
    return set(split_history) | set(ids)


def initialize(saved, games, history, *, resuming=False, family=None, data_positions=None):
    progress = (
        checkpoint_progress(saved)
        if saved is not None
        else {"family": "cg1", "samples_seen": 0, "data_positions": 0}
    )
    if family is not None:
        if resuming and family != progress["family"]:
            raise ValueError("resume restores --network-family; conflicting override")
        progress["family"] = family
    if not resuming and progress["data_positions"] is not None:
        progress["data_positions"] += sum(len(g.plies) for g in games if g.game_id not in history)
    if data_positions is not None:
        previous = progress["data_positions"]
        if previous is None:
            raise ValueError("cannot override unknown ancestral data progress")
        if data_positions < previous or resuming and data_positions != previous:
            raise ValueError("--data-positions conflicts with inherited/generated data progress")
        progress["data_positions"] = data_positions
    return validate(progress)


def network_name(saved):
    progress = checkpoint_progress(saved)
    samples, data = (progress[k] for k in ["samples_seen", "data_positions"])
    config = saved["config"]
    if any(type(config[k]) is not int or config[k] <= 0 for k in ["blocks", "channels"]):
        raise ValueError("network blocks/channels must be positive integers")
    return (
        f"{progress['family']}-b{config['blocks']}c{config['channels']}"
        f"-s{samples if samples is not None else 'unknown'}"
        f"-d{data if data is not None else 'unknown'}"
    )
