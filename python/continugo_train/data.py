"""SGF corpus metadata, game-level holdout, bounded concurrent real features.

Only Rust parses/replays games. Threads each own a persistent Rust process; no
CUDA fork or 40-MB tensor IPC between Python processes. One game is replayed once
per worker visit. Ordered queues make sampling independent of worker completion
speed. Selected-cell tensors are buffered in bounded memory, never cached as a corpus.
"""

import hashlib
import json
import queue
import random
import sys
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from .feature_cache import SampleReader
from .features import inspect_game
from .sampling import position_quotas, weighted_sizes


@dataclass(frozen=True)
class GameEntry:
    path: str
    game_id: str
    file_sha256: str
    board_size: int
    diameter: int
    resolution: int
    plies: tuple[int, ...]
    # None preserves legacy all-policy manifests; cheap positions still train value.
    policy_plies: tuple[int, ...] | None = None


@dataclass
class Corpus:
    train: list[GameEntry]
    validation: list[GameEntry]
    skipped: dict[str, int]
    fingerprint: str
    # Keep ancestral assignments even when a game leaves the current replay window.
    split_history: dict[str, str]

    def manifest(self) -> list[dict]:
        return [
            dict(asdict(game), split=split)
            for split, games in [("train", self.train), ("validation", self.validation)]
            for game in games
        ]


def inspect_corpus(inputs: list[str], density: int, binary=None):
    """Validate/deduplicate eligible games without requiring a holdout yet."""
    paths = set()
    for name in inputs:
        path = Path(name).resolve()
        if path.is_dir():
            paths.update(
                p.resolve() for p in path.rglob("*") if p.is_file() and p.suffix.lower() == ".sgf"
            )
        elif path.is_file() and path.suffix.lower() == ".sgf":
            paths.add(path)
        else:
            raise ValueError(f"SGF input does not exist or is not an SGF: {path}")
    if not paths:
        raise ValueError("no SGF files found")
    eligible = []
    skipped = {"unlabeled": 0, "no_policy": 0, "duplicate_game": 0}
    for path in sorted(paths):
        info = inspect_game(path, density, binary)
        if not 9 <= info["board_size"] <= 19:
            raise ValueError(f"{path}: training board sizes must be in 9..19")
        if info.get("result") == "0":
            print(f"WARNING DRAW SGF size={info['board_size']}: {path}", file=sys.stderr)
        if not info["labeled"]:
            print(
                f"WARNING UNLABELED/TRUNCATED SGF excluded (size={info['board_size']}, completed={info['completed']}): {path}",
                file=sys.stderr,
            )
            skipped["unlabeled"] += 1
        elif not info["eligible_plies"]:
            skipped["no_policy"] += 1
        else:
            eligible.append(
                GameEntry(
                    str(path),
                    info["game_id"],
                    info["file_sha256"],
                    info["board_size"],
                    info["diameter"],
                    info["resolution"],
                    tuple(info["eligible_plies"]),
                    tuple(info.get("policy_training_plies", info["eligible_plies"])),
                )
            )
    # Deduplicate decoded board+plays, even if copied games have different comments.
    unique = {}
    for game in sorted(eligible, key=lambda g: (g.game_id, g.file_sha256, g.path)):
        if game.game_id in unique:
            skipped["duplicate_game"] += 1
        else:
            unique[game.game_id] = game
    return list(unique.values()), skipped


def scan_corpus(
    inputs: list[str],
    density: int,
    seed: int,
    validation_fraction: float,
    binary=None,
    *,
    split_history: dict[str, str] | None = None,
) -> Corpus:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation fraction must be between 0 and 1")
    history = dict(split_history or {})
    if any(split not in ("train", "validation") for split in history.values()):
        raise ValueError("invalid split history")
    eligible, skipped = inspect_corpus(inputs, density, binary)
    buckets = defaultdict(list)
    for game in eligible:
        buckets[game.resolution].append(game)
    train, validation = [], []
    rng = random.Random(seed)
    for resolution in sorted(buckets):
        games = sorted(buckets[resolution], key=lambda g: g.game_id)
        rng.shuffle(games)
        count = (
            max(1, min(len(games) - 1, round(len(games) * validation_fraction)))
            if len(games) > 1
            else 0
        )
        new_games = [game for game in games if game.game_id not in history]
        held_out = sum(history.get(game.game_id) == "validation" for game in games)
        new_validation = max(0, min(len(new_games), count - held_out))
        for index, game in enumerate(new_games):
            history[game.game_id] = "validation" if index < new_validation else "train"
        # Preserve shuffled order on both the first scan and subsequent resumes.
        validation.extend(game for game in games if history[game.game_id] == "validation")
        train.extend(game for game in games if history[game.game_id] == "train")
    if not train or not validation:
        raise ValueError(
            "need unique completed MCTS games, including at least two in one resolution bucket, for train/validation; inherited split assignments cannot be reassigned"
        )
    # Relocating an unchanged corpus is allowed; changing targets/content is not.
    identity = [
        {"game_id": g.game_id, "file_sha256": g.file_sha256, "plies": g.plies, "split": split}
        for split, games in [("train", train), ("validation", validation)]
        for g in games
    ]
    fingerprint = hashlib.sha256(
        json.dumps(sorted(identity, key=lambda g: g["game_id"]), sort_keys=True).encode()
    ).hexdigest()
    return Corpus(train, validation, skipped, fingerprint, history)


@dataclass
class TrainingSample:
    coarse: Tensor
    exact: Tensor
    legal: Tensor
    indices: Tensor
    visits: Tensor
    outcome: float
    game_id: str
    ply: int
    actions: Tensor
    board_size: int
    score: float
    ownership: Tensor
    policy_weight: float = 1.0


@dataclass
class Batch:
    coarse: Tensor
    exact: Tensor
    legal: Tensor
    actions: Tensor
    indices: Tensor
    visits: Tensor
    outcomes: Tensor
    board_sizes: Tensor
    scores: Tensor
    ownership: Tensor
    policy_weights: Tensor

    def to(self, device) -> "Batch":
        return Batch(*(value.to(device) for value in vars(self).values()))


def collate(samples: list[TrainingSample]) -> Batch:
    if not samples:
        raise ValueError("empty batch")
    width = max(s.visits.numel() for s in samples)
    count = max(s.actions.numel() for s in samples)
    indices = torch.full((len(samples), width), count, dtype=torch.int64)
    visits = torch.zeros(len(samples), width)
    exact = samples[0].exact.new_zeros((len(samples), count, samples[0].exact.shape[-1]))
    legal = torch.zeros(len(samples), count, dtype=torch.bool)
    actions = torch.zeros(len(samples), count, dtype=torch.int64)
    for row, sample in enumerate(samples):
        if sample.coarse.shape != samples[0].coarse.shape:
            raise ValueError("batch mixes spatial resolutions")
        n = sample.actions.numel()
        exact[row, :n] = sample.exact
        legal[row, :n] = sample.legal
        actions[row, :n] = sample.actions
        indices[row, : sample.indices.numel()] = torch.where(
            sample.indices == n, count, sample.indices
        )
        visits[row, : sample.visits.numel()] = sample.visits
    return Batch(
        torch.stack([s.coarse for s in samples]),
        exact,
        legal,
        actions,
        indices,
        visits,
        torch.tensor([s.outcome for s in samples], dtype=torch.float32),
        torch.tensor([s.board_size for s in samples], dtype=torch.int64),
        torch.tensor([s.score for s in samples], dtype=torch.float32),
        torch.stack([s.ownership for s in samples]),
        torch.tensor([s.policy_weight for s in samples], dtype=torch.float32),
    )


def _plans(games, seed, epoch, training, limit, size_weights="uniform", batch_size=1):
    rng = random.Random(seed + (epoch * 1000003 if training else 0))
    games = list(games)
    if training:
        rng.shuffle(games)
    plans = []
    for game in games:
        plies = list(game.plies)
        if training:
            rng.shuffle(plies)
        elif limit and len(plies) > limit:
            plies = sorted(random.Random(seed ^ int(game.game_id[:16], 16)).sample(plies, limit))
        if limit:
            plies = plies[:limit]
        plans.append((game, plies))
    if size_weights == "uniform":
        return plans
    # Sample references before extracting cell features. No exact features
    # are computed for surplus long-board positions.
    references = defaultdict(list)
    for game, plies in plans:
        references[game.board_size].extend((game, ply) for ply in plies)
    quotas = position_quotas({size: len(items) for size, items in references.items()}, size_weights)
    pools = {}
    for size, count in sorted(quotas.items()):
        items = references[size]
        if training:
            rng.shuffle(items)
        elif len(items) > count:
            items = random.Random(seed + size * 100003).sample(items, count)
        pools[size] = deque(items[:count])
    cycle = weighted_sizes(sorted(pools), size_weights)
    index = (seed + epoch) % len(cycle) if training else 0
    ordered = []
    while pools:
        size = cycle[index % len(cycle)]
        index += 1
        if size not in pools:
            continue
        for _ in range(min(batch_size, len(pools[size]))):
            game, ply = pools[size].popleft()
            ordered.append((game, [ply]))
        if not pools[size]:
            del pools[size]
    return ordered


def _serial_samples(plans, density, binary, feature_profile=None, feature_cache="off"):
    with SampleReader(binary, cache=feature_cache, profile=feature_profile is not None) as client:
        for game, plies in plans:
            for ply in plies:
                try:
                    sample = client.load(
                        game.path, ply, density=density, file_sha256=game.file_sha256
                    )
                    if feature_profile is not None and client.last_profile is not None:
                        feature_profile.add(client.process.pid, client.last_profile)
                    if (
                        sample.full_domain
                        or sample.outcome is None
                        or sample.score is None
                        or sample.ownership is None
                        or sample.resolution != game.resolution
                        or sample.board_size != game.board_size
                        or sample.diameter != game.diameter
                    ):
                        raise ValueError(
                            "training requires labeled target-cell samples in the inspected resolution"
                        )
                    indices, visits = sample.policy_targets()
                    yield TrainingSample(
                        sample.coarse,
                        sample.exact,
                        sample.legal,
                        indices,
                        visits,
                        sample.outcome,
                        game.game_id,
                        ply,
                        sample.actions,
                        game.board_size,
                        sample.score,
                        sample.ownership,
                        float(game.policy_plies is None or ply in game.policy_plies),
                    )
                except Exception as error:
                    raise ValueError(f"{game.path} ply {ply}: {error}") from error


def _parallel_samples(plans, density, workers, binary, feature_profile=None, feature_cache="off"):
    if not plans:
        return
    if not workers:
        yield from _serial_samples(plans, density, binary, feature_profile, feature_cache)
        return
    count = min(workers, len(plans))
    stop = threading.Event()
    queues = [queue.Queue(maxsize=1) for _ in range(count)]

    def put(channel, value):
        while not stop.is_set():
            try:
                channel.put(value, timeout=0.1)
                return
            except queue.Full:
                pass

    def produce(shard, channel):
        try:
            with closing(
                _serial_samples(shard, density, binary, feature_profile, feature_cache)
            ) as stream:
                for sample in stream:
                    if stop.is_set():
                        break
                    put(channel, sample)
        except Exception as error:  # noqa: BLE001 -- re-raised in the consumer thread.
            put(channel, error)
        finally:
            put(channel, None)

    with ThreadPoolExecutor(max_workers=count) as pool:
        for index, channel in enumerate(queues):
            pool.submit(produce, plans[index::count], channel)
        try:
            active = list(queues)
            while active:
                for channel in list(active):
                    value = channel.get()
                    if value is None:
                        active.remove(channel)
                    elif isinstance(value, Exception):
                        raise value
                    else:
                        yield value
        finally:
            stop.set()


def batches(
    games,
    *,
    batch_size,
    seed,
    epoch,
    training,
    density,
    workers,
    shuffle_buffer,
    limit=0,
    binary=None,
    size_weights="uniform",
    feature_profile=None,
    feature_cache="off",
):
    """Bounded, deterministic shuffling and homogeneous-resolution batches.

    Close the generator when stopping an epoch early, so Rust workers are reaped.
    """
    plans = _plans(games, seed, epoch, training, limit, size_weights, batch_size)
    # Weighted plans already shuffle references within each size. A global
    # shuffle would undo the early-step budget mix and waste extraction work.
    if size_weights != "uniform":
        shuffle_buffer = 0
    rng = random.Random(seed + epoch * 1000033)
    buffered = []
    buckets = defaultdict(list)

    def enqueue(sample):
        key = tuple(sample.coarse.shape)
        bucket = buckets[key]
        bucket.append(sample)
        if len(bucket) == batch_size:
            batch = collate(bucket)
            bucket.clear()
            return batch
        return None

    with closing(
        _parallel_samples(plans, density, workers, binary, feature_profile, feature_cache)
    ) as stream:
        for sample in stream:
            if training and shuffle_buffer:
                if len(buffered) < shuffle_buffer:
                    buffered.append(sample)
                    continue
                index = rng.randrange(len(buffered))
                sample, buffered[index] = buffered[index], sample
            batch = enqueue(sample)
            if batch is not None:
                yield batch
        rng.shuffle(buffered)
        for sample in buffered:
            batch = enqueue(sample)
            if batch is not None:
                yield batch
        buffered.clear()
        for key in sorted(buckets):
            if buckets[key]:
                yield collate(buckets[key])
