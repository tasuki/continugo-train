"""SGF corpus metadata, game-level holdout, bounded concurrent real features.

Only Rust parses/replays games. Threads each own a persistent Rust process; no
CUDA fork or 40-MB tensor IPC between Python processes. One game is replayed once
per worker visit. Ordered queues make sampling independent of worker completion
speed. Dense tensors are buffered in bounded memory, never cached as a corpus.
"""

import hashlib
import json
import queue
import random
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from .features import PASS, FeatureClient, inspect_game


@dataclass(frozen=True)
class GameEntry:
    path: str
    game_id: str
    file_sha256: str
    board_size: int
    diameter: int
    resolution: int
    plies: tuple[int, ...]


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
        if not info["labeled"]:
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
                )
            )
    # Deduplicate decoded board+plays, even if copied games have different comments.
    unique = {}
    for game in sorted(eligible, key=lambda g: (g.game_id, g.file_sha256, g.path)):
        if game.game_id in unique:
            skipped["duplicate_game"] += 1
        else:
            unique[game.game_id] = game
    buckets = defaultdict(list)
    for game in unique.values():
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


@dataclass
class Batch:
    coarse: Tensor
    exact: Tensor
    legal: Tensor
    indices: Tensor
    visits: Tensor
    outcomes: Tensor

    def to(self, device) -> "Batch":
        return Batch(*(value.to(device) for value in vars(self).values()))


def collate(samples: list[TrainingSample]) -> Batch:
    if not samples:
        raise ValueError("empty batch")
    width = max(s.visits.numel() for s in samples)
    indices = torch.full((len(samples), width), PASS, dtype=torch.int64)
    visits = torch.zeros(len(samples), width)
    for row, sample in enumerate(samples):
        if (
            sample.coarse.shape != samples[0].coarse.shape
            or sample.exact.shape != samples[0].exact.shape
        ):
            raise ValueError("batch mixes spatial resolutions or action domains")
        indices[row, : sample.indices.numel()] = sample.indices
        visits[row, : sample.visits.numel()] = sample.visits
    return Batch(
        torch.stack([s.coarse for s in samples]),
        torch.stack([s.exact for s in samples]),
        torch.stack([s.legal for s in samples]),
        indices,
        visits,
        torch.tensor([s.outcome for s in samples], dtype=torch.float32),
    )


def _plans(games, seed, epoch, training, limit):
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
    return plans


def _serial_samples(plans, density, binary):
    with FeatureClient(binary) as client:
        for game, plies in plans:
            for ply in plies:
                try:
                    sample = client.load(
                        game.path, ply, density=density, file_sha256=game.file_sha256
                    )
                    if (
                        not sample.full_domain
                        or sample.outcome is None
                        or sample.resolution != game.resolution
                        or sample.board_size != game.board_size
                        or sample.diameter != game.diameter
                    ):
                        raise ValueError(
                            "training requires labeled full-domain samples in the inspected resolution"
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
                    )
                except Exception as error:
                    raise ValueError(f"{game.path} ply {ply}: {error}") from error


def _parallel_samples(plans, density, workers, binary):
    if not plans:
        return
    if not workers:
        yield from _serial_samples(plans, density, binary)
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
            with closing(_serial_samples(shard, density, binary)) as stream:
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
):
    """Bounded, deterministic shuffling and homogeneous-resolution batches.

    Close the generator when stopping an epoch early, so Rust workers are reaped.
    """
    plans = _plans(games, seed, epoch, training, limit)
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

    with closing(_parallel_samples(plans, density, workers, binary)) as stream:
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
