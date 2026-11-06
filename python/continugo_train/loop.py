"""Restartable native self-play -> bounded replay -> warm-start -> ONNX loop.

SGFs remain the game records. loop.pt is an atomic orchestration checkpoint, not
an alternate game format. Training, experimental actors and accepted networks
are independent. Matches never promote: publication needs explicit user approval.
"""

import argparse
import fcntl
import hashlib
import io
import math
import os
import signal
import subprocess
import sys
import tempfile
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch

from . import archive, gate, plot_metrics, promote, ratings
from .data import inspect_corpus, scan_corpus
from .features import feature_binary, inspect_game
from .model import NetworkConfig, PolicyValueNet
from .paths import new_run_directory
from .progress import checkpoint_progress, data_history, network_name
from .progress import initialize as initialize_progress
from .runtime import inference_environment
from .sampling import DEFAULT_SIZE_WEIGHTS, weighted_sizes
from .train import Options, atomic_checkpoint, checkpoint_splits, feature_spec

VERSION = 6
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class LoopConfig:
    sizes: tuple[int, ...] = (9, 13, 19)
    size_weights: str = DEFAULT_SIZE_WEIGHTS
    games_per_iteration: int = 10
    seed: int = 0
    simulations: int = 32
    max_moves: int = 512
    cpuct: float = 1.5
    widening: float = 2.0
    widening_exponent: float = 0.5
    proposal_attempts: int = 128
    independent_exploration: float = 0.25
    score_weight: float = 0.3
    concurrent_games: int = 12
    inference_batch_size: int = 12
    inference_batch_wait_ms: int = 2
    inference_threads: int = 1
    feature_threads: int = 6
    cache_positions: int = 64
    replay_iterations: int = 4
    replay_games: int = 256
    train_epochs: int = 1
    steps_per_epoch: int = 20
    positions_per_game: int = 8
    validation_positions_per_game: int = 4
    batch_size: int = 2
    workers: int = Options().workers
    shuffle_buffer: int = 8
    validation_fraction: float = 0.2
    lr: float | None = None
    warm_start_optimizer: str = "keep"
    device: str = "cuda"
    threads: int = 4
    selfplay_binary: str = str(ROOT / "target/release/continugo-train")
    feature_binary: str = str(ROOT / "target/release/features")
    ort_library: str | None = None
    arena_binary: str = str(ROOT / "target/release/arena")
    rating_binary: str = str(ROOT / "target/release/rating")
    reference_games: int = 10
    reference_max_moves: int = 2048
    gate_games: int = 20
    gate_simulations: int = 32
    gate_max_moves: int = 512
    acceptance_rate: float = 0.5
    actor_policy: str = "latest"
    evaluation_interval: int = 1
    model_store: str = str(ROOT / "models")
    adaptive_max_games: int = 600
    adaptive_target_rd: float = 25.0
    adaptive_min_pairs: int = 60
    adaptive_min_opponents: int = 2
    adaptive_min_opponent_pairs: int = 10
    adaptive_batch_games: int = 4

    def validate(self):
        weighted_sizes(self.sizes, self.size_weights)
        if any(
            not 0 <= count < 2**32 or count % 2 for count in [self.gate_games, self.reference_games]
        ):
            raise ValueError(
                "gate-games/reference-games must be even nonnegative native u32 counts (0 disables)"
            )
        if self.reference_max_moves <= 0:
            raise ValueError("reference-max-moves must be positive")
        if (
            self.adaptive_max_games < 0
            or not 0 < self.adaptive_batch_games < 2**32
            or self.adaptive_batch_games % 2
            or self.adaptive_max_games % self.adaptive_batch_games
            or not math.isfinite(self.adaptive_target_rd)
            or self.adaptive_target_rd <= 0
            or min(
                self.adaptive_min_pairs,
                self.adaptive_min_opponents,
                self.adaptive_min_opponent_pairs,
            )
            <= 0
        ):
            raise ValueError("invalid adaptive rating precision/evidence limits or game budget")
        if (
            not 0 < self.gate_simulations < 2**32
            or self.gate_max_moves <= 0
            or not 0.5 <= self.acceptance_rate <= 1
        ):
            raise ValueError("positive gate budgets and acceptance-rate in 0.5..1 required")
        if not self.sizes or any(size not in range(9, 20) for size in self.sizes):
            raise ValueError("sizes must be nonempty and in 9..19")
        if not 1 <= self.games_per_iteration < 2**32 or not 1 <= self.simulations < 2**32:
            raise ValueError("game count and simulations must fit positive native u32 limits")
        positive = [
            "games_per_iteration",
            "simulations",
            "max_moves",
            "proposal_attempts",
            "inference_threads",
            "feature_threads",
            "replay_iterations",
            "train_epochs",
            "batch_size",
            "threads",
            "evaluation_interval",
        ]
        if any(getattr(self, name) <= 0 for name in positive) or self.replay_games < 2:
            raise ValueError(
                "positive game/search/training/thread limits and replay-games >= 2 required"
            )
        nonnegative = [
            "seed",
            "steps_per_epoch",
            "positions_per_game",
            "validation_positions_per_game",
            "workers",
            "shuffle_buffer",
            "cache_positions",
        ]
        if any(getattr(self, name) < 0 for name in nonnegative):
            raise ValueError("seed, training caps, workers, buffer and cache must be nonnegative")
        if not 1 <= self.concurrent_games <= 64 or not 1 <= self.inference_batch_size <= 32:
            raise ValueError("concurrent-games must be in 1..64 and inference-batch-size in 1..32")
        if not 0 <= self.inference_batch_wait_ms <= 1000:
            raise ValueError("inference-batch-wait-ms must be in 0..1000")
        if (
            not math.isfinite(self.cpuct)
            or self.cpuct <= 0
            or not math.isfinite(self.widening)
            or self.widening <= 0
            or not 0 < self.widening_exponent < 1
            or not 0 < self.independent_exploration <= 1
            or not math.isfinite(self.score_weight)
            or not 0 <= self.score_weight <= 1
            or not 0 < self.validation_fraction < 1
            or self.lr is not None
            and (not math.isfinite(self.lr) or self.lr <= 0)
        ):
            raise ValueError("invalid search, validation fraction or learning rate")
        if self.actor_policy not in {"latest", "gate"}:
            raise ValueError("actor-policy must be latest or gate")
        if self.device not in {"cpu", "cuda"} or self.warm_start_optimizer not in {"keep", "reset"}:
            raise ValueError("invalid device or warm-start optimizer policy")

    def search(self):
        return {
            name: getattr(self, name)
            for name in [
                "simulations",
                "cpuct",
                "widening",
                "widening_exponent",
                "proposal_attempts",
                "independent_exploration",
                "score_weight",
            ]
        }

    def training_options(self, parent, iteration):
        options = dict(parent["options"])
        for name in [
            "batch_size",
            "workers",
            "shuffle_buffer",
            "positions_per_game",
            "validation_positions_per_game",
            "steps_per_epoch",
            "validation_fraction",
            "size_weights",
        ]:
            options[name] = getattr(self, name)
        options["seed"] = self.seed + iteration - 1
        if self.lr is not None:
            options["lr"] = self.lr
        result = Options(**options)
        result.validate()
        return result


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_bytes(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".snapshot-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


@contextmanager
def loop_lock(directory):
    # Never unlink this inode: a waiting/starting writer must lock the same file.
    descriptor = os.open(directory / ".loop.lock", os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(descriptor, "r+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("loop is locked by another runner or an in-flight child") from error
        stream.seek(0)
        stream.truncate()
        stream.write(str(os.getpid()))
        stream.flush()
        # Children inherit this lock; a killed supervisor cannot race a live child.
        yield stream.fileno()


def run_process(command, logfile, lock_fd, environment=None):
    with logfile.open("a") as stream:
        stream.write(f"command={command!r}\n")
        stream.flush()
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
            pass_fds=(lock_fd,) if isinstance(lock_fd, int) else tuple(lock_fd),
        )
        try:
            code = process.wait()
        except BaseException:
            # SIGINT lets the trainer close feature workers and release its lock.
            for sig, timeout in [(signal.SIGINT, 20), (signal.SIGTERM, 5), (signal.SIGKILL, None)]:
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=timeout)
                    break
                except subprocess.TimeoutExpired:
                    continue
            raise
    if code:
        raise RuntimeError(f"stage exited {code}; see {logfile}")


def load_training(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "options",
        "config",
        "model",
        "optimizer",
        "epoch",
        "global_step",
        "run_id",
        "feature_spec",
        "corpus_fingerprint",
        "best_validation",
        "rng_cpu",
        "rng_python",
        "rng_cuda",
    }
    if not isinstance(saved, dict) or not required.issubset(saved):
        raise ValueError(f"not a training checkpoint: {path}")
    data_history(saved, checkpoint_splits(saved))
    checkpoint_progress(saved)
    options = Options(**saved["options"])
    options.validate()
    config = NetworkConfig(**saved["config"])
    if saved["feature_spec"] != feature_spec(options.samples_per_diameter):
        raise ValueError("checkpoint feature semantics are incompatible")
    if config.input_planes != len(saved["feature_spec"]["coarse"]) or config.move_features != len(
        saved["feature_spec"]["moves"]
    ):
        raise ValueError("checkpoint feature widths are incompatible")
    if saved["epoch"] <= 0 or saved["global_step"] < 0:
        raise ValueError("expected a completed training checkpoint")
    return saved


def verify_export(directory, checkpoint, saved):
    """Adopt only a fully published, compatible encoder/scorer pair."""
    import json

    import onnx

    from .export import METADATA_KEY, verify

    if {p.name for p in directory.iterdir()} != {"encoder.onnx", "policy.onnx"}:
        raise ValueError(f"incomplete/unexpected export artifacts: {directory}")
    expected = {
        "checkpoint_sha256": digest(checkpoint),
        "epoch": saved["epoch"],
        "network_name": network_name(saved),
        "progress": checkpoint_progress(saved),
        "config": saved["config"],
        "feature_spec": saved["feature_spec"],
    }
    for name in ["encoder.onnx", "policy.onnx"]:
        graph = onnx.load(directory / name)
        onnx.checker.check_model(graph)
        metadata = {item.key: item.value for item in graph.metadata_props}
        if json.loads(metadata.get(METADATA_KEY, "null")) != expected:
            raise ValueError(f"export does not match its training checkpoint: {directory}")
    model = PolicyValueNet(NetworkConfig(**saved["config"])).eval()
    model.load_state_dict(saved["model"])
    verify(model, directory)


def select_replay(records, limit, size_weights=DEFAULT_SIZE_WEIGHTS):
    """Newest candidates first, balanced round-robin by board size, unique plays."""
    buckets = defaultdict(deque)
    seen = set()
    for record in records:
        if record["eligible"] and record["game_id"] not in seen:
            seen.add(record["game_id"])
            buckets[record["board_size"]].append(record)
    if not buckets:
        return []
    cycle = weighted_sizes(sorted(buckets), size_weights)
    buckets = {size: items for size, items in buckets.items() if size in cycle}
    selected = []
    index = 0
    while buckets and len(selected) < limit:
        size = cycle[index % len(cycle)]
        index += 1
        if size not in buckets:
            continue
        selected.append(buckets[size].popleft())
        if not buckets[size]:
            del buckets[size]
    return selected


class LoopRunner:
    def __init__(self, root, state, lock_fd):
        self.root, self.state, self.lock_fd = root, state, lock_fd
        self.config = LoopConfig(**state["config"])

    def save(self):
        atomic_checkpoint(self.root / "loop.pt", self.state)

    def verify_runtime(self):
        expected = self.state["runtime_fingerprints"]
        current = {
            "selfplay": digest(self.config.selfplay_binary),
            "features": digest(self.config.feature_binary),
        }
        if current != expected:
            raise ValueError(
                "generation/feature executable changed; fork a new loop instead of mixing protocols"
            )
        if (
            self.state["rating_protocol"] is not None
            and ratings.protocol(self.config) != self.state["rating_protocol"]
        ):
            raise ValueError(
                "evaluation/rating executable changed; fork a new loop instead of mixing protocols"
            )

    def execute(self, command, logfile, environment=None):
        self.verify_runtime()
        run_process(command, logfile, self.lock_fd, environment)
        self.verify_runtime()

    def log(self, text):
        print(text, flush=True)
        with (self.root / "loop.log").open("a") as stream:
            stream.write(text + "\n")

    def path(self, relative):
        path = self.root / relative
        if not path.resolve().is_relative_to(self.root) or path.is_symlink():
            raise ValueError(f"artifact path escapes loop or is a symlink: {relative}")
        return path

    def record(self, path):
        return {"path": str(path.relative_to(self.root)), "sha256": digest(path)}

    def check(self, record):
        path = self.path(record["path"])
        if not path.is_file() or digest(path) != record["sha256"]:
            raise ValueError(f"committed artifact missing or changed: {path}")
        return path

    def check_model(self, record):
        self.check(record["encoder"])
        self.check(record["policy"])
        return self.path(record["directory"])

    def snapshots(self):
        records = [
            self.state["initial_checkpoint"],
            self.state["initial_actor_checkpoint"],
            *self.state["bootstrap"],
        ]
        for record in records:
            destination = self.path(record["path"])
            if destination.exists() or self.state["snapshots_ready"]:
                self.check(record)
                continue
            content = Path(record["source"]).read_bytes()
            if hashlib.sha256(content).hexdigest() != record["sha256"]:
                raise ValueError(f"snapshot source changed: {record['source']}")
            atomic_bytes(destination, content)
        if not self.state["snapshots_ready"]:
            self.state["snapshots_ready"] = True
            self.save()

    def verify_committed(self):
        self.snapshots()
        if self.state["initial_model"]:
            self.check_model(self.state["initial_model"])
        for iteration in self.state["iterations"]:
            for record in iteration["games"]:
                if record:
                    self.check(record)
            for record in iteration["replay"] or []:
                self.check(record)
            if iteration.get("generated_data"):
                self.check(iteration["generated_data"])
            if iteration["checkpoint"]:
                self.check(iteration["checkpoint"])
            if iteration["model"]:
                self.check_model(iteration["model"])
            for key in ["gate", "reference"]:
                if iteration.get(key):
                    for record in iteration[key]["games"]:
                        if record:
                            self.check(record)
            for key in ["training_archive", "experimental_archive"]:
                if iteration[key]:
                    archive.verify_bundle(iteration[key])
        initial_reference = self.state.get("initial_reference")
        if initial_reference and initial_reference.get("reference"):
            for record in initial_reference["reference"]["games"]:
                if record:
                    self.check(record)
        for key in [
            "initial_training_archive",
            "initial_actor_training_archive",
            "initial_experimental_archive",
        ]:
            if self.state[key]:
                archive.verify_bundle(self.state[key])
        for role in ["training", "actor"]:
            if self.state[role]:
                self.check(self.state[role]["checkpoint"])
                archive.verify_bundle(self.state[role]["archive"])
                if role == "actor":
                    self.check_model(self.state[role]["model"])

    def recovery(self, checkpoint, kind, *, model=None, iteration=None):
        candidates = [str(self.check(record)) for record in self.state["bootstrap"]]
        # Recovery packs can be used even after their original corpus paths vanish.
        source = Path(checkpoint.get("source", self.check(checkpoint)))
        candidates += [str(path) for path in (source.parent / "replay").glob("*.sgf")]
        evaluation = []
        if iteration:
            evaluation = [
                str(self.check(r))
                for key in ["gate", "reference"]
                for r in (iteration.get(key) or {}).get("games", [])
                if r
            ]
        reference = archive.publish(
            self.config.model_store,
            kind,
            self.check(checkpoint),
            model=self.check_model(model) if model else None,
            game_candidates=candidates,
            evaluation=evaluation,
            context={
                "loop": str(self.root),
                "iteration": iteration["number"] if iteration else 0,
                "config": asdict(self.config),
                "next_seed": self.config.seed
                + (iteration["number"] if iteration else 0) * self.config.games_per_iteration,
                "gate": iteration["gate"]["report"] if iteration and iteration["gate"] else None,
                "reference": (iteration.get("reference") or {}).get("report")
                if iteration
                else None,
                "rating_protocol": self.state["rating_protocol"],
            },
        )
        self.log(f"RECOVERY {kind}: {reference['directory']}")
        return reference

    def export(self, checkpoint, directory):
        saved = load_training(self.check(checkpoint))
        output = self.path(directory)
        if not output.exists():
            self.log(f"export {checkpoint['path']} -> {directory}")
            self.execute(
                [
                    sys.executable,
                    "-m",
                    "continugo_train.export",
                    "--checkpoint",
                    str(self.check(checkpoint)),
                    "--output",
                    str(output),
                    "--threads",
                    str(self.config.threads),
                ],
                output.parent / f"{output.name}-export.log",
            )
        # Covers interruption after atomic directory rename, before loop commit.
        verify_export(output, self.check(checkpoint), saved)
        return {
            "directory": directory,
            "checkpoint_sha256": checkpoint["sha256"],
            "network_name": network_name(saved),
            "encoder": self.record(output / "encoder.onnx"),
            "policy": self.record(output / "policy.onnx"),
        }

    def plan(self, iteration, index):
        cycle = weighted_sizes(self.config.sizes, self.config.size_weights)
        size = cycle[
            ((iteration["number"] - 1) * self.config.games_per_iteration + index) % len(cycle)
        ]
        seed = (
            self.config.seed + (iteration["number"] - 1) * self.config.games_per_iteration + index
        )
        return size, seed

    def adopt_games(self, iteration):
        density = self.state["feature_spec"]["samples_per_diameter"]
        for attempt in iteration["selfplay_attempts"]:
            if attempt.get("adopted"):
                continue
            for local, index in enumerate(attempt["indices"], 1):
                if iteration["games"][index]:
                    continue
                size, seed = self.plan(iteration, index)
                path = self.path(attempt["path"]) / f"game-{local:06d}-size{size}-seed{seed}.sgf"
                if not path.exists():
                    if attempt["status"] == "succeeded":
                        raise ValueError(f"successful self-play omitted {path}")
                    continue
                try:
                    info = inspect_game(path, density, self.config.feature_binary)
                except ValueError as error:
                    if attempt["status"] == "succeeded":
                        raise
                    self.log(f"retain invalid partial record without adopting: {path}: {error}")
                    continue
                root = info.get("root_comment")
                if (
                    not isinstance(root, dict)
                    or info["board_size"] != size
                    or root.get("seed") != seed
                    or root.get("model") != iteration["parent_model"]["checkpoint_sha256"]
                    or root.get("opponent") is not None
                    or root.get("search") != self.config.search()
                    or not 0 < info["moves"] <= self.config.max_moves
                    or info.get("policy_visit_totals") != [self.config.simulations] * info["moves"]
                    or info["completed"] != info["labeled"]
                    or not info["completed"]
                    and info["moves"] != self.config.max_moves
                ):
                    raise ValueError(
                        f"self-play record does not match its planned job/model: {path}"
                    )
                iteration["games"][index] = {
                    **self.record(path),
                    "game_id": info["game_id"],
                    "board_size": size,
                    "seed": seed,
                    "moves": info["moves"],
                    "eligible": bool(info["eligible_plies"]),
                    "eligible_positions": len(info["eligible_plies"]),
                    "result": info["result"],
                }
                if info["result"] == "0":
                    self.log(f"WARNING DRAW in self-play: {path}")
                if not info["labeled"]:
                    self.log(f"WARNING TRUNCATED self-play game: {path}")
                self.save()
            attempt["adopted"] = True
            self.save()

    def selfplay(self, iteration):
        self.adopt_games(iteration)
        missing = [index for index, record in enumerate(iteration["games"]) if record is None]
        # Contiguous jobs preserve native size cycling and seed/index assignment.
        groups = []
        for index in missing:
            if not groups or index != groups[-1][-1] + 1:
                groups.append([])
            groups[-1].append(index)
        for indices in groups:
            number = len(iteration["selfplay_attempts"]) + 1
            relative = f"iteration-{iteration['number']:04d}/selfplay/attempt-{number:04d}"
            directory = self.path(relative)
            directory.mkdir(parents=True, exist_ok=False)
            attempt = {"path": relative, "indices": indices, "status": "running", "adopted": False}
            iteration["selfplay_attempts"].append(attempt)
            self.save()
            cycle = weighted_sizes(self.config.sizes, self.config.size_weights)
            rotation = (
                (iteration["number"] - 1) * self.config.games_per_iteration + indices[0]
            ) % len(cycle)
            sizes = cycle[rotation:] + cycle[:rotation]
            _, seed = self.plan(iteration, indices[0])
            library, environment = inference_environment(
                self.config.device, self.config.ort_library
            )
            command = [
                self.config.selfplay_binary,
                "--engine",
                "neural",
                "--device",
                self.config.device,
                "--ort-library",
                str(library),
                "--model",
                str(self.check_model(iteration["parent_model"])),
                "--games",
                str(len(indices)),
                "--sizes",
                ",".join(map(str, sizes)),
                "--size-weights",
                "uniform",
                "--seed",
                str(seed),
                "--max-moves",
                str(self.config.max_moves),
                "--output",
                str(directory),
            ]
            settings = {
                **self.config.search(),
                **{
                    name: getattr(self.config, name)
                    for name in [
                        "concurrent_games",
                        "inference_batch_size",
                        "inference_batch_wait_ms",
                        "inference_threads",
                        "feature_threads",
                        "cache_positions",
                    ]
                },
            }
            for name, value in settings.items():
                command += ["--" + name.replace("_", "-"), str(value)]
            self.log(
                f"iteration {iteration['number']}: self-play {len(indices)} missing games, seed={seed}"
            )
            try:
                self.execute(command, directory / "selfplay.log", environment)
            except BaseException:
                attempt["status"] = "interrupted"
                self.save()
                raise
            attempt["status"] = "succeeded"
            self.save()
            self.adopt_games(iteration)
        initial = self.state["initial_data_positions"]
        iteration["data_positions"] = (
            initial
            + sum(
                record["eligible_positions"]
                for previous in self.state["iterations"][: iteration["number"]]
                for record in previous["games"]
            )
            if initial is not None
            else None
        )
        self.save()
        if not any(record["eligible"] for record in iteration["games"]):
            raise ValueError(
                "iteration has no completed policy-labeled games; use a larger move budget in a new loop"
            )

    def replay(self, iteration, parent):
        options = self.config.training_options(parent, iteration["number"])
        if iteration["replay"] is None:
            recent = self.state["iterations"][
                max(0, iteration["number"] - self.config.replay_iterations) : iteration["number"]
            ]
            candidates = [record for previous in reversed(recent) for record in previous["games"]]
            selected = select_replay(
                [*candidates, *self.state["bootstrap"]],
                self.config.replay_games,
                self.config.size_weights,
            )
        else:
            selected = iteration["replay"]
        paths = [str(self.check(record)) for record in selected]
        if not paths:
            raise ValueError("no eligible replay games")
        corpus = scan_corpus(
            paths,
            options.samples_per_diameter,
            options.seed,
            options.validation_fraction,
            self.config.feature_binary,
            split_history=checkpoint_splits(parent),
        )
        expected_hashes = {str(self.check(r)): r["sha256"] for r in selected}
        if any(expected_hashes[g.path] != g.file_sha256 for g in corpus.train + corpus.validation):
            raise ValueError("replay changed during selection")
        if (
            iteration["replay"] is not None
            and iteration["corpus_fingerprint"] != corpus.fingerprint
        ):
            raise ValueError("frozen replay corpus/split changed")
        if iteration["replay"] is None:
            iteration["replay"] = selected
            iteration["corpus_fingerprint"] = corpus.fingerprint
            self.save()
            self.log(
                f"iteration {iteration['number']}: replay {len(selected)} unique games; train={len(corpus.train)} validation={len(corpus.validation)}"
            )
        return options

    def generation_progress(self, iteration):
        expected = {
            "version": 1,
            "data_positions": iteration["data_positions"],
            "game_ids": sorted(
                {
                    record["game_id"]
                    for record in [*self.state["bootstrap"], *iteration["games"]]
                    if record["eligible"]
                }
            ),
        }
        path = self.path(f"iteration-{iteration['number']:04d}/generation-progress.pt")
        if iteration.get("generated_data"):
            self.check(iteration["generated_data"])
        if path.exists():
            if torch.load(path, map_location="cpu", weights_only=True) != expected:
                raise ValueError("frozen generation progress changed")
        else:
            atomic_checkpoint(path, expected)
        if not iteration.get("generated_data"):
            iteration["generated_data"] = self.record(path)
            self.save()
        return path

    def epoch_snapshots(self, iteration, attempt, parent, options):
        directory = self.path(attempt["path"])
        snapshots = sorted(
            directory.glob("epoch-[0-9]*.pt"), key=lambda p: int(p.stem.split("-")[1])
        )
        run_id = None
        ancestral_splits = checkpoint_splits(parent)
        expected_progress = checkpoint_progress(parent)
        expected_progress["data_positions"] = iteration["data_positions"]
        generated = torch.load(self.generation_progress(iteration), weights_only=True)
        expected_data_ids = data_history(parent, ancestral_splits) | set(generated["game_ids"])
        for number, path in enumerate(snapshots, 1):
            saved = load_training(path)
            history = checkpoint_splits(saved)
            if expected_progress["samples_seen"] is not None:
                expected_progress["samples_seen"] += saved["metrics"]["train"]["samples"]
            if (
                checkpoint_progress(saved) != expected_progress
                or data_history(saved, history) != expected_data_ids
                or path.name != f"epoch-{number:04d}.pt"
                or saved["epoch"] != number
                or number > self.config.train_epochs
                or saved["options"] != asdict(options)
                or NetworkConfig(**saved["config"]) != NetworkConfig(**parent["config"])
                or saved["feature_spec"] != parent["feature_spec"]
                or saved["corpus_fingerprint"] != iteration["corpus_fingerprint"]
                or saved.get("initialization", {}).get("source", {}).get("sha256")
                != iteration["parent_checkpoint"]["sha256"]
                or saved.get("initialization", {}).get("optimizer")
                != self.config.warm_start_optimizer
                or saved.get("initialization", {}).get("mode") != "warm_start"
                or saved["run_id"] == parent["run_id"]
                or any(
                    history.get(identity) != split for identity, split in ancestral_splits.items()
                )
                or run_id is not None
                and saved["run_id"] != run_id
            ):
                raise ValueError(f"training epoch does not match its frozen job: {path}")
            run_id = saved["run_id"]
        return snapshots

    def clear_stale_train_lock(self, directory):
        path = directory / ".train.lock"
        if path.exists():
            pid = int(path.read_text())
            if pid <= 0:
                raise ValueError(f"invalid training lock: {path}")
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                path.unlink()
            else:
                raise ValueError(f"trainer PID {pid} may still be alive; refusing to remove {path}")

    def training(self, iteration, parent, options):
        if iteration["checkpoint"]:
            self.check(iteration["checkpoint"])
            return
        attempt = iteration["train_attempts"][-1] if iteration["train_attempts"] else None
        snapshots = self.epoch_snapshots(iteration, attempt, parent, options) if attempt else []
        if len(snapshots) != self.config.train_epochs:
            if not snapshots:
                # A failed first epoch has no resumable state; preserve the attempt.
                relative = f"iteration-{iteration['number']:04d}/train-attempt-{len(iteration['train_attempts']) + 1:04d}"
                attempt = {"path": relative}
                iteration["train_attempts"].append(attempt)
                self.save()
            else:
                self.clear_stale_train_lock(self.path(attempt["path"]))
            directory = self.path(attempt["path"])
            command = [sys.executable, "-m", "continugo_train.train", "--data"]
            command += [str(self.check(record)) for record in iteration["replay"]]
            command += [
                "--output",
                str(directory),
                "--epochs",
                str(self.config.train_epochs),
                "--device",
                self.config.device,
                "--threads",
                str(self.config.threads),
                "--feature-binary",
                self.config.feature_binary,
            ]
            command += ["--generated-data", str(self.generation_progress(iteration))]
            for name, value in asdict(options).items():
                command += ["--" + name.replace("_", "-"), str(value)]
            if snapshots:
                command += ["--resume", str(snapshots[-1])]
            else:
                command += [
                    "--warm-start",
                    str(self.check(iteration["parent_checkpoint"])),
                    "--warm-start-optimizer",
                    self.config.warm_start_optimizer,
                ]
            self.log(
                f"iteration {iteration['number']}: train from epoch {len(snapshots)} to {self.config.train_epochs}"
            )
            self.execute(command, directory.with_suffix(".log"))
            snapshots = self.epoch_snapshots(iteration, attempt, parent, options)
            if len(snapshots) != self.config.train_epochs:
                raise ValueError("trainer exited without the requested completed epoch snapshots")
        # Epoch snapshots, not possibly lagging latest.pt/best.pt aliases, are authoritative.
        iteration["checkpoint"] = self.record(snapshots[-1])
        self.save()

    def rate_evaluation(self, job, key, checkpoints):
        match = job[key]
        if not match["games"] or match.get("rating_event"):
            return
        result = ratings.ingest(
            self.config.model_store,
            self.state["rating_protocol"],
            [self.check(record) for record in match["games"]],
            checkpoints=[self.check(record) for record in checkpoints],
            kind=key,
            decision=match["report"].get("decision"),
            rating_binary=self.config.rating_binary,
            feature_binary=self.config.feature_binary,
        )
        match["rating_event"] = result
        self.save()
        self.log(f"RATINGS {key}: {result}")

    def adaptive_evaluation(self, iteration):
        """Precision-targeted evaluation of this exact candidate, never actor selection."""
        if iteration.get("adaptive_rating") is not None:
            return iteration["adaptive_rating"]
        from . import adaptive_ratings

        with ratings.transaction(self.config.model_store) as (tables, _):
            count = sum(n != ratings.REFERENCE_ID for n in tables.networks)
        if count <= self.config.adaptive_min_opponents:
            report = {
                "stop_reason": "insufficient-opponents; confidence target NOT met",
                "games": 0,
            }
        else:
            relative = f"iteration-{iteration['number']:04d}/adaptive-ratings"
            output = self.path(relative)
            args = ["--output", str(output)]
            if (output / "loop.pt").exists():
                args += ["--resume"]
            else:
                # A separate high-bit stream keeps these games distinct from both
                # generation/gate seeds and the one-off backfill evaluator.
                seed = (
                    (1 << 63)
                    + self.config.seed
                    + (iteration["number"] - 1) * (self.config.adaptive_max_games // 2)
                )
                args += [
                    "--loop",
                    str(self.root),
                    "--targets",
                    iteration["checkpoint"]["sha256"],
                    "--latest",
                    "1",
                    "--seed",
                    str(seed),
                ]
                for option, value in {
                    "max-games": self.config.adaptive_max_games,
                    "target-rd": self.config.adaptive_target_rd,
                    "min-pairs": self.config.adaptive_min_pairs,
                    "min-opponents": self.config.adaptive_min_opponents,
                    "min-opponent-pairs": self.config.adaptive_min_opponent_pairs,
                    "batch-games": self.config.adaptive_batch_games,
                }.items():
                    args += ["--" + option, str(value)]
            self.verify_runtime()
            result = adaptive_ratings.run(
                adaptive_ratings.parse_args(args), parent_lock_fd=self.lock_fd
            )
            self.verify_runtime()
            if result["targets"] != [iteration["checkpoint"]["sha256"]]:
                raise ValueError("adaptive evaluator targeted the wrong candidate")
            report = {
                "path": relative,
                "stop_reason": result["stop_reason"],
                "progress": result["progress"],
                "games": len(result["jobs"]) * result["config"]["gate_games"],
            }
        iteration["adaptive_rating"] = report
        self.save()
        self.log(f"ADAPTIVE RATINGS iteration={iteration['number']}: {report}")
        return report

    def actor_pointer(self):
        archive.latest_pointer(
            Path(self.config.model_store),
            "experimental",
            Path(self.state["actor"]["archive"]["directory"]),
        )

    def run(self, stop_after_iteration=None):
        self.verify_committed()
        # Initial checkpoints are unaccepted experimental inputs, not implicit
        # permission to publish an accepted baseline.
        if self.state["initial_model"] is None:
            self.state["initial_model"] = self.export(
                self.state["initial_actor_checkpoint"], "initial-model"
            )
            self.save()
        if self.state["initial_actor_training_archive"] is None:
            self.state["initial_actor_training_archive"] = self.recovery(
                self.state["initial_actor_checkpoint"], "training"
            )
            self.save()
        if self.state["initial_training_archive"] is None:
            self.state["initial_training_archive"] = self.recovery(
                self.state["initial_checkpoint"], "training"
            )
            self.save()
        if self.state["initial_experimental_archive"] is None:
            self.state["initial_experimental_archive"] = self.recovery(
                self.state["initial_actor_checkpoint"],
                "experimental",
                model=self.state["initial_model"],
            )
            self.save()
        if self.state["training"] is None:
            self.state["training"] = {
                "checkpoint": self.state["initial_checkpoint"],
                "archive": self.state["initial_training_archive"],
            }
        if self.state["actor"] is None:
            self.state["actor"] = {
                "checkpoint": self.state["initial_actor_checkpoint"],
                "model": self.state["initial_model"],
                "archive": self.state["initial_experimental_archive"],
            }
        self.state["accepted"] = promote.accepted_head(self.config.model_store)
        self.save()
        self.actor_pointer()
        ratings.refresh(self.config.model_store)
        if self.state.get("initial_reference") is None:
            self.state["initial_reference"] = {
                "number": 1,
                "initial": True,
                "model": self.state["initial_model"],
                "reference": None,
            }
            self.save()
        initial = self.state["initial_reference"]
        if self.config.reference_games:
            ratings.register_checkpoint(
                self.config.model_store, self.check(self.state["initial_actor_checkpoint"])
            )
        gate.evaluate(self, initial, reference=True)
        self.rate_evaluation(initial, "reference", [self.state["initial_actor_checkpoint"]])
        stop = min(
            stop_after_iteration or self.state["target_iterations"], self.state["target_iterations"]
        )
        for number in range(1, stop + 1):
            if len(self.state["iterations"]) < number:
                self.state["iterations"].append(
                    {
                        "number": number,
                        "parent_checkpoint": self.state["training"]["checkpoint"],
                        "parent_actor_checkpoint": self.state["actor"]["checkpoint"],
                        "parent_model": self.state["actor"]["model"],
                        "games": [None] * self.config.games_per_iteration,
                        "selfplay_attempts": [],
                        "replay": None,
                        "corpus_fingerprint": None,
                        "train_attempts": [],
                        "checkpoint": None,
                        "model": None,
                        "gate": None,
                        "reference": None,
                        "decision": None,
                        "training_archive": None,
                        "experimental_archive": None,
                        "actor_updated": None,
                        "completed": False,
                    }
                )
                self.save()
            iteration = self.state["iterations"][number - 1]
            if iteration["completed"]:
                continue
            parent = load_training(self.check(iteration["parent_checkpoint"]))
            self.selfplay(iteration)
            plot_metrics.update(self.root)
            options = self.replay(iteration, parent)
            self.training(iteration, parent, options)
            plot_metrics.update(self.root)
            if iteration["training_archive"] is None:
                iteration["training_archive"] = self.recovery(
                    iteration["checkpoint"], "training", iteration=iteration
                )
                self.save()
            self.state["training"] = {
                "checkpoint": iteration["checkpoint"],
                "archive": iteration["training_archive"],
            }
            self.save()
            if iteration["model"] is None:
                iteration["model"] = self.export(
                    iteration["checkpoint"], f"iteration-{number:04d}/model"
                )
                self.save()
            if number % self.config.evaluation_interval == 0:
                if (
                    self.config.gate_games
                    or self.config.reference_games
                    or self.config.adaptive_max_games
                ):
                    ratings.register_checkpoint(
                        self.config.model_store, self.check(iteration["checkpoint"])
                    )
                if self.config.gate_games:
                    ratings.register_checkpoint(
                        self.config.model_store, self.check(iteration["parent_actor_checkpoint"])
                    )
                report = gate.evaluate(self, iteration)
                self.rate_evaluation(
                    iteration,
                    "gate",
                    [iteration["checkpoint"], iteration["parent_actor_checkpoint"]],
                )
                gate.evaluate(self, iteration, reference=True)
                self.rate_evaluation(iteration, "reference", [iteration["checkpoint"]])
                decision = report["decision"]
            else:
                decision = "not-evaluated"
            if iteration["experimental_archive"] is None:
                iteration["experimental_archive"] = self.recovery(
                    iteration["checkpoint"],
                    "experimental",
                    model=iteration["model"],
                    iteration=iteration,
                )
                self.save()
            if self.config.adaptive_max_games and number % self.config.evaluation_interval == 0:
                # The immutable inference bundle must exist before the adaptive
                # evaluator snapshots its candidate/opponent pool. Ledger SGFs,
                # not training replay, retain its subsequent rated evidence.
                self.adaptive_evaluation(iteration)
            update_actor = self.config.actor_policy == "latest" or decision in {
                "passed",
                "disabled",
            }
            if update_actor:
                self.state["actor"] = {
                    "checkpoint": iteration["checkpoint"],
                    "model": iteration["model"],
                    "archive": iteration["experimental_archive"],
                }
            iteration["actor_updated"] = update_actor
            iteration["decision"] = decision
            iteration["completed"] = True
            # Reading an approved publication never switches training or actors.
            self.state["accepted"] = promote.accepted_head(self.config.model_store)
            self.save()
            self.actor_pointer()
            self.log(
                f"iteration {number} complete: match={decision}; experimental actor {'advanced' if update_actor else 'held'}; "
                f"training progress retained at {iteration['checkpoint']['path']}; no automatic promotion"
            )
        if stop < self.state["target_iterations"]:
            self.log(
                f"loop paused after iteration {stop}; total target remains {self.state['target_iterations']}; "
                "resume without --stop-after-iteration to continue"
            )
        else:
            self.log(
                f"loop complete: {self.state['target_iterations']} iterations; accepted publication requires explicit user approval"
            )
        return self.state


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="new runs default to runs/YYYYMMDD-HHMMSS-loop; required with --resume",
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="initial offline-training checkpoint (new loop only)"
    )
    parser.add_argument(
        "--actor-checkpoint",
        "--incumbent-checkpoint",
        dest="actor_checkpoint",
        type=Path,
        help="optional different initial experimental self-play checkpoint; never marks it accepted",
    )
    parser.add_argument(
        "--bootstrap-data", nargs="+", help="optional SGFs snapshotted into a new loop"
    )
    parser.add_argument(
        "--resume", action="store_true", help="resume the existing loop.pt with frozen settings"
    )
    parser.add_argument(
        "--iterations", type=int, help="total target; default 1, or saved target on resume"
    )
    parser.add_argument(
        "--enable-head-to-head",
        type=int,
        metavar="GAMES",
        help="on resume at a completed round boundary, enable rated actor matches without changing latest-actor selection",
    )
    parser.add_argument(
        "--enable-adaptive-ratings",
        type=int,
        metavar="MAX_GAMES",
        help="on resume at a completed round boundary, enable automatic adaptive evaluation for future scheduled candidates",
    )
    parser.add_argument(
        "--stop-after-iteration",
        type=int,
        help="exit cleanly after this iteration (including evaluation/archive); does not lower the saved total target",
    )
    for field in fields(LoopConfig):
        name = "--" + field.name.replace("_", "-")
        if field.name == "sizes":
            parser.add_argument(name, type=lambda s: tuple(int(size) for size in s.split(",")))
        else:
            kind = (
                float
                if field.name == "lr"
                else str
                if field.default is None
                else type(field.default)
            )
            parser.add_argument(name, type=kind, help=f"default: {field.default}")
    return parser.parse_args(argv)


def run_loop(args):
    if args.output is None:
        if args.resume:
            raise ValueError("--resume requires --output naming the existing loop")
        if args.checkpoint is None:
            raise ValueError("a new loop requires --checkpoint")
        root = new_run_directory("loop")
        print(f"Loop output: {root}", flush=True)
    else:
        root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with loop_lock(root) as lock_fd:
        head_to_head_backup = None
        adaptive_backup = None
        if (
            args.enable_head_to_head is not None or args.enable_adaptive_ratings is not None
        ) and not args.resume:
            raise ValueError("--enable-head-to-head/--enable-adaptive-ratings require --resume")
        if args.resume:
            if (
                args.checkpoint is not None
                or args.bootstrap_data is not None
                or args.actor_checkpoint is not None
            ):
                raise ValueError(
                    "--checkpoint/--bootstrap-data are new-loop inputs, not resume overrides"
                )
            state = torch.load(root / "loop.pt", map_location="cpu", weights_only=True)
            if (
                not isinstance(state, dict)
                or state.get("version") != VERSION
                or state.get("root") != str(root)
            ):
                raise ValueError(
                    "incompatible/relocated loop checkpoint; start a new v6 loop with a score/ownership checkpoint"
                )
            for field in fields(LoopConfig):
                # Preserve frozen old-loop behavior; the enabled default applies
                # to new campaigns, not implicit upgrades of saved loops.
                if field.name.startswith("adaptive_"):
                    state["config"].setdefault(
                        field.name, 0 if field.name == "adaptive_max_games" else field.default
                    )
                value = getattr(args, field.name)
                if value is not None and value != state["config"][field.name]:
                    raise ValueError(
                        f"resume restores --{field.name.replace('_', '-')}; conflicting override"
                    )
            if args.enable_head_to_head is not None:
                count = args.enable_head_to_head
                if not 0 < count < 2**32 or count % 2:
                    raise ValueError("--enable-head-to-head needs a positive even game count")
                if state["config"]["actor_policy"] != "latest":
                    raise ValueError("--enable-head-to-head preserves latest-actor selection only")
                if any(not item["completed"] for item in state["iterations"]):
                    raise ValueError("enable head-to-head only at a completed round boundary")
                previous = state["config"]["gate_games"]
                if previous not in {0, count}:
                    raise ValueError("head-to-head matches are already enabled with another count")
                if previous == 0:
                    head_to_head_backup = (root / "loop.pt").read_bytes()
                    state.setdefault("config_changes", []).append(
                        {
                            "after_iteration": len(state["iterations"]),
                            "setting": "gate_games",
                            "before": 0,
                            "after": count,
                            "reason": "enable rated head-to-head evaluation; keep latest actor",
                        }
                    )
                    state["config"]["gate_games"] = count
            if args.enable_adaptive_ratings is not None:
                budget = args.enable_adaptive_ratings
                if budget <= 0:
                    raise ValueError("--enable-adaptive-ratings needs a positive game budget")
                if state["config"]["actor_policy"] != "latest":
                    raise ValueError(
                        "enable adaptive ratings without changing latest-actor selection only"
                    )
                if any(not item["completed"] for item in state["iterations"]):
                    raise ValueError("enable adaptive ratings only at a completed round boundary")
                previous = state["config"]["adaptive_max_games"]
                if previous not in {0, budget}:
                    raise ValueError("adaptive ratings already enabled with another budget")
                if previous == 0:
                    adaptive_backup = (root / "loop.pt").read_bytes()
                    state.setdefault("config_changes", []).append(
                        {
                            "after_iteration": len(state["iterations"]),
                            "setting": "adaptive_max_games",
                            "before": 0,
                            "after": budget,
                            "reason": "automatically adaptively rate future scheduled candidates; keep latest actor",
                        }
                    )
                    state["config"]["adaptive_max_games"] = budget
            target = args.iterations if args.iterations is not None else state["target_iterations"]
            if target < state["target_iterations"]:
                raise ValueError("iteration target may only stay unchanged or increase")
            state["target_iterations"] = target
        else:
            if any(path.name != ".loop.lock" for path in root.iterdir()):
                raise ValueError("output is not empty; choose a new loop or --resume")
            if args.checkpoint is None:
                raise ValueError("a new loop requires --checkpoint")
            supplied = {
                field.name: getattr(args, field.name)
                for field in fields(LoopConfig)
                if getattr(args, field.name) is not None
            }
            for name in [
                "selfplay_binary",
                "feature_binary",
                "ort_library",
                "arena_binary",
                "rating_binary",
                "model_store",
            ]:
                if name in supplied:
                    supplied[name] = str(Path(supplied[name]).resolve())
            config = LoopConfig(**supplied)
            config.validate()
            source = args.checkpoint.resolve()
            content = source.read_bytes()
            saved = load_training(io.BytesIO(content))
            initial_hash = hashlib.sha256(content).hexdigest()
            del content
            actor_source = (args.actor_checkpoint or args.checkpoint).resolve()
            if actor_source == source:
                actor_record = {
                    "path": "initial.pt",
                    "sha256": initial_hash,
                    "source": str(source),
                }
            else:
                content = actor_source.read_bytes()
                actor_saved = load_training(io.BytesIO(content))
                if actor_saved["feature_spec"] != saved["feature_spec"]:
                    raise ValueError(
                        "actor and training checkpoints need identical feature semantics/density"
                    )
                actor_record = {
                    "path": "initial-actor.pt",
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "source": str(actor_source),
                }
                del content
            bootstrap, skipped = (
                inspect_corpus(
                    args.bootstrap_data,
                    saved["feature_spec"]["samples_per_diameter"],
                    config.feature_binary,
                )
                if args.bootstrap_data
                else ([], {})
            )
            state = {
                "version": VERSION,
                "root": str(root),
                "config": asdict(config),
                "target_iterations": args.iterations if args.iterations is not None else 1,
                "feature_spec": saved["feature_spec"],
                "snapshots_ready": False,
                "initial_checkpoint": {
                    "path": "initial.pt",
                    "sha256": initial_hash,
                    "source": str(source),
                },
                "initial_model": None,
                "initial_data_positions": initialize_progress(
                    saved, bootstrap, data_history(saved, checkpoint_splits(saved))
                )["data_positions"],
                "initial_actor_checkpoint": actor_record,
                "initial_training_archive": None,
                "initial_actor_training_archive": None,
                "initial_experimental_archive": None,
                "training": None,
                "actor": None,
                "accepted": None,
                "bootstrap": [],
                "bootstrap_skipped": skipped,
                "iterations": [],
            }
            for index, game in enumerate(bootstrap):
                state["bootstrap"].append(
                    {
                        "path": f"bootstrap/game-{index:06d}.sgf",
                        "sha256": game.file_sha256,
                        "source": game.path,
                        "game_id": game.game_id,
                        "board_size": game.board_size,
                        "eligible_positions": len(game.plies),
                        "eligible": True,
                    }
                )
        config = LoopConfig(**state["config"])
        config.validate()
        store = Path(config.model_store).resolve()
        shared = (ROOT / "models").resolve()
        if store != shared and store.is_relative_to(shared):
            raise ValueError(
                "all campaigns share models/; per-training model-store subdirectories are not supported"
            )
        if (
            store.is_relative_to(root)
            or root.is_relative_to(store)
            or store.is_relative_to(ROOT / "runs")
        ):
            raise ValueError(
                "model-store must be separate from the loop output and outside disposable runs/"
            )
        target = state["target_iterations"]
        if (
            target <= 0
            or config.seed
            + target
            * max(config.games_per_iteration, config.gate_games // 2, config.reference_games // 2)
            > 2**64
        ):
            raise ValueError("positive iteration target and nonoverflowing u64 game seeds required")
        if config.adaptive_max_games and (
            config.seed + target * (config.adaptive_max_games // 2) >= 2**63
        ):
            raise ValueError("automatic adaptive rating seeds overflow their reserved u64 stream")
        if args.stop_after_iteration is not None and (
            args.stop_after_iteration <= 0
            or args.stop_after_iteration > target
            or len(state["iterations"]) > args.stop_after_iteration
        ):
            raise ValueError(
                "stop-after-iteration must be positive, within the total target, and not precede recorded work"
            )
        if not Path(config.selfplay_binary).is_file():
            raise ValueError("self-play binary missing; build release binaries first")
        if config.gate_games or config.reference_games or config.adaptive_max_games:
            if not Path(config.arena_binary).is_file() or not Path(config.rating_binary).is_file():
                raise ValueError("arena/rating binary missing; build release binaries first")
            current_protocol = ratings.protocol(config)
        else:
            current_protocol = None
        if args.resume and state.get("rating_protocol") != current_protocol:
            raise ValueError(
                "evaluation/rating executable changed; fork a new loop instead of mixing protocols"
            )
        state["rating_protocol"] = current_protocol
        feature_binary(config.feature_binary)
        fingerprints = {
            "selfplay": digest(config.selfplay_binary),
            "features": digest(config.feature_binary),
        }
        if args.resume and state.get("runtime_fingerprints") != fingerprints:
            raise ValueError(
                "generation/feature executable changed; fork a new loop instead of mixing protocols"
            )
        state["runtime_fingerprints"] = fingerprints
        torch.set_num_threads(config.threads)
        runner = LoopRunner(root, state, lock_fd)
        if head_to_head_backup is not None:
            backup = root / "before-head-to-head.pt"
            if backup.exists():
                raise ValueError(f"refusing to overwrite previous configuration backup: {backup}")
            atomic_bytes(backup, head_to_head_backup)
        if adaptive_backup is not None:
            backup = root / "before-adaptive-ratings.pt"
            if backup.exists():
                raise ValueError(f"refusing to overwrite previous configuration backup: {backup}")
            atomic_bytes(backup, adaptive_backup)
        runner.save()
        return runner.run(args.stop_after_iteration)


def main(argv=None):
    try:
        run_loop(parse_args(argv))
    except KeyboardInterrupt:
        print("Loop interrupted; restart with --resume and the same --output.", file=sys.stderr)
        raise SystemExit(130) from None
    except (ValueError, OSError, RuntimeError) as error:
        print(f"loop error: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
