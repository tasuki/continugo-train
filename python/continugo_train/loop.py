"""Restartable native self-play -> bounded replay -> warm-start -> ONNX loop.

SGFs remain the game records. loop.pt is an atomic orchestration checkpoint, not
an alternate game format. This initial loop promotes latest weights, not strength.
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

from .data import inspect_corpus, scan_corpus
from .features import feature_binary, inspect_game
from .model import NetworkConfig, PolicyValueNet
from .runtime import inference_environment
from .train import Options, atomic_checkpoint, checkpoint_splits, feature_spec

VERSION = 1
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class LoopConfig:
    sizes: tuple[int, ...] = (9, 13, 19)
    games_per_iteration: int = 12
    seed: int = 0
    simulations: int = 32
    max_moves: int = 512
    cpuct: float = 1.5
    widening: float = 2.0
    widening_exponent: float = 0.5
    proposal_attempts: int = 128
    independent_exploration: float = 0.25
    concurrent_games: int = 6
    inference_batch_size: int = 8
    inference_batch_wait_ms: int = 2
    inference_threads: int = 1
    feature_threads: int = 4
    cache_positions: int = 64
    replay_iterations: int = 4
    replay_games: int = 256
    train_epochs: int = 1
    steps_per_epoch: int = 20
    positions_per_game: int = 8
    validation_positions_per_game: int = 4
    batch_size: int = 2
    workers: int = 2
    shuffle_buffer: int = 8
    validation_fraction: float = 0.2
    lr: float | None = None
    warm_start_optimizer: str = "keep"
    device: str = "cuda"
    threads: int = 4
    selfplay_binary: str = str(ROOT / "target/release/continugo-train")
    feature_binary: str = str(ROOT / "target/release/features")
    ort_library: str | None = None

    def validate(self):
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
            or not 0 < self.validation_fraction < 1
            or self.lr is not None
            and (not math.isfinite(self.lr) or self.lr <= 0)
        ):
            raise ValueError("invalid search, validation fraction or learning rate")
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
            pass_fds=(lock_fd,),
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
    checkpoint_splits(saved)
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


def select_replay(records, limit):
    """Newest candidates first, balanced round-robin by board size, unique plays."""
    buckets = defaultdict(deque)
    seen = set()
    for record in records:
        if record["eligible"] and record["game_id"] not in seen:
            seen.add(record["game_id"])
            buckets[record["board_size"]].append(record)
    selected = []
    while buckets and len(selected) < limit:
        for size in sorted(buckets):
            selected.append(buckets[size].popleft())
            if not buckets[size]:
                del buckets[size]
            if len(selected) == limit:
                break
    return selected


class LoopRunner:
    def __init__(self, root, state, lock_fd):
        self.root, self.state, self.lock_fd = root, state, lock_fd
        self.config = LoopConfig(**state["config"])

    def save(self):
        atomic_checkpoint(self.root / "loop.pt", self.state)

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
        records = [self.state["initial_checkpoint"], *self.state["bootstrap"]]
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
            if iteration["checkpoint"]:
                self.check(iteration["checkpoint"])
            if iteration["model"]:
                self.check_model(iteration["model"])

    def export(self, checkpoint, directory):
        saved = load_training(self.check(checkpoint))
        output = self.path(directory)
        if not output.exists():
            self.log(f"export {checkpoint['path']} -> {directory}")
            run_process(
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
                self.lock_fd,
            )
        # Covers interruption after atomic directory rename, before loop commit.
        verify_export(output, self.check(checkpoint), saved)
        return {
            "directory": directory,
            "checkpoint_sha256": checkpoint["sha256"],
            "encoder": self.record(output / "encoder.onnx"),
            "policy": self.record(output / "policy.onnx"),
        }

    def plan(self, iteration, index):
        size = self.config.sizes[index % len(self.config.sizes)]
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
                    or root.get("model") != iteration["parent_checkpoint"]["sha256"]
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
                }
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
            rotation = indices[0] % len(self.config.sizes)
            sizes = self.config.sizes[rotation:] + self.config.sizes[:rotation]
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
                run_process(command, directory / "selfplay.log", self.lock_fd, environment)
            except BaseException:
                attempt["status"] = "interrupted"
                self.save()
                raise
            attempt["status"] = "succeeded"
            self.save()
            self.adopt_games(iteration)
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
                [*candidates, *self.state["bootstrap"]], self.config.replay_games
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

    def epoch_snapshots(self, iteration, attempt, parent, options):
        directory = self.path(attempt["path"])
        snapshots = sorted(
            directory.glob("epoch-[0-9]*.pt"), key=lambda p: int(p.stem.split("-")[1])
        )
        run_id = None
        ancestral_splits = checkpoint_splits(parent)
        for number, path in enumerate(snapshots, 1):
            saved = load_training(path)
            history = checkpoint_splits(saved)
            if (
                path.name != f"epoch-{number:04d}.pt"
                or saved["epoch"] != number
                or number > self.config.train_epochs
                or saved["options"] != asdict(options)
                or saved["config"] != parent["config"]
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
            run_process(command, directory.with_suffix(".log"), self.lock_fd)
            snapshots = self.epoch_snapshots(iteration, attempt, parent, options)
            if len(snapshots) != self.config.train_epochs:
                raise ValueError("trainer exited without the requested completed epoch snapshots")
        # Epoch snapshots, not possibly lagging latest.pt/best.pt aliases, are authoritative.
        iteration["checkpoint"] = self.record(snapshots[-1])
        self.save()

    def run(self):
        self.verify_committed()
        if self.state["initial_model"] is None:
            self.state["initial_model"] = self.export(
                self.state["initial_checkpoint"], "initial-model"
            )
            self.save()
        for number in range(1, self.state["target_iterations"] + 1):
            if len(self.state["iterations"]) < number:
                previous = self.state["iterations"][-1] if self.state["iterations"] else None
                self.state["iterations"].append(
                    {
                        "number": number,
                        "parent_checkpoint": previous["checkpoint"]
                        if previous
                        else self.state["initial_checkpoint"],
                        "parent_model": previous["model"]
                        if previous
                        else self.state["initial_model"],
                        "games": [None] * self.config.games_per_iteration,
                        "selfplay_attempts": [],
                        "replay": None,
                        "corpus_fingerprint": None,
                        "train_attempts": [],
                        "checkpoint": None,
                        "model": None,
                    }
                )
                self.save()
            iteration = self.state["iterations"][number - 1]
            if iteration["model"]:
                continue
            parent = load_training(self.check(iteration["parent_checkpoint"]))
            self.selfplay(iteration)
            options = self.replay(iteration, parent)
            self.training(iteration, parent, options)
            iteration["model"] = self.export(
                iteration["checkpoint"], f"iteration-{number:04d}/model"
            )
            self.save()
            self.log(
                f"iteration {number} complete; promoted latest epoch: {iteration['checkpoint']['path']}"
            )
        self.log(
            f"loop complete: {self.state['target_iterations']} iterations; no playing-strength claim"
        )
        return self.state


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, help="initial offline-training checkpoint (new loop only)"
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
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with loop_lock(root) as lock_fd:
        if args.resume:
            if args.checkpoint is not None or args.bootstrap_data is not None:
                raise ValueError(
                    "--checkpoint/--bootstrap-data are new-loop inputs, not resume overrides"
                )
            state = torch.load(root / "loop.pt", map_location="cpu", weights_only=True)
            if (
                not isinstance(state, dict)
                or state.get("version") != VERSION
                or state.get("root") != str(root)
            ):
                raise ValueError("incompatible or relocated loop checkpoint")
            for field in fields(LoopConfig):
                value = getattr(args, field.name)
                if value is not None and value != state["config"][field.name]:
                    raise ValueError(
                        f"resume restores --{field.name.replace('_', '-')}; conflicting override"
                    )
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
            for name in ["selfplay_binary", "feature_binary", "ort_library"]:
                if name in supplied:
                    supplied[name] = str(Path(supplied[name]).resolve())
            config = LoopConfig(**supplied)
            config.validate()
            source = args.checkpoint.resolve()
            content = source.read_bytes()
            saved = load_training(io.BytesIO(content))
            initial_hash = hashlib.sha256(content).hexdigest()
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
        target = state["target_iterations"]
        if target <= 0 or config.seed + target * config.games_per_iteration > 2**64:
            raise ValueError("positive iteration target and nonoverflowing u64 game seeds required")
        if not Path(config.selfplay_binary).is_file():
            raise ValueError("self-play binary missing; build release binaries first")
        feature_binary(config.feature_binary)
        torch.set_num_threads(config.threads)
        runner = LoopRunner(root, state, lock_fd)
        runner.save()
        return runner.run()


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
