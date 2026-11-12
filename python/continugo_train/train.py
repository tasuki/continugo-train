"""Offline FP32 policy/value training from completed, annotated SGFs.

No ONNX or neural self-play here. Rust owns game parsing, rules and features.
Checkpoints are atomic, at epoch boundaries; --epochs is the total epoch target.
--resume continues a fixed-corpus run; --warm-start initializes a new replay run.
"""

import argparse
import hashlib
import io
import math
import os
import random
import sys
import tempfile
import time
import uuid
from collections import Counter
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch

from .data import batches, scan_corpus
from .features import (
    COARSE_PLANES,
    MOVE_FEATURES,
    SCHEMA_VERSION,
    TARGET_VERSION,
    FeatureProfile,
    coarse_planes,
    feature_binary,
    move_features,
    ownership_weights,
)
from .losses import joint_loss
from .memory import GpuMemorySampler
from .model import POLICY_VERSION, NetworkConfig, PolicyValueNet
from .paths import new_run_directory
from .progress import data_history, network_name
from .progress import initialize as initialize_progress
from .sampling import DEFAULT_SIZE_WEIGHTS, parse_weights


@dataclass
class Options:
    batch_size: int = 32
    workers: int = 6
    shuffle_buffer: int = 8
    samples_per_diameter: int = 4
    positions_per_game: int = 0
    validation_positions_per_game: int = 0
    steps_per_epoch: int = 0
    seed: int = 0
    validation_fraction: float = 0.2
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    size_weights: str = DEFAULT_SIZE_WEIGHTS
    score_loss_weight: float = 0.25
    ownership_loss_weight: float = 0.25
    feature_cache: str = "auto"
    augmentation: str = "d4"

    def validate(self):
        from .feature_cache import MODES

        if self.feature_cache not in MODES:
            raise ValueError(f"feature_cache must be one of {MODES}")
        if self.augmentation not in ("none", "d4"):
            raise ValueError("augmentation must be none or d4")
        parse_weights(self.size_weights)
        if any(
            not math.isfinite(w) or w < 0
            for w in (self.score_loss_weight, self.ownership_loss_weight)
        ):
            raise ValueError("auxiliary loss weights must be finite and nonnegative")
        if self.batch_size <= 0 or self.samples_per_diameter <= 0:
            raise ValueError("batch size and sample density must be positive")
        if any(
            getattr(self, name) < 0
            for name in [
                "workers",
                "shuffle_buffer",
                "positions_per_game",
                "validation_positions_per_game",
                "steps_per_epoch",
                "seed",
            ]
        ):
            raise ValueError("worker/buffer/position/step limits and seed must be nonnegative")
        if (
            not 0 < self.validation_fraction < 1
            or not math.isfinite(self.lr)
            or self.lr <= 0
            or not 0 <= self.momentum < 1
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
            or not math.isfinite(self.grad_clip)
            or self.grad_clip < 0
        ):
            raise ValueError("invalid validation fraction or optimizer settings")


def feature_spec(density, *, schema=SCHEMA_VERSION):
    planes = coarse_planes(schema)
    return {
        "schema": schema,
        "policy": POLICY_VERSION,
        "targets": TARGET_VERSION,
        "coarse": list(planes),
        "moves": list(move_features(schema)),
        "samples_per_diameter": density,
    }


def checkpoint_splits(saved):
    """Conservative game-level holdout ledger, including older checkpoints."""
    history = saved.get("split_history", {})
    if (
        not isinstance(history, dict)
        or not isinstance(saved.get("manifest"), list)
        or not saved["manifest"]
    ):
        raise ValueError("checkpoint is missing a valid corpus/split manifest")
    history = dict(history)
    for game in saved["manifest"]:
        if not isinstance(game, dict) or not {"game_id", "split"}.issubset(game):
            raise ValueError("checkpoint contains an invalid corpus/split manifest")
        identity, split = game["game_id"], game["split"]
        if not isinstance(identity, str) or split not in ("train", "validation"):
            raise ValueError("checkpoint contains an invalid corpus/split manifest")
        if identity in history and history[identity] != split:
            raise ValueError("checkpoint contains conflicting split history")
        history[identity] = split
    if any(
        not isinstance(identity, str)
        or len(identity) != 64
        or any(c not in "0123456789abcdef" for c in identity)
        or split not in ("train", "validation")
        for identity, split in history.items()
    ):
        raise ValueError("checkpoint contains invalid split history")
    return history


def atomic_checkpoint(path, state):
    descriptor, name = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(state, name)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


@contextmanager
def run_lock(directory):
    path = directory / ".train.lock"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError(
            "run is locked; if no trainer is alive, remove .train.lock manually"
        ) from error
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


@contextmanager
def training_output(directory, resuming):
    with run_lock(directory):
        # Recheck freshness while holding the writer lock, before appending a log.
        if not resuming and any(path.name != ".train.lock" for path in directory.iterdir()):
            raise ValueError("output directory is not empty; choose a new run or --resume")
        with (directory / "training.log").open("a") as logfile:
            yield logfile


def parent_peak_rss():
    try:
        import resource
    except ImportError:
        return None
    divisor = 2**20 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_phase(
    model, optimizer, games, options, epoch, device, binary, training, *, profile_stages=False
):
    model.train(training)
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    samples = steps = policy_samples = 0
    samples_by_size = Counter()
    totals = [0.0] * 5
    score_abs = score_bias = consistency = 0.0
    prediction_sums = torch.zeros(3, dtype=torch.float64)
    prediction_squares = torch.zeros(3, dtype=torch.float64)
    wait = compute = 0.0
    stage_seconds = Counter()
    feature_profile = FeatureProfile() if profile_stages else None

    @contextmanager
    def stage(name):
        if not profile_stages:
            yield
            return
        sync(device)
        tick = time.perf_counter()
        yield
        sync(device)
        stage_seconds[name] += time.perf_counter() - tick

    started = time.perf_counter()
    stream = batches(
        games,
        batch_size=options.batch_size,
        seed=options.seed,
        epoch=epoch,
        training=training,
        density=options.samples_per_diameter,
        workers=options.workers,
        shuffle_buffer=options.shuffle_buffer if training else 0,
        limit=options.positions_per_game if training else options.validation_positions_per_game,
        binary=binary,
        size_weights=options.size_weights,
        feature_profile=feature_profile,
        feature_cache=options.feature_cache,
        augmentation=options.augmentation,
    )
    memory = GpuMemorySampler(enabled=profile_stages and device.type == "cuda")
    with memory, closing(stream), torch.set_grad_enabled(training):
        while not (training and options.steps_per_epoch and steps >= options.steps_per_epoch):
            tick = time.perf_counter()
            try:
                cpu_batch = next(stream)
            except StopIteration:
                break
            wait += time.perf_counter() - tick
            tick = time.perf_counter()
            samples_by_size.update(cpu_batch.board_sizes.tolist())
            with stage("transfer"):
                batch = cpu_batch.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            if profile_stages:
                # Same operations as PolicyValueNet.forward_all; synchronization is
                # diagnostic-only and may change overlap/throughput.
                with stage("encoder"):
                    spatial, pass_logit, values, scores, ownership = model.encode_training(
                        batch.coarse
                    )
                with stage("scorer"):
                    logits = model.policy(
                        spatial, pass_logit, batch.exact, batch.actions, batch.legal
                    )
                del spatial, pass_logit
            else:
                logits, values, scores, ownership = model.forward_all(
                    batch.coarse, batch.exact, batch.actions, batch.legal
                )
            with stage("loss_backward_optimizer"):
                loss = joint_loss(
                    logits,
                    values,
                    scores,
                    ownership,
                    batch.indices,
                    batch.visits,
                    batch.outcomes,
                    batch.scores,
                    batch.ownership,
                    score_weight=options.score_loss_weight,
                    ownership_weight=options.ownership_loss_weight,
                    policy_weights=batch.policy_weights,
                )
                if not torch.isfinite(loss.total):
                    raise RuntimeError("nonfinite joint training loss")
                if training:
                    loss.total.backward()
                    norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        options.grad_clip if options.grad_clip else math.inf,
                        error_if_nonfinite=True,
                    )
                    if not torch.isfinite(norm):
                        raise RuntimeError("nonfinite gradient norm")
                    optimizer.step()
            sync(device)
            compute += time.perf_counter() - tick
            count = batch.outcomes.numel()
            policy_count = int(cpu_batch.policy_weights.sum().item())
            policy_samples += policy_count
            for index, term in enumerate(loss):
                totals[index] += term.detach().item() * (policy_count if index == 1 else count)
            with torch.no_grad():
                error = scores - batch.scores
                score_abs += error.abs().sum().item()
                score_bias += error.sum().item()
                weights = ownership_weights(ownership.shape[-1], device=device)
                integrated = (ownership * weights).sum(dim=(-2, -1))
                consistency += (scores - integrated).abs().sum().item()
                predictions = torch.stack((values, scores, integrated)).double().cpu()
                prediction_sums += predictions.sum(dim=1)
                prediction_squares += predictions.square().sum(dim=1)
            samples += count
            steps += 1
            del batch, cpu_batch, logits, values, scores, ownership, loss
    if samples == 0:
        raise ValueError("phase produced no training positions")
    elapsed = time.perf_counter() - started
    return {
        "samples": samples,
        "steps": steps,
        "policy_samples": policy_samples,
        "samples_by_size": dict(sorted(samples_by_size.items())),
        "loss": totals[0] / samples,
        "policy_ce": totals[1] / policy_samples if policy_samples else 0.0,
        "value_mse": totals[2] / samples,
        "score_mse": totals[3] / samples,
        "ownership_mse": totals[4] / samples,
        "score_mae": score_abs / samples,
        "score_bias": score_bias / samples,
        "score_ownership_mae": consistency / samples,
        "prediction_std": (
            (prediction_squares / samples - (prediction_sums / samples).square())
            .clamp_min(0)
            .sqrt()
            .tolist()
        ),
        "seconds": elapsed,
        "positions_per_second": samples / elapsed,
        "load_wait_seconds": wait,
        "compute_seconds": compute,
        "cuda_peak_mib": torch.cuda.max_memory_allocated(device) / 2**20
        if device.type == "cuda"
        else None,
        "parent_peak_rss_mib": parent_peak_rss(),
        "cuda_reserved_peak_mib": torch.cuda.max_memory_reserved(device) / 2**20
        if device.type == "cuda"
        else None,
        **(
            {
                "stage_seconds": dict(stage_seconds),
                "feature_profile": feature_profile.summary(),
                **memory.summary(),
            }
            if profile_stages
            else {}
        ),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", nargs="+", required=True, help="SGF files or directories (recursive)"
    )
    parser.add_argument(
        "--output", type=Path, help="new runs default to runs/YYYYMMDD-HHMMSS-train"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--resume", type=Path, help="continue the same fixed-corpus run")
    source.add_argument(
        "--warm-start", type=Path, help="initialize a new replay run from a training checkpoint"
    )
    parser.add_argument(
        "--warm-start-optimizer",
        choices=["reset", "keep"],
        help="warm-start SGD momentum: reset (default) or keep; local steps/RNG restart, lineage totals persist",
    )
    parser.add_argument(
        "--epochs", type=int, help="total epochs in this run; default 3, or one more when resuming"
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--feature-binary", type=Path)
    parser.add_argument(
        "--network-family", help="readable training family; default cg1 or inherited"
    )
    parser.add_argument(
        "--data-positions",
        type=int,
        help="orchestrator's cumulative eligible self-play rows, including data outside replay",
    )
    parser.add_argument(
        "--generated-data",
        type=Path,
        help="loop's frozen generation totals/game-ID metadata, including games outside replay",
    )
    for field in fields(Options):
        parser.add_argument("--" + field.name.replace("_", "-"), type=type(field.default))
    for name in [
        "channels",
        "blocks",
        "policy_hidden",
        "value_hidden",
        "chunk_size",
        "top_cells",
        "exploration_cells",
    ]:
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    parser.add_argument(
        "--value-activation",
        choices=["relu", "leaky_relu"],
        help="value-head activation (leaky_relu slope 0.1); changing a checkpoint requires --warm-start",
    )
    args = parser.parse_args(argv)
    if args.warm_start_optimizer is not None and args.warm_start is None:
        parser.error("--warm-start-optimizer requires --warm-start")
    return args


def train(args):
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    binary = feature_binary(args.feature_binary)
    resuming = args.resume is not None
    source = args.resume or args.warm_start
    source_bytes = source.read_bytes() if source else None
    saved = (
        torch.load(io.BytesIO(source_bytes), map_location="cpu", weights_only=True)
        if source
        else None
    )
    if source is not None and (
        not isinstance(saved, dict)
        or not {
            "options",
            "config",
            "model",
            "optimizer",
            "epoch",
            "global_step",
            "run_id",
            "feature_spec",
            "corpus_fingerprint",
            "manifest",
            "best_validation",
            "rng_cpu",
            "rng_python",
            "rng_cuda",
        }.issubset(saved)
    ):
        raise ValueError("not an offline-training checkpoint")
    history = checkpoint_splits(saved) if saved is not None else {}
    known_data = data_history(saved, history)
    generated = None
    if args.generated_data is not None:
        generated = torch.load(args.generated_data, map_location="cpu", weights_only=True)
        if (
            not isinstance(generated, dict)
            or generated.get("version") != 1
            or not {"game_ids", "data_positions"}.issubset(generated)
        ):
            raise ValueError("incompatible generation progress metadata")
        known_generated = data_history({"data_game_ids": generated.get("game_ids")}, {})
        if args.data_positions is not None and args.data_positions != generated["data_positions"]:
            raise ValueError("--data-positions disagrees with generation progress metadata")
        args.data_positions = generated["data_positions"]
    restored_options = dict(saved["options"]) if saved is not None else {}
    # Old fixed-corpus checkpoints retain their original exhaustive ordering.
    # New jobs/warm-starts use the new 40/40/20 default unless explicitly set.
    if resuming and "size_weights" not in restored_options:
        restored_options["size_weights"] = "uniform"
    if resuming and "augmentation" not in restored_options:
        restored_options["augmentation"] = "none"
    options = Options(**restored_options)
    if not resuming:
        # A new training job uses today's cache default, not its parent's I/O policy.
        options.feature_cache = Options().feature_cache
        options.augmentation = Options().augmentation
    mutable = {
        "workers",
        "batch_size",
        "shuffle_buffer",
        "positions_per_game",
        "steps_per_epoch",
        "lr",
        "grad_clip",
        "feature_cache",
    }
    for field in fields(Options):
        value = getattr(args, field.name)
        if value is not None:
            if resuming and field.name not in mutable and value != getattr(options, field.name):
                raise ValueError(
                    f"resume restores --{field.name.replace('_', '-')}; conflicting override"
                )
            setattr(options, field.name, value)
    options.validate()
    config = NetworkConfig(**saved["config"]) if saved else NetworkConfig()
    settings = asdict(config)
    for name in [
        "channels",
        "blocks",
        "policy_hidden",
        "value_hidden",
        "chunk_size",
        "top_cells",
        "exploration_cells",
    ]:
        value = getattr(args, name)
        if value is not None:
            if saved and name != "chunk_size" and value != settings[name]:
                raise ValueError(
                    f"checkpoint architecture conflicts with --{name.replace('_', '-')}"
                )
            settings[name] = value
    if args.value_activation is not None:
        if resuming and args.value_activation != config.value_activation:
            raise ValueError("resume restores --value-activation; use --warm-start to change it")
        settings["value_activation"] = args.value_activation
    config = NetworkConfig(**settings)
    if config.input_planes != len(COARSE_PLANES) or config.move_features != len(MOVE_FEATURES):
        raise ValueError("checkpoint feature widths do not match the real extractor")
    spec = feature_spec(options.samples_per_diameter)
    if saved and saved["feature_spec"] != spec:
        raise ValueError("checkpoint uses incompatible feature semantics")
    completed = saved["epoch"] if resuming else 0
    epochs = args.epochs if args.epochs is not None else completed + (1 if resuming else 3)
    if epochs <= completed:
        raise ValueError("total epochs must exceed completed checkpoint epochs")
    output = (
        args.output or (args.resume.parent if resuming else new_run_directory("train"))
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not resuming and any(output.iterdir()):
        raise ValueError("output directory is not empty; choose a new run or --resume")
    if resuming:
        if output != args.resume.resolve().parent:
            raise ValueError("resume in the checkpoint's existing run directory")
        if any(int(p.stem.split("-")[1]) > completed for p in output.glob("epoch-[0-9]*.pt")):
            raise ValueError("newer epoch checkpoints exist; resume the latest checkpoint")
        if (output / "latest.pt").is_file():
            latest = torch.load(output / "latest.pt", map_location="cpu", weights_only=True)
            if latest["run_id"] != saved["run_id"]:
                raise ValueError("output directory belongs to another run")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    random.seed(options.seed)
    torch.manual_seed(options.seed)
    model = PolicyValueNet(config).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=options.lr,
        momentum=options.momentum,
        weight_decay=options.weight_decay,
    )
    if saved is not None:
        model.load_state_dict(saved["model"])
    keep_optimizer = resuming or args.warm_start_optimizer == "keep"
    if keep_optimizer:
        optimizer.load_state_dict(saved["optimizer"])
        for group in optimizer.param_groups:
            group.update(
                lr=options.lr, momentum=options.momentum, weight_decay=options.weight_decay
            )
    if resuming:
        torch.set_rng_state(saved["rng_cpu"])
        random.setstate(saved["rng_python"])
        if device.type == "cuda" and len(saved["rng_cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(saved["rng_cuda"])
    initialization = (
        saved.get("initialization", {"mode": "fresh"}) if resuming else {"mode": "fresh"}
    )
    if args.warm_start is not None:
        initialization = {
            "mode": "warm_start",
            "optimizer": args.warm_start_optimizer or "reset",
            "source": {
                "path": str(source.resolve()),
                "sha256": hashlib.sha256(source_bytes).hexdigest(),
                "run_id": saved["run_id"],
                "epoch": saved["epoch"],
                "global_step": saved["global_step"],
            },
        }
        previous_activation = NetworkConfig(**saved["config"]).value_activation
        if previous_activation != config.value_activation:
            initialization["value_activation_change"] = {
                "from": previous_activation,
                "to": config.value_activation,
                "negative_slope": 0.1 if config.value_activation == "leaky_relu" else 0.0,
            }
    del source_bytes
    with training_output(output, resuming) as logfile:

        def log(message):
            print(message, flush=True)
            logfile.write(message + "\n")
            logfile.flush()

        corpus = scan_corpus(
            args.data,
            options.samples_per_diameter,
            options.seed,
            options.validation_fraction,
            binary,
            split_history=history,
        )
        if resuming and corpus.fingerprint != saved["corpus_fingerprint"]:
            raise ValueError("eligible SGF corpus/targets changed since checkpoint")
        progress = initialize_progress(
            saved,
            corpus.train + corpus.validation,
            known_data,
            resuming=resuming,
            family=args.network_family,
            data_positions=args.data_positions,
        )
        if generated is not None and progress["data_positions"] != generated["data_positions"]:
            raise ValueError("generation progress conflicts with ancestral data total")
        all_data_ids = known_data | {g.game_id for g in corpus.train + corpus.validation}
        if generated is not None:
            if resuming and not known_generated.issubset(known_data):
                raise ValueError("generation history changed since checkpoint")
            all_data_ids |= known_generated
        weights = parse_weights(options.size_weights)
        if weights is not None and any(
            not any(weights.get(g.board_size, 1) for g in games)
            for games in [corpus.train, corpus.validation]
        ):
            raise ValueError("both train and validation need positive-weight board sizes")
        log(
            f"OFFLINE FP32; device={device}; torch={torch.__version__}; parameters={sum(p.numel() for p in model.parameters())}; config={asdict(config)}"
        )
        log(f"options={asdict(options)}")
        if options.size_weights != "uniform":
            log(
                "Sampling uses board-weighted position references (without replacement); surplus rows are not extracted; dense shuffle-buffer is bypassed."
            )
        log(f"initialization={initialization}; inherited_split_games={len(history)}")
        log(
            f"games train={len(corpus.train)} validation={len(corpus.validation)}; eligible positions train={sum(len(g.plies) for g in corpus.train)} validation={sum(len(g.plies) for g in corpus.validation)}; skipped={corpus.skipped}; corpus={corpus.fingerprint}"
        )
        log(
            "Validation is by complete unique game, stratified by resolution where possible; no synthetic policy/outcome labels."
        )
        if options.feature_cache == "auto":
            from .feature_cache import ensure_caches

            log("Feature cache: auto; preparing missing sidecars before training.")
            built = ensure_caches(
                corpus.train + corpus.validation,
                density=options.samples_per_diameter,
                binary=binary,
                workers=min(4, max(1, options.workers)),
            )
            log(f"Feature cache: {built} built; all selected games ready for cached reads.")
        step = saved["global_step"] if resuming else 0
        best = saved["best_validation"] if resuming else math.inf
        run_id = saved["run_id"] if resuming else uuid.uuid4().hex
        for epoch in range(completed, epochs):
            metrics = {}
            for training, name, games in [
                (True, "train", corpus.train),
                (False, "validation", corpus.validation),
            ]:
                result = run_phase(
                    model, optimizer, games, options, epoch, device, binary, training
                )
                metrics[name] = result
                if training:
                    step += result["steps"]
                    if progress["samples_seen"] is not None:
                        progress["samples_seen"] += result["samples"]
                log(
                    f"epoch={epoch + 1}/{epochs} {name}: loss={result['loss']:.6f} policy_ce={result['policy_ce']:.6f} value_mse={result['value_mse']:.6f} score_mse={result['score_mse']:.6f} ownership_mse={result['ownership_mse']:.6f} score_mae={result['score_mae']:.6f} score_bias={result['score_bias']:.6f} prediction_std={result['prediction_std']} samples={result['samples']} policy_samples={result['policy_samples']} steps={result['steps']} positions/s={result['positions_per_second']:.2f} load_wait={result['load_wait_seconds']:.3f}s compute+transfer={result['compute_seconds']:.3f}s cuda_peak_mib={result['cuda_peak_mib']} parent_peak_rss_mib={result['parent_peak_rss_mib']}"
                )
            improved = metrics["validation"]["loss"] < best
            best = min(best, metrics["validation"]["loss"])
            state = {
                "run_id": run_id,
                "initialization": initialization,
                "progress": dict(progress),
                "data_game_ids": sorted(all_data_ids),
                "split_history": corpus.split_history,
                "epoch": epoch + 1,
                "global_step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": asdict(config),
                "options": asdict(options),
                "feature_spec": spec,
                "corpus_fingerprint": corpus.fingerprint,
                "manifest": corpus.manifest(),
                "best_validation": best,
                "metrics": metrics,
                "torch_version": str(torch.__version__),
                "rng_cpu": torch.get_rng_state(),
                "rng_python": random.getstate(),
                "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            }
            name = network_name(state)
            destination = output / f"epoch-{epoch + 1:04d}.pt"
            if destination.exists():
                raise ValueError(f"refusing to overwrite {destination}")
            atomic_checkpoint(destination, state)
            atomic_checkpoint(output / "latest.pt", state)
            if improved:
                atomic_checkpoint(output / "best.pt", state)
            log(
                f"samples_by_size: train={metrics['train']['samples_by_size']} validation={metrics['validation']['samples_by_size']}"
            )
            log(
                f"saved {destination}; network={name}; best_validation={best:.6f}; global_step={step}"
            )
            from .plot_metrics import update

            update(output)
    return output / "latest.pt"


def main(argv=None):
    try:
        train(parse_args(argv))
    except KeyboardInterrupt:
        print(
            "Interrupted. Resume the latest completed epoch checkpoint; partial epochs are not saved.",
            file=sys.stderr,
        )
        raise SystemExit(130) from None
    except (ValueError, OSError, RuntimeError) as error:
        print(f"training error: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
