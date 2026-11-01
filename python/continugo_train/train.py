"""Offline FP32 policy/value training from completed, annotated SGFs.

No ONNX or neural self-play here. Rust owns game parsing, rules and features.
Checkpoints are atomic, at epoch boundaries; --epochs is the total epoch target.
"""

import argparse
import math
import os
import random
import sys
import tempfile
import time
import uuid
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch

from .data import batches, scan_corpus
from .features import COARSE_PLANES, MOVE_FEATURES, PASS, SCHEMA_VERSION, feature_binary
from .losses import policy_value_loss
from .model import NetworkConfig, PolicyValueNet


@dataclass
class Options:
    batch_size: int = 2
    workers: int = 2
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

    def validate(self):
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


def feature_spec(density):
    return {
        "schema": SCHEMA_VERSION,
        "coarse": list(COARSE_PLANES),
        "moves": list(MOVE_FEATURES),
        "samples_per_diameter": density,
    }


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


def run_phase(model, optimizer, games, options, epoch, device, binary, actions, training):
    model.train(training)
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    samples = steps = 0
    totals = [0.0, 0.0, 0.0]
    wait = compute = 0.0
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
    )
    with closing(stream), torch.set_grad_enabled(training):
        while not (training and options.steps_per_epoch and steps >= options.steps_per_epoch):
            tick = time.perf_counter()
            try:
                cpu_batch = next(stream)
            except StopIteration:
                break
            wait += time.perf_counter() - tick
            tick = time.perf_counter()
            batch = cpu_batch.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits, values = model(batch.coarse, batch.exact, actions, batch.legal)
            loss = policy_value_loss(logits, values, batch.indices, batch.visits, batch.outcomes)
            if not torch.isfinite(loss.total):
                raise RuntimeError("nonfinite policy/value loss")
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
            for index, term in enumerate(loss):
                totals[index] += term.detach().item() * count
            samples += count
            steps += 1
            del batch, cpu_batch, logits, values, loss
    if samples == 0:
        raise ValueError("phase produced no training positions")
    elapsed = time.perf_counter() - started
    return {
        "samples": samples,
        "steps": steps,
        "loss": totals[0] / samples,
        "policy_ce": totals[1] / samples,
        "value_mse": totals[2] / samples,
        "seconds": elapsed,
        "positions_per_second": samples / elapsed,
        "load_wait_seconds": wait,
        "compute_seconds": compute,
        "cuda_peak_mib": torch.cuda.max_memory_allocated(device) / 2**20
        if device.type == "cuda"
        else None,
        "parent_peak_rss_mib": parent_peak_rss(),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", nargs="+", required=True, help="SGF files or directories (recursive)"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--epochs", type=int, help="total epochs; default 3, or one more when resuming"
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--feature-binary", type=Path)
    for field in fields(Options):
        parser.add_argument("--" + field.name.replace("_", "-"), type=type(field.default))
    for name in ["channels", "blocks", "policy_hidden", "value_hidden", "chunk_size"]:
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    return parser.parse_args(argv)


def train(args):
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    binary = feature_binary(args.feature_binary)
    saved = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if saved is not None and (
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
            "best_validation",
            "rng_cpu",
            "rng_python",
            "rng_cuda",
        }.issubset(saved)
    ):
        raise ValueError("not an offline-training checkpoint")
    options = Options(**saved["options"]) if saved else Options()
    mutable = {
        "workers",
        "batch_size",
        "shuffle_buffer",
        "positions_per_game",
        "steps_per_epoch",
        "lr",
        "grad_clip",
    }
    for field in fields(Options):
        value = getattr(args, field.name)
        if value is not None:
            if saved and field.name not in mutable and value != getattr(options, field.name):
                raise ValueError(
                    f"resume restores --{field.name.replace('_', '-')}; conflicting override"
                )
            setattr(options, field.name, value)
    options.validate()
    config = NetworkConfig(**saved["config"]) if saved else NetworkConfig()
    settings = asdict(config)
    for name in ["channels", "blocks", "policy_hidden", "value_hidden", "chunk_size"]:
        value = getattr(args, name)
        if value is not None:
            if saved and name != "chunk_size" and value != settings[name]:
                raise ValueError(f"resume architecture conflicts with --{name.replace('_', '-')}")
            settings[name] = value
    config = NetworkConfig(**settings)
    if config.input_planes != len(COARSE_PLANES) or config.move_features != len(MOVE_FEATURES):
        raise ValueError("checkpoint feature widths do not match the real extractor")
    spec = feature_spec(options.samples_per_diameter)
    if saved and saved["feature_spec"] != spec:
        raise ValueError("checkpoint uses incompatible feature semantics")
    completed = saved["epoch"] if saved else 0
    epochs = args.epochs if args.epochs is not None else completed + (1 if saved else 3)
    if epochs <= completed:
        raise ValueError("total epochs must exceed completed checkpoint epochs")
    output = (args.output or (args.resume.parent if args.resume else Path("runs/train"))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not saved and any(output.iterdir()):
        raise ValueError("output directory is not empty; choose a new run or --resume")
    if saved:
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
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = options.lr
        torch.set_rng_state(saved["rng_cpu"])
        random.setstate(saved["rng_python"])
        if device.type == "cuda" and len(saved["rng_cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(saved["rng_cuda"])
    with run_lock(output), (output / "training.log").open("a") as logfile:

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
        )
        if saved and corpus.fingerprint != saved["corpus_fingerprint"]:
            raise ValueError("eligible SGF corpus/targets changed since checkpoint")
        log(
            f"OFFLINE FP32; device={device}; torch={torch.__version__}; parameters={sum(p.numel() for p in model.parameters())}; config={asdict(config)}"
        )
        log(f"options={asdict(options)}")
        log(
            f"games train={len(corpus.train)} validation={len(corpus.validation)}; eligible positions train={sum(len(g.plies) for g in corpus.train)} validation={sum(len(g.plies) for g in corpus.validation)}; skipped={corpus.skipped}; corpus={corpus.fingerprint}"
        )
        log(
            "Validation is by complete unique game, stratified by resolution where possible; no synthetic policy/outcome labels."
        )
        actions = torch.arange(PASS, device=device)
        step = saved["global_step"] if saved else 0
        best = saved["best_validation"] if saved else math.inf
        run_id = saved["run_id"] if saved else uuid.uuid4().hex
        for epoch in range(completed, epochs):
            metrics = {}
            for training, name, games in [
                (True, "train", corpus.train),
                (False, "validation", corpus.validation),
            ]:
                result = run_phase(
                    model, optimizer, games, options, epoch, device, binary, actions, training
                )
                metrics[name] = result
                if training:
                    step += result["steps"]
                log(
                    f"epoch={epoch + 1}/{epochs} {name}: loss={result['loss']:.6f} policy_ce={result['policy_ce']:.6f} value_mse={result['value_mse']:.6f} samples={result['samples']} steps={result['steps']} positions/s={result['positions_per_second']:.2f} load_wait={result['load_wait_seconds']:.3f}s compute+transfer={result['compute_seconds']:.3f}s cuda_peak_mib={result['cuda_peak_mib']} parent_peak_rss_mib={result['parent_peak_rss_mib']}"
                )
            improved = metrics["validation"]["loss"] < best
            best = min(best, metrics["validation"]["loss"])
            state = {
                "run_id": run_id,
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
            destination = output / f"epoch-{epoch + 1:04d}.pt"
            if destination.exists():
                raise ValueError(f"refusing to overwrite {destination}")
            atomic_checkpoint(destination, state)
            atomic_checkpoint(output / "latest.pt", state)
            if improved:
                atomic_checkpoint(output / "best.pt", state)
            log(f"saved {destination}; best_validation={best:.6f}; global_step={step}")
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
