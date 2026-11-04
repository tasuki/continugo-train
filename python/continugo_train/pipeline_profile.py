"""Real-position pipeline diagnostics, not playing-strength or training trials.

Requires completed SGFs at sizes 9/13/19 and a matching checkpoint/ONNX bundle.
Generates neural self-play with genuine visits/outcomes, then benchmarks the real
loader and SGD path. No models are published and no training checkpoints saved.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from .data import scan_corpus
from .features import FeatureClient, FeatureProfile, inspect_game
from .memory import GpuMemorySampler
from .model import NetworkConfig, PolicyValueNet
from .paths import ROOT, new_run_directory
from .runtime import inference_environment
from .train import Options, feature_spec, run_phase

SIZES = (9, 13, 19)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def representative_plies(moves):
    """Pre-move snapshots: 10% through play, halfway, before the final two pass turns.

    Completed games can include earlier passes, so these are phase proxies, not
    inferred tactical categories. Reject tiny smoke fixtures rather than calling
    a one-stone board a representative endgame.
    """
    if moves < 10:
        raise ValueError("representative games need at least 10 moves")
    return {"early": max(1, (moves - 2) // 10), "middle": (moves - 2) // 2, "late": moves - 2}


def select_positions(paths):
    candidates = {size: [] for size in SIZES}
    files = set()
    for path in paths:
        path = path.resolve()
        files.update(path.rglob("*.sgf") if path.is_dir() else [path])
    for path in sorted(files):
        info = inspect_game(path)
        if info["board_size"] in candidates and info["completed"] and info["moves"] >= 10:
            candidates[info["board_size"]].append((path, info))
    selected = []
    for size in SIZES:
        if not candidates[size]:
            raise ValueError(f"need a completed size-{size} SGF with at least 10 moves")
        # Deterministic selection; longer games are preferable to early-pass smoke tests.
        path, info = min(candidates[size], key=lambda item: (-item[1]["moves"], str(item[0])))
        for phase, ply in representative_plies(info["moves"]).items():
            selected.append(
                {
                    "size": size,
                    "phase": phase,
                    "ply": ply,
                    "sgf": str(path),
                    "file_sha256": sha256(path),
                    "game_id": info["game_id"],
                    "moves": info["moves"],
                }
            )
    return selected


def measured_command(command, stem, environment, *, profile_gpu=False):
    """GNU time peak RSS is process-lifetime, including loading/replay/warmup.

    This launches native binaries directly, so Python/Torch launchers and cargo
    compilation are not incorrectly counted as native inference memory.
    """
    timer = Path("/usr/bin/time")
    timed = timer.is_file() and platform.system() == "Linux"
    actual = [str(timer), "-f", "%M", "-o", str(stem) + ".rss", *command] if timed else command
    started = time.perf_counter()
    with subprocess.Popen(
        actual, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment
    ) as process:
        with GpuMemorySampler(process.pid, enabled=profile_gpu) as memory:
            stdout, stderr = process.communicate()
            seconds = time.perf_counter() - started
        result = subprocess.CompletedProcess(actual, process.returncode, stdout, stderr)
    Path(str(stem) + ".stderr").write_text(result.stderr)
    Path(str(stem) + ".stdout").write_text(result.stdout)
    result.check_returncode()
    return result.stdout, {
        "process_seconds": seconds,
        "process_peak_rss_mib": float(Path(str(stem) + ".rss").read_text()) / 1024
        if timed
        else None,
        **memory.summary(),
    }


def transport_samples(games, iterations):
    """Warm persistent production feature protocol, including decoding/tensor copies.

    Includes Rust extraction/serialization/pipe wait; not a pure pipe bandwidth
    measurement. Uses real policy target cells, not empty random-bot targets.
    """
    results = []
    with FeatureClient(profile=True) as client:
        for size in SIZES:
            game = next((g for g in games if g.board_size == size), None)
            if game is None:
                raise ValueError(f"training data has no labeled policy positions at size {size}")
            plies = game.plies
            for phase, ply in zip(
                ("early", "middle", "late"),
                (plies[0], plies[len(plies) // 2], plies[-1]),
                strict=True,
            ):
                client.load(game.path, ply, file_sha256=game.file_sha256)
                profile = FeatureProfile()
                started = time.perf_counter()
                for _ in range(iterations):
                    sample = client.load(game.path, ply, file_sha256=game.file_sha256)
                    profile.add(client.process.pid, client.last_profile)
                seconds = time.perf_counter() - started
                results.append(
                    {
                        "size": size,
                        "phase": phase,
                        "ply": ply,
                        "sgf": game.path,
                        "file_sha256": game.file_sha256,
                        "iterations": iterations,
                        "seconds": seconds,
                        "exact_coordinates": sample.actions.numel(),
                        "feature_profile": profile.summary(),
                    }
                )
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--positions",
        type=Path,
        nargs="+",
        required=True,
        help="completed representative SGFs/directories (e.g. dense random games)",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data", type=Path, nargs="+", help="training corpus; defaults to generated self-play"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--feature-threads", type=int, default=6)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=Options().workers)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument(
        "--positions-per-game", type=int, default=12, help="0 uses all eligible positions"
    )
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--max-moves", type=int, default=512)
    parser.add_argument("--seed", type=int, default=800)
    args = parser.parse_args(argv)
    if (
        min(
            args.iterations,
            args.feature_threads,
            args.threads,
            args.batch_size,
            args.steps,
            args.warmup_steps,
            args.simulations,
            args.max_moves,
        )
        <= 0
        or min(args.seed, args.workers, args.positions_per_game) < 0
    ):
        parser.error("counts must be positive; seed/workers/positions-per-game must be nonnegative")
    for binary in ("infer", "features", "continugo-train"):
        if not (ROOT / "target/release" / binary).is_file():
            parser.error("build first: cargo build --locked --release --bins")
    library, environment = inference_environment(args.device)
    positions = select_positions(args.positions)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["feature_spec"] != feature_spec(4):
        raise ValueError("profiler currently requires sample density 4 and current feature schema")
    # Verify provenance, not just tensor shapes, before comparing training/inference.
    import onnx

    for name in ("encoder.onnx", "policy.onnx"):
        metadata = {p.key: p.value for p in onnx.load(args.model / name).metadata_props}
        if json.loads(metadata["continugo"])["checkpoint_sha256"] != sha256(args.checkpoint):
            raise ValueError("checkpoint and ONNX model do not match")
    if args.output:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
    else:
        output = new_run_directory("pipeline-profile")
    torch.set_num_threads(args.threads)
    report = {
        "arguments": {key: str(value) for key, value in vars(args).items()},
        "platform": platform.platform(),
        "torch": torch.__version__,
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "shared_rules": {
            "revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT.parent / "continugo", text=True
            ).strip(),
            "git_status": subprocess.check_output(
                ["git", "status", "--short"], cwd=ROOT.parent / "continugo", text=True
            ),
            "source_sha256": {
                str(path.relative_to(ROOT.parent / "continugo")): sha256(path)
                for path in sorted((ROOT.parent / "continugo/backend/rules/src").rglob("*.rs"))
            },
        },
        "checkpoint_sha256": sha256(args.checkpoint),
        "model_sha256": {
            name: sha256(args.model / name) for name in ("encoder.onnx", "policy.onnx")
        },
        "runtime": str(library),
        "binary_sha256": {
            name: sha256(ROOT / "target/release" / name)
            for name in ("infer", "features", "continugo-train")
        },
        "config": checkpoint["config"],
        "positions": [],
        "limitations": [
            "Phase labels are move-count proxies; inspect source SGFs for tactical coverage.",
            "Single selected game per size, not multi-seed statistics or playing strength.",
            "IPC residual includes dispatch/framing/scheduling; server writes overlap client reads.",
            "Feature worker wall-time sums include prefetch and are not exposed loader wait.",
            "Sum of Rust child lifetime RSS peaks is not a simultaneous process-tree peak.",
            "Native RSS includes model loading/replay/warmup; driver GPU memory is sampled, not an exact peak.",
            "Training parent RSS is cumulative; diagnostic Rust child RSS is reported separately.",
            "Synchronized training stage diagnostics may perturb throughput; baseline is separate.",
        ],
    }

    def save():
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"Output: {output}", flush=True)
    save()
    for position in positions:
        for value_only in (False, True):
            label = f"{position['size']}-{position['phase']}-{'value' if value_only else 'policy'}"
            print(label, flush=True)
            command = [
                str(ROOT / "target/release/infer"),
                "--model",
                str(args.model.resolve()),
                "--ort-library",
                str(library),
                "--device",
                args.device,
                "--sgf",
                position["sgf"],
                "--ply",
                str(position["ply"]),
                "--feature-threads",
                str(args.feature_threads),
                "--threads",
                str(args.threads),
                "--cache-positions",
                "0",
                "--profile-json",
                "--warmup",
                "1",
                "--iterations",
                str(args.iterations),
            ]
            if value_only:
                command.append("--value-only")
            stdout, process = measured_command(
                command, output / label, environment, profile_gpu=args.device == "cuda"
            )
            report["positions"].append(
                {**position, "value_only": value_only, **process, **json.loads(stdout)}
            )
            save()
    print("neural self-play (six games; uniform size schedule)", flush=True)
    games_path = output / "selfplay"
    stats_path = output / "selfplay.json"
    _, process = measured_command(
        [
            str(ROOT / "target/release/continugo-train"),
            "--engine",
            "neural",
            "--model",
            str(args.model.resolve()),
            "--ort-library",
            str(library),
            "--device",
            args.device,
            "--games",
            "6",
            "--sizes",
            "9,13,19",
            "--size-weights",
            "uniform",
            "--seed",
            str(args.seed),
            "--simulations",
            str(args.simulations),
            "--max-moves",
            str(args.max_moves),
            "--feature-threads",
            str(args.feature_threads),
            "--inference-threads",
            str(args.threads),
            "--output",
            str(games_path),
            "--stats-json",
            str(stats_path),
        ],
        output / "selfplay",
        environment,
        profile_gpu=args.device == "cuda",
    )
    report["selfplay"] = {**json.loads(stats_path.read_text()), **process}
    save()
    corpus = scan_corpus([str(p) for p in (args.data or [games_path])], 4, args.seed, 0.25)
    report["corpus"] = {
        "manifest": corpus.manifest(),
        "skipped": corpus.skipped,
        "fingerprint": corpus.fingerprint,
    }
    report["training_transport"] = transport_samples(
        corpus.train + corpus.validation, args.iterations
    )
    save()
    options = Options(
        batch_size=args.batch_size,
        workers=args.workers,
        steps_per_epoch=args.steps,
        positions_per_game=args.positions_per_game,
        validation_positions_per_game=4,
        seed=args.seed,
        size_weights="uniform",
    )
    report["training_options"] = asdict(options)
    report["training_warmup_steps"] = args.warmup_steps
    device = torch.device(args.device)
    for detailed in (False, True):
        print(f"training {'synchronized stages' if detailed else 'baseline'}", flush=True)
        torch.manual_seed(args.seed)
        model = PolicyValueNet(NetworkConfig(**checkpoint["config"])).to(device)
        model.load_state_dict(checkpoint["model"])
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=options.lr,
            momentum=options.momentum,
            weight_decay=options.weight_decay,
        )
        # Initialize kernels/optimizer before timing, then restore parameters,
        # BatchNorm buffers, RNG and fresh optimizer state for each measurement.
        run_phase(
            model,
            optimizer,
            corpus.train,
            replace(options, steps_per_epoch=args.warmup_steps),
            0,
            device,
            None,
            True,
        )
        model.load_state_dict(checkpoint["model"])
        torch.manual_seed(args.seed)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=options.lr,
            momentum=options.momentum,
            weight_decay=options.weight_decay,
        )
        metrics = run_phase(
            model, optimizer, corpus.train, options, 0, device, None, True, profile_stages=detailed
        )
        report["training_stages" if detailed else "training_baseline"] = metrics
        save()
        del model, optimizer
    print(
        json.dumps(
            {key: report[key] for key in ("selfplay", "training_baseline", "training_stages")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
