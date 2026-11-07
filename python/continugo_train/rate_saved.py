"""Fresh adaptive rating of saved milestones; never a round-robin schedule.

A small reference calibration is followed by near-strength matchmaking. Incompatible
feature schemas use separate frozen native runtimes and never play each other.
"""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from . import adaptive_ratings, archive, gate, ratings
from .loop import LoopConfig, LoopRunner, loop_lock
from .paths import ROOT
from .round_robin import snapshot_models
from .saved_models import saved_sources
from .train import atomic_checkpoint

VERSION = "saved-model-adaptive-ratings-v1"


def initialize(root, args):
    if args.max_games <= 0:
        raise ValueError("--max-games must be positive")
    store = args.model_store.resolve()
    sources = saved_sources(store)
    if not sources:
        raise ValueError("save campaign models before scheduling ratings")
    groups = {}
    for source in sources.values():
        saved = torch.load(source["checkpoint"]["path"], map_location="cpu", weights_only=True)
        schema = ratings.canonical(saved["feature_spec"])
        groups.setdefault(schema, []).append(source)
    states = []
    for index, (schema, pool) in enumerate(sorted(groups.items()), 1):
        pool.sort(key=lambda s: (s["campaign"], s["round"]))
        newest = max(pool, key=lambda s: s["campaign"])
        source = newest.get("source_context")
        if source is None:  # Early snapshots recorded only the original loop path.
            source = torch.load(
                Path(newest["source_loop"]) / "loop.pt", map_location="cpu", weights_only=True
            )
        config = replace(
            LoopConfig(**source["config"]),
            model_store=str(store),
            gate_games=args.batch_games,
            reference_games=args.reference_games,
            gate_simulations=32,
            gate_max_moves=1024,
            reference_max_moves=2048,
            seed=args.seed + index * 1_000_000,
            device=args.device,
            adaptive_max_games=args.max_games,
            adaptive_target_rd=args.target_rd,
            adaptive_min_pairs=args.min_pairs,
            adaptive_min_opponents=args.min_opponents,
            adaptive_min_opponent_pairs=args.min_opponent_pairs,
            adaptive_batch_games=args.batch_games,
        )
        config.validate()
        settings = ratings.protocol(config, legacy=source["version"] == 5)
        directory = root / f"cohort-{index:02d}"
        directory.mkdir()
        models, checkpoints = snapshot_models(directory, source, pool)
        jobs = [
            {
                "number": i + 1,
                "model": model,
                "checkpoint": checkpoint,
                "initial": False,
                "reference": None,
            }
            for i, (model, checkpoint) in enumerate(zip(models, checkpoints))
        ]
        state = {
            "version": VERSION,
            "root": str(directory),
            "config": asdict(config),
            "feature_spec": json.loads(schema),
            "rating_protocol": settings,
            "runtime_fingerprints": source["runtime_fingerprints"],
            "jobs": jobs,
            "targets": [m["checkpoint_sha256"] for m in models],
        }
        atomic_checkpoint(directory / "loop.pt", state)
        states.append(state)
    manifest = {
        "version": VERSION,
        "root": str(root),
        "model_store": str(store),
        "cohorts": [str(Path(s["root"]).relative_to(root)) for s in states],
        "models": len(sources),
        "policy": {
            key: getattr(args, key)
            for key in [
                "target_rd",
                "min_pairs",
                "min_opponents",
                "min_opponent_pairs",
                "max_games",
                "batch_games",
            ]
        },
        "reference_games": args.reference_games,
        "maximum_games": len(sources) * args.reference_games + len(states) * args.max_games,
    }
    atomic_checkpoint(root / "schedule.pt", manifest)
    return manifest


def run(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with loop_lock(root) as lock_fd:
        if args.resume:
            manifest = torch.load(root / "schedule.pt", map_location="cpu", weights_only=True)
            if manifest.get("version") != VERSION or manifest.get("root") != str(root):
                raise ValueError("incompatible or relocated rating schedule")
        else:
            if any(p.name != ".loop.lock" for p in root.iterdir()):
                raise ValueError("output is not empty; use --resume")
            archive.initialize_store(args.model_store)
            manifest = initialize(root, args)
        print(
            f"Adaptive rating of {manifest['models']} models in {len(manifest['cohorts'])} compatible cohorts; at most {manifest['maximum_games']} games, not a round robin",
            flush=True,
        )
        for cohort in manifest["cohorts"]:
            directory = root / cohort
            state = torch.load(directory / "loop.pt", map_location="cpu", weights_only=True)
            with loop_lock(directory) as cohort_lock:
                runner = LoopRunner(directory, state, (lock_fd, cohort_lock))
                runner.verify_runtime()
                for job in state["jobs"]:
                    ratings.register_checkpoint(
                        runner.config.model_store,
                        runner.check(job["checkpoint"]),
                        model=runner.check_model(job["model"]),
                    )
                    runner.log(
                        f"CALIBRATION {job['number']}/{len(state['jobs'])}: {job['model']['network_name']}"
                    )
                    for record in (job.get("reference") or {}).get("games", []):
                        if record:
                            runner.check(record)
                    gate.evaluate(runner, job, reference=True)
                    runner.rate_evaluation(job, "reference", [job["checkpoint"]])
        # Calibrate every campaign before refining any cohort, so all run curves
        # become visible promptly. The next opponents are chosen adaptively.
        for cohort in manifest["cohorts"]:
            directory = root / cohort
            state = torch.load(directory / "loop.pt", map_location="cpu", weights_only=True)
            with loop_lock(directory) as cohort_lock:
                runner = LoopRunner(directory, state, (lock_fd, cohort_lock))
                runner.verify_runtime()
                if len(state["targets"]) <= manifest["policy"]["min_opponents"]:
                    runner.log("insufficient compatible opponents; confidence target NOT met")
                    continue
                output = directory / "adaptive"
                command = ["--output", str(output)]
                if (output / "loop.pt").exists():
                    command += ["--resume"]
                else:
                    command += [
                        "--loop",
                        str(directory),
                        "--targets",
                        *state["targets"],
                        "--latest",
                        str(len(state["targets"])),
                        "--seed",
                        str(runner.config.seed + 500_000),
                    ]
                    for key, value in manifest["policy"].items():
                        command += ["--" + key.replace("_", "-"), str(value)]
                result = adaptive_ratings.run(
                    adaptive_ratings.parse_args(command), parent_lock_fd=(lock_fd, cohort_lock)
                )
                print(f"{cohort}: {result['stop_reason']}\n{result['progress']}", flush=True)
        print("Fresh adaptive ratings finished; no training or model approval", flush=True)
        return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-store", type=Path, default=ROOT / "models")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reference-games", type=int, default=10)
    parser.add_argument("--target-rd", type=float, default=75)
    parser.add_argument("--min-pairs", type=int, default=20)
    parser.add_argument("--min-opponents", type=int, default=2)
    parser.add_argument("--min-opponent-pairs", type=int, default=4)
    parser.add_argument(
        "--max-games", type=int, default=360, help="additional cap per compatible cohort"
    )
    parser.add_argument("--batch-games", type=int, default=4)
    parser.add_argument("--seed", type=int, default=97000000)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    return parser.parse_args(argv)


def main(argv=None):
    try:
        run(parse_args(argv))
    except KeyboardInterrupt:
        raise SystemExit("Interrupted; resume this schedule with --resume") from None
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit(f"rate-saved error: {error}") from error


if __name__ == "__main__":
    main()
