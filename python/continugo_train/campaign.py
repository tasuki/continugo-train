"""Recipe-driven, frozen training campaigns. See docs/current/campaigns.md."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

BINARIES = ("continugo-train", "features", "arena", "rating")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def verify_files(run, hashes):
    for name, digest in hashes.items():
        if sha(run / name) != digest:
            raise ValueError(f"Frozen campaign file changed: {name}")


def freeze(root, run, recipe):
    """Never replace a frozen runtime, including on prepare-only/resume."""
    manifest = run / "runtime.json"
    if manifest.exists():
        hashes = json.loads(manifest.read_text())["hashes"]
        verify_files(run, hashes)
        if json.loads((run / "recipe.json").read_text()) != recipe:
            raise ValueError("Recipe changed; fork a new campaign")
        return
    if (run / "loop").exists():
        raise ValueError("Refusing to freeze over an existing loop")
    subprocess.run(["cargo", "build", "--locked", "--release", "--bins"], cwd=root, check=True)
    package = run / "runtime/continugo_train"
    shutil.copytree(
        root / "python/continugo_train",
        package,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (run / "bin").mkdir(exist_ok=True)
    for name in BINARIES:
        shutil.copy2(root / "target/release" / name, run / "bin" / name)
    dump(run / "recipe.json", recipe)
    files = [run / "recipe.json", *package.glob("*.py"), *(run / "bin").iterdir()]
    hashes = {str(p.relative_to(run)): sha(p) for p in sorted(files)}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dump(manifest, {"hashes": hashes, "git_revision": revision})


def fork_checkpoint(parent, family, source, digest):
    """Change identity only: model, momentum, counters and holdouts stay intact."""
    import copy

    fork = copy.deepcopy(parent)
    fork["progress"]["family"] = family
    fork["campaign_fork"] = {
        "source": str(source),
        "sha256": digest,
        "optimizer": "keep",
        "weights_unchanged": True,
    }
    return fork


def seed_cache(record, binary, density):
    """Reuse only exact extractor matches; rebuild locally without touching ancestors."""
    from .feature_cache import CacheReader, build_cache, sidecar_path

    original, target = Path(record["source"]), Path(record["path"])
    destination = sidecar_path(target)
    if not destination.exists() and not destination.is_symlink():
        try:
            with CacheReader(
                original, density=density, binary=binary, file_sha256=record["sha256"]
            ):
                pass
        except (FileNotFoundError, ValueError):
            build_cache(target, density=density, binary=binary)
        else:
            destination.parent.mkdir(exist_ok=True)
            destination.symlink_to(sidecar_path(original).resolve())
    with CacheReader(target, density=density, binary=binary, file_sha256=record["sha256"]):
        pass
    return str(destination.resolve())


def prepare_fresh(root, run, recipe, hashes, lock_fd):
    """Create a default checkpoint from newly generated MCTS games, never ancestors."""
    import math
    from dataclasses import asdict

    import torch

    from . import loop
    from .data import inspect_corpus
    from .model import NetworkConfig
    from .train import Options, feature_spec

    store = root / "artifacts" / recipe["lineage"]
    if store.exists():
        raise ValueError("Fresh lineage store already exists; inspect or discard this attempt")
    config = loop.LoopConfig(
        selfplay_binary=str(run / "bin/continugo-train"),
        feature_binary=str(run / "bin/features"),
        model_store=str(store),
    )
    config.validate()

    def execute(command, name):
        verify_files(run, hashes)
        print(f"Fresh campaign {name}: {command}", flush=True)
        loop.run_process(command, run / f"{name}.log", lock_fd, dict(os.environ))

    bootstrap = run / "bootstrap"
    bootstrap_receipt = run / "bootstrap-complete.json"
    if not bootstrap_receipt.exists():
        if bootstrap.exists():
            raise ValueError("Partial bootstrap exists; inspect or discard this attempt")
        settings = recipe["bootstrap"]
        if settings["games"] < 2 or settings["rollouts"] < 1:
            raise ValueError("Fresh bootstrap needs multiple games and completed-rollout evaluation")
        execute([
            str(run / "bin/continugo-train"), "--engine", "mcts",
            "--games", str(settings["games"]), "--rollouts", str(settings["rollouts"]),
            "--output", str(bootstrap), "--stats-json", str(run / "bootstrap-stats.json"),
        ], "bootstrap")
        stats = json.loads((run / "bootstrap-stats.json").read_text())
        if stats["completed"] != settings["games"]:
            raise ValueError("Bootstrap contains unfinished games")
        records, skipped = inspect_corpus(
            [str(bootstrap)], Options().samples_per_diameter, config.feature_binary
        )
        if len(records) < 2 or skipped["unlabeled"] or skipped["no_policy"]:
            raise ValueError(f"Bootstrap lacks usable labeled games: {skipped}")
        dump(bootstrap_receipt, {"stats": stats, "eligible_games": len(records), "skipped": skipped})

    checkpoint = run / "initial-training/latest.pt"
    training_receipt = run / "initial-training-complete.json"
    if not training_receipt.exists():
        if checkpoint.parent.exists():
            raise ValueError("Uncommitted initial training exists; inspect or discard this attempt")
        execute([
            sys.executable, "-m", "continugo_train.train", "--data", str(bootstrap),
            "--output", str(checkpoint.parent), "--network-family", recipe["family"],
            "--feature-binary", config.feature_binary,
        ], "initial-training")
        dump(training_receipt, {"checkpoint_sha256": sha(checkpoint)})
    if sha(checkpoint) != json.loads(training_receipt.read_text())["checkpoint_sha256"]:
        raise ValueError("Fresh initial checkpoint changed")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        saved["initialization"] != {"mode": "fresh"}
        or saved["progress"]["family"] != recipe["family"]
        or saved["feature_spec"] != feature_spec(Options().samples_per_diameter)
        or saved["options"] != asdict(Options())
        or saved["config"] != asdict(NetworkConfig())
    ):
        raise ValueError("Fresh initialization does not match current defaults")
    if not all(torch.isfinite(tensor).all() for tensor in saved["model"].values()) or not all(
        math.isfinite(saved["metrics"][phase]["loss"]) for phase in ("train", "validation")
    ):
        raise ValueError("Nonfinite initial training state")
    receipt = {
        "family": recipe["family"], "fresh": True,
        "source_checkpoint": None, "inherited_corpus": [],
        "boundary_checkpoint": str(checkpoint.relative_to(run)),
        "boundary_checkpoint_sha256": sha(checkpoint),
        "bootstrap_data": str(bootstrap), "config": dict(asdict(config), sizes=list(config.sizes)),
        "runtime_hashes": hashes, "rating_every": recipe["rating_every"],
        "automatic_publication": False,
    }
    dump(run / "launch.json", receipt)
    return receipt


def prepare(root, run, recipe, lock_fd=None):
    import torch

    from . import loop
    from .train import atomic_checkpoint, checkpoint_splits

    hashes = json.loads((run / "runtime.json").read_text())["hashes"]
    verify_files(run, hashes)
    # Once committed, preparation no longer depends on mutable ancestral loop state.
    launch = run / "launch.json"
    if launch.exists():
        receipt = json.loads(launch.read_text())
        if sha(run / receipt.get("boundary_checkpoint", "parent.pt")) != receipt["boundary_checkpoint_sha256"]:
            raise ValueError("Campaign boundary checkpoint changed")
        return receipt
    if recipe.get("fresh"):
        return prepare_fresh(root, run, recipe, hashes, lock_fd)
    source = root / recipe["source_loop"]
    state = torch.load(source / "loop.pt", map_location="cpu", weights_only=True)
    iteration = state["iterations"][recipe["source_round"] - 1]
    if not iteration["completed"]:
        raise ValueError("Parent round is not complete")
    record = iteration["checkpoint"]
    parent_path = source / record["path"]
    if record["sha256"] != recipe["parent_sha256"] or sha(parent_path) != recipe["parent_sha256"]:
        raise ValueError("Parent checkpoint changed")
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    if parent["progress"]["family"] == recipe["family"]:
        raise ValueError("Fork requires a new model family")
    records = iteration["replay"]
    if len({r["game_id"] for r in records}) != len(records):
        raise ValueError("Duplicate seed games")
    history = checkpoint_splits(parent)
    provenance = []
    counts = {s: {"games": 0, "positions": 0} for s in ("train", "validation")}
    (run / "seed-data").mkdir(exist_ok=True)
    for rank, record in enumerate(records):
        original = source / record["path"]
        target = run / "seed-data" / f"rank-{rank:06d}.sgf"
        if sha(original) != record["sha256"]:
            raise ValueError(f"Seed game changed: {original}")
        if not target.exists():
            shutil.copyfile(original, target)
        if sha(target) != record["sha256"]:
            raise ValueError(f"Seed snapshot changed: {target}")
        split = history[record["game_id"]]
        counts[split]["games"] += 1
        counts[split]["positions"] += record["eligible_positions"]
        provenance.append(
            {
                "rank": rank,
                "source": str(original),
                "path": str(target),
                "sha256": record["sha256"],
                "game_id": record["game_id"],
                "split": split,
            }
        )
    print(f"Preparing {len(provenance)} seed caches with the frozen feature extractor", flush=True)
    pool = ThreadPoolExecutor(max_workers=max(1, state["config"]["workers"]))
    try:

        def cache(record):
            return seed_cache(
                record, run / "bin/features", parent["options"]["samples_per_diameter"]
            )

        for number, (record, cached) in enumerate(zip(provenance, pool.map(cache, provenance)), 1):
            record["feature_cache"] = cached
            if number % 25 == 0 or number == len(provenance):
                print(f"Seed caches ready: {number}/{len(provenance)}", flush=True)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    derived = run / "parent.pt"
    expected = fork_checkpoint(parent, recipe["family"], parent_path, recipe["parent_sha256"])
    atomic_checkpoint(derived, expected)
    config = dict(loop.training_config(state["config"]))
    config.update(recipe["overrides"])
    config.update(
        selfplay_binary=str(run / "bin/continugo-train"),
        feature_binary=str(run / "bin/features"),
        model_store=str(
            root / "artifacts" / recipe["lineage"]
            if "lineage" in recipe
            else state["config"]["model_store"]
        ),
        warm_start_optimizer="keep",
    )
    config["sizes"] = tuple(config["sizes"])
    frozen = loop.LoopConfig(**config)
    frozen.validate()
    frozen.training_options(expected, 1)
    receipt = {
        "family": recipe["family"],
        "source_checkpoint": str(parent_path),
        "source_sha256": recipe["parent_sha256"],
        "boundary_checkpoint_sha256": sha(derived),
        "config": config,
        "initial_replay_counts": counts,
        "runtime_hashes": hashes,
        "rating_every": recipe["rating_every"],
        "automatic_publication": False,
    }
    dump(run / "source-records.json", provenance)
    dump(launch, receipt)
    print(
        f"Prepared {recipe['family']}: preserved weights, momentum, counters and holdouts; {counts}",
        flush=True,
    )
    return receipt


def rating_arguments(store, family, every):
    """Initialize a new lineage evaluator; resume only after its state exists."""
    store = Path(store)
    mode = ["--resume"] if (store / "ratings/evaluator.pt").is_file() else ["--family", family]
    return [
        *mode,
        "--model-store", str(store),
        "--include-family", family,
        "--family-evaluation-every", str(every),
    ]


def campaign_runner(run, receipt):
    """Shared window ordering/cache propagation and optional boundary ratings."""
    from . import loop, plot_timing, timing
    from .feature_cache import sidecar_path

    class CampaignLoop(loop.LoopRunner):
        def __init__(self, root, state, lock_fd):
            super().__init__(root, state, lock_fd)
            state["bootstrap"].sort(key=lambda r: Path(r["source"]).name)

        def verify_runtime(self):
            super().verify_runtime()
            verify_files(run, receipt["runtime_hashes"])

        def snapshots(self):
            super().snapshots()
            for record in self.state["bootstrap"]:
                source = sidecar_path(Path(record["source"])).resolve()
                target = sidecar_path(self.path(record["path"]))
                target.parent.mkdir(exist_ok=True)
                if not target.exists() and not target.is_symlink():
                    target.symlink_to(source)
                if target.resolve() != source or not target.is_file():
                    raise ValueError(f"Missing or changed bootstrap cache: {target}")

        def save_milestone(self, iteration):
            super().save_milestone(iteration)
            every = receipt["rating_every"]
            number = iteration["number"]
            if not every or number % every:
                return
            name = f"{receipt['family']}-r{number:04d}"
            store = Path(self.config.model_store)
            registry_path = store / "champions.json"

            def complete():
                registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
                return registry.get("evaluated", {}).get(name, {}).get("complete")

            if complete():
                return
            journal = timing.journal(self.root, number)
            try:
                with timing.measure(journal, "evaluation"):
                    self.execute(
                        [
                            sys.executable,
                            "-m",
                            "continugo_train.rate_saved",
                            *rating_arguments(store, receipt["family"], every),
                        ],
                        self.path(f"iteration-{number:04d}/ratings.log"),
                        dict(os.environ, **{timing.ENVIRONMENT: str(journal)}),
                    )
            finally:
                if self._timing_iteration is None:
                    plot_timing.update(self.root)
            if not complete():
                raise ValueError(f"Champion evaluation not complete: {name}")

    return CampaignLoop


def frozen_environment(root, run):
    """Resolve external viewer assets before moving into the frozen package."""
    environment = dict(os.environ, PYTHONPATH=str(run / "runtime"))
    bundle = environment.get("CONTINUGO_VIEWER_BUNDLE")
    environment["CONTINUGO_VIEWER_BUNDLE"] = str(
        Path(bundle).resolve()
        if bundle
        else root.parent / "continugo/frontend/viewer/dist/continugo-viewer.js"
    )
    return environment


def stop(_signum, _frame):
    raise KeyboardInterrupt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--iterations", type=int, help="total target, not additional rounds")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--frozen", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    recipe_path = args.recipe.resolve()
    recipe = json.loads(recipe_path.read_text())
    root = Path.cwd().resolve()
    run = root / recipe["output"]
    iterations = args.iterations if args.iterations is not None else recipe["iterations"]
    if iterations < 1:
        parser.error("iterations must be positive")
    run.mkdir(parents=True, exist_ok=True)
    with (run / ".campaign.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not args.frozen:
            freeze(root, run, recipe)
        else:
            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            (run / "run.pid").write_text(str(os.getpid()) + "\n")

            def status(value, **extra):
                dump(
                    run / "status.json",
                    dict(
                        status=value,
                        pid=os.getpid(),
                        utc=datetime.now(UTC).isoformat(),
                        target_iterations=iterations,
                        **extra,
                    ),
                )

            try:
                from . import loop

                status("preparing")
                if args.prepare_only and recipe.get("fresh"):
                    status("runtime-prepared")
                    return
                receipt = prepare(root, run, recipe, lock.fileno())
                if args.prepare_only:
                    status("prepared")
                    return
                os.environ["CONTINUGO_FEATURE_BIN"] = str(run / "bin/features")
                status("running")
                base = loop.LoopRunner
                loop.LoopRunner = campaign_runner(run, receipt)
                try:
                    command = ["--output", str(run / "loop"), "--iterations", str(iterations)]
                    if (run / "loop/loop.pt").exists():
                        command += ["--resume"]
                    else:
                        command += [
                            "--checkpoint",
                            str(run / receipt.get("boundary_checkpoint", "parent.pt")),
                            "--bootstrap-data",
                            receipt.get("bootstrap_data", str(run / "seed-data")),
                        ]
                        for key, value in receipt["config"].items():
                            if value is not None:
                                if key == "sizes":
                                    value = ",".join(map(str, value))
                                command += ["--" + key.replace("_", "-"), str(value)]
                    loop.run_loop(loop.parse_args(command))
                finally:
                    loop.LoopRunner = base
                status("complete")
            except BaseException as exc:
                status(
                    "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                    error=repr(exc),
                )
                raise
            return
    # Run both the supervisor and all children from the same immutable package.
    environment = frozen_environment(root, run)
    command = [
        sys.executable,
        "-m",
        "continugo_train.campaign",
        str(run / "recipe.json"),
        "--frozen",
        "--iterations",
        str(iterations),
    ]
    if args.prepare_only:
        command.append("--prepare-only")
    os.execve(sys.executable, command, environment)


if __name__ == "__main__":
    main()
