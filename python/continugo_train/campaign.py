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


def prepare(root, run, recipe):
    import torch

    from . import loop
    from .train import atomic_checkpoint, checkpoint_splits

    hashes = json.loads((run / "runtime.json").read_text())["hashes"]
    verify_files(run, hashes)
    # Once committed, preparation no longer depends on mutable ancestral loop state.
    launch = run / "launch.json"
    if launch.exists():
        receipt = json.loads(launch.read_text())
        if sha(run / "parent.pt") != receipt["boundary_checkpoint_sha256"]:
            raise ValueError("Campaign boundary checkpoint changed")
        return receipt
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
    # A new fork adopts today's window policy, not an ancestor's legacy window.
    config["training_window_fraction"] = loop.LoopConfig().training_window_fraction
    config.update(recipe["overrides"])
    config.update(
        selfplay_binary=str(run / "bin/continugo-train"),
        feature_binary=str(run / "bin/features"),
        model_store=str(root / "artifacts"),
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
            registry_path = store / "ratings/champions.json"

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
                            "--resume",
                            "--model-store",
                            str(store),
                            "--include-family",
                            receipt["family"],
                            "--family-evaluation-every",
                            str(every),
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
                receipt = prepare(root, run, recipe)
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
                            str(run / "parent.pt"),
                            "--bootstrap-data",
                            str(run / "seed-data"),
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
