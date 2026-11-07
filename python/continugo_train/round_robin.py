"""Restartable rated head-to-head matches, without training or actor selection."""

import argparse
import shutil
from dataclasses import asdict, replace
from itertools import combinations
from pathlib import Path

import torch

from . import archive, gate, ratings
from .loop import LoopConfig, LoopRunner, digest, loop_lock
from .train import atomic_checkpoint

VERSION = "rated-round-robin-v1"


def available_models(store, *, feature_spec=None):
    """Rated models still available in campaigns or deliberately retained bundles."""
    with ratings.transaction(store) as (tables, _):
        rows = [r for r in tables.network_rows() if r["network_id"] != ratings.REFERENCE_ID]
    from .saved_models import saved_sources

    saved = saved_sources(store)
    available = []
    for row in rows:
        checksum = row["network_id"]
        # Prefer durable publications when available; never resurrect missing runs.
        source = None
        for kind in ["accepted", "retained", "experimental"]:
            source = archive.find_bundle(Path(store) / kind, kind, checksum)
            if source is not None:
                break
        if source is None:
            source = saved.get(checksum)
        if source is None:
            source = ratings.inference_source(store, checksum)
        if source is not None:
            if feature_spec is not None:
                checkpoint = (
                    source["checkpoint"]["path"]
                    if isinstance(source, dict)
                    else source / "checkpoint.pt"
                )
                saved_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if saved_checkpoint["feature_spec"] != feature_spec:
                    continue
            available.append((row, source))
    return available


def latest_bundles(store, count):
    if count < 2:
        raise ValueError("--latest must be at least 2")
    available = available_models(store)
    if len(available) < count:
        raise ValueError(
            "not enough available registered networks; restore a run or retain its models first"
        )
    return [source for _, source in available[:count]]


def initialize(root, source, bundles, games, seed):
    if len(bundles) < 2:
        raise ValueError("at least two bundles are required")
    config = replace(LoopConfig(**source["config"]), gate_games=games, seed=seed)
    config.validate()
    if not games:
        raise ValueError("--games must be positive")
    pairs = len(bundles) * (len(bundles) - 1) // 2
    if seed + pairs * (games // 2) > 2**64:
        raise ValueError("evaluation game seeds overflow u64")
    store = Path(config.model_store).resolve()
    if root.is_relative_to(store) or store.is_relative_to(root):
        raise ValueError("evaluation output must be separate from the model store")
    settings = ratings.protocol(config, legacy=source["rating_protocol"]["version"] == 1)
    if settings != source["rating_protocol"]:
        raise ValueError("source loop evaluation binaries/settings changed")
    models, checkpoints = snapshot_models(root, source, bundles)
    jobs = [
        {
            "number": number,
            "model": models[a],
            "parent_model": models[b],
            "checkpoint": checkpoints[a],
            "parent_actor_checkpoint": checkpoints[b],
            "gate": None,
        }
        for number, (a, b) in enumerate(combinations(range(len(models)), 2), 1)
    ]
    return {
        "version": VERSION,
        "root": str(root),
        "config": asdict(config),
        "feature_spec": source["feature_spec"],
        "rating_protocol": settings,
        "runtime_fingerprints": source["runtime_fingerprints"],
        "jobs": jobs,
    }


def snapshot_models(root, source, bundles):
    models, checkpoints = [], []
    for index, bundle in enumerate(bundles):
        if isinstance(bundle, dict):
            metadata = bundle
            inputs = {
                "checkpoint.pt": bundle["checkpoint"],
                "encoder.onnx": bundle["encoder"],
                "policy.onnx": bundle["policy"],
            }
        else:
            bundle = Path(bundle).resolve()
            metadata = archive.verify_bundle(
                {"directory": str(bundle), "sha256": digest(bundle / "recovery.pt")}
            )
            inputs = {
                name: {"path": str(bundle / name), "sha256": digest(bundle / name)}
                for name in ["checkpoint.pt", "encoder.onnx", "policy.onnx"]
            }
        saved = torch.load(inputs["checkpoint.pt"]["path"], map_location="cpu", weights_only=True)
        if saved["feature_spec"] != source["feature_spec"]:
            raise ValueError("all models must share the source loop's feature specification")
        checksum = metadata["checkpoint_sha256"]
        if checksum in {m["checkpoint_sha256"] for m in models}:
            raise ValueError("duplicate checkpoint in round-robin")
        directory = root / f"models/{index:04d}"
        directory.mkdir(parents=True)
        records = {}
        for name in ["encoder.onnx", "policy.onnx", "checkpoint.pt"]:
            path = directory / name
            source_file = Path(inputs[name]["path"])
            if source_file.is_symlink() or digest(source_file) != inputs[name]["sha256"]:
                raise ValueError("inference artifact missing or changed before copying")
            shutil.copyfile(source_file, path)
            records[name] = {"path": str(path.relative_to(root)), "sha256": digest(path)}
            if records[name]["sha256"] != inputs[name]["sha256"]:
                raise ValueError("inference artifact changed while copying")
        if records["checkpoint.pt"]["sha256"] != checksum:
            raise ValueError("checkpoint identity mismatch")
        checkpoints.append(records["checkpoint.pt"])
        models.append(
            {
                "directory": str(directory.relative_to(root)),
                "checkpoint_sha256": checksum,
                "network_name": metadata["network_name"],
                "encoder": records["encoder.onnx"],
                "policy": records["policy.onnx"],
            }
        )
    return models, checkpoints


def run(args):
    root = args.output.resolve()
    if args.ingest_only and not args.resume:
        raise ValueError("--ingest-only requires --resume")
    root.mkdir(parents=True, exist_ok=True)
    with loop_lock(root) as lock_fd:
        if args.resume:
            if (
                args.loop
                or args.bundles
                or args.latest is not None
                or args.games is not None
                or args.seed is not None
            ):
                raise ValueError("resume restores the frozen schedule; do not supply new settings")
            state = torch.load(root / "loop.pt", map_location="cpu", weights_only=True)
            if state.get("version") != VERSION or state.get("root") != str(root):
                raise ValueError("incompatible/relocated round-robin")
        else:
            if any(p.name != ".loop.lock" for p in root.iterdir()):
                raise ValueError("output is not empty; use --resume or a fresh directory")
            if args.loop is None:
                raise ValueError("--loop is required to supply evaluation settings")
            if args.bundles and args.latest is not None:
                raise ValueError("choose --bundles or --latest, not both")
            source = torch.load(args.loop / "loop.pt", map_location="cpu", weights_only=True)
            bundles = args.bundles or latest_bundles(
                source["config"]["model_store"], args.latest if args.latest is not None else 4
            )
            state = initialize(
                root,
                source,
                bundles,
                args.games if args.games is not None else 20,
                args.seed if args.seed is not None else 9100000,
            )
            atomic_checkpoint(root / "loop.pt", state)
        runner = LoopRunner(root, state, lock_fd)
        runner.verify_runtime()
        for job in state["jobs"]:
            if args.ingest_only and not job.get("gate"):
                continue
            checkpoints = [job["checkpoint"], job["parent_actor_checkpoint"]]
            for record in checkpoints:
                runner.check(record)
            runner.check_model(job["model"])
            runner.check_model(job["parent_model"])
            for record in (job.get("gate") or {}).get("games", []):
                if record:
                    runner.check(record)
            runner.log(
                f"RATED MATCH {job['number']}/{len(state['jobs'])}: "
                f"{job['model']['network_name']} vs {job['parent_model']['network_name']}"
            )
            gate.evaluate(runner, job, adopt_only=args.ingest_only)
            if args.ingest_only:
                records = [r for r in job["gate"]["games"] if r]
                if records:
                    result = ratings.ingest(
                        runner.config.model_store,
                        state["rating_protocol"],
                        [runner.check(r) for r in records],
                        checkpoints=[runner.check(r) for r in checkpoints],
                        kind="gate",
                        rating_binary=runner.config.rating_binary,
                        feature_binary=runner.config.feature_binary,
                    )
                    runner.log(f"Preserved completed evidence: {result}")
            else:
                runner.rate_evaluation(job, "gate", checkpoints)
        runner.log(
            "Evidence ingested; no new games launched"
            if args.ingest_only
            else "Rated round-robin complete; no training, actor changes, or publication"
        )
        return state


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--loop", type=Path, help="source loop directory; read-only evaluation settings"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="on resume, preserve completed partial-match SGFs without launching games",
    )
    parser.add_argument("--latest", type=int, help="newest registered networks (default 4)")
    parser.add_argument("--bundles", type=Path, nargs="+")
    parser.add_argument("--games", type=int, help="paired-color games per matchup (default 20)")
    parser.add_argument("--seed", type=int, help="fresh evaluation seed (default 9100000)")
    return parser.parse_args(argv)


def main(argv=None):
    try:
        run(parse_args(argv))
    except KeyboardInterrupt:
        raise SystemExit("Interrupted; resume with --output SAME_DIRECTORY --resume") from None
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit(f"round-robin error: {error}") from error


if __name__ == "__main__":
    main()
