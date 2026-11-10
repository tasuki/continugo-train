"""One independent model per family/round, with small portable metadata."""

import argparse
import filecmp
import json
import re
import shutil
import tempfile
from pathlib import Path

import torch

from . import archive
from .paths import ROOT
from .progress import checkpoint_progress, network_name
from .train import checkpoint_splits

FILES = ("checkpoint.pt", "encoder.onnx", "policy.onnx")


def campaign_name(root):
    root = Path(root).resolve()
    return root.parent.name if root.name == "loop" else root.name


def checked(root, record):
    path = root / record["path"]
    if (
        path.is_symlink()
        or not path.resolve().is_relative_to(root)
        or not path.is_file()
        or archive.digest(path) != record["sha256"]
    ):
        raise ValueError(f"campaign artifact missing or changed: {path}")
    return path


def verify(directory):
    directory = Path(directory)
    if directory.is_symlink():
        raise ValueError("saved model cannot be a symlink")
    metadata = json.loads((directory / "metadata.json").read_text())
    if set(metadata) != {"family", "round", "network_name", "progress", "source"}:
        raise ValueError("unexpected saved model metadata")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", metadata["family"]) or metadata["round"] < 1:
        raise ValueError("invalid model family/round")
    if directory.parent.name != metadata["family"] or directory.name != f"r{metadata['round']:04d}":
        raise ValueError("model directory disagrees with metadata")
    if {path.name for path in directory.iterdir()} != {*FILES, "metadata.json"}:
        raise ValueError("unexpected saved model files")
    for name in FILES:
        path = directory / name
        if path.is_symlink() or not path.is_file() or not path.stat().st_size:
            raise ValueError(f"saved model artifact missing: {path}")
        if path.stat().st_nlink != 1:
            raise ValueError(f"saved model must be an independent file: {path}")
    return metadata


def save_round(root, state, job, store=None):
    root = Path(root).resolve()
    if not job.get("completed") or not job.get("model") or not job.get("checkpoint"):
        raise ValueError("only completed, exported rounds can be saved")
    if state.get("version") not in {5, 6} or state.get("root") != str(root):
        raise ValueError("saving requires an unrelocated v5/v6 campaign")
    inputs = {"checkpoint.pt": checked(root, job["checkpoint"])}
    inputs.update(
        {f"{name}.onnx": checked(root, job["model"][name]) for name in ["encoder", "policy"]}
    )
    saved = torch.load(inputs["checkpoint.pt"], map_location="cpu", weights_only=True)
    progress = checkpoint_progress(saved)
    metadata = {
        "family": progress["family"],
        "round": job["number"],
        "network_name": network_name(saved),
        "progress": {
            "training_games": sum(split == "train" for split in checkpoint_splits(saved).values()),
            "samples_seen": progress["samples_seen"],
            "data_positions": progress["data_positions"],
        },
        "source": {"run": campaign_name(root), "iteration": job["number"]},
    }
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", metadata["family"]):
        raise ValueError("invalid model family")
    store = archive.initialize_store(store or state["config"]["model_store"])
    parent = store / "models" / metadata["family"]
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / f"r{job['number']:04d}"
    import fcntl

    with (store / ".archive.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            if verify(destination) != metadata or any(
                not filecmp.cmp(source, destination / name, shallow=False)
                for name, source in inputs.items()
            ):
                raise ValueError(
                    f"saved-model name conflicts with different or changed data: {destination}"
                )
            return destination
        with tempfile.TemporaryDirectory(prefix=".saving-", dir=parent) as temporary:
            stage = Path(temporary)
            for name, source in inputs.items():
                shutil.copyfile(source, stage / name)
                archive.seal_file(stage / name)
            (stage / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            archive.seal_file(stage / "metadata.json")
            archive.sync_directory(stage)
            stage.rename(destination)
        destination.chmod(0o555)
        archive.sync_directory(parent)
    return destination


def save_campaign(root, every=10, *, store=None, iteration=None):
    if every <= 0:
        raise ValueError("save interval must be positive")
    root = Path(root).resolve()
    state = torch.load(root / "loop.pt", map_location="cpu", weights_only=True)
    jobs = [
        job
        for job in state["iterations"]
        if job["completed"]
        and (job["number"] == iteration if iteration is not None else job["number"] % every == 0)
    ]
    if iteration is not None and not jobs:
        raise ValueError("requested round is not completed")
    return [save_round(root, state, job, store) for job in jobs]


def source_loop(store, run):
    """Locate training provenance without putting machine-specific paths in metadata."""
    for path in [
        ROOT / "runs" / run / "loop",
        Path(store).parent / run,
        Path(store).parent / run / "loop",
    ]:
        if (path / "loop.pt").is_file():
            return path.resolve()
    return None


def saved_sources(store):
    result, contexts = {}, {}
    for path in sorted(Path(store).glob("models/*/r*/metadata.json")):
        metadata = verify(path.parent)
        records = {
            name: {
                "path": str((path.parent / file).resolve()),
                "sha256": archive.digest(path.parent / file),
            }
            for name, file in [
                ("checkpoint", "checkpoint.pt"),
                ("encoder", "encoder.onnx"),
                ("policy", "policy.onnx"),
            ]
        }
        checksum = records["checkpoint"]["sha256"]
        run = metadata["source"]["run"]
        loop = source_loop(store, run)
        if loop is not None and run not in contexts:
            contexts[run] = torch.load(loop / "loop.pt", map_location="cpu", weights_only=True)
        # Runtime identities are computed in memory for native ONNX validation;
        # metadata.json and filenames contain no content hashes.
        result[checksum] = {
            **metadata,
            **records,
            "id": f"{metadata['family']}-r{metadata['round']:04d}",
            "checkpoint_sha256": checksum,
            "campaign": run,
            "training_games": metadata["progress"]["training_games"],
            "source_loop": str(loop) if loop else None,
            "source_context": contexts.get(run),
        }
    return result


def campaign_index(runs):
    result = {}
    for file in sorted(Path(runs).glob("*/loop/loop.pt")):
        state = torch.load(file, map_location="cpu", weights_only=True)
        if state.get("version") not in {5, 6}:
            continue
        for job in state["iterations"]:
            if job["completed"]:
                result[job["checkpoint"]["sha256"]] = {
                    "campaign": campaign_name(file.parent),
                    "round": job["number"],
                    "checkpoint": file.parent / job["checkpoint"]["path"],
                }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--loop", type=Path)
    source.add_argument("--runs", type=Path)
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--model-store", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args(argv)
    if args.iteration is not None and args.loop is None:
        parser.error("--iteration requires --loop")
    loops = [args.loop] if args.loop else sorted(args.runs.glob("*/loop"))
    try:
        paths = [
            path
            for root in loops
            for path in save_campaign(
                root, args.every, store=args.model_store, iteration=args.iteration
            )
        ]
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit(f"save-models error: {error}") from error
    for path in paths:
        print(path)
    print(f"Saved/verified {len(paths)} model snapshots; no training or evaluation launched.")


if __name__ == "__main__":
    main()
