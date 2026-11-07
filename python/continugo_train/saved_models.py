"""Save selected campaign rounds as independent, hash-verified model copies."""

import argparse
import fcntl
import json
import shutil
import tempfile
from pathlib import Path

import torch

from . import archive
from .paths import ROOT, utc_now
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
    if metadata.get("version") != 1 or set(metadata.get("files", {})) != set(FILES):
        raise ValueError("incompatible saved model metadata")
    if {p.name for p in directory.iterdir()} != {*FILES, "metadata.json"}:
        raise ValueError("unexpected saved model files")
    for name in FILES:
        path = directory / name
        if (
            path.is_symlink()
            or not path.is_file()
            or archive.digest(path) != metadata["files"][name]
        ):
            raise ValueError(f"saved model artifact missing or changed: {path}")
        if path.stat().st_nlink != 1:
            raise ValueError(f"saved model must be an independent file: {path}")
    if metadata["files"]["checkpoint.pt"] != metadata["checkpoint_sha256"]:
        raise ValueError("saved checkpoint identity mismatch")
    return metadata


def save_round(root, state, job, store=None):
    root = Path(root).resolve()
    if not job.get("completed") or job.get("model") is None or job.get("checkpoint") is None:
        raise ValueError("only completed, exported rounds can be saved")
    if state.get("version") not in {5, 6} or state.get("root") != str(root):
        raise ValueError("saving requires an unrelocated v5/v6 campaign")
    checkpoint = checked(root, job["checkpoint"])
    inputs = {"checkpoint.pt": checkpoint}
    inputs.update(
        {f"{name}.onnx": checked(root, job["model"][name]) for name in ["encoder", "policy"]}
    )
    checksum = job["checkpoint"]["sha256"]
    if job["model"]["checkpoint_sha256"] != checksum:
        raise ValueError("export belongs to a different checkpoint")
    name = campaign_name(root)
    destination_name = f"{name}-r{job['number']:04d}"
    store = archive.initialize_store(store or state["config"]["model_store"])
    destination = store / destination_name
    expected = {file: archive.digest(path) for file, path in inputs.items()}
    with (store / ".archive.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists() or destination.is_symlink():
            metadata = verify(destination)
            if (
                metadata["campaign"] != name
                or metadata["round"] != job["number"]
                or metadata["source_loop"] != str(root)
                or metadata["files"] != expected
            ):
                raise ValueError(
                    f"saved-model name conflicts with different provenance: {destination}"
                )
            return destination
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        metadata = {
            "version": 1,
            "campaign": name,
            "round": job["number"],
            "source_loop": str(root),
            "source_context": {
                key: state[key]
                for key in [
                    "version",
                    "config",
                    "feature_spec",
                    "runtime_fingerprints",
                    "rating_protocol",
                ]
            },
            "checkpoint_sha256": checksum,
            "network_name": network_name(checkpoint_data),
            "saved_at": utc_now().isoformat(),
            "progress": checkpoint_progress(checkpoint_data),
            "training_games": sum(
                split == "train" for split in checkpoint_splits(checkpoint_data).values()
            ),
            "accepted": False,
            "files": expected,
        }
        with tempfile.TemporaryDirectory(prefix=".saving-", dir=store) as temporary:
            stage = Path(temporary)
            for file, source in inputs.items():
                shutil.copyfile(source, stage / file)
                if archive.digest(stage / file) != expected[file]:
                    raise ValueError("model changed while saving")
                archive.seal_file(stage / file)
            (stage / "metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n"
            )
            archive.seal_file(stage / "metadata.json")
            verify(stage)
            archive.sync_directory(stage)
            stage.rename(destination)
        destination.chmod(0o555)
        archive.sync_directory(store)
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


def saved_sources(store):
    result = {}
    for path in sorted(Path(store).glob("*-r*/metadata.json")):
        metadata = verify(path.parent)
        result[metadata["checkpoint_sha256"]] = {
            **metadata,
            **{
                name: {
                    "path": str((path.parent / file).resolve()),
                    "sha256": metadata["files"][file],
                }
                for name, file in [
                    ("checkpoint", "checkpoint.pt"),
                    ("encoder", "encoder.onnx"),
                    ("policy", "policy.onnx"),
                ]
            },
        }
    return result


def campaign_index(runs):
    """Attribute checkpoint identities to their producing campaign, not a child fork."""
    result = {}
    states = []
    for file in sorted(Path(runs).glob("*/loop/loop.pt")):
        state = torch.load(file, map_location="cpu", weights_only=True)
        if state.get("version") in {5, 6} and state.get("root") == str(file.parent.resolve()):
            states.append((file.parent.resolve(), state))
    for completed in [False, True]:
        for root, state in states:
            records = (
                [(0, state.get("initial_actor_checkpoint"))]
                if not completed
                else [
                    (job["number"], job.get("checkpoint"))
                    for job in state["iterations"]
                    if job["completed"]
                ]
            )
            for number, record in records:
                if record:
                    result[record["sha256"]] = {
                        "campaign": campaign_name(root),
                        "round": number,
                        "checkpoint": root / record["path"],
                    }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--loop", type=Path)
    source.add_argument(
        "--runs", type=Path, help="backfill every campaign directly under this directory"
    )
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--model-store", type=Path, default=ROOT / "models")
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
