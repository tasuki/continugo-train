"""Write-once recovery bundles outside disposable runs/. No automatic deletion.

Accepted ONNX is accompanied by the exact trainable checkpoint and replay SGFs.
Every training candidate is backed up too, including rejected candidates. SGF
blobs are immutable copies in this store; hardlinks only connect store-owned data.
"""

import fcntl
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

import torch

from .paths import bundle_directory, utc_now
from .progress import checkpoint_progress, network_name
from .train import atomic_checkpoint


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def seal_file(path):
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
    path.chmod(0o444)


def checked_file(root, record):
    path = root / record["path"]
    if (
        path.is_symlink()
        or not path.resolve().is_relative_to(root.resolve())
        or not path.is_file()
        or digest(path) != record["sha256"]
    ):
        raise ValueError(f"recovery artifact missing or changed: {path}")
    return path


def verify_bundle(reference):
    root = Path(reference["directory"])
    metadata = root / "recovery.pt"
    if digest(metadata) != reference["sha256"]:
        raise ValueError(f"recovery manifest changed: {metadata}")
    saved = torch.load(metadata, map_location="cpu", weights_only=True)
    if saved.get("version") != 1:
        raise ValueError("incompatible recovery bundle")
    for record in saved["files"]:
        checked_file(root, record)
    return saved


def seal_directories(destination):
    for directory in [destination / "replay", destination / "evaluation", destination]:
        if directory.exists():
            directory.chmod(0o555)
            sync_directory(directory)


def latest_pointer(store, kind, destination):
    descriptor, temporary = tempfile.mkstemp(prefix=".latest-", dir=store)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(str(destination.relative_to(store)) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, store / f"LATEST-{kind.upper()}")
        sync_directory(store)
    finally:
        Path(temporary).unlink(missing_ok=True)


def find_bundle(parent, kind, checksum):
    # The name is for browsing, not identity. Verify the full digest in metadata.
    candidates = list(parent.glob(f"*-{checksum[:12]}"))
    legacy = parent / checksum
    if legacy.exists():
        candidates.append(legacy)
    matches = []
    for path in sorted(candidates):
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"unexpected archive entry: {path}")
        metadata = torch.load(path / "recovery.pt", map_location="cpu", weights_only=True)
        if metadata.get("checkpoint_sha256") == checksum:
            if metadata.get("kind") != kind:
                raise ValueError("archive checkpoint kind mismatch")
            matches.append(path)
    if len(matches) > 1:
        raise ValueError("multiple recovery bundles have the same checkpoint identity")
    return matches[0] if matches else None


def publish(
    store, kind, checkpoint, *, model=None, context=None, game_candidates=(), evaluation=()
):
    """Return a pinned bundle reference. Existing bundles are verified, not rewritten."""
    store, checkpoint = Path(store).resolve(), Path(checkpoint).resolve()
    if kind not in {"accepted", "training"}:
        raise ValueError("invalid archive kind")
    store.mkdir(parents=True, exist_ok=True)
    checksum = digest(checkpoint)
    parent = store / kind
    with (store / ".archive.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = store / "STORE.pt"
        if marker.exists():
            if torch.load(marker, weights_only=True) != {
                "format": "continugo-recovery-store",
                "version": 1,
            }:
                raise ValueError("unrecognized model store; choose a different --model-store")
        else:
            if any(path.name != ".archive.lock" for path in store.iterdir()):
                raise ValueError(
                    "model store is nonempty without a STORE.pt marker; refusing to overwrite existing data"
                )
            atomic_checkpoint(marker, {"format": "continugo-recovery-store", "version": 1})
            seal_file(marker)
        overview = store / "README.md"
        if not overview.exists():
            overview.write_text(
                "# ContinuGo durable model store\n\n"
                "accepted/YYYYMMDD-HHMMSS-SEQ-NAME-ID/ contains accepted encoder.onnx + policy.onnx, plus a full "
                "checkpoint.pt and replay/ snapshot for recovery. training/ uses the same UTC-dated names and preserves "
                "every completed training candidate, even when rejected or inconclusive.\n\n"
                "LATEST-ACCEPTED and LATEST-TRAINING are convenience pointers to the most "
                "recently selected bundle in each category. They can differ intentionally.\n\n"
                "ONNX alone is inference-only. Read each bundle's README.md for training "
                "recovery; code/rules and a compatible environment are still required.\n\n"
                "Bundles are write-once and read-only, with no automatic pruning. SGF blobs "
                "are store-owned copies, not links to disposable runs/. Back up this entire "
                "store off-machine (for example rsync -aH); a local directory cannot protect "
                "against disk loss. Do not chmod/delete bundles used by an active loop.\n"
            )
            seal_file(overview)
        parent.mkdir(exist_ok=True)
        destination = find_bundle(parent, kind, checksum)
        if destination is not None:
            reference = {
                "directory": str(destination),
                "sha256": digest(destination / "recovery.pt"),
            }
            metadata = verify_bundle(reference)
            if metadata["checkpoint_sha256"] != checksum or metadata["kind"] != kind:
                raise ValueError("archive checkpoint identity mismatch")
            seal_directories(destination)
            latest_pointer(store, kind, destination)
            return reference
        created = utc_now()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        name = network_name(saved)
        destination = bundle_directory(parent, checksum, created, name)
        fallbacks = {digest(path): Path(path) for path in game_candidates}
        files = []
        blobs = store / "sgf-blobs"
        blobs.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".publishing-", dir=destination.parent
        ) as temporary:
            stage = Path(temporary)

            def record(path):
                files.append({"path": str(path.relative_to(stage)), "sha256": digest(path)})

            def copy(source, relative):
                target = stage / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                seal_file(target)
                record(target)

            def sgf(source, expected, relative):
                source = Path(source)
                if not source.is_file() or digest(source) != expected:
                    raise ValueError(f"recovery SGF missing/changed: {source}")
                blob = blobs / f"{expected}.sgf"
                if blob.exists():
                    if blob.is_symlink() or digest(blob) != expected:
                        raise ValueError(f"recovery SGF blob corrupted: {blob}")
                else:
                    descriptor, name = tempfile.mkstemp(prefix=".sgf-", dir=blobs)
                    os.close(descriptor)
                    try:
                        shutil.copyfile(source, name)
                        if digest(name) != expected:
                            raise ValueError("SGF changed while archiving")
                        seal_file(Path(name))
                        os.rename(name, blob)
                    finally:
                        Path(name).unlink(missing_ok=True)
                target = stage / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(blob, target)
                record(target)

            copy(checkpoint, "checkpoint.pt")
            if digest(stage / "checkpoint.pt") != checksum:
                raise ValueError("checkpoint changed while archiving")
            if model is not None:
                for name in ["encoder.onnx", "policy.onnx"]:
                    copy(Path(model) / name, name)
            for game in saved["manifest"]:
                expected = game["file_sha256"]
                original = Path(game["path"])
                relocated = checkpoint.parent / "replay" / f"{expected}.sgf"
                source = (
                    original
                    if original.is_file() and digest(original) == expected
                    else (relocated if relocated.is_file() else fallbacks.get(expected))
                )
                if source is None:
                    raise ValueError(
                        f"cannot archive full training recovery: missing SGF {expected}"
                    )
                sgf(source, expected, f"replay/{expected}.sgf")
            for source in evaluation:
                expected = digest(source)
                sgf(source, expected, f"evaluation/{expected}.sgf")
            readme = stage / "README.md"
            readme.write_text(
                "# Immutable model recovery bundle\n\n"
                f"Published (UTC): {created.isoformat()}\n\n"
                f"Kind: {kind}\nNetwork: {name}\nCheckpoint SHA-256: {checksum}\n\n"
                f"Cumulative progress: {checkpoint_progress(saved)}\n\n"
                f"Epoch: {saved['epoch']}; local optimizer step: {saved['global_step']}\n\n"
                f"Loop target mix: {(context or {}).get('config', {}).get('size_weights', 'see checkpoint')}\n\n"
                f"Checkpoint sampling: {saved['options'].get('size_weights', 'uniform (legacy)')}\n\n"
                f"Acceptance: {(context or {}).get('gate') or 'initial checkpoint / training snapshot'}\n\n"
                "ONNX alone is inference-only. checkpoint.pt contains weights, BatchNorm, "
                "SGD state, RNG, options and holdout history; replay/ contains its exact SGFs.\n\n"
                "To continue learning, start a new loop with --checkpoint checkpoint.pt "
                "--bootstrap-data replay/ --warm-start-optimizer keep. To restore the latest "
                "training progress while retaining a different accepted opponent, also pass "
                "--incumbent-checkpoint PATH/TO/ACCEPTED/checkpoint.pt.\n\n"
                "For strict offline continuation, copy checkpoint.pt into a fresh run directory "
                "and use train --resume COPIED_CHECKPOINT --data THIS_BUNDLE/replay --epochs N. "
                "N must exceed the checkpoint epoch. Never train directly inside this bundle.\n\n"
                "The original loop's control/attempt history and the software environment are "
                "not reconstructed here. Keep the repository/rules and an off-machine backup. "
                "No local directory protects against disk loss.\n"
            )
            seal_file(readme)
            record(readme)
            atomic_checkpoint(
                stage / "recovery.pt",
                {
                    "version": 1,
                    "kind": kind,
                    "checkpoint_sha256": checksum,
                    "network_name": name,
                    "progress": checkpoint_progress(saved),
                    "created_at": created.isoformat(),
                    "files": files,
                    "context": context or {},
                },
            )
            seal_file(stage / "recovery.pt")
            for directory in [stage / "replay", stage / "evaluation"]:
                if directory.exists():
                    sync_directory(directory)
            sync_directory(stage)
            stage.rename(destination)
        # Read-only directories resist accidental recursive deletion. Owners can
        # explicitly chmod them for deliberate maintenance; this is not a backup.
        seal_directories(destination)
        sync_directory(blobs)
        sync_directory(destination.parent)
        reference = {"directory": str(destination), "sha256": digest(destination / "recovery.pt")}
        verify_bundle(reference)
        latest_pointer(store, kind, destination)
        return reference
