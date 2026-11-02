"""Write-once recovery bundles outside disposable runs/. No automatic deletion.

Accepted ONNX is accompanied by the exact trainable checkpoint and replay SGFs.
Every training candidate is backed up too, including rejected candidates. SGF
blobs are immutable copies in this store; hardlinks only connect store-owned data.
Experimental actors remain unaccepted. Accepted publication requires a matching
explicit user authorization, and only committed approval receipts can move its head.
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
    if saved.get("version") not in {1, 2} or saved.get("kind") not in {
        "training",
        "experimental",
        "accepted",
    }:
        raise ValueError("incompatible recovery bundle")
    for record in saved["files"]:
        checked_file(root, record)
    if saved["version"] == 2:
        if saved["kind"] != "accepted":
            raise ValueError("approval format is only valid for accepted bundles")
        from .promote import verify_embedded

        verify_embedded(root, saved)
    return saved


def seal_directories(destination):
    for directory in [destination / "replay", destination / "evaluation", destination]:
        if directory.exists():
            directory.chmod(0o555)
            sync_directory(directory)


def latest_pointer(store, kind, destination):
    if kind == "accepted":
        metadata = verify_bundle(
            {"directory": str(destination), "sha256": digest(destination / "recovery.pt")}
        )
        if (
            metadata.get("version") != 2
            or not metadata.get("approval")
            or destination.resolve().parent != Path(store).resolve() / "accepted"
        ):
            raise ValueError("accepted pointer changes require an explicitly approved bundle")
        from .promote import validate_head_update

        validate_head_update(store, destination, metadata)
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


def _publish(
    store,
    kind,
    checkpoint,
    *,
    model=None,
    context=None,
    game_candidates=(),
    evaluation=(),
    approval=None,
):
    """Return a pinned bundle reference. Existing bundles are verified, not rewritten."""
    store, checkpoint = Path(store).resolve(), Path(checkpoint).resolve()
    if kind not in {"accepted", "training", "experimental"}:
        raise ValueError("invalid archive kind")
    if kind == "accepted" and approval is None:
        raise ValueError("accepted publication requires explicit user approval")
    if kind == "experimental" and model is None:
        raise ValueError("experimental recovery requires a frozen inference model")
    store.mkdir(parents=True, exist_ok=True)
    checksum = digest(checkpoint)
    approval_path = None
    parent = store / kind
    if parent.is_symlink():
        raise ValueError("archive kind directory cannot be a symlink")
    with (store / ".archive.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if kind == "accepted":
            from .promote import validate_authorization

            approval_path, _ = validate_authorization(store, approval, checksum, model)
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
                "experimental/ contains unaccepted frozen actors with full recovery data. "
                "LATEST-EXPERIMENTAL tracks the selected self-play actor independently of training. "
                "Accepted publication requires a hash-pinned explicit user approval; "
                "approvals/ retains immutable approval intents and publication receipts.\n\n"
                "ratings/networks.csv lists all networks (filter accepted=1); "
                "ratings/matches.csv retains processed SGF hashes and rating history. "
                "ratings/games/ retains authoritative SGF evidence. "
                "Ratings are distinct from the acceptance gate.\n\n"
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
            if kind == "accepted" and metadata.get("approval") != approval:
                raise ValueError("existing accepted bundle has a different or historical approval")
            seal_directories(destination)
            if kind == "training":
                latest_pointer(store, kind, destination)
            return reference
        created = utc_now()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        label = network_name(saved)
        destination = bundle_directory(parent, checksum, created, label)
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
                blob = _sgf_blob(blobs, source, expected)
                target = stage / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(blob, target)
                record(target)

            if approval_path is not None:
                copy(approval_path, "approval.pt")
                if digest(stage / "approval.pt") != approval["sha256"]:
                    raise ValueError("approval changed during publication")
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
                f"Kind: {kind}\nNetwork: {label}\nCheckpoint SHA-256: {checksum}\n\n"
                f"Cumulative progress: {checkpoint_progress(saved)}\n\n"
                f"Epoch: {saved['epoch']}; local optimizer step: {saved['global_step']}\n\n"
                f"Loop target mix: {(context or {}).get('config', {}).get('size_weights', 'see checkpoint')}\n\n"
                f"Checkpoint sampling: {saved['options'].get('size_weights', 'uniform (legacy)')}\n\n"
                f"Experimental match evidence (not approval): {(context or {}).get('gate')}\n\n"
                f"User approval: {approval or 'NONE: unaccepted recovery artifact'}\n\n"
                "ONNX alone is inference-only. checkpoint.pt contains weights, BatchNorm, "
                "SGD state, RNG, options and holdout history; replay/ contains its exact SGFs.\n\n"
                "To continue learning, start a new loop with --checkpoint checkpoint.pt "
                "--bootstrap-data replay/ --warm-start-optimizer keep. To restore the latest "
                "training progress while retaining a different experimental actor, also pass "
                "--actor-checkpoint PATH/TO/EXPERIMENTAL/checkpoint.pt. "
                "Neither operation marks a model accepted; use the separate manual approval command.\n\n"
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
                    "version": 2 if kind == "accepted" else 1,
                    "kind": kind,
                    **({"approval": approval} if approval is not None else {}),
                    "checkpoint_sha256": checksum,
                    "network_name": label,
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
        if kind == "training":
            latest_pointer(store, kind, destination)
        return reference


def _sgf_blob(blobs, source, expected):
    """Caller holds the archive lock. Never link disposable run-owned files."""
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
    return blob


def retain_evidence(store, records, destination):
    """Write-once, self-contained rating SGFs sharing only store-owned inodes."""
    store, destination = Path(store), Path(destination)
    if not (store / "STORE.pt").is_file():
        raise ValueError("rating evidence needs an initialized model store")
    blobs = store / "sgf-blobs"
    with (store / ".archive.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        blobs.mkdir(exist_ok=True)
        destination.mkdir(parents=True, exist_ok=True)
        for source, expected in records:
            blob = _sgf_blob(blobs, source, expected)
            target = destination / blob.name
            if target.exists():
                if target.is_symlink() or digest(target) != expected:
                    raise ValueError(f"rating SGF missing/changed: {target}")
            else:
                os.link(blob, target)
        sync_directory(destination)
        sync_directory(blobs)


def publish(store, kind, checkpoint, **kwargs):
    reference = _publish(store, kind, checkpoint, **kwargs)
    from .ratings import register_accepted, register_checkpoint

    # Release the archive lock before taking the rating-ledger lock.
    if kind == "accepted":
        register_accepted(store, reference)
    else:
        register_checkpoint(store, Path(reference["directory"]) / "checkpoint.pt")
    return reference
