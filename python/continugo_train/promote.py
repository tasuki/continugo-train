"""Explicit, hash-pinned user approval. Never called by the training loop.

An immutable approval intent precedes publication; a receipt follows it. Recovery
can only finish already authorized intents. The latest receipt owns the accepted
head, so retrying an older approval cannot roll that head backwards.
"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import torch

from . import archive, ratings
from .paths import utc_now
from .train import atomic_checkpoint

ROOT = Path(__file__).resolve().parents[2]


def identity(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@contextmanager
def publication_lock(store):
    store = Path(store).resolve()
    if store.is_relative_to(ROOT / "runs"):
        raise ValueError("approval store must be outside disposable runs/")
    marker = store / "STORE.pt"
    if not marker.is_file() or torch.load(marker, weights_only=True) != {
        "format": "continugo-recovery-store",
        "version": 1,
    }:
        raise ValueError("approval requires an initialized model store")
    with (store / ".promotion.lock").open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield store


def load_intent(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("approval intent cannot be a symlink")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("version") != 1
        or payload.get("authorization") != "explicit-user-approval"
        or payload.get("reference_strength_confirmed") is not True
        or not isinstance(payload.get("sequence"), int)
        or payload["sequence"] < 1
        or not re.fullmatch(r"[0-9a-f]{64}", payload.get("checkpoint_sha256", ""))
        or not payload.get("approved_by", "").strip()
        or not payload.get("reason", "").strip()
        or path.name != f"{payload['sequence']:08d}-{identity(payload)}.pt"
    ):
        raise ValueError("invalid or changed approval intent")
    return payload


def validate_authorization(store, reference, checksum, model):
    """Called by the archive writer *before* any accepted publication/reuse."""
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("accepted publication requires explicit user approval")
    store = Path(store).resolve()
    path = Path(reference["path"])
    if (
        path.is_symlink()
        or path.resolve().parent != store / "approvals"
        or archive.digest(path) != reference["sha256"]
    ):
        raise ValueError("approval must be an unchanged store-owned intent")
    payload = load_intent(path)
    if payload["store"] != str(store) or payload["checkpoint_sha256"] != checksum or model is None:
        raise ValueError("approval does not authorize this checkpoint/store/model")
    source = archive.verify_bundle(payload["source"])
    directory = Path(payload["source"]["directory"]).resolve()
    if (
        source["kind"] != "experimental"
        or directory.parent != store / "experimental"
        or Path(model).resolve() != directory
    ):
        raise ValueError("approval requires the pinned experimental recovery bundle")
    if source["checkpoint_sha256"] != checksum or any(
        archive.digest(directory / name) != value for name, value in payload["model"].items()
    ):
        raise ValueError("approved model identity changed")
    return path, payload


def verify_embedded(root, metadata):
    """Accepted bundles retain their approval even when copied off the machine."""
    path = root / "approval.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        metadata.get("approval", {}).get("sha256") != archive.digest(path)
        or payload.get("authorization") != "explicit-user-approval"
        or payload.get("reference_strength_confirmed") is not True
        or payload.get("checkpoint_sha256") != metadata["checkpoint_sha256"]
        or not payload.get("approved_by")
        or not payload.get("reason")
        or Path(metadata["approval"]["path"]).name
        != f"{payload['sequence']:08d}-{identity(payload)}.pt"
        or set(payload.get("model", {})) != {"encoder.onnx", "policy.onnx"}
        or any(archive.digest(root / name) != value for name, value in payload["model"].items())
    ):
        raise ValueError("accepted bundle has no matching explicit approval")


def verify_model(directory):
    # Export verification is independent of approval. Neither a moving alias nor
    # a graph from a different training checkpoint may be authorized accidentally.
    from .loop import load_training, verify_export

    checkpoint = directory / "checkpoint.pt"
    # verify_export expects a directory containing only the two inference graphs;
    # durable bundles contain other recovery files, so verify a private view.
    with tempfile.TemporaryDirectory(prefix=".approval-model-", dir=directory.parent) as temporary:
        view = Path(temporary)
        for name in ["encoder.onnx", "policy.onnx"]:
            os.link(directory / name, view / name)
        verify_export(view, checkpoint, load_training(checkpoint))


def _intents(store):
    directory = store / "approvals"
    if directory.is_symlink():
        raise ValueError("approval directory cannot be a symlink")
    directory.mkdir(exist_ok=True)
    archive.sync_directory(store)
    if any(
        not path.with_name(path.name.replace(".receipt.pt", ".pt")).is_file() or path.is_symlink()
        for path in directory.glob("*.receipt.pt")
    ):
        raise ValueError("approval receipt is missing its immutable intent")
    records = [
        (path, load_intent(path))
        for path in sorted(directory.glob("[0-9]*.pt"))
        if not path.name.endswith(".receipt.pt")
    ]
    if [payload["sequence"] for _, payload in records] != list(range(1, len(records) + 1)):
        raise ValueError("approval sequence is missing or conflicting")
    if len({payload["checkpoint_sha256"] for _, payload in records}) != len(records):
        raise ValueError("duplicate checkpoint approval intents")
    return records


def _recover(store):
    last = None
    results = {}
    for path, payload in _intents(store):
        receipt = path.with_suffix(".receipt.pt")
        reference = {"path": str(path), "sha256": archive.digest(path)}
        if receipt.exists():
            saved = torch.load(receipt, map_location="cpu", weights_only=True)
            if saved.get("approval") != reference:
                raise ValueError("approval receipt changed")
            accepted = saved["accepted"]
            metadata = archive.verify_bundle(accepted)
            if (
                metadata.get("approval") != reference
                or metadata["checkpoint_sha256"] != payload["checkpoint_sha256"]
            ):
                raise ValueError("approval receipt does not match its accepted bundle")
            ratings.register_accepted(store, accepted)
        else:
            directory = Path(payload["source"]["directory"])
            source = archive.verify_bundle(payload["source"])
            accepted = archive.publish(
                store,
                "accepted",
                directory / "checkpoint.pt",
                model=directory,
                approval=reference,
                evaluation=sorted((directory / "evaluation").glob("*.sgf")),
                context={**source["context"], "user_approval": payload},
            )
            atomic_checkpoint(receipt, {"approval": reference, "accepted": accepted})
            archive.seal_file(receipt)
            archive.sync_directory(path.parent)
        last = accepted
        results[payload["checkpoint_sha256"]] = accepted
    if last is not None:
        archive.latest_pointer(store, "accepted", Path(last["directory"]))
    return results


def recover(store):
    """Finish existing authorizations only; cannot create permission."""
    store = Path(store).resolve()
    with publication_lock(store):
        return _recover(store)


def approve(
    store, candidate, checkpoint_sha256, *, approved_by, reason, reference_strength_confirmed=False
):
    if (
        not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256)
        or not isinstance(approved_by, str)
        or not approved_by.strip()
        or not isinstance(reason, str)
        or not reason.strip()
        or reference_strength_confirmed is not True
    ):
        raise ValueError(
            "explicit user identity, reason, full checkpoint hash and reference-strength confirmation are required"
        )
    store = Path(store).resolve()
    with publication_lock(store):
        candidate = Path(candidate)
        if candidate.is_symlink():
            raise ValueError("candidate cannot be a symlink")
        candidate = candidate.resolve()
        if candidate.parent != store / "experimental":
            raise ValueError("candidate must be a store-owned experimental bundle")
        source_reference = {
            "directory": str(candidate),
            "sha256": archive.digest(candidate / "recovery.pt"),
        }
        source = archive.verify_bundle(source_reference)
        if source["kind"] != "experimental" or source["checkpoint_sha256"] != checkpoint_sha256:
            raise ValueError("candidate does not match the explicitly approved checkpoint hash")
        request = {
            "store": str(store),
            "source": source_reference,
            "checkpoint_sha256": checkpoint_sha256,
            "approved_by": approved_by.strip(),
            "reason": reason.strip(),
            "reference_strength_confirmed": True,
            "authorization": "explicit-user-approval",
            "model": {
                name: archive.digest(candidate / name) for name in ["encoder.onnx", "policy.onnx"]
            },
        }
        existing = _intents(store)
        for _, payload in existing:
            if payload["checkpoint_sha256"] == checkpoint_sha256:
                if any(payload.get(key) != value for key, value in request.items()):
                    raise ValueError("this checkpoint already has a different immutable approval")
                return _recover(store)[checkpoint_sha256]
        _recover(store)
        verify_model(candidate)
        payload = {
            "version": 1,
            **request,
            "sequence": len(existing) + 1,
            "approved_at": utc_now().isoformat(),
        }
        path = store / "approvals" / f"{payload['sequence']:08d}-{identity(payload)}.pt"
        atomic_checkpoint(path, payload)
        archive.seal_file(path)
        archive.sync_directory(path.parent)
        return _recover(store)[checkpoint_sha256]


def validate_head_update(store, destination, metadata):
    records = _intents(Path(store).resolve())
    if not records:
        raise ValueError("accepted head requires a committed user-approval receipt")
    path, payload = records[-1]
    receipt = path.with_suffix(".receipt.pt")
    if not receipt.is_file():
        raise ValueError("latest approval is not committed; recover publication first")
    saved = torch.load(receipt, map_location="cpu", weights_only=True)
    reference = {"path": str(path), "sha256": archive.digest(path)}
    expected = {
        "directory": str(Path(destination).resolve()),
        "sha256": archive.digest(Path(destination) / "recovery.pt"),
    }
    if (
        saved.get("approval") != reference
        or saved.get("accepted") != expected
        or metadata.get("approval") != reference
        or metadata["checkpoint_sha256"] != payload["checkpoint_sha256"]
    ):
        raise ValueError("accepted head must identify the latest committed approval")


def accepted_head(store):
    """Read, never create, an accepted model. Historical bundles remain valid."""
    store = Path(store).resolve()
    pointer = store / "LATEST-ACCEPTED"
    if not pointer.exists():
        return None
    if pointer.is_symlink():
        raise ValueError("accepted pointer cannot be a symlink")
    relative = Path(pointer.read_text().strip())
    directory = store / relative
    if (
        relative.is_absolute()
        or directory.is_symlink()
        or directory.resolve().parent != store / "accepted"
    ):
        raise ValueError("accepted pointer escapes its store")
    reference = {
        "directory": str(directory.resolve()),
        "sha256": archive.digest(directory / "recovery.pt"),
    }
    metadata = archive.verify_bundle(reference)
    if metadata["kind"] != "accepted":
        raise ValueError("accepted pointer is not an accepted bundle")
    return {
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "archive": reference,
        "approval": metadata.get("approval"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-store", type=Path, required=True)
    parser.add_argument(
        "--recover", action="store_true", help="finish only previously authorized publications"
    )
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--approved-by")
    parser.add_argument("--reason", help="user approval and reviewed reference-strength evidence")
    parser.add_argument("--confirm-much-stronger-than-reference", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.recover:
            if any(
                [
                    args.candidate,
                    args.checkpoint_sha256,
                    args.approved_by,
                    args.reason,
                    args.confirm_much_stronger_than_reference,
                ]
            ):
                raise ValueError("--recover cannot issue new approvals")
            result = recover(args.model_store)
        else:
            if not all([args.candidate, args.checkpoint_sha256, args.approved_by, args.reason]):
                raise ValueError("promotion requires explicit candidate/hash/user/reason arguments")
            result = approve(
                args.model_store,
                args.candidate,
                args.checkpoint_sha256,
                approved_by=args.approved_by,
                reason=args.reason,
                reference_strength_confirmed=args.confirm_much_stronger_than_reference,
            )
        print(result)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"promotion error: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
