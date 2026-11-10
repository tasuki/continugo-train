"""Read-only loop progress and independent training/actor/accepted identities."""

import argparse
import shutil
from collections import deque
from pathlib import Path

import torch

from . import promote
from .loop import VERSION
from .progress import checkpoint_progress, network_name


def inspect_loop(output):
    root = Path(output).resolve()
    state = torch.load(root / "loop.pt", map_location="cpu", weights_only=True)
    if state.get("version") != VERSION or state.get("root") != str(root):
        raise ValueError("status requires an unrelocated v6 manual-approval loop")
    jobs = state["iterations"]
    current = jobs[-1] if jobs else None
    if current is None:
        stage = "initialization"
    elif current["completed"]:
        stage = (
            "complete" if current["number"] == state["target_iterations"] else "between iterations"
        )
    elif not all(current["games"]):
        stage = "self-play"
    elif current["replay"] is None:
        stage = "replay selection"
    elif current["checkpoint"] is None:
        stage = "training"
    elif current["model"] is None:
        stage = "export"
    else:
        stage = "actor update / completion"
    training = state["training"]
    checkpoint = (
        root / training["checkpoint"]["path"]
        if training
        else root / state["initial_checkpoint"]["path"]
    )
    metrics = progress = name = None
    if checkpoint.is_file():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        metrics = saved.get("metrics")
        progress = checkpoint_progress(saved)
        name = network_name(saved)
    logs = []
    if current and current["train_attempts"]:
        logs.append(root / current["train_attempts"][-1]["path"] / "training.log")
    if current and current["selfplay_attempts"]:
        logs.append(root / current["selfplay_attempts"][-1]["path"] / "selfplay.log")
    return {
        "root": str(root),
        "stage": stage,
        "completed_iterations": sum(job["completed"] for job in jobs),
        "target_iterations": state["target_iterations"],
        "training": training,
        "training_name": name,
        "progress": progress,
        "metrics": metrics,
        "actor": state["actor"],
        "actor_policy": "latest",
        "accepted": promote.accepted_head(state["config"]["model_store"]),
        "current_iteration": current["number"] if current else None,
        "committed_games": sum(record is not None for record in current["games"]) if current else 0,
        "planned_games": len(current["games"]) if current else 0,
        "free_gib": shutil.disk_usage(root).free / 2**30,
        "logs": [str(path) for path in logs if path.is_file()],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    info = inspect_loop(args.output)
    print(f"Loop: {info['root']}\nStage: {info['stage']}")
    print(
        f"Iterations: {info['completed_iterations']}/{info['target_iterations']}; current={info['current_iteration']}"
    )
    print(f"Training: {info['training_name']} {info['progress']}")
    actor = info["actor"]
    print(
        f"Experimental actor ({info['actor_policy']}): {actor['model']['network_name'] if actor else 'initializing'}"
    )
    print(f"Actor checkpoint: {actor['checkpoint']['sha256'] if actor else 'not yet selected'}")
    accepted = info["accepted"]
    print(
        f"Accepted: {accepted['checkpoint_sha256'] if accepted else 'NONE (no user-approved publication in this store)'}"
    )
    if accepted and accepted["approval"] is None:
        print("Accepted head is historical, predating the explicit-approval protocol.")
    print(
        f"Committed self-play records: {info['committed_games']}/{info['planned_games']}; free disk={info['free_gib']:.1f} GiB"
    )
    print(f"Last completed training metrics: {info['metrics']}")
    for path in info["logs"]:
        print(f"\n--- {path} ---")
        with Path(path).open(errors="replace") as stream:
            print("".join(deque(stream, maxlen=8)).rstrip())
    print("No status/evaluation command grants permission to promote.")


if __name__ == "__main__":
    main()
