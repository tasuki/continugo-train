"""Deliberately retain one campaign model with its exact checkpoint and replay.

Retention is not approval and never changes the campaign's training or actor.
"""

import argparse
from pathlib import Path

import torch

from .loop import VERSION, LoopRunner


def retain(output, iteration=None):
    root = Path(output).resolve()
    state = torch.load(root / "loop.pt", map_location="cpu", weights_only=True)
    if state.get("version") != VERSION or state.get("root") != str(root):
        raise ValueError("retention requires an unrelocated v6 campaign")
    runner = LoopRunner(root, state, None)
    if iteration == 0:
        checkpoint, model, job = state["initial_actor_checkpoint"], state["initial_model"], None
    else:
        jobs = [job for job in state["iterations"] if job["completed"]]
        if iteration is not None:
            jobs = [job for job in jobs if job["number"] == iteration]
        if not jobs:
            raise ValueError(
                "select a completed iteration (or --iteration 0 for the initial actor)"
            )
        job = jobs[-1]
        checkpoint, model = job["checkpoint"], job["model"]
    if model is None:
        raise ValueError("selected model has not been exported")
    # Read a single atomic state snapshot. Committed checkpoint, graph and SGF
    # files are immutable; archive.publish verifies their hashes while copying.
    runner.check(checkpoint)
    runner.check_model(model)
    return runner.recovery(checkpoint, "retained", model=model, iteration=job)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop", type=Path, required=True)
    parser.add_argument(
        "--iteration", type=int, help="completed round; default latest; 0 = initial actor"
    )
    args = parser.parse_args(argv)
    try:
        reference = retain(args.loop, args.iteration)
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit(f"retention error: {error}") from error
    print(reference["directory"])


if __name__ == "__main__":
    main()
