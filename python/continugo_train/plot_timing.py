"""Plot per-iteration active time; reconstruct explicitly estimated historical stages."""

import argparse
import ast
import fcntl
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from . import timing
from .paths import plot_directory
from .plot_metrics import pyplot, save_figure

PHASES = {
    "selfplay": "Self-play + feature caches",
    "window": "Window preparation",
    "training": "Training + validation/checkpoints",
    "arena": "Arena games",
    "reporting": "Evaluation/reporting",
    "export": "Export",
    "saving": "Model saving",
    "other": "Other overhead",
}


@dataclass
class RoundTiming:
    number: int
    seconds: dict[str, float]
    estimated: bool
    complete: bool


def read_events(path):
    events = []
    if not path.exists():
        return events
    for line in path.read_text().splitlines(keepends=True):
        if not line.endswith("\n") or not line.strip():
            continue  # Concurrent writer or interrupted final append.
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue  # A killed writer can leave a fragment before the next append.
        if not isinstance(event, dict) or not {"phase", "seconds", "completed"}.issubset(event):
            raise ValueError(f"invalid timing event in {path}")
        seconds = event["seconds"]
        if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"invalid duration in {path}")
        events.append(event)
    return events


def log_spans(paths):
    """Birth-to-last-write spans, not ctime (ctime changes when files are copied).

    GNU stat exposes filesystem birth times on Linux. Unknown birth times are
    omitted rather than inventing durations. Historical spans can include pauses.
    """
    paths = [p for p in paths if p.is_file()]
    if not paths:
        return {}
    try:
        result = subprocess.run(
            ["stat", "-c", "%W|%Y|%n", "--", *map(str, paths)],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return {}
    spans = {}
    for line in result.stdout.splitlines():
        start, end, name = line.split("|", 2)
        start, end = int(start), int(end)
        if 0 < start <= end:
            spans[Path(name)] = end - start
    return spans


def arena_history(store, family):
    """Recover actual per-game durations from the shared native evaluator log."""
    totals = defaultdict(float)
    path = Path(store) / "ratings/evaluator.log"
    if not path.is_file():
        return totals
    number = None
    with path.open() as stream:
        for line in stream:
            if line.startswith("command="):
                number = None
                try:
                    command = ast.literal_eval(line[len("command=") :])
                    candidate = Path(command[command.index("--candidate") + 1])
                    if candidate.parent.name == family and re.fullmatch(r"r\d+", candidate.name):
                        number = int(candidate.name[1:])
                except (ValueError, SyntaxError, IndexError, TypeError):
                    pass
            elif number is not None:
                match = re.search(r"^arena game .* elapsed=([\d.]+)s", line)
                if match:
                    seconds = float(match[1])
                    if math.isfinite(seconds):
                        totals[number] += seconds
    return totals


def collect(run, *, estimates=True):
    import torch

    run = Path(run).resolve()
    root = run / "loop" if (run / "loop/loop.pt").is_file() else run
    state = torch.load(root / "loop.pt", map_location="cpu", weights_only=True)
    launch = root.parent / "launch.json"
    family = json.loads(launch.read_text())["family"] if launch.exists() else None
    historical_arena = arena_history(state["config"]["model_store"], family) if estimates else {}
    rounds = []
    for iteration in state["iterations"]:
        number = iteration["number"]
        directory = root / f"iteration-{number:04d}"
        events = read_events(timing.journal(root, number))
        totals = defaultdict(float)
        for event in events:
            totals[event["phase"]] += event["seconds"]
        estimated = False
        # Old completed rounds have no journal. Do not mix a live historical
        # stage's growing logfile lifetime into an apparently finished total.
        if estimates and not events and iteration.get("completed"):
            logs = {
                "selfplay": list(directory.glob("selfplay/attempt-*/selfplay.log")),
                "training": list(directory.glob("train-attempt-*.log")),
                "export": [directory / "model-export.log"],
                "evaluation": [directory / "ratings.log"],
            }
            spans = log_spans([p for paths in logs.values() for p in paths])
            for phase, paths in logs.items():
                totals[phase] = sum(spans.get(p, 0) for p in paths)
            totals["arena"] = historical_arena.get(number, 0)
            estimated = True
        if not events and not estimated:
            continue
        # Arena is a subphase of evaluation, not an extra additive duration.
        totals["reporting"] = max(0, totals["evaluation"] - totals["arena"])
        known = sum(totals[phase] for phase in PHASES if phase != "other")
        totals["other"] = max(0, totals["iteration"] - known)
        rounds.append(
            RoundTiming(
                number,
                {p: totals[p] for p in PHASES},
                estimated,
                bool(iteration.get("completed"))
                and (
                    estimated or any(e["phase"] == "iteration" and e["completed"] for e in events)
                ),
            )
        )
    if not rounds:
        raise ValueError(f"no iteration timing evidence under {run}")
    return rounds


def plot(rounds, output, title="Iteration timing"):
    plt = pyplot(output)
    figure, axis = plt.subplots(figsize=(14, 7))
    x = [r.number for r in rounds]
    bottom = [0.0] * len(rounds)
    for phase, label in PHASES.items():
        values = [r.seconds[phase] / 60 for r in rounds]
        bars = axis.bar(x, values, bottom=bottom, label=label, width=0.8)
        for bar, row in zip(bars, rounds, strict=True):
            if row.estimated:
                bar.set_hatch("//")
            if not row.complete:
                bar.set_alpha(0.45)
        bottom = [a + b for a, b in zip(bottom, values, strict=True)]
    axis.set(xlabel="Iteration", ylabel="Minutes (sum of recorded stage time)", title=title)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    axis.set_ylim(0, max(1, max(bottom)) * 1.08)
    axis.legend(loc="upper right", fontsize=8, ncols=2)
    figure.text(
        0.5,
        0.02,
        "Hatched: historical estimates from log lifetimes (may include pauses; unknown overhead omitted).\n"
        "Measured: active elapsed time, including retries; excludes downtime between resumes. Faded: partial.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    save_figure(plt, figure, output)


def update(run):
    run = Path(run)
    if (run / "loop/loop.pt").is_file():
        run = run / "loop"
    try:
        rows = collect(run)
        output = plot_directory(run)
        plot(rows, output / "timing.png", f"Iteration timing — {output.name}")
    except (OSError, ValueError, RuntimeError) as error:
        print(f"WARNING: timing plot not updated: {error}", file=sys.stderr)


def watch(run, pid, interval=30):
    """Refresh charts until the supervising process exits, without changing training.

    A lock prevents duplicate watchers. Normal campaign runtimes record stages and
    update plots directly; this is an explicitly invoked plotting-tool operation.
    """
    if pid <= 0 or not math.isfinite(interval) or interval <= 0:
        raise ValueError("watch PID and interval must be positive")
    run = Path(run).resolve()
    root = run / "loop" if (run / "loop/loop.pt").is_file() else run
    with (root / ".timing-watch.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return  # An existing watcher already owns this output.
        previous = None
        while True:
            control = (root / "loop.pt").stat()
            signature = (
                control.st_mtime_ns,
                control.st_size,
                tuple(
                    (p.parent.name, p.stat().st_mtime_ns)
                    for p in sorted(root.glob("iteration-*/timing.jsonl"))
                ),
            )
            if signature != previous:
                update(root)
                previous = signature
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                update(root)
                return
            time.sleep(interval)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--measured-only", action="store_true", help="omit historical estimates")
    parser.add_argument(
        "--watch-pid",
        type=int,
        help="refresh campaign-root timing image until this supervisor exits",
    )
    parser.add_argument(
        "--interval", type=float, default=30, help="watch polling interval in seconds"
    )
    args = parser.parse_args(argv)
    if args.watch_pid is not None and (args.output is not None or args.measured_only):
        parser.error("--watch-pid cannot be combined with --output or --measured-only")
    try:
        if args.watch_pid is not None:
            watch(args.run, args.watch_pid, args.interval)
            return
        rows = collect(args.run, estimates=not args.measured_only)
        output = args.output or plot_directory(args.run) / "timing.png"
        plot(rows, output, f"Iteration timing — {args.run.resolve().name}")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"timing plot error: {error}\n")
    print(f"Plotted timing for {len(rows)} iterations: {output}")


if __name__ == "__main__":
    main()
