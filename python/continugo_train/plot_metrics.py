"""Plot saved-epoch losses and self-play draw proportions from a run directory.

Reads training.log and loop.pt only; never changes a running job.
Install the optional plotting dependency with: pip install -e '.[plot]'
"""

import argparse
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

EPOCH = re.compile(r"^epoch=(\d+)/\d+ (train|validation): (.*)$")
SAVED = re.compile(r"^saved .*[/\\]epoch-(\d+)\.pt;")
METRICS = (
    ("loss", "Total loss"),
    ("policy_ce", "Policy cross-entropy"),
    ("value_mse", "Value MSE"),
    ("score_mse", "Score MSE"),
    ("ownership_mse", "Ownership MSE"),
)


@dataclass
class Job:
    label: str
    logs: list[Path]
    epochs: dict[int, dict[str, dict[str, float]]]


def read_epochs(path):
    """Ignore incomplete/live epochs; later committed repeats replace earlier ones."""
    pending = {}
    committed = {}
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            # A live writer may not have finished the final line yet.
            if not line.endswith("\n"):
                continue
            match = EPOCH.match(line)
            if match:
                epoch, phase, values = match.groups()
                fields = dict(re.findall(r"(\w+)=([^\s]+)", values))
                try:
                    metrics = {name: float(fields[name]) for name, _ in METRICS}
                    if not all(math.isfinite(value) for value in metrics.values()):
                        raise ValueError("nonfinite loss")
                except (KeyError, ValueError) as error:
                    raise ValueError(f"{path}:{number}: invalid loss metrics") from error
                pending.setdefault(int(epoch), {})[phase] = metrics
            match = SAVED.match(line)
            if match:
                epoch = int(match.group(1))
                phases = pending.pop(epoch, {})
                if set(phases) != {"train", "validation"}:
                    raise ValueError(f"{path}:{number}: saved epoch lacks train/validation metrics")
                committed[epoch] = phases
    return committed


def collect_jobs(run):
    """Discover known trainer layouts; do not count replay/archive copies of logs."""
    run = Path(run)
    if not run.is_dir():
        raise ValueError(f"run directory does not exist: {run}")
    if (run / "training.log").is_file():
        groups = [("Training", [run / "training.log"])]
    else:
        groups = []
        initial = run / "initial-training" / "training.log"
        if initial.is_file():
            groups.append(("Bootstrap", [initial]))
        loop = run / "loop" if (run / "loop").is_dir() else run
        iterations = sorted(
            (p for p in loop.glob("iteration-*") if re.fullmatch(r"iteration-\d+", p.name)),
            key=lambda p: int(p.name.split("-")[1]),
        )
        for iteration in iterations:
            attempts = sorted(
                (
                    p
                    for p in iteration.glob("train-attempt-*")
                    if re.fullmatch(r"train-attempt-\d+", p.name)
                ),
                key=lambda p: int(p.name.rsplit("-", 1)[1]),
            )
            logs = [p / "training.log" for p in attempts if (p / "training.log").is_file()]
            if logs:
                groups.append((f"Round {int(iteration.name.split('-')[1])}", logs))
    jobs = []
    for label, logs in groups:
        epochs = {}
        for log in logs:
            # Recovery attempts may repeat local epoch numbers. Plot each once,
            # preferring the latest attempt that actually logged a saved epoch.
            epochs.update(read_epochs(log))
        if epochs:
            jobs.append(Job(label, logs, epochs))
    if not jobs:
        raise ValueError(f"no saved-epoch loss metrics found under {run}")
    return jobs


@dataclass
class DrawRate:
    iteration: int
    games: int
    draws: int
    truncated: int


def collect_draws(run):
    """Use one atomic loop snapshot; count actual games, not deduplicated replay."""
    run = Path(run)
    loop = run / "loop" if (run / "loop").is_dir() else run
    path = loop / "loop.pt"
    if not path.is_file():
        return []
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    rates = []
    for iteration in state["iterations"]:
        games = iteration.get("games", [])
        # Adoption can lag SGF writes. Include only fully recorded batches,
        # regardless of whether their later training/evaluation has finished.
        if not games or any(game is None for game in games):
            continue
        rates.append(
            DrawRate(
                iteration=iteration["number"],
                games=len(games),
                draws=sum(game["result"] == "0" for game in games),
                truncated=sum(game["result"] is None for game in games),
            )
        )
    return sorted(rates, key=lambda rate: rate.iteration)


def pyplot(output):
    if Path(output).suffix.lower() not in {".png", ".svg", ".pdf"}:
        raise ValueError("plot output must end in .png, .svg or .pdf")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ValueError(
            "plotting requires: .venv/bin/python -m pip install -e '.[plot]'"
        ) from error

    return plt


def save_figure(plt, figure, output):
    output = Path(output)
    temporary = None
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".plot-", suffix=output.suffix, dir=output.parent)
        os.close(fd)
        temporary = Path(name)
        figure.savefig(temporary, dpi=150)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        plt.close(figure)


def plot(jobs, output, title):
    plt = pyplot(output)
    # Connect saved epochs across jobs to expose campaign trends. Round markers
    # and the footer retain the warning that replay datasets change between jobs.
    figure, axes = plt.subplots(len(METRICS), 1, figsize=(13, 3 * len(METRICS)), sharex=True)
    offset = 0
    starts = []
    full_x = []
    series = {(metric, phase): [] for metric, _ in METRICS for phase in ('train', 'validation')}
    for index, job in enumerate(jobs):
        epochs = sorted(job.epochs)
        x = [offset + epoch for epoch in epochs]
        starts.append((x[0], job.label))
        full_x.extend(range(x[0], x[-1] + 1))
        for axis, (metric, label) in zip(axes, METRICS, strict=True):
            for phase, color in [("train", "tab:blue"), ("validation", "tab:orange")]:
                # NaNs leave holes for missing epochs rather than joining across them.
                series[metric, phase].extend([
                    job.epochs[epoch][phase][metric] if epoch in job.epochs else math.nan
                    for epoch in range(epochs[0], epochs[-1] + 1)
                ])
            axis.set_ylabel(label)
            axis.grid(alpha=0.2)
            if index:
                axis.axvline(x[0] - 0.5, color="grey", linewidth=0.7, alpha=0.35)
        offset += epochs[-1]
    for axis, (metric, _) in zip(axes, METRICS, strict=True):
        for phase, color in [('train', 'tab:blue'), ('validation', 'tab:orange')]:
            axis.plot(full_x, series[metric, phase], color=color, marker='.',
                      markersize=4, linewidth=1.2, label=phase.capitalize())
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Recorded training epoch (local epoch counters stitched across jobs)")
    # Limit label density on long campaigns while retaining the final round.
    stride = max(1, math.ceil(len(starts) / 14))
    labels = starts[::stride]
    if labels[-1] != starts[-1]:
        labels.append(starts[-1])
    top = axes[0].secondary_xaxis("top")
    top.set_xticks([x for x, _ in labels], [label for _, label in labels], rotation=45, ha="left")
    top.tick_params(labelsize=8)
    figure.suptitle(title, y=0.995)
    figure.text(
        0.5,
        0.01,
        "Saved epochs only • Replay changes between rounds • Loss is not playing strength",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.95))
    save_figure(plt, figure, output)


def plot_draws(rates, output, title):
    if not rates:
        raise ValueError("no fully recorded self-play batches to plot")
    plt = pyplot(output)
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    figure, axis = plt.subplots(figsize=(12, 5))
    x = [rate.iteration for rate in rates]
    y = [rate.draws / rate.games for rate in rates]
    axis.plot(x, y, marker="o", color="tab:blue", label="Natural draws")
    for rate, fraction in zip(rates, y, strict=True):
        axis.annotate(
            f"{rate.draws}/{rate.games}",
            (rate.iteration, fraction),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    if any(rate.truncated for rate in rates):
        axis.plot(
            x,
            [rate.truncated / rate.games for rate in rates],
            marker="x",
            linestyle="--",
            color="grey",
            label="Truncated / unknown (not draws)",
        )
    axis.set(title=title, xlabel="Self-play iteration", ylabel="Proportion of generated games")
    axis.set_ylim(0, 1.08)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    figure.text(
        0.5,
        0.015,
        "Fully recorded self-play batches only • Counts are draws / all games, including truncations",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(plt, figure, output)


def update(run):
    """Refresh campaign plots after epochs/batches; plotting cannot stop learning."""
    run = Path(run).resolve()
    for parent in (run, *run.parents):
        if (parent / "loop.pt").is_file():
            run = parent.parent if parent.name == "loop" else parent
            break
        if (parent / "loop" / "loop.pt").is_file():
            run = parent
            break
    try:
        try:
            jobs = collect_jobs(run)
        except ValueError as error:
            if "no saved-epoch" not in str(error):
                raise
            jobs = []
        if jobs:
            plot(jobs, run / "metrics.png", f"Training losses — {run.name}")
        rates = collect_draws(run)
        if rates:
            plot_draws(rates, run / "draws.png", f"Self-play draws — {run.name}")
    except Exception as error:  # noqa: BLE001 — diagnostics must never abort training
        print(f"WARNING: metrics plots not updated: {error}", file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, required=True, help="campaign, loop or offline job directory"
    )
    parser.add_argument("--output", type=Path, help="PNG/SVG/PDF; default RUN/metrics.png")
    parser.add_argument("--title", help="override the loss figure title")
    parser.add_argument(
        "--draws-output",
        type=Path,
        help="draw-rate plot; default draws.FORMAT beside the loss plot",
    )
    args = parser.parse_args(argv)
    try:
        jobs = collect_jobs(args.run)
        rates = collect_draws(args.run)
        if args.draws_output and not rates:
            raise ValueError("no fully recorded self-play batches found for the draw-rate plot")
        output = args.output or args.run / "metrics.png"
        draws_output = args.draws_output or output.with_name(f"draws{output.suffix}")
        if rates and draws_output.resolve() == output.resolve():
            raise ValueError("loss and draw-rate output paths must differ")
        plot(jobs, output, args.title or f"Training losses — {args.run.resolve().name}")
        if rates:
            plot_draws(rates, draws_output, f"Self-play draws — {args.run.resolve().name}")
            print(f"Plotted draw proportions for {len(rates)} iterations: {draws_output}")
    except (OSError, ValueError) as error:
        print(f"plot error: {error}", file=sys.stderr)
        return 1
    count = sum(len(job.epochs) for job in jobs)
    print(f"Plotted {count} saved epochs across {len(jobs)} jobs: {output}")
    print("Replay corpora change between rounds; these are not fixed-dataset learning curves.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
