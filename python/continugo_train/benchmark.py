"""Synthetic compute/memory benchmark, NOT SGF training or self-play throughput.

Separates coarse inference, selected-cell scoring with precomputed coarse features,
combined inference, coarse backward/SGD, and hierarchical policy/value backward/SGD.
Inputs/targets are generated on-device outside timings. CUDA timings synchronize
at both boundaries. FP32/eager mode is deliberate (including on Pascal GPUs).
"""

import argparse
import os
import platform
import statistics
import time
from dataclasses import asdict

import torch

from .losses import hierarchical_policy_value_loss
from .model import (
    COORDINATE_ACTIONS,
    NetworkConfig,
    PolicyValueNet,
    cell_actions,
    resolution_for_diameter,
)


def positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def positive_list(value: str) -> list[int]:
    return [positive(part) for part in value.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--diameters", type=positive_list, default=[110, 76, 52])
    parser.add_argument("--batch-sizes", type=positive_list, default=[1, 4])
    parser.add_argument(
        "--actions",
        type=positive,
        help="approximate coordinate budget, rounded to complete cells; default uses model cell budget",
    )
    parser.add_argument("--samples-per-diameter", type=positive, default=4)
    parser.add_argument("--iterations", type=positive, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--threads", type=positive, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--skip-training", action="store_true")
    for name in asdict(NetworkConfig()):
        kind = (
            {"choices": ["relu", "leaky_relu"]}
            if name == "value_activation"
            else {"type": positive}
        )
        parser.add_argument(
            "--" + name.replace("_", "-"), default=getattr(NetworkConfig(), name), **kind
        )
    args = parser.parse_args()
    if (
        (args.actions and args.actions > COORDINATE_ACTIONS)
        or args.warmup < 0
        or max(args.diameters) > 1000
    ):
        parser.error("actions <= 1000000, diameters <= 1000, warmup >= 0 required")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    config = NetworkConfig(**{name: getattr(args, name) for name in asdict(NetworkConfig())})
    model = PolicyValueNet(config).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
    print(
        f"SYNTHETIC FP32 eager; torch={torch.__version__}; device={device} {name}; "
        f"threads={args.threads}; parameters={parameters}; warmup={args.warmup}; "
        f"iterations={args.iterations}; samples_per_diameter={args.samples_per_diameter}",
        flush=True,
    )
    print(f"config={asdict(config)}", flush=True)
    print("diameter resolution batch actions stage median_ms positions/s peak_MiB", flush=True)

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def measure(call, diameter, resolution, batch, stage, count) -> None:
        for _ in range(args.warmup):
            call()
        sync()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        seconds = []
        for _ in range(args.iterations):
            sync()
            start = time.perf_counter()
            call()
            sync()
            seconds.append(time.perf_counter() - start)
        median = statistics.median(seconds)
        peak = (
            f"{torch.cuda.max_memory_allocated(device) / 2**20:.1f}"
            if device.type == "cuda"
            else "-"
        )
        print(
            f"{diameter} {resolution} {batch} {count} {stage} "
            f"{median * 1000:.3f} {batch / median:.2f} {peak}",
            flush=True,
        )

    def run_case(diameter: int, batch: int) -> None:
        model.zero_grad(set_to_none=True)
        resolution = resolution_for_diameter(diameter, args.samples_per_diameter)
        coarse = torch.randn(batch, config.input_planes, resolution, resolution, device=device)
        cells = config.top_cells + config.exploration_cells
        if args.actions:
            cells = max(1, round(args.actions * resolution**2 / COORDINATE_ACTIONS))
        chosen = torch.linspace(0, resolution**2 - 1, min(cells, resolution**2)).long().tolist()
        actions = cell_actions(chosen, resolution).to(device)
        count = actions.numel()
        exact = torch.randn(batch, count, config.move_features, device=device)
        legal = torch.rand(batch, count, device=device) > 0.1
        selected = [0, count // 2, count - 1]
        legal[:, selected] = True
        indices = torch.tensor([selected + [count]], device=device).expand(batch, -1)
        visits = torch.tensor([[4.0, 2.0, 1.0, 1.0]], device=device).expand(batch, -1)
        outcomes = torch.linspace(-1, 1, batch, device=device)
        model.eval()
        with torch.inference_mode():
            spatial, pass_logit, _ = model.encode(coarse)

            def encode():
                context, pass_score, value = model.encode(coarse)
                return context, pass_score, value, model.cell_head(context)

            measure(encode, diameter, resolution, batch, "coarse_infer", count)
            measure(
                lambda: model.policy(spatial, pass_logit, exact, actions, legal),
                diameter,
                resolution,
                batch,
                "policy_infer",
                count,
            )
            measure(
                lambda: model(coarse, exact, actions, legal),
                diameter,
                resolution,
                batch,
                "full_infer",
                count,
            )
        if args.skip_training:
            return
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

        def coarse_step() -> None:
            # Synthetic squared-output objective isolates coarse backward cost.
            optimizer.zero_grad(set_to_none=True)
            spatial, pass_logit, value = model.encode(coarse)
            loss = (
                spatial.square().mean()
                + pass_logit.square().mean()
                + value.square().mean()
                + model.cell_head(spatial).square().mean()
            )
            loss.backward()
            optimizer.step()

        def full_step() -> None:
            optimizer.zero_grad(set_to_none=True)
            logits, values = model(coarse, exact, actions, legal)
            losses = hierarchical_policy_value_loss(logits, values, indices, visits, outcomes)
            if not torch.isfinite(losses.total):
                raise RuntimeError("nonfinite training loss")
            losses.total.backward()
            optimizer.step()

        measure(coarse_step, diameter, resolution, batch, "coarse_step", count)
        measure(full_step, diameter, resolution, batch, "full_step", count)

    for diameter in args.diameters:
        for batch in args.batch_sizes:
            run_case(diameter, batch)


if __name__ == "__main__":
    main()
