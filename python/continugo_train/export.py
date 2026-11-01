"""Export a checkpoint as dynamic ONNX encoder + coordinate scorer (opset 17).

Splitting avoids recomputing the trunk for each coordinate chunk. Raw placement
logits are masked/normalized by native inference using authoritative Rust legality.
Feature/architecture/provenance metadata lives inside both graphs, not sidecars.
"""

import argparse
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .model import NetworkConfig, PolicyValueNet
from .paths import new_run_directory
from .train import feature_spec

METADATA_KEY = "continugo"


class Encoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, coarse):
        return self.model.encode(coarse)


class Scorer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, spatial, exact, actions):
        return self.model.score_chunk(spatial, exact, actions)


def runtime_path():
    spec = importlib.util.find_spec("onnxruntime")
    if spec is None:
        raise ValueError("install the inference extra: pip install -e '.[inference]'")
    capi = Path(spec.origin).parent / "capi"
    libraries = [
        *capi.glob("libonnxruntime.so*"),
        *capi.glob("libonnxruntime*.dylib"),
        *capi.glob("onnxruntime.dll"),
    ]
    if not libraries:
        raise ValueError(f"ONNX Runtime shared library not found in {capi}")
    return min(libraries).resolve()


def verify(model, directory):
    """Check dynamic batch/spatial/action axes against eager FP32 PyTorch."""
    import onnxruntime as ort

    settings = ort.SessionOptions()
    settings.intra_op_num_threads = 1
    settings.inter_op_num_threads = 1
    encoder = ort.InferenceSession(
        str(directory / "encoder.onnx"), settings, providers=["CPUExecutionProvider"]
    )
    scorer = ort.InferenceSession(
        str(directory / "policy.onnx"), settings, providers=["CPUExecutionProvider"]
    )
    rng = torch.Generator().manual_seed(0)
    maximum = 0.0
    with torch.inference_mode():
        for batch, resolution, count in [(1, 38, 7), (1, 54, 73), (2, 78, 19)]:
            coarse = torch.rand(
                batch, model.config.input_planes, resolution, resolution, generator=rng
            )
            exact = torch.rand(batch, count, model.config.move_features, generator=rng)
            actions = torch.cat(
                [
                    torch.tensor([0, 999, 1000, 999999, 500500, 500501]),
                    torch.randint(1000000, (count - 6,), generator=rng),
                ]
            )
            spatial, pass_logit, value = model.encode(coarse)
            expected = [spatial, pass_logit, value, model.score_chunk(spatial, exact, actions)]
            actual = encoder.run(None, {"coarse": coarse.numpy()})
            actual += scorer.run(
                None, {"spatial": actual[0], "exact": exact.numpy(), "actions": actions.numpy()}
            )
            for reference, output in zip(expected, actual, strict=True):
                np.testing.assert_allclose(output, reference.numpy(), rtol=1e-4, atol=1e-5)
                maximum = max(maximum, float(np.max(np.abs(output - reference.numpy()))))
    return maximum


def export_checkpoint(checkpoint, output):
    import onnx

    checkpoint, output = Path(checkpoint), Path(output)
    if output.exists():
        raise ValueError("export output already exists; use a fresh model directory")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = NetworkConfig(**saved["config"])
    spec = saved["feature_spec"]
    if spec != feature_spec(spec["samples_per_diameter"]):
        raise ValueError("checkpoint has incompatible features")
    if config.input_planes != len(spec["coarse"]) or config.move_features != len(spec["moves"]):
        raise ValueError("checkpoint architecture has incompatible feature widths")
    model = PolicyValueNet(config).eval()
    model.load_state_dict(saved["model"])
    metadata = json.dumps(
        {
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "epoch": saved["epoch"],
            "config": saved["config"],
            "feature_spec": spec,
        },
        sort_keys=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".onnx-", dir=output.parent) as temporary:
        directory = Path(temporary)
        coarse = torch.zeros(1, config.input_planes, 38, 38)
        with torch.inference_mode():
            spatial, _, _ = model.encode(coarse)
            torch.onnx.export(
                Encoder(model).eval(),
                (coarse,),
                str(directory / "encoder.onnx"),
                input_names=["coarse"],
                output_names=["spatial", "pass_logit", "value"],
                dynamic_axes={
                    "coarse": {0: "batch", 2: "height", 3: "width"},
                    "spatial": {0: "batch", 2: "height", 3: "width"},
                    "pass_logit": {0: "batch"},
                    "value": {0: "batch"},
                },
                opset_version=17,
                dynamo=False,
            )
            torch.onnx.export(
                Scorer(model).eval(),
                (spatial, torch.zeros(1, 7, config.move_features), torch.arange(7)),
                str(directory / "policy.onnx"),
                input_names=["spatial", "exact", "actions"],
                output_names=["logits"],
                dynamic_axes={
                    "spatial": {0: "batch", 2: "height", 3: "width"},
                    "exact": {0: "batch", 1: "actions"},
                    "actions": {0: "actions"},
                    "logits": {0: "batch", 1: "actions"},
                },
                opset_version=17,
                dynamo=False,
            )
        for name in ["encoder.onnx", "policy.onnx"]:
            graph = onnx.load(directory / name)
            onnx.helper.set_model_props(graph, {METADATA_KEY: metadata})
            onnx.checker.check_model(graph)
            onnx.save(graph, directory / name)
        error = verify(model, directory)
        directory.rename(output)
    return error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--output", type=Path, help="default: a model bundle inside runs/YYYYMMDD-HHMMSS-export"
    )
    parser.add_argument(
        "--runtime-path",
        action="store_true",
        help="print the installed native runtime library path",
    )
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.runtime_path:
        print(runtime_path())
        return
    if args.checkpoint is None or args.threads <= 0:
        parser.error("export requires --checkpoint and positive --threads")
    if args.output is None:
        args.output = new_run_directory("export") / "model"
    torch.set_num_threads(args.threads)
    error = export_checkpoint(args.checkpoint, args.output)
    print(f"Exported {args.output}; dynamic ONNX/PyTorch parity max_abs_error={error:.8g}")


if __name__ == "__main__":
    main()
