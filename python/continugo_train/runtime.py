"""Locate CPU/CUDA runtimes and launch native commands; never evaluate moves here.

The optional CUDA 11.8 runtime and cuDNN 8 live under .venv/ort-gpu, isolated
from PyTorch's CUDA 11.8/cuDNN 9 installation and the CPU ONNX Python package.
Both providers use the same ONNX model. No implicit fallback on CUDA failure.
"""

import argparse
import importlib.metadata
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

from .export import runtime_path

CUDA_INDEX = "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/"
GPU_VERSION = "1.20.1"
CUDNN_VERSION = "8.9.6.50"


def gpu_root():
    return Path(sys.prefix) / "ort-gpu"


def gpu_library():
    libraries = list((gpu_root() / "onnxruntime/capi").glob("libonnxruntime.so*"))
    if not libraries:
        raise ValueError(
            "CUDA runtime is not installed; run mise run setup-gpu (or select --device cpu)"
        )
    return min(libraries).resolve()


def inference_environment(device, library=None):
    if device not in {"cpu", "cuda"}:
        raise ValueError("inference device must be cpu or cuda")
    chosen = library or os.environ.get("ORT_DYLIB_PATH")
    if chosen is None:
        chosen = gpu_library() if device == "cuda" else runtime_path()
    chosen = Path(chosen).resolve()
    environment = os.environ.copy()
    environment["ORT_DYLIB_PATH"] = str(chosen)
    if device == "cuda" and chosen.parent == (gpu_root() / "onnxruntime/capi").resolve():
        cudnn = gpu_root() / "nvidia/cudnn/lib"
        nvidia = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
        if not (cudnn / "libcudnn.so.8").is_file():
            raise ValueError("isolated cuDNN 8 is missing; run mise run setup-gpu")
        if not (nvidia / "cublas/lib/libcublas.so.11").is_file():
            raise ValueError("CUDA 11 libraries from the cu118 PyTorch setup are missing")
        directories = [
            cudnn,
            *(
                nvidia / name / "lib"
                for name in [
                    "cublas",
                    "cuda_runtime",
                    "cufft",
                    "curand",
                    "cusolver",
                    "cusparse",
                    "cuda_nvrtc",
                ]
            ),
        ]
        previous = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = ":".join(str(p) for p in directories if p.is_dir())
        if previous:
            environment["LD_LIBRARY_PATH"] += ":" + previous
    return chosen, environment


def setup_gpu():
    if sys.platform != "linux":
        raise ValueError(
            "automatic CUDA 11 setup currently supports Linux; provide your own runtime on other platforms"
        )
    nvidia = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    if not (nvidia / "cublas/lib/libcublas.so.11").is_file():
        raise ValueError(
            "setup-gpu requires the project's CUDA 11.8 PyTorch libraries; run mise run setup first"
        )
    installed = {
        d.metadata["Name"].lower(): d.version
        for d in importlib.metadata.distributions(path=[str(gpu_root())])
    }
    packages = [
        ("onnxruntime-gpu", GPU_VERSION, ["--index-url", CUDA_INDEX]),
        ("nvidia-cudnn-cu11", CUDNN_VERSION, []),
    ]
    for name, version, extra in packages:
        if installed.get(name) == version:
            print(f"{name}=={version} already installed in {gpu_root()}", flush=True)
            continue
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--no-deps",
                "--target",
                str(gpu_root()),
                *extra,
                f"{name}=={version}",
            ],
            check=True,
        )
    print(f"GPU runtime ready: {gpu_library()}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("command", choices=["setup", "selfplay", "infer"])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--ort-library", type=Path)
    args, remaining = parser.parse_known_args(argv)
    try:
        if args.command == "setup":
            setup_gpu()
            return
        command = ["cargo", "run", "--locked", "--release"]
        command += ["--bin", "infer"] if args.command == "infer" else []
        command += ["--"] + (["--engine", "neural"] if args.command == "selfplay" else [])
        if "--help" in remaining or "-h" in remaining:
            os.execvpe(command[0], [*command, *remaining], os.environ.copy())
        library, environment = inference_environment(args.device, args.ort_library)
        command += ["--device", args.device, "--ort-library", str(library), *remaining]
        os.execvpe(command[0], command, environment)
    except (ValueError, OSError) as error:
        print(f"inference setup error: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
