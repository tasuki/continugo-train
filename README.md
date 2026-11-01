# ContinuGo Train, choo-choo

Rust self-play experiments using shared ContinuGo rules, plus a configurable PyTorch policy/value network.

## Requirements

- [mise](https://mise.jdx.dev/)
- The base `continugo` repository checked out alongside this one at `../continugo`.

All generated artifacts stay here: timestamped experiments in **`runs/`**, durable
accepted models in **`models/accepted/`**, and training recovery packs in
**`models/training/`**. Names start with UTC `YYYYMMDD-HHMMSS`; counters handle
collisions. ONNX alone cannot resume training. `models/` is Git-ignored: keep an
off-machine backup and exclude it from destructive Git-clean commands.

## Setup

```sh
mise trust
mise install
mise run setup
```

## Usage

```sh
mise run test
mise run check
mise run selfplay -- --engine random --games 3 --output runs/random
mise run selfplay -- --help
mise run model-bench -- --device cuda
mise run train -- --help
mise run loop -- --help
mise run arena -- --help
```

Self-play saves self-contained SGF records; nonstandard training data goes in comments. Existing files are not overwritten.

See [features.md](docs/features.md) for real Rust inputs and SGF loading,
[training.md](docs/training.md) for offline training, resume and replay warm-start,
[inference.md](docs/inference.md) for ONNX export and neural self-play,
[batching.md](docs/batching.md) for concurrent games and inference batching,
[loop.md](docs/loop.md) for restartable gated training with a 40/40/20 board mix,
[evaluation.md](docs/evaluation.md) for acceptance matches,
[model-store.md](docs/model-store.md) for durable ONNX + training recovery packs,
[python/README.md](python/README.md) for the network, [todo.md](docs/todo.md) for the plan, and [benchmarks.md](docs/benchmarks.md)
for measurements.
