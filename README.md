# ContinuGo Train, choo-choo

Rust self-play experiments using shared ContinuGo rules, plus a configurable PyTorch policy/value network.

## Current operating policy

Experimental actors may advance independently, but accepted networks require
[explicit hash-pinned approval](docs/current/model-store.md#approval-operation)
and convincing strength against the reference. The [loop](docs/current/loop.md)
never promotes automatically.

## Requirements

- [mise](https://mise.jdx.dev/)
- The base `continugo` repository checked out alongside this one at `../continugo`.

All generated artifacts stay here: timestamped experiments in **`runs/`**, durable
accepted models in **`models/accepted/`**, training recovery packs in
**`models/training/`**, and unaccepted actors in **`models/experimental/`**. Names start with UTC `YYYYMMDD-HHMMSS`; counters handle
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
mise run loop-status -- --help
mise run promote -- --help
mise run arena -- --help
```

Self-play saves self-contained SGF records; nonstandard training data goes in comments. Existing files are not overwritten.

## Documentation

- [Current implementation and operating instructions](docs/current/)
- [Open work and proposed experiments](docs/todo.md)
- [Benchmarks](docs/benchmarks.md)
- [Network architecture](docs/current/network.md)

Update current docs alongside code changes. Keep proposals in `docs/todo.md` and
dated run results in `docs/history/`, not mixed into current instructions.
