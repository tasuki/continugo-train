# ContinuGo Train, choo-choo

Rust self-play experiments using shared ContinuGo rules, plus a configurable PyTorch policy/value network.

The [neural policy](docs/current/network.md) selects non-overlapping coarse cells
before exact coordinate analysis, with whole-board exploration and pass. The
`cells-v1` model format requires fresh training; old dense-policy checkpoints are
not supported, but their validated SGF training data remains usable.

## Current operating policy

Experimental actors may advance independently, but accepted networks require
[explicit hash-pinned approval](docs/current/model-store.md#approval-operation)
and convincing strength against the reference. The [loop](docs/current/loop.md)
never promotes automatically.

New campaigns automatically use [adaptive rated matchmaking](docs/current/loop.md#automatic-adaptive-rating-precision)
for scheduled candidates, targeting RD ≤ 25 plus substantial paired-color evidence,
without gating the latest actor. `--adaptive-max-games 0` explicitly opts out for
smoke tests; existing saved loops retain their frozen settings.

## Next work

[Open work](docs/todo.md) lists only pending tasks toward the small-board learning
trial. Completed work and measurements belong in [history](docs/history/), not in
the todo or this README.

## Requirements

- [mise](https://mise.jdx.dev/)
- The base `continugo` repository checked out alongside this one at `../continugo`.

All generated artifacts stay here: timestamped experiments in **`runs/`**, durable
accepted models in **`models/accepted/`**, training recovery packs in
**`models/training/`**, and unaccepted actors in **`models/experimental/`**. Names start with UTC `YYYYMMDD-HHMMSS`; counters handle
collisions. All campaigns share this single store and global ratings; do not
nest separate stores by training run. Only scheduled/evaluated networks enter
the rating registry. ONNX alone cannot resume training. `models/` is Git-ignored: keep an
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
mise run pipeline-profile -- --help
mise run train -- --help
mise run loop -- --help
mise run loop-status -- --help
mise run plot-ratings
mise run plot-metrics -- --run runs/YOUR_CAMPAIGN
mise run promote -- --help
mise run arena -- --help
mise run rate-models -- --help
```

Self-play saves self-contained SGF records; nonstandard training data goes in comments. Existing files are not overwritten.

## Documentation

- [Current implementation and operating instructions](docs/current/)
- [Representative pipeline profiling](docs/current/profiling.md)
- [Benchmarks](docs/benchmarks.md)
- [Network architecture](docs/current/network.md)

Update current docs alongside code changes. Keep proposals in `docs/todo.md` and
dated run results in `docs/history/`, not mixed into current instructions.
