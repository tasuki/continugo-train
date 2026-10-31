# ContinuGo Train, choo-choo

Rust self-play experiments using shared ContinuGo rules, plus a configurable PyTorch policy/value network.

## Requirements

- [mise](https://mise.jdx.dev/)
- The base `continugo` repository checked out alongside this one at `../continugo`.

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
```

Self-play saves self-contained SGF records; nonstandard training data goes in comments. Existing files are not overwritten.

See [python/README.md](python/README.md) for the network and synthetic benchmarks,
[TODO.md](TODO.md) for the plan, and [BENCHMARKS.md](BENCHMARKS.md) for measurements.
