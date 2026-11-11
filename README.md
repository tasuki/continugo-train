# ContinuGo Train, choo-choo

Rust self-play experiments using shared ContinuGo rules, plus a configurable PyTorch policy/value network.

The [neural policy](docs/current/network.md) selects non-overlapping coarse cells
before exact coordinate analysis, with whole-board exploration and pass.
New training uses [twelve coarse input planes](docs/current/features.md)
(schema 5): antialiased stones, connected group size, liberty centers,
liberty footprints and borders—without scalar liberty area or capturability flags. Existing schema-4 models retain their original inputs. The
eleven exact move inputs include a binary **friendly-region splitting** bit:
using actual captures, does the move divide one friendly-bounded available-center
region into at least two nonempty regions? It preserves single-coordinate splits
without footprint integration. The user explicitly reopened unused schema 5 for
this replacement; older footprint metadata/tensors are incompatible. The network also learns [final score and ownership](docs/current/auxiliary.md).
Fresh score/ownership training is required; old policy/value-only checkpoints,
exports and loop controls are not migrated. Validated source SGFs remain usable.
Search defaults to `0.5*outcome + 0.5*normalized_score`; ownership is auxiliary.
Winning passes are not forced: they compete with continuations on the same utility.

## Current operating policy

Experimental actors may advance independently, but accepted networks require
[explicit hash-pinned approval](docs/current/model-store.md#approval-operation)
and convincing strength against the reference. The [loop](docs/current/loop.md)
never promotes automatically.

Training saves each tenth-round model but runs **no rating matches**. Evaluate saved
models separately with `mise run rate-saved -- --family cg5`, or resume with
`mise run rate-saved -- --resume`. [Ratings](docs/current/evaluation.md) belong under
`artifacts/ratings/`, and games under `artifacts/sgfs/YYYY-MM-DD/`,
not inside training iterations; the latest actor always advances. The evaluator
uses a 7/10 → 28/40-win challenge on rounds 5, 10, 15, … and automatically advances champions, starting
from the random bot. Only those multiples of five get any evaluation, including
ten random games; other rounds are skipped entirely. Neural rating moves use 128
simulations. **Publication** remains a separate explicit user decision.

## Next work

[Open work](docs/todo.md) lists only pending tasks toward the small-board learning
trial. Completed work and measurements belong in [history](docs/history/), not in
the todo or this README.

## Requirements

- [mise](https://mise.jdx.dev/)
- The base `continugo` repository checked out alongside this one at `../continugo`.

Training campaigns live in **`runs/`**; experiments/debugging in **`scratch/`**.
The durable store is **`artifacts/`**, containing only **`models/`**, **`ratings/`**
and date-grouped **`sgfs/`** (plus hidden control/lock files). Models are saved as
`artifacts/models/cg5/r0001/` through `r0100/`; the evaluator references them directly.
There are no model copies under ratings, per-run evaluator trees or duplicate match
CSV ledger. SGFs are authoritative; `networks.csv` is their rebuildable summary.
The shared `evaluator.log` is retained and gitignored. Dated PNGs are preserved.
Currently only cg5 is included. Keep an off-machine backup of campaigns and
artifacts; ONNX alone cannot resume training.

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
mise run selfplay -- --engine random --games 3 --output scratch/random
mise run selfplay -- --help
mise run model-bench -- --device cuda
mise run pipeline-profile -- --help
mise run train -- --help
mise run loop -- --help
mise run loop-status -- --help
mise run plot-ratings
mise run plot-metrics -- --run runs/YOUR_CAMPAIGN
mise run nn-view -- --sgf GAME.sgf --checkpoint MODEL/checkpoint.pt
mise run save-models -- --runs runs
mise run rate-saved -- --help
mise run retain-model -- --help
mise run promote -- --help
mise run arena -- --help
mise run rate-models -- --help
```

Self-play saves self-contained SGF records; nonstandard training data goes in comments. Existing files are not overwritten.

To see what the network sees, use the [neural game microscope](docs/current/nn-viewer.md):
`mise run nn-view` for existing training/rated SGFs, or add
`--nn-html --nn-checkpoint FILE` to neural self-play/arena. Each `.nn.html` is a
standalone desktop dashboard with embedded images and ←/→ move navigation.
Rated games get these sidecars beside the SGFs in `artifacts/sgfs/YYYY-MM-DD/`, from each
mover's own network; random or unsupported turns are skipped.

Use timestamped names in both `runs/` and `scratch/`. A short training sanity check
belongs in `scratch/`; a continuing training campaign belongs in `runs/`.
Standalone export, neural self-play, arena and profiling outputs default to
`scratch/`; training/loop outputs default to `runs/`. Explicit paths stay literal.
See [artifact organization](docs/current/artifacts.md).

## Documentation

- [Current implementation and operating instructions](docs/current/)
- [Artifact organization](docs/current/artifacts.md)
- [Representative pipeline profiling](docs/current/profiling.md)
- [Benchmarks](docs/benchmarks.md)
- [Network architecture](docs/current/network.md)
- [Self-contained neural game viewer](docs/current/nn-viewer.md)
- [Score/ownership labels, utility and diagnostics](docs/current/auxiliary.md)

Update current docs alongside code changes. Keep proposals in `docs/todo.md` and
dated run results in `docs/history/`, not mixed into current instructions.
