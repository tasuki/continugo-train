# Offline policy/value training

This is a real SGF → Rust features → PyTorch training job, without ONNX. It learns
from saved MCTS games; this command does **not** generate self-play or establish
playing strength. Use the separate [native neural self-play command](inference.md)
after exporting a checkpoint. Training loss and checkpoint selection are not
playing-strength measurements.

## First run

After `mise run setup`, generate a small corpus with policy visits and scored RE:

```sh
mise run selfplay -- --engine mcts --games 24 --sizes 9,13,19 --seed 100 --simulations 32 --output runs/bootstrap
mise run train -- --data runs/bootstrap --output runs/train-001 --device cuda --epochs 3 --steps-per-epoch 20 --positions-per-game 8 --validation-positions-per-game 4
```

Use a new output directory, or omit `--output` to reserve
`runs/YYYYMMDD-HHMMSS-train` here automatically (UTC, collision-safe).
Explicit output paths remain literal; resume defaults to its checkpoint's existing
directory rather than allocating a new one. `--device auto` chooses CUDA when available,
otherwise CPU; `--device cpu` forces CPU. Inspect options with
`mise run train -- --help`. Architecture options include `--channels`, `--blocks`, `--policy-hidden`,
`--value-hidden`, and `--chunk-size`. Defaults are the existing 32-channel/4-block
network, FP32 eager mode, SGD lr 0.01 / momentum 0.9 / weight decay 1e-4, gradient
norm clipping at 5, batch size 2, and two feature workers. Execution is FP32 eager;
memory use depends on batch size, feature buffers, and resolution.

The example deliberately limits the run. With omitted limits (all default to
zero), all eligible positions are candidates before board-mixture downsampling.
`--size-weights uniform` instead uses every eligible position once and evaluates
all held-out positions. Nonzero `--positions-per-game` chooses a fresh reproducible
subset each training epoch. `--validation-positions-per-game` chooses a fixed
seeded subset per held-out game. `--steps-per-epoch` caps optimizer batches;
otherwise an epoch exhausts the data. Final partial batches are retained.

### Board mixture: 40% / 40% / 20%

New training/warm-start runs default to `--size-weights 9:2,13:2,19:1`, emphasizing
9/13 equivalents while retaining 20% for size 19. This controls **actual positions**:
a long 19 game cannot dominate merely by providing many more plies. Surplus
positions are downsampled without replacement before extracting million-row
features, and homogeneous batches follow the weighted schedule. There are no
extra loss multipliers, duplicate synthetic samples or conditional action softmaxes.

The default cycle is 9,13,19,9,13. Very small budgets round; unavailable sizes
renormalize and tiny buckets retain coverage where possible. Zero weight excludes
a size, and unlisted supported sizes have weight one. At least one positive-weight
size is required in both train and validation. Actual `samples_by_size` is saved
and logged for both phases. These fractions are samples, not elapsed CPU/GPU time.

Weighted mode shuffles lightweight references within sizes and bypasses the dense
global `--shuffle-buffer` so early-step caps retain their intended board mix. It
can revisit a worker's game snapshot more often than contiguous uniform loading;
worker queues remain bounded. `uniform` retains the original exhaustive ordering
and dense shuffle-buffer behavior. Strict resume freezes this sampling option;
legacy checkpoints without it restore `uniform`, rather than silently changing an
old experiment. Warm-start/new runs use `9:2,13:2,19:1` unless otherwise specified.

**Random SGFs have no MCTS policy labels.** They are not silently assigned played-
move targets. The joint policy/value trainer uses only positions with
both real visits and a completed-game outcome. It reports skipped unlabeled,
no-policy and duplicate games, and explicitly warns about draws and unlabeled/
truncated SGFs. Invalid moves, policies or scores fail rather than
being silently skipped. At least two eligible unique games in one resolution
bucket are needed to obtain a held-out set. Train board sizes are 9..19.

## Network names and cumulative progress

Checkpoints carry a `progress` record, inherited by warm-starts regardless of SGD
reset/keep. Readable names such as `cg1-b4c32-s1248000-d550347` identify:

- `cg1`: training family (`--network-family`, default `cg1`; inherited when omitted).
- `b4c32`: four residual blocks, 32 trunk channels. Full architecture/head settings
  remain in the checkpoint and ONNX metadata.
- `s`: cumulative **optimized training rows**, including repeated rows across
  epochs and warm-starts. Validation rows do not increment it. This counts actual
  minibatch rows, not optimizer steps multiplied by an assumed batch size.
- `d`: cumulative **eligible pre-move rows from completed policy-labeled self-play**,
  before position caps or board-mixture downsampling, including held-out rows.
  Bootstrap MCTS games count; random diagnostics and truncated games do not.

For standalone training, `d` counts eligible imported rows from previously unseen
canonical game IDs; copying or reintroducing ancestral games does not count again.
The loop instead tracks each planned generation job once, including completed
repeated games and games evicted from replay. It supplies frozen totals/game IDs
in `generation-progress.pt` via `--generated-data`; this is orchestration metadata,
not a game-record sidecar. Checkpoints retain `data_game_ids` even for generated
games never selected for training, separately from the holdout ledger, so later
offline imports of those games do not count them again. An external orchestrator
can also supply `--data-positions`, but must use `--generated-data` to preserve
IDs for rows outside the corpus. Totals cannot be smaller than inherited progress
plus newly imported eligible rows. Recovery copies do not create new generation.
This counter is not the number of unique
board positions, raw played moves, or all rows ever generated outside the lineage.

Strict resume preserves the family/data total and advances `s` only for newly
completed training epochs. Rejection, optimizer reset, and replay eviction cannot
reset these totals. Older checkpoints lacking counters produce names such as
`cg1-b4c32-sunknown-dunknown`; continuation does not invent ancestral totals.

Names appear in training logs, ONNX metadata, SGF `PB`/`PW`, and durable bundle
names/manifests. SGF root comments still carry full checkpoint hashes. Names are
for browsing, **not identity**: full hashes disambiguate checkpoints with identical
architecture/counters. Bundle directories retain their UTC datetime prefix and
short-hash suffix; inner epoch/checkpoint filenames stay stable.

## Data loading and validation

- Rust validates **every** move/policy in each SGF and checks scored RE against
  authoritative final scoring. Value targets are side-to-move win/loss/draw,
  not score margins. Missing RE means no value label, even after two passes.
- A canonical identity hashes board size plus decoded plays. Copies with changed
  names/comments/seeds cannot land in both train and validation. Duplicate games
  use one deterministic representative (lowest raw-file hash, then path).
- A seeded split holds out **whole unique games**, stratified by input resolution
  when a bucket has at least two games. Singleton buckets stay in training. Common
  openings may still occur in different games; positions are not globally deduped.
- Each background thread owns a persistent Rust process, which retains immutable
  pre-move snapshots of one validated game. A game is replayed/scored once per
  worker visit, not once per sampled position. Workers never touch CUDA.
- Position references are shuffled within board size and scheduled by the configured
  mixture; workers are consumed in stable round-robin order. In `uniform` mode, the
  original bounded dense shuffle buffer (default 8) mixes output. Positions are
  batched by actual resolution without spatial padding.
- Every sample includes **all million coordinates plus pass** for exact policy
  normalization, not a sampled/conditional action softmax. Sparse visit targets
  are padded with zero counts; no dense target distribution is allocated.

SGFs stay the primary game records. The Rust `features --inspect` command emits
metadata only. `features --serve` accepts local JSON request lines containing a
path, ply, density and expected file hash; replies are status-u8 / length-u64
framed CGFEAT01 tensors or UTF-8 errors. These are ephemeral IPC commands/tensors,
**not JSONL game records or SGF sidecars**. The tensor schema did not change.

Corpora are immutable during a job. Workers check the inspected raw-file SHA-256
when loading a game, then serve that snapshot. Do not point a run at files being
rewritten by self-play. The implementation stores per-game snapshots, so unusually
long games consume more worker memory. Dense feature packets are never cached as
an entire corpus: one packet is about 45 MB. Reduce `--shuffle-buffer`,
`--batch-size`, or `--workers` for less host memory (`--shuffle-buffer` applies only
in uniform mode). Worker queues are bounded;
early epoch termination closes readers and reaps Rust processes.

## Checkpoints and resume

An epoch produces `epoch-0001.pt`, etc., plus atomic `latest.pt` and `best.pt`.
Best means the lowest held-out **policy CE + value MSE**, not playing strength.
The [loop's experimental matches](evaluation.md) measure playing strength;
[accepted publication requires explicit user approval](model-store.md#approval-operation), never
just a lower loss or a favorable match verdict.
Checkpoints contain weights and BatchNorm buffers, SGD momentum/state, optimizer
step count, completed epochs, RNG states, architecture/options, feature names,
existing compatibility marker/density, and an embedded corpus/split manifest.
No separate game-data manifest file is written.

```sh
mise run train -- --data runs/bootstrap --resume runs/train-001/latest.pt --epochs 5
# Continue with fuller training passes while preserving the same held-out set:
mise run train -- --data runs/bootstrap --resume runs/train-001/latest.pt --epochs 6 --steps-per-epoch 0 --positions-per-game 0
```

`--epochs` is a **total target**, not an additional count. Without it, a resume
runs one more epoch. Resume in the checkpoint's existing run directory; newer
epoch files prevent accidentally rewinding/overwriting progress. Training settings
are restored. Batch size, workers, shuffle buffer, training position/step caps,
learning rate, gradient clipping and policy chunk size may be explicitly changed.
Feature semantics/density, architecture weights, split seed/fraction and validation
sample selection and board-mixture policy must match. Changed eligible SGF contents/targets are rejected;
relocating an otherwise identical corpus is allowed.

Checkpoints are at **completed epoch boundaries**, after validation. Ctrl-C does
not save a partial epoch. Resume the latest completed one; if none completed,
start a fresh run directory. A `.train.lock` prevents concurrent writers; after a
hard crash, remove it only after confirming no trainer is alive. Writes use
same-directory temporary files plus atomic replacement. Existing epoch snapshots
are not overwritten. GPU training is not promised bitwise deterministic
(`grid_sample` backward uses atomics); unchanged single-thread CPU continuation
is tested against an uninterrupted run, including optimizer state. Changing
workers/device/batch settings can also change numerical or sampling order.

Load a trained model for later inference/export:

```python
import torch
from continugo_train.model import NetworkConfig, PolicyValueNet

saved = torch.load("runs/train-001/best.pt", map_location="cpu", weights_only=True)
model = PolicyValueNet(NetworkConfig(**saved["config"]))
model.load_state_dict(saved["model"])
model.eval()
```

Resume refuses incompatible feature definitions.

## Warm-start on a changing replay corpus

Use `--warm-start`, **not** `--resume`, when adding/removing self-play games or
changing replay targets. It initializes a new run from an existing offline-training
checkpoint; it does not weaken fixed-corpus resume checks.

```sh
mise run train -- --data runs/bootstrap runs/neural-001 --warm-start runs/train-001/best.pt --output runs/train-002 --epochs 3 --device cuda
# Explicitly retain the source's SGD momentum instead of resetting it:
mise run train -- --data runs/replay-003 --warm-start runs/train-002/latest.pt --warm-start-optimizer keep --output runs/train-003 --epochs 2
# Continue this new run with its corpus unchanged:
mise run train -- --data runs/replay-003 --resume runs/train-003/latest.pt --epochs 3
```

`--warm-start` and `--resume` are mutually exclusive. Warm-start uses a **new/empty**
output, automatically dated if `--output` is omitted; the source run is never overwritten. It can fork
from an earlier epoch or `best.pt` even when the source run has newer checkpoints.
The new run gets its own lock, log, epoch snapshots, `latest.pt` and `best.pt`.

### Initialization contract

- Copy **all weights and BatchNorm buffers**. Architecture and feature semantics,
  including sample density, must match. Policy chunk size may change without
  changing weights. No partial loading, feature migration or network resizing.
- Inherit training options by default, including learning rate, seed and sampling
  caps. Warm-start may explicitly override any training option except incompatible
  feature density; unlike resume, its split seed/fraction and validation sampling
  limits may change. Be explicit about caps when moving from a smoke run to a
  longer run.
- `--warm-start-optimizer reset` is the default: start fresh SGD state, without
  source momentum. `keep` restores momentum/state. In either mode, selected learning
  rate, momentum and weight decay apply; explicit overrides are not silently lost
  when loading optimizer parameter groups. This flag requires `--warm-start`.
- **Local epoch and optimizer-step counters restart at zero**, even with `keep`.
  Cumulative lineage sample/data counters are inherited, not reset.
  `--epochs` is the total target for the **new** run, default three; it is not an
  increment to the source epoch. New RNG states come from the selected seed,
  rather than restoring source RNG states. GPU backward remains nondeterministic.
- Reset best validation loss to infinity. Parent and child validation losses are
  not directly comparable across changing data; `best.pt` selects only among
  completed epochs in the new run, not against the parent's best loss.
- Embed `initialization` provenance in every new checkpoint: source path and
  SHA-256, source run/epoch/step and optimizer policy. Resume preserves it. The
  source file is loaded and hashed from the same byte snapshot. No metadata
  sidecar or new game-record format is introduced.

### Holdout safety across replay windows

Blindly re-splitting overlapping data would put previously trained games into
validation. Checkpoints therefore carry a canonical-game-ID `split_history` ledger:

- All ancestral train/validation assignments stay pinned, including games absent
  from the current replay window. A removed game keeps its split when reintroduced.
- New games are assigned deterministically with the selected seed, trying to fill
  the validation-fraction target per resolution after inherited holdouts. Existing
  assignments take precedence over that fraction; it is not an exact quota.
- Duplicates/changed comments still share one canonical game identity. A copied
  training game cannot become validation. The current corpus is still fully
  replayed, checked and deduplicated by Rust-backed loading.
- A window without both train and validation games **fails** instead of moving
  known training games into validation. Include suitable inherited holdouts or
  enough new completed games to create a holdout. Singleton new resolution buckets
  remain in training, as before.
- Older offline checkpoints without `split_history` are supported by deriving
  assignments from their embedded manifest. Invalid/conflicting history is
  rejected. The ledger stores IDs/splits, not tensors; it grows with the lineage's
  unique games and is not pruned when the replay window shrinks.

The current corpus/split manifest and fingerprint still describe only this run's
selected eligible games. A warm-started run's own `--resume` rejects changed data
and retains the same split, optimizer/RNG state and completed-epoch semantics.
Holdout losses remain diagnostics/model-selection metrics, not playing-strength
measurements or a fully untouched benchmark.

The standalone trainer does **not** generate games, select a replay window or
export a model. The separate [`mise run loop` supervisor](loop.md) orchestrates
these stages with frozen replay selection and interruption recovery. Keep each
job's SGF inputs immutable; changing the corpus is allowed **between** warm-start
runs, not while a job is loading it.

## Measurements

`training.log` and stdout report policy cross-entropy, value MSE, sample/batch
counts, end-to-end positions/s, loader wait, compute-plus-transfer time, CUDA peak
live allocation and parent peak RSS. Metrics are weighted by **positions**, not
batches, so leftovers do not bias validation. CUDA timings synchronize. Phase
elapsed time includes loader startup/shuffle fill and shutdown of prefetched work;
wait and compute subtotals need not sum to it.

CUDA memory excludes reserved allocator/context/desktop usage. Parent peak RSS is
the process lifetime high-water mark including Python threads, **not** Rust child
RSS. More workers/buffering use more memory. Feature extraction and binary
transport are included in end-to-end timings; historical measurements belong in
[benchmarks.md](../benchmarks.md).
