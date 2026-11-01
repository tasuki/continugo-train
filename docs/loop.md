# Restartable gated neural self-play loop

`mise run loop` runs native self-play → bounded SGF replay → warm-start training →
ONNX export → paired-color acceptance matches. The structure follows
[KataGo's synchronous training pattern](https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md),
with a game-result gate rather than loss-based promotion. This is still a small
single-machine implementation, not an asynchronous distributed trainer.

## Small run

Requires an existing full training checkpoint, inference dependencies and the
CPU/CUDA setup in [inference.md](inference.md). The task builds native binaries.
The supervisor requires POSIX process groups and `flock`.

```sh
mise run loop -- --checkpoint runs/TRAIN_RUN/best.pt --bootstrap-data runs/BOOTSTRAP_RUN --iterations 2 --games-per-iteration 5 --seed 400 --simulations 8 --max-moves 64 --train-epochs 1 --steps-per-epoch 5 --positions-per-game 1 --validation-positions-per-game 1 --batch-size 1 --workers 1 --gate-games 10 --gate-simulations 8 --gate-max-moves 512
# Use the dated output path printed by the first command:
mise run loop -- --output runs/YYYYMMDD-HHMMSS-loop --resume
# Increase the total target, not an additional iteration count:
mise run loop -- --output runs/YYYYMMDD-HHMMSS-loop --resume --iterations 3
```

New loops automatically reserve `runs/YYYYMMDD-HHMMSS-loop` in this repository
(UTC; same-second collisions get a counter). An explicit `--output` is used literally;
`--resume` must always identify the existing output. Inner iteration/game/epoch names
stay stable beneath that dated run. Example input paths above are placeholders:
the old smoke artifacts were removed for a clean start.

CUDA is the default; `--device cpu` selects CPU explicitly. Export verification is
CPU. Native play, training and export are supervised **separate processes**, so
GPU inference libraries do not replace the training stack. New loops default to
one iteration, ten self-play games, one training epoch capped at 20 batches, and
20 gate games. These are setup budgets, not strength-tuned settings.

Resume uses the saved total target and frozen settings; explicit conflicting
overrides fail. The target may increase, never decrease. Keep the loop at the same
absolute path and compatible code/rules/runtime versions. **v1 automatic-promotion
loops are not silently migrated**: their files remain untouched; start a new v2
loop from their latest immutable training checkpoint. Standalone legacy training
checkpoints remain loadable.

## The 40/40/20 board budget

`--sizes 9,13,19 --size-weights 9:2,13:2,19:1` is the default. Integer weights
`9:40,13:40,19:20` are equivalent. The deterministic smooth cycle is
`9,13,19,9,13`: four size-9 and four size-13 games for every two size-19 games.
Generation continues the cycle across iteration boundaries, even when an iteration
contains fewer than five games. Weights are configurable; `uniform` restores
literal size cycling. Zero weights exclude a size; unlisted sizes have weight one.

The same policy drives:

- Self-play game planning (default ten games = 4/4/2).
- Bounded replay selection, preferring newer unique games within each size.
- **Actual training and validation position selection**, not just loss multipliers.
  Surplus positions from long games are removed before exact-feature extraction.
- Evaluation **pairs**, so both colors on each board get the same budget. Default
  20 gate games = 8/8/4; ten games = 4/4/2.

Tiny budgets round, missing sizes renormalize, and insufficient data can prevent
an exact ratio. `samples_by_size` reports actual optimized/validated counts. These
are game/sample fractions, **not wall-clock fractions**: a size-19 position can
still cost more. Shape/parity tests continue to cover all three sizes.

## Training progress and accepted models are different

Each iteration freezes two independent parents:

- **Training parent:** the latest completed training candidate, even if rejected.
- **Self-play/evaluation incumbent:** the last accepted ONNX model.

Warm-start defaults to `--warm-start-optimizer keep`; weights and momentum continue
from the latest training checkpoint. Epoch/step counters restart locally per run,
not cumulatively across the loop. Learning rate can be set with `--lr`; architecture,
feature density, momentum, weight decay and gradient clipping inherit from the
training parent. Training seed is `seed+iteration-1`.

The trainer's `best.pt` remains a within-job validation-loss diagnostic. The loop
exports the **latest completed epoch**, then gates it by actual match outcomes.
Rejection or an inconclusive test **does not roll training back**. The next round
continues training from that candidate while self-play keeps the incumbent.
`--incumbent-checkpoint` optionally supplies a different initially accepted model
when restoring a newer, unaccepted training checkpoint.

See [evaluation.md](evaluation.md) for match settings, draw/truncation handling and
conservative acceptance bounds. `--gate-games 0` explicitly disables gating; it is
not the default and is useful only for plumbing diagnostics.

## Replay and labels

`--replay-iterations` (default 4) admits the current and previous N-1 generated
iterations. Eligible bootstrap snapshots are older fallback candidates, not a
protected quota. `--replay-games` (default 256) caps the selected **unique labeled
games**, including bootstrap data. Newest records win duplicate decoded-play
identities; weighted round-robin chooses sizes. This is game-based windowing, not
KataGo's adaptive row reservoir.

Rust validates every move, policy, score and generation/model identity. Truncated
SGFs retain real visit targets but no invented outcome and are excluded from replay.
Draws and truncations produce explicit warnings, including when inspecting a corpus.
Each iteration must produce at least one completed policy-labeled game.

Ancestral train/validation assignments remain pinned even after games leave the
window. The selected corpus must have both splits; include enough new unique games
or suitable `--bootstrap-data`. A source checkpoint's manifest does not implicitly
load its original SGFs into replay. Gate games are **not** used as training data.
Corpora remain immutable within each job. See [training.md](training.md).

## Durable model store

By default, models and recovery data stay **in this repository's `models/`**, not in
`runs/` and not in a sibling directory:

```text
models/
  LATEST-ACCEPTED               relative path to an accepted bundle
  LATEST-TRAINING               latest backed-up training candidate, even if rejected
  accepted/YYYYMMDD-HHMMSS-SEQ-ID/   ONNX + checkpoint + exact replay SGFs
  training/YYYYMMDD-HHMMSS-SEQ-ID/   checkpoint + replay for every candidate
```

Choose another disk/location with `--model-store PATH`; it must be separate from
loop output and outside disposable `runs/`. Bundle names start with UTC publication
time, then a same-second sequence and short checkpoint ID; full hashes remain in
the metadata and republishing preserves the original name. Bundles are write-once/read-only, hashed,
and never automatically pruned. Both initial parents and every completed candidate
are backed up before further progress/promotion. Initial checkpoints are trusted
baselines, not magically gate-tested. Accepted bundles can be used directly with
native `--model` inference.

**ONNX alone cannot resume training.** The adjacent `checkpoint.pt` and `replay/`
provide model/BatchNorm, optimizer/RNG, options, holdout history and exact corpus
state. See [model-store.md](model-store.md) for recovery and backup commands. Keep
an off-machine backup: local read-only files do not protect against disk loss.
`models/` is Git-ignored; exclude it from destructive Git-clean operations.

## Process interruption and immutable artifacts

`loop.pt` is an atomic control checkpoint containing frozen config, artifact hashes,
replay metadata, accepted pointer, candidates and gate decisions. SGFs are the game
records; no JSON/JSONL game sidecars are written. Completed epochs, exports and
committed SGFs are verified rather than overwritten.

- **Input snapshots:** source checkpoint/eligible bootstrap files are copied exactly.
  Initial source files must keep matching their hashes until copying finishes;
  afterward the originals need not remain.
- **Self-play:** game `j` in iteration `i` uses seed
  `seed+(i-1)*games_per_iteration+j`. On restart, adopt valid finished SGFs and
  regenerate only missing/invalid partial files into new attempts.
- **Training:** reuse the newest compatible contiguous epoch snapshot, including
  publication gaps before `latest.pt`/`best.pt` aliases. A failed first epoch starts
  a fresh attempt. Partial epochs are not resumed; durable candidate backups occur
  once the requested training phase is complete.
- **Export:** verify and adopt an atomically published bundle if the supervisor
  checkpoint lagged behind publication.
- **Gate:** resume missing paired games without rerunning training. Preserve every
  completed match and its authoritative result. A durable accepted bundle must be
  complete before the accepted pointer advances.
- **Recovery publication:** stage and fsync checkpoint/data/models, then atomically
  publish a read-only bundle. Already published packs are verified and reused.

Ctrl-C interrupts the active child process group, waits for cleanup, and escalates
if needed. An inherited OS lock prevents a new supervisor racing a still-live child
if the old supervisor is killed. Never unlink `.loop.lock` to bypass it. A stale
trainer lock in an owned resumable attempt is removed only when its PID is absent.
Live, malformed or permission-inaccessible locks are not blindly deleted.

This is process-interruption safety, not an end-to-end power-loss guarantee. Batch
shape/backend changes can alter regenerated games; GPU backward is not bitwise
deterministic. Idempotent resume reuses committed work, not interrupted trajectories.
The replay cap bounds active selection, not archive disk usage or the holdout ledger.

Tests cover board ratios, model separation on rejection, unknown outcomes, gate
restart, recovery publication, complete-run loss, and strict CPU continuation from
archived data after deleting the original run. Real CPU and CUDA smokes are recorded
in [benchmarks.md](benchmarks.md). Next is calibrated strength evaluation and budget
tuning—not claiming that a small acceptance match proves improvement.
