# Restartable synchronous neural self-play loop

`mise run loop` connects the existing native self-play, SGF loader, warm-start
trainer and ONNX exporter. It follows the small-machine synchronous pattern in
[KataGo's self-play training guide](https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md):
generate games, select/shuffle replay data, train, export, repeat. This initial
implementation **accepts each latest trained model without strength gating**.
It is not asynchronous training or evidence of improved playing strength.

## First small run

Requires an existing compatible **offline-training checkpoint**, the inference
extra, and the same CPU/CUDA runtime setup as [inference.md](inference.md).
`mise run loop` builds the native self-play and feature binaries first. The loop
supervisor currently requires **POSIX** process groups and `flock` (Linux/macOS).

```sh
mise run loop -- --checkpoint runs/offline-training/best.pt --bootstrap-data runs/offline-bootstrap --output runs/loop-001 --iterations 2 --games-per-iteration 6 --sizes 9,13,19 --seed 200 --simulations 8 --max-moves 64 --concurrent-games 6 --inference-batch-size 6 --replay-iterations 2 --replay-games 32 --train-epochs 1 --steps-per-epoch 2 --positions-per-game 1 --validation-positions-per-game 1 --batch-size 1 --workers 1 --shuffle-buffer 0
# Resume an interrupted run with its original total target:
mise run loop -- --output runs/loop-001 --resume
# Extend a completed run to a larger total target:
mise run loop -- --output runs/loop-001 --resume --iterations 3
```

Use a fresh output directory. CUDA is the default for both self-play and training;
add `--device cpu` for CPU. There is no implicit CUDA fallback. Export verification
runs on CPU. Native self-play, training and export run in **separate, supervised
processes**, so incompatible CUDA/cuDNN inference libraries do not replace the
training environment and GPU allocations are released between phases.

`--iterations` is a **total target**, not an additional count. New runs default to
one iteration. Resume defaults to the saved target; it may be increased, never
reduced. Already completed iterations are verified and skipped. Other loop settings
are frozen: explicit conflicting resume overrides fail. Source checkpoint and
bootstrap inputs are new-run inputs, not resume overrides. Keep the loop at the
same absolute path and software/rules/runtime versions for continuation; relocating
an entire loop is not supported in this first version.

## Stages and handoff

1. Snapshot the initial checkpoint and optional eligible bootstrap SGFs into the
   loop directory. Hashes and source paths are recorded before copying. Resume can
   finish this snapshot stage while original inputs still match their hashes;
   once copied, original inputs need not remain available.
2. Export the initial checkpoint as an atomic encoder/scorer bundle. Both graphs
   must match the checkpoint identity/features/configuration and pass PyTorch
   parity verification before adoption.
3. Generate the configured number of games with that iteration's **frozen** model.
   Game `j` (zero-based) of iteration `i` uses seed
   `seed + (i-1)*games_per_iteration + j`; sizes cycle from the beginning of `--sizes`
   each iteration. Seed overflow is rejected. Batching is as in [batching.md](batching.md).
4. Validate generated SGFs using the shared Rust codec/rules/scorer. The supervisor
   checks size, seed, model identity, search settings and every move's visit total
   through Rust inspection metadata. Completed games need authoritative RE; fully
   truncated games retain real policies but receive **no invented outcome**.
5. Freeze a bounded replay selection and its corpus/split fingerprint, then
   warm-start training from the parent checkpoint.
6. Export the **latest completed epoch**, atomically record its bundle and checkpoint
   in `loop.pt`, and use them for the next iteration. `best.pt` is still produced by
   the trainer but is not a promotion gate or the loop's weight handoff.

A model is never switched midway through a game's generation. Source and iteration
checkpoints are immutable inputs; warm-start creates separate training runs.
Global loop progress and local epoch/optimizer-step counters are distinct.

## Replay and training controls

- `--replay-iterations` (default 4): only the current and most recent N-1 generated
  iterations are replay candidates. Eligible bootstrap snapshots remain older
  fallback candidates, not a permanently protected allocation.
- `--replay-games` (default 256): cap on selected **unique completed, policy-labeled
  games**, including bootstrap games. Newer iterations precede older iterations;
  duplicate decoded plays use the newest candidate. Selection is deterministic,
  round-robin across board sizes, preferring newer games within each size. It is
  game-based, not a uniform position reservoir or KataGo's adaptive row window.
- Split assignments come from the parent checkpoint's ancestral ledger, including
  removed games. New games are assigned deterministically using the existing
  resolution-stratified splitter. A training game cannot become validation when
  the replay window changes. See [training.md](training.md#holdout-safety-across-replay-windows).
- The selected corpus must contain both training and validation games. Supply
  `--bootstrap-data` or enough unique new completed games per resolution. A source
  checkpoint's manifest does **not** implicitly import its original SGFs. Windows
  containing only ancestral training games fail rather than manufacture a holdout.
- Each iteration must produce at least one completed policy-labeled game, even if
  old replay data is available. If every game hits the move cap, the loop stops
  instead of silently training only on stale data. Start a new loop with a suitable
  move/search budget if the frozen configuration cannot produce usable data.

Training defaults deliberately keep a cycle small: one epoch, at most 20 batches,
eight sampled training positions per game and four validation positions per game,
batch two, two feature workers and shuffle buffer eight. Caps are configurable;
zero position/step limits mean the trainer's normal full-pass behavior. These
settings are **not tuned for strength** or an optimal training/new-data ratio.

The loop defaults to `--warm-start-optimizer keep` to continue SGD momentum across
iterations; select `reset` explicitly to discard it. This differs from the standalone
trainer's conservative reset default. Architecture, feature density, momentum,
weight decay and gradient clipping inherit from the source; `--lr` optionally sets
the learning rate. Iteration `i` uses training/split seed `seed+i-1`. Local epoch,
step, RNG and best-loss tracking follow the [warm-start contract](training.md#warm-start-on-a-changing-replay-corpus).
Training within a resumed iteration uses strict fixed-corpus `--resume` instead.

## Artifacts

```text
loop.pt                         atomic control checkpoint, frozen config/artifact hashes
loop.log                        supervisor diagnostics
.loop.lock                      persistent flock inode; informational supervisor PID
initial.pt                      exact source checkpoint snapshot
initial-model/{encoder,policy}.onnx
bootstrap/*.sgf                  exact eligible source SGF snapshots (optional)
iteration-0001/
  selfplay/attempt-0001/*.sgf    authoritative game records, including truncations
  selfplay/attempt-0001/selfplay.log
  train-attempt-0001/           trainer epoch snapshots, aliases, lock and training.log
  train-attempt-0001.log        supervised process output
  model/{encoder,policy}.onnx   published next model
  model-export.log
```

`loop.pt` stores only orchestration state, artifact identities and replay metadata;
it is **not an alternate game record**. SGFs remain self-contained. No JSON/JSONL
game sidecars or external model metadata manifests are written. Resume reads torch
checkpoints with `weights_only=True`. Artifact hashes are verified before reuse;
changed or missing committed files fail rather than silently regenerate.

The replay cap bounds **active training selection**, not archive disk usage or the
ancestral split ledger. Old iterations, attempts and snapshots are retained for
recovery/audit. No garbage collector is implemented; do not delete committed files
from an active loop. Native game concurrency, inference queues and feature buffers
retain their existing bounds. A game cap is not a bound on positions or total RSS.

## Interruptions and recovery

Ctrl-C sends SIGINT to the active child process group and waits for cleanup, with
TERM/KILL escalation if needed. Resume with the same output directory:

- **Games:** inspect interrupted attempts, adopt valid finished SGFs, and regenerate
  only missing/invalid uncommitted games into new attempt directories. Partial SGFs
  are retained for diagnostics, never overwritten or included in replay. Successful
  jobs with missing/invalid output fail as errors instead of hiding backend bugs.
- **Training:** reuse the newest contiguous, compatible immutable epoch snapshot,
  even if `latest.pt`/`best.pt` publication lagged behind. Completed epochs are not
  overwritten. An interruption before any epoch finishes starts a new training
  attempt, preserving the old one. Partial epochs are not resumed.
- **Export:** a published bundle is checked and adopted if interruption occurred
  after its atomic directory rename but before the loop checkpoint update. It is
  not overwritten or needlessly re-exported.

The supervisor holds an OS file lock which is inherited by its direct children.
If the supervisor is killed, a still-running child keeps the lock: another runner
cannot race it. Wait for that child to finish, or stop it before restarting. **Do
not unlink `.loop.lock`** to bypass a live lock. After acquiring it, the runner may
remove a stale trainer lock in its own resumable attempt only when the recorded PID
is demonstrably absent. Live, malformed or permission-inaccessible trainer locks
are not blindly removed. Other standalone trainer lock rules remain unchanged.

This is process-interruption safety using atomic replacement, not a power-loss or
storage-failure durability guarantee. Independent game seeds are stable, but a
regenerated game may differ due to batch-shape/backend floating-point changes.
GPU backward is not bitwise deterministic. An idempotent resume guarantees reuse
of committed artifacts, not an identical interrupted mid-game/mid-epoch trajectory.

## Verification and next steps

Tests inject interruptions during snapshots, self-play, first/incremental epochs
and export publication; check immutable output, strict settings, inherited locks,
model lineage, replay limits and idempotent resume. A real CPU test interrupts the
supervisor with SIGINT, resumes, and completes two iterations across 9/13/19. The
real CUDA two-iteration smoke is recorded in [benchmarks.md](benchmarks.md).

Next is reproducible **playing-strength evaluation** across sizes and budgets.
There is no gatekeeper, arena, automatic capacity scaling, adaptive train/new-data
budget, disk pruning, browser integration or multi-machine asynchronous scheduler
in this milestone. Tune those only after measuring strength per unit of compute.
