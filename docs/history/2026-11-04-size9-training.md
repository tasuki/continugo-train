# 2026-11-04 fresh size-9 training campaign

Started **2026-11-04 15:14:54 UTC** in
`runs/20261104-151420-size9-training/`. This is a launch report, not a completed
campaign result or evidence of playing strength.

## Schedule correction and clean restart

The user clarified the intended schedule: **60 self-play rounds, four training
epochs per round**, with ratings every ten rounds, matching the previous
campaign's round/epoch budget. There are also **five initial bootstrap-training
epochs**, for 245 training epochs across the campaign.

The mistakenly configured 60-total-epoch campaign in
`runs/20261104-150017-size9-training/` was interrupted during initial reference
evaluation, before any neural self-play/training rounds. At the user's explicit
request, its entire run directory and dedicated model/rating store were deleted
after confirming the processes exited and locks were released. No accepted
models existed in that store. Its weights, optimizer state, SGFs and rating
evidence are **not reused**. The new campaign uses new bootstrap, training and
self-play seeds; only the tested software/binaries are reused.

## Configuration

- Fresh `cells-v1` weights, BatchNorm and SGD state. Network: 32 channels, four
  blocks, 80,116 parameters; top 32 cells plus eight exploration cells, density 4.
- All generation, training/validation sampling and evaluation are size 9 only;
  explicit weights `9:1,13:0,19:0`.
- Bootstrap: 500 fresh MCTS games on four CPU processes, seeds
  1,100,000–1,100,499; 32 simulations/move, cap 1,024 moves, explicit
  `--bootstrap-avoid-eyes`.
- Bootstrap-only guidance uses shared `SmallEyeAnalysis` for tree placement
  proposals, with captures exempt. Legality, neural masks, terminal-pass checks,
  real visit accounting and natural outcome labeling are unchanged. Guidance is
  forbidden in neural search/evaluation and cannot be combined with rollouts.
  Enabled guidance is recorded in SGF root search configuration; disabled fields
  are omitted to preserve normal neural job/protocol checks.
- Initial training: five epochs from random initialization, seed 410001.
- Each training epoch: up to 1,000 batches of two rows (2,000 optimized rows),
  up to eight sampled positions/game and four validation positions/held-out game.
  Whole-game ancestral holdouts remain pinned. Six feature workers, four PyTorch
  threads, FP32 CUDA; SGD lr 0.01, momentum 0.9, weight decay 1e-4, clipping 5.
- Loop: **60 rounds**, each generating **50 fresh neural self-play games** then
  training **four epochs**. Search: 32 simulations/move, cap 1,024; starting seed
  2,100,000. Latest experimental actor advances without a strength gate, and SGD
  state is kept. Replay: 512 unique labeled games, up to 16 rounds, with bootstrap
  fallback. Planned loop generation: 3,000 fresh games.
- Neural execution: 12 concurrent games / batch limit 12 / two-ms collection
  wait, six feature threads, one ONNX thread, 64 cached contexts.
- Ratings: initial actor after five bootstrap epochs, then rounds
  **10/20/30/40/50/60** (cumulative training epochs 45/85/125/165/205/245).
  Each batch: 20 paired-color games against `random-no-eye-fill-v1`, neural
  maximum-visit play, 32 simulations/move, cap 2,048.
- Parent-actor matches disabled (`--gate-games 0`); no actor-match threshold is
  used to select actors. No automatic acceptance/publication.
- Frozen read-only binaries under the run's `bin/`; source patch, source/rules
  commits, binary SHA-256 hashes and exact stage commands retained in the run.
- Dedicated durable store: `models/20261104-151420-size9-training/`, separate from
  disposable runs and the preexisting unowned `models/README.md`.

## Inspection and uncertainty

Before this clean restart, the same implementation passed an inspected fresh
one-epoch pilot, native export parity, four naturally completed unguided neural
games (including one opening draw), and bootstrap SGF/visit/outcome validation.
Those deleted artifacts are not inputs to the new campaign. Inspect new losses,
samples, completion/pass behavior, SGFs and reference outcomes as they arrive.
A low loss is not evidence of useful values, eye preservation or strength.

Twenty-game reference batches are coarse progress diagnostics, not calibrated
promotion tests. For illustration, 10 wins in 20 independent decisive games gives
roughly a 30–70% Wilson 95% interval; paired seeds and draws complicate that
interpretation. Glicko RD is not itself a 95% confidence interval. Report actual
W/D/L, color coverage, rating/RD and truncations.

Capped games remain unknown, not draws or forced scores, and are excluded from
training outcomes and skill updates. Inspect planned-score bounds and completed-
only scores: outcome-dependent truncation can bias the latter and is not covered
by ordinary sampling intervals or rating deviation. Size-9-only ratings do not
establish larger-board strength. No result authorizes acceptance by itself.

## Operation

From the repository root:

```sh
# Continue/resume; do not run alongside a live launcher:
.venv/bin/python -u runs/20261104-151420-size9-training/run.py

# Follow stage transitions and initial training:
tail -f runs/20261104-151420-size9-training/run.log
tail -f runs/20261104-151420-size9-training/initial-training/training.log

# Once loop initialization has started:
mise run loop-status -- --output runs/20261104-151420-size9-training/loop
mise run ratings -- --model-store models/20261104-151420-size9-training
```

The launcher resets background-shell SIGINT ignoring, has an inherited process
lock and supervised process groups. Interrupt its PID with SIGINT for orderly
child cleanup; never remove live lock files. Resume uses completed offline epochs
and loop recovery, not a fresh optimizer. Do not replace run-owned binaries.
Local durable bundles still need an off-machine backup.

## Validation

Rust tests, shared rules/analysis workspace tests, Clippy and Ruff passed. All
139 Python tests passed on full rerun, including native inference/export parity
and rating/storage tests. The first Python run had one real-loop interruption
timing failure; it passed on isolated rerun too, without test changes.

## New campaign's initial inspection

Bootstrap finished at **15:15:32 UTC**: 500/500 natural completions, 257 black
wins, 241 white wins, two draws, no truncations. All records passed Rust replay,
outcome, guidance-flag and 32-visits-per-move checks. There were 30,606 moves
(mean 61.21, range 2–101). One duplicate canonical game was excluded, leaving
499 unique games: 399 train / 100 validation, 24,470 / 6,134 eligible positions.

The first fresh epoch saved `cg1-b4c32-s2000-d30604`: 2,000 optimized rows,
400 validation rows, all size 9. Training policy CE/value MSE were
11.158035/1.040038; validation 11.045247/1.123541. Initialization is fresh, with
no inherited split games. These are loss diagnostics, not strength evidence.

Append later measurements as results arrive.
