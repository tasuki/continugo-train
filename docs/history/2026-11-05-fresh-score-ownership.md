# Fresh score/ownership size-9 campaign — 2026-11-05

User-authorized fresh start in
`runs/20261105-214158-score-ownership-size9/`, launched **21:43:53 UTC**.
This is an operating receipt, not evidence of improved playing strength.

## Archive and initialization

- Confirmed no old training/evaluation process was running. Acquired both model
  archive and rating writer locks, then moved `models/` intact to **`models_20261105/`**.
- Created a new empty repository-root `models/` for this campaign and subsequent
  compatible campaigns. Added `models_*/` to `.gitignore`; no old artifacts or
  rating evidence were deleted or migrated. Old controls must not resume against
  the replacement store.
- Fresh score/ownership network, BatchNorm, SGD, RNG and split ledger, family
  `cg2`; no warm-start checkpoint or historical SGFs imported. Architecture:
  32 channels / four residual blocks, leaky-ReLU outcome head, 32-hidden score
  head and 16-channel ownership head. Auxiliary loss weights 0.25 each.
- Bootstrap: **500 newly generated untrained/random-proposal MCTS games**, four
  independent 125-game CPU shards, size 9, 32 simulations, bootstrap eye avoidance,
  cap 1024, seeds 4,100,000–4,100,499. Literal random-only SGFs have no policy
  targets; the usual MCTS bootstrap provides real visits and natural final labels.
- Five initial training epochs from random weights, seed 510,001; then **60
  self-play iterations**, not 60 total epochs and not a round-30 stop.

## Frozen budgets

- Size 9 only, weights `9:1,13:0,19:0`.
- **50 fresh games + four training epochs per iteration**; planned 3000 neural
  self-play games and 245 total training epochs including initialization.
- Each epoch: up to 1000 batches of two positions, eight positions/game and four
  validation positions/game; six feature workers, four PyTorch threads, CUDA FP32.
  SGD lr 0.01 / momentum 0.9 / weight decay 1e-4 / clip 5; momentum retained between
  rounds, not imported from an old lineage.
- Self-play: 32 simulations, cap 1024, first seed 6,100,000; score weight **0.3**,
  utility `(outcome + 0.3*normalized_score)/1.3`. No neural eye guidance or rollout
  evaluation. Existing root-visit sampling unchanged.
- Replay: 512 unique labeled games / 16 rounds, with this campaign's bootstrap
  fallback; ancestral whole-game holdouts stay pinned within the new lineage.
- 12 neural games / batch limit 12 / collection wait 2 ms, six feature threads,
  one inference thread, 64 cached contexts. Latest actor, no automatic acceptance.
- Ratings initially against the fixed random-no-eye-fill reference, then rounds
  **10/20/30/40/50/60**. Each scheduled round has 20 parent-actor and 20 reference
  games plus the existing adaptive precision phase: target RD 25, 60 complete
  color-swapped pairs, two opponents with ten pairs each, up to 600 additional
  games per scheduled candidate. Reference cap 2048, other matches cap 1024,
  32 simulations, no handcrafted neural guidance. Evaluation is outside replay.

## Operation

Launcher PID at startup: **445233**, in `run.pid` (verify live process identity
before signaling). Native binaries are private read-only copies under `bin/`.
`plan.pt` records seeds, full loop settings and binary/Python hashes. Source/rules
commits, source patch/snapshot and passing `mise run test` / `mise run check` logs
are retained under the run. Do not change Python code during the active campaign.

```sh
tail -f runs/20261105-214158-score-ownership-size9/run.log
mise run loop-status -- --output runs/20261105-214158-score-ownership-size9/loop
mise run ratings
# Resume only with no live launcher or inherited child locks:
.venv/bin/python -u runs/20261105-214158-score-ownership-size9/run.py
```

Interrupt the launcher with SIGINT for supervised process-group cleanup. Resume
adopts completed bootstrap shards, initial epochs and loop stages without resetting
SGD or counters. Never remove a live lock. The source hash checks reject a mixed
implementation on restart. Strength, eye preservation, calibration, truncations,
wall-clock cost and memory still need measurement; low loss is not strength.

## Initial verification

Bootstrap completed at 21:44:32 UTC and passed full replay/protocol validation by
21:44:38: **500/500 natural completions, three draws, no truncations**, 30,658
eligible rows before canonical deduplication. Two duplicate games were excluded
from training; 398 train / 100 validation games provide 24,542 / 6,112 eligible
positions. No old game IDs or holdouts were inherited.

Verified the new CUDA trainer's saved epoch-2 checkpoint: initialization is
`fresh`, family `cg2`, cumulative optimized rows 4000 and generated rows 30,654
(after canonical deduplication). Held-out 400-row losses were policy CE 10.984957,
outcome MSE 1.001224, score MSE 0.008166 and ownership MSE 0.594322. These are
startup diagnostics, not a strength result. The launcher remains active, finishing
the five initial epochs before the initial reference match and 60-round loop.
