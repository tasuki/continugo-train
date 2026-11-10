# Fresh schema-5 campaign — 2026-11-08

User requested training from scratch with the new input feature planes. Launched
at **15:53:20 UTC** in `runs/20261108-154636-schema5-scratch/`, launcher PID
**2990796** at startup. This is a launch receipt, not playing-strength evidence.

Family `cg5` starts from random weights with no historical optimizer, BatchNorm,
replay, holdouts or tensor caches. Revised schema 5 has twelve coarse planes and
eleven exact fields, including binary `splits_friendly_region`. Source commit
`74d10cb`; previous campaigns and the shared `models/` ledger are untouched.
Incompatible schema-4 models are excluded from the opponent pool.

## Plan

Retains the previous size-9 campaign's learning/search budgets: 500 fresh
non-neural MCTS bootstrap games without eye guidance, five initial epochs, then
60 rounds of 50 neural games and four epochs. Generation randomizes 32/256
simulations at 75%/25%; only expensive positions train policy. Utility is equal
outcome/normalized-score weighting. Latest actors advance without publication.

Default 32-channel/four-block network, leaky outcome head, score/ownership loss
weights 1/0.25, CUDA FP32, batch two, six feature workers, four Torch threads,
1,000 batches/epoch, eight training/four held-out positions per game. Replay
covers up to 512 unique games / 16 rounds plus this campaign's bootstrap.

Initial reference evaluation: 20 games. Every tenth round saves a milestone and
runs 20 parent plus 20 reference games, fixed 32 simulations. Uses **current
moderate adaptive-rating defaults** rather than the previous campaign's larger
budget: RD <= 75, 20 paired-color seeds, two opponents/four pairs each, cap 360
additional games. Insufficient compatible opponents is explicitly reported.
Nothing is automatically accepted.

## Initial verification

Bootstrap validated at **15:53:43 UTC**: **500/500 natural completions**, two draws,
zero truncations, 18,617 eligible generated rows and 4,574 policy-enabled rows.
Trainer deduplication excluded one repeated two-pass game: 399 training and 100
validation games, 14,763/3,852 eligible rows. CUDA training reports 81,766
parameters, twelve coarse/eleven exact inputs, `initialization={'mode': 'fresh'}`
and `inherited_split_games=0`.

Preflight: 120 Rust tests passed, thirteen optional tests ignored; release native
binaries built. Of 236 Python tests, 235 passed in the background invocation;
the real interruption/restart test inherited ignored SIGINT from the background
shell and failed its expected exit-code assertion. That exact test passed when
rerun in the foreground (including real two-round resume). Both logs and this
qualification are retained in `verification.json`. No production-code change was
needed. CUDA matrix multiplication and isolated ONNX Runtime availability passed.

## Visibility and recovery

The campaign's `README.md` is its operating card; `plan.json`/`plan.pt` record
budgets, feature names, seeds and hashes. `bin/` contains private independent
read-only binaries. `verified.pt` pins the launcher/runtime; source snapshots and
commit IDs are retained. Do not change the Python implementation during the run.

```sh
tail -f runs/20261108-154636-schema5-scratch/run.log
tail -f runs/20261108-154636-schema5-scratch/initial-training/training.log
# After loop initialization:
mise run loop-status -- --output runs/20261108-154636-schema5-scratch/loop
```

Verify `run.pid` against its live command before SIGINT. After launcher/children
have stopped, rerun the campaign's `run.py` to adopt completed work. Historical
campaign launchers must not be resumed with this runtime.
