# Cold-start passing investigation — 2026-11-02

The [first overnight run](overnight-2026-11-01.md) produced a pass-heavy network:
94/100 gate games were two opening passes; the other six lasted four moves.
This document records the search changes and bounded follow-up experiment, not
an assertion of increased playing strength.

## Exact terminal pass checks

Both neural and unguided MCTS now examine legal pass continuations before neural
leaf estimates or progressive widening can obscure them:

- If passing **ends the game with a win**, take that proven win regardless of
  policy prior, leaf-value bias, or the widening budget.
- If passing **loses immediately**, or allows the opponent to **win immediately
  by passing**, that action is a proven loss. Prefer an available placement to
  it, rather than allowing a large exploration bonus to select the known loss.
- A draw remains a legal choice. Being ahead on an unfinished board does **not**
  prove a win: the opponent need not cooperate by passing. An incomplete set of
  sampled placement children never proves that the whole position is lost.

Only actual terminal hypothetical games are scored by shared rules. There is no
unfinished-territory value bonus, early-pass ban, invented label, eye filter,
feature-schema change, network resizing, or learning-rate change. These checks
are applied to the existing pass action; full-policy normalization and independent
coordinate exploration remain unchanged. The default widening coefficient is
still 2.0; a narrower coefficient alone did not rescue the failed network.

Search behaviour/seeds and arena fingerprints change. **Do not resume the old
loop with rebuilt evaluation binaries**; fork a new experiment instead. Its
immutable bundles, CSV ratings, SGFs and old binaries remain available.

## Optional rollout evaluator (experimental, standalone only)

Self-play and arena expose:

- `--rollouts N`: zero (default) retains neutral unguided / neural leaf values;
  positive N evaluates each new leaf with N hypothetical completed-game playouts.
- `--rollout-max-moves 512`: safety cap per playout.
- `--rollout-pass-probability 0.03125`: random pass probability, otherwise bounded
  full-domain legal coordinate proposals using `--proposal-attempts`. A failed
  bounded placement proposal also passes; this does not claim no legal move exists.
- Immediate winning passes are taken exactly. The direct random reference's
  narrow eye-avoidance filter is **not** used in rollouts or MCTS.

In neural rollout mode the network supplies policy, **not leaf values**. Rollouts
use shared legality and natural terminal scoring. Unknown capped playouts have no
observed value and are not counted as draws; completed-only evidence is averaged
separately from structural search visits. If no completed evidence exists, the
selection fallback is neutral/unknown, not a labeled draw. Mixed censoring can
still bias these estimates: inspect completed/truncated counts and measure cost.

SGF root comments record the complete rollout configuration and aggregate
`rollout_stats` (completed/truncated, wins/losses/draws from each leaf's perspective,
and simulated plies). These are diagnostic work counters, not additional played
or training games. Actual policy visits and natural actual-game outcomes remain
the only training targets. A truncated actual game never acquires RE because its
hypothetical rollouts completed.

```sh
mise run selfplay -- --engine mcts --games 10 --simulations 32 --rollouts 2 --rollout-max-moves 256 --widening 0.5 --output runs/rollout-test
# Bootstrap-search reference matches, both candidate colors, no ONNX dependency:
mise run arena -- --unguided --reference-random --games 10 --simulations 32 --output runs/bootstrap-reference
```

Unguided diagnostic matches use maximum root visits, just like neural evaluation;
ordinary self-play still samples visits at temperature one. Reference turns have
no invented search-policy targets. Evaluation SGFs must stay out of replay.

Rollouts are not yet exposed by the gated loop or its rating protocol. Do not
ingest rollout/unguided matches into an existing neural-rating pool. Existing
rating ingestion checks the complete recorded search settings and player IDs.

## Experiment and early findings

Artifacts: `runs/20261102-102710-search-calibration/`. Frozen original executables
are in `baseline-bin/`; the fresh-training diagnostic freezes its own executables
in `trial-bin/`. The detached `trial.py` records stage commands in `trial.log`,
stage output in `trial-*.log`, and status in `trial-status.txt`. It originally
had a 45-minute diagnostic limit. At the user's request that whole-batch limit
was removed on 2026-11-02: the active arena continued uninterrupted under
`finish-reference.py`, with progress in `reference-watch.log`. Only the deadline
guardian was replaced; no game was restarted or model promoted. The finite
10-game target and per-game 512-move safety cap remain. No model is automatically
accepted or promoted by this diagnostic.

Same seeds 32000..32099, 100 games, 32 simulations/move, 512-move cap, 40/40/20 mix:

| Bootstrap | Moves | Mean moves/game | Completed | Draws / two opening passes |
|---|---:|---:|---:|---:|
| Original neutral search | 2,303 | 23.03 | 100 | 0 |
| Neutral search + exact terminal checks | 12,712 | 127.12 | 100 | 1 |

Longer games are not proof of stronger play. The changed teacher now avoids a
provably losing pass while a sampled placement remains available; the old teacher
frequently ended on those short, avoidable losses.

Two frozen failed-network self-matches at size 9 still drew by opening passes
when narrowing widening from 2 to 0.5, including a cpuct reduction from 1.5 to 0.5.
A rollout-backed opening probe on that already pass-biased network also retained
pass as the largest individual visit count. Hence merely continuing its weights
is not assumed to repair the feedback loop.

A preliminary size-9 reference comparison used **temperature-one MCTS candidate
moves**, before the new diagnostic was corrected to maximum visits: exact checks
alone scored 1 win / 9 losses in 3.47s; two rollouts/leaf, cap 256, widening 0.5
scored 1 win / 9 losses in 34.88s. Both had zero draws/truncations. This is a tiny,
non-promotional cost probe, not evidence of equal strength or a calibrated rating.
It does not justify enabling rollouts by default. A subsequent maximum-visit
comparison, same size-9 pairs/seeds/budgets, scored **2 wins / 8 losses** for each
configuration: 3.99s without rollouts versus 34.11s with rollouts. These ran while
the fresh neural reference batch was also active, so wall times are diagnostic,
not isolated performance benchmarks. This sample still does not establish equal
strength or justify the additional rollout cost as a default.

## Fresh-network diagnostic

The simpler exact-terminal-check bootstrap is being tested first. Fresh family
`cold1`, unchanged 80,099-parameter 32-channel/4-block network, SGD lr 0.01,
momentum 0.9, batch 2, two feature workers, CUDA, seed 14001:

1. Generate the two 100-game bootstrap corpora above; train only on the corrected
   corpus, preserving whole-game holdouts (80 training / 20 validation games).
2. Train three epochs, at most 500 batches each, up to 32 training positions/game
   and 16 fixed validation positions/held-out game. No inherited weights/optimizer.
3. Back up the complete checkpoint/corpus into the isolated durable store
   `models/cold-start-trial-20261102/`, without changing the existing accepted model.
4. Export and check 10 size-weighted opening-search probes. These deliberately
   stop after one actual move, have no outcome label, and are never training data.
5. Play 10 paired-color reference matches at 32 simulations, actual-game cap 512.
   Inspect censoring and per-size outcomes rather than inferring strength from loss.

Fresh training completed in 14m 53s (including validation, with concurrent test
load); all three epoch checkpoints were retained. The latest checkpoint was
backed up as `cold1-b4c32-s3000-d12712` in the isolated model store. Validation
value MSE was 1.003 / 1.010 / 1.091, so neural values are **not yet demonstrated
reliable**; these are diagnostic experiments, not authorization for a longer
neural self-play run.

The epoch-2 checkpoint's empty size-9 P(pass) was 1.57%, versus 29.39% for the old
initial network and 67.34% for the rejected candidate. More importantly, all **10
latest-checkpoint opening-search probes** (4/4/2 by size) strictly preferred a
placement by maximum visits: pass had 0 visits on size 9 and 2 visits on sizes
13/19; the leading placement always had 3. These are root-policy diagnostics,
not ten completed games. The played one-ply sampling itself is not the evaluation
chooser; the comparison uses its recorded root visits.

The reference batch is still in progress. Its first three completed games were
124 / 149 / 175 moves, with **1 win / 2 losses**, no draws or truncations. Do not
interpret this partial small sample as an established strength gain or compare
its ratings with the old executable's pool. No model has been promoted.

Read progress from the repository root:

```sh
.venv/bin/python runs/20261102-102710-search-calibration/progress.py
```

Tests include misleading values/priors, winning-pass selection, provably losing
passes, refusal to assume cooperative replies, rollout perspectives/censoring,
unknown-value accounting, valid sparse visits and reference-side labeling.
All **53 Rust tests** and **88 Python tests**, including CPU/CUDA parity, gate
recovery and archived training continuation, passed. Clippy, formatting, Ruff and
Git whitespace checks also passed. Benchmark/environment-dependent Rust tests
remain explicitly ignored; the Python suite exercises the native inference tests.
