# First overnight training run — 2026-11-01

Launched at **22:39 UTC**. Target: about six hours, with graceful interruption at
**2026-11-02 04:39 UTC**, or earlier completion/failure. This is real training,
not a claim of measured improvement or calibrated evaluation. Finished normally
at **2026-11-02 01:55:02 UTC**, after **3h 15m 38s**, reaching the work limit
before the deadline.

Run: `runs/20261101-223811-overnight/`. The detached runner's PID is recorded in
`runner.pid`; `status.txt` records stage transitions and the terminal state.
The full launch script, commands, settings and stop/resume notes are in that
run's `run.py`, `run.log`, and `README.md`. These generated files are Git-ignored.
Durable loop recovery packs and ratings go into this repository's `models/`.

## Budget

- Existing FP32 32-channel / 4-block network (80,099 parameters); no resizing.
- Default 40/40/20 size-9/13/19 coverage throughout.
- **100 bootstrap MCTS games**, seeds 10000..10099, 32 simulations/move,
  512-move cap. Generation finished: all 100 naturally completed, no draws or
  truncations, 2,371 eligible rows; 80 training / 20 validation games.
- **Initial training:** 3 epochs, at most 500 batches/epoch, batch 2, up to 16
  training positions/game/epoch and 16 fixed validation positions/held-out game.
  CUDA, two feature workers, four PyTorch CPU threads, seed 12001; SGD defaults.
- **One neural loop iteration:** 300 self-play games, seeds 20000..20299,
  32 simulations/move, 512-move cap, six concurrent games / inference batch limit
  six, four feature threads, one inference helper thread, 64 cached contexts.
- **Replay/training:** at most 400 unique eligible games, 3 epochs of up to 1000
  batches each, batch 2, same 16-position caps, keep optimizer progress.
- **Gate:** 100 paired-color games, 32 simulations/move, 512-move cap;
  require 60% of planned points using conservative bounds for unknown outcomes.
- **Reference:** 10 matches for the initial baseline and, if promoted, the
  candidate; 2048-move cap. Results inform ratings, not promotion.

At most **one candidate promotion** is possible because there is one iteration.
The general durable daily promotion cap is **not implemented**; do not blindly
increase the iteration target. The initial bootstrap model is a trusted baseline,
not a gate-qualified improvement. Budgets/thresholds are provisional.

## Results and draw audit

Initial training took 11 minutes. All 300 neural self-play games completed
naturally: 6,934 moves in 8,321 seconds (2h 18m 41s), with 55 draws and no
truncations. Training continued for three epochs; the candidate was rejected
with 0 wins, 6 losses and 94 draws (47% points against the 60% requirement).
The initial network lost all 10 reference matches. No candidate was promoted;
its training/optimizer progress remains archived.

All **510 SGFs** (100 bootstrap + 300 neural self-play + 100 gate + 10 reference)
were subsequently replayed/validated through the authoritative Rust inspector.
**Every draw was a two-move game: Black passes, White passes, empty board.**
There were no draws after any stone placements in this experiment.

| Gate board size | Games | Two-opening-pass draws | Other results |
|---|---:|---:|---|
| 9 | 40 | 40 | None |
| 13 | 40 | 40 | None |
| 19 | 20 | 14 | Six candidate losses, each only four moves |

All 55 neural self-play draws were also two-opening-pass games (41 size 9,
12 size 13, two size 19). Bootstrap and reference games had no draws.
For example, gate `game-000097-size9-seed20048-candidate-black.sgf` contains
only `B[];W[]`. A four-move loss,
`game-000005-size19-seed20002-candidate-black.sgf`, is Black pass, White
placement, Black pass, White pass, scored W+470.9.

The gate is dominated by degenerate early passing, not close games ending in
improbably equal territory. Inspect pass/search/value behavior before extending
training or interpreting the gate/ratings as useful strength calibration.
Do not suppress these draws or invent replacement outcomes.

### Opening inference diagnostics

Real Rust features fed to the saved eager models (evaluation mode, CUDA,
full-domain policy normalization) show:

| Size | Initial P(pass), empty board | Candidate P(pass), empty board |
|---|---:|---:|
| 9 | 29.39% | 67.34% |
| 13 | 18.35% | 52.13% |
| 19 | 11.82% | 44.50% |

For the size-19 four-move loss example, the initial model predicts value
+0.127843 both on the empty board and for Black facing White's single stone
(rounded to six decimals). The candidate predicts -0.294410 and -0.294220,
respectively. These tests indicate very weak position discrimination in these
opening examples; they do not identify the value error over all positions.

Empty-board coarse/exact/legality/action inputs are **identical before and after
Black's opening pass** on all three sizes. Current features omit pass history;
the search/rules still retain the actual history and detect terminal outcomes.

Neural MCTS uses network values at nonterminal leaves, not completed-game
rollouts. At 32 simulations the default widening schedule admits roughly a
dozen coordinate children. A high pass prior plus weak leaf values can leave
pass the largest individual visit count, even when placements collectively
receive more visits. Evaluation selects that single maximum-visit action.
These observations explain why search cannot be assumed to repair poor
bootstrap play; they are not yet an ablation quantifying each cause. Investigate
cold-start supervision, pass-state observability, and search-budget sensitivity
rather than silently filtering natural outcomes or forbidding legal moves.

## Inspect progress

From the repository root:

```sh
.venv/bin/python runs/20261101-223811-overnight/progress.py
watch -n 60 .venv/bin/python runs/20261101-223811-overnight/progress.py
mise run ratings
```

The progress summary tails training losses, loop transitions, generation and
arena logs. Detailed epoch metrics/checkpoints include policy CE, value MSE,
total loss, actual per-size sample counts, throughput, loader wait, compute time
and memory. Raw match outcomes, per-size gate reports, rating/RD, truncations,
and SGFs are needed to assess playing strength; loss alone is not sufficient.

If the deadline lands during an epoch, partial optimizer work is not saved.
Completed epochs/games remain; complete loop candidates are backed up before
gating. Initial offline training checkpoints remain under `runs/` until the loop
archives its initial parent, so do not remove the run while it is in flight.

After the detached runner has stopped, resume an existing loop without increasing
its target:

```sh
mise run loop -- --output runs/20261101-223811-overnight/loop --resume
```

Manual resume has no six-hour deadline unless separately supervised. Do not
rerun the one-off launch script over existing output. If stopped before the loop
starts, resume the latest completed offline epoch instead; see the run README
and [training](training.md). Never unlink a live lock file.

Preflight: native release binaries built; 22 acceptance/rating tests passed;
native CPU/CUDA heavy-operator parity test passed; real PyTorch CUDA
forward/backward succeeded.
