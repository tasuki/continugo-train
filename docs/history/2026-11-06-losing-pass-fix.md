# Losing-pass exclusion removed; corrected continuation — 2026-11-06

The user identified eye-filling/self-destructive continuation as the priority:
passing must remain available even when passing loses. They authorized stopping
the current campaign and incorporating the correction without starting over.

## Search correction

`src/search.rs` previously excluded a pass from PUCT whenever a placement child
existed and passing either lost immediately or allowed the opponent an immediate
winning pass. There was no evidence that the placement was better. This prevented
score-aware utility from choosing a smaller terminal loss over destructive play.

Both neural and unguided search now retain losing passes in normal selection.
The unused hypothetical first-pass loss probe was removed; first-pass values
come from search. Exact terminal winning-pass shortcuts remain. No game rule,
label, reward, neural eye filter, training objective or head architecture changed.
Ownership remains auxiliary supervision; outcome and score directly form search
utility. This fixes an action-eligibility defect, not demonstrated playing strength.

Regression coverage checks that first/second losing passes can beat worse-valued
continuations in both searches, plus visit conservation and the winning-pass
shortcut. **74 Rust and 194 Python tests passed**, with formatting and Clippy.

## Authorized handoff

- Source campaign: `runs/20261105-214158-score-ownership-size9/`.
- Corrected handoff/trial: `runs/20261106-111101-losing-pass-fix/`.
- Original supervisor was gracefully interrupted in round 32. Its 23 completed
  games remain preserved; the handoff resumes only missing work and finishes
  old round 32 using its original private binaries, then pauses before round 33.
- After that completed boundary, the launcher forks the durable training
  checkpoint plus exact replay. Weights, BatchNorm, SGD momentum, lineage
  counters and ancestral validation assignments are retained. Nothing is reset
  or overwritten in the old campaign; normal boundary completion writes its
  ordinary training/recovery artifacts.
- Corrected binaries are separate read-only copies. Search/training/game budgets
  remain fixed: size 9, 50 games/round, 32 simulations, score weight 0.3, four
  epochs, 1000 batches of two rows, lr 0.01, replay 512 games/16 rounds.
- The generation seed advances to **6,101,600**, without reusing the old stream.
  Local round 1 is global round 33. The original 60-round plan becomes a 28-round
  fork target, but the launcher **always pauses after five corrected rounds**
  (global round 37 / 250 fresh games) for review. No automatic acceptance.
- Shared `models/` archives/ratings remain intact. Executable fingerprints use
  the ordinary new-loop protocol. Evaluation interval remains ten local rounds,
  so the short trial has an initial reference batch but no scheduled endpoint
  match. Endpoint strength needs separate comparison at review.

The run's README contains restart/status commands and exact source/test receipts.
`fork.pt` is deliberately absent until original round 32 completes; its presence
pins the actual source checkpoint/replay/configuration. `run.log` distinguishes
boundary completion from corrected training. An armed handoff is not a claim
that corrected training or the trial has already completed.
