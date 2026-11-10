# Round-70 pass-versus-capture pilot — 2026-11-09

User requested the small forced-first-move continuation experiment using the
ongoing schema-5 campaign's **round-70** network. No campaign setting, production
source, training weight, optimizer, replay or publication was changed. A separate
sequential diagnostic shared GPU time with the ongoing run.

Artifacts: `scratch/20261109-173918-r70-branch-pilot/`, including independent model
copies, hashes, selection evidence, native/Python diagnostic source, eight SGFs,
JSON traces, `summary.json`, `README.md` and **`comparison.png`**.

## Model and selected position

- `cg5-b4c32-s570000-d177833`, checkpoint SHA-256
  `b3fd8e849cd18b5074ed3441bb8f53c7f89cd3ff69a130d91992d0907638c305`.
- Source: round-70 reference game
  `game-001394-size9-seed12100696-candidate-white.sgf`, **ply 65 / move 66**.
  Source actor hash is verified against the pinned model. White actually passed
  with **45 stones** on the board.
- Legal alternative **W[pAbe]**, coordinate **(806, 56)**, captures **13 Black
  stones**. Current-board White margin rises by 55.89 stone-area units. This is a
  selection diagnostic, not a nonterminal training reward or proof of final gain.
- Selection was fixed before continuations: replay the 20 round-70 reference
  SGFs; among recorded legal non-eye alternatives to neural passes with at least
  30 stones, choose the largest immediate capture, breaking ties by earliest ply,
  path, visits and action. This produced 66 eligible pass positions and one
  failure-conditioned witness—not an unbiased population estimate.

## Paired continuations

Force pass or capture, then use the frozen round-70 actor on **both** sides,
32 simulations per move, temperature-one visit sampling, unchanged default
search parameters including 50/50 outcome/score utility. No heuristics, eye
filter or pass restriction. Four paired seeds; 256-continuation-move safety cap.
**All eight naturally completed.** Shared Rust rules/SGF decoding independently
verify the full source prefix, forced action, legality, terminal RE, length and
reported score perspective. No Python SGF/rules implementation is used.

Final **White margins in stone-area units**:

| Seed | Force pass | Force capture | Capture − pass |
|---|---:|---:|---:|
| 7001 | +73.17 | +57.64 | −15.53 |
| 7002 | +56.88 | +69.28 | +12.40 |
| 7003 | +69.19 | −61.07 | −130.26 |
| 7004 | +35.18 | +88.07 | +52.89 |

Capture has a larger margin in **2/4** pairs. Means: pass +58.61, capture +38.48;
mean paired difference **−20.12**, dominated by a large later reversal. White wins
4/4 pass branches and 3/4 capture branches. These tiny failure-selected samples
do **not** establish that passing is better or that capturing is reliably better.

Continuations took about **5.1 minutes** total compute, excluding model loads,
root probes and harness development. The first CUDA continuation took about 52
seconds including loading. The initial CPU attempt aborted on strict encoder
output validation and supplied no final result; it is retained but excluded.
No validation was relaxed; reported runs use the campaign's CUDA backend.

## Critic and search evidence

Eager-CPU inference with frozen native features cross-checks CUDA estimates:

- Root coarse pass probability: **27.64%**.
- White-perspective predicted normalized final score after pass: **+0.312**.
- After capture: **+0.488**, about **18.56** more predicted stone-area units.
- Outcome predicts a near-certain White win for both children.

Thus the critic **does recognize the capture's benefit** on the immediate child
inputs. This witness does not establish that the critic cannot see captures.

Four independent 32-visit root probes:

| Seed | Pass visits | Visits to any 13-stone capture |
|---|---:|---:|
| 7001 | 4/32 | 0/32 (not proposed as a root child) |
| 7002 | 5/32 | 6/32 |
| 7003 | 4/32 | 2/32 |
| 7004 | 5/32 | 0/32 (not proposed as a root child) |

Pass is maximum-visit after tie-breaking in seeds 7003/7004. In the original
source, both pass and the selected capture had 4/32 visits; the pass was sampled.
These are fresh seeded searches, not original-game RNG replays. Absence of a
root child does not locate the omission specifically in coarse-cell selection
rather than coordinate sampling/progressive widening.

## Interpretation

The continuation test does **not** provide a clean action-quality verdict:
later tactical swings swamp the immediate gain. After forced pass in seed 7001,
White later captures a 34-stone group; after forced capture in seed 7003, Black
later captures a 34-stone White group. Outcomes reflect this weak actor's future
play, not perfect-play action values.

More concrete findings are a substantial learned pass prior, intermittent failure
to instantiate the large capture at the root, and a critic that already ranks
its immediate child above pass. This justifies investigating proposal coverage
and visit allocation/sampling on fixed witnesses; it does not prove any particular
search change will improve playing strength. Another feature reset is not
supported by this experiment.

Continuation players differ from the source reference opponent, and budgets are
fixed at 32 rather than the campaign's randomized 32/256 schedule. Matching seeds
does not make later random choices identical across different branches. Four
pairs from one selected board cannot establish general strength or prevalence.

The [terminal-pass follow-up](2026-11-09-r70-terminal-pass-pilot.md) uses an
independent endgame position and an exact pass baseline, avoiding the two-noisy-
branches comparison here.
