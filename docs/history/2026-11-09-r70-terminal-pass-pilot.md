# Round-70 exact terminal-pass baseline — 2026-11-09

Follow-up requested after the [first branch pilot](2026-11-09-r70-branch-pilot.md)
was too noisy. Same frozen **round-70** network, independent late-game position,
one exact terminal-pass score and four forced-placement continuations. No
production code, campaign setting, replay, weight, optimizer or publication was
changed. A separate process shared GPU time with the ongoing campaign.

Artifacts: `scratch/20261109-185203-r70-terminal-pass-pilot/`, including pinned
model copies/hashes, native/Python diagnostic source, selection evidence, five
validated SGFs, full traces, `summary.json`, `README.md` and **`comparison.png`**.

## Model, position and selection

- Model `cg5-b4c32-s570000-d177833`, checkpoint SHA-256
  `b3fd8e849cd18b5074ed3441bb8f53c7f89cd3ff69a130d91992d0907638c305`.
- Round-70 reference source
  `game-001399-size9-seed12100699-candidate-black.sgf`, **ply 84 / move 85**.
  Source actor hash is verified against the model. Black to move, **48 stones**;
  White just passed. Black actually passed, ending the source game.
- Alternative **B[rtrG]**, coordinate **(903, 916)**, legally captures **eight
  White stones**. Immediate current-board margin gain is 34.71 stone-area units;
  this is a selection diagnostic, not a final-score label.
- Before any new continuation results, choose the largest immediate margin gain
  among recorded legal non-eye captures at actual terminal neural passes, with
  at least 30 stones, excluding the first pilot's source game. Fourteen eligible
  terminal-pass positions. Ties use path, ply, action ID. This is a selected
  witness, not an unbiased prevalence estimate.

## Exact baseline and continuations

Applying pass immediately produces authoritative **Black +67.4691638**. The
baseline SGF naturally terminates with zero further moves. It uses neither a
neural score nor an assumption about an opponent's future reply.

Force the capture instead, then let round 70 play both sides at 32 simulations
per move, temperature-one visit sampling, default campaign search parameters and
50/50 outcome/score utility. No eye filter, pass ban or guidance; 256 subsequent
moves maximum. All four naturally completed:

| Seed | Final Black margin | Gain over exact pass | Subsequent moves |
|---|---:|---:|---:|
| 7101 | +83.52 | +16.05 | 17 |
| 7102 | +88.56 | +21.09 | 12 |
| 7103 | +89.71 | +22.24 | 16 |
| 7104 | +82.57 | +15.10 | 12 |

Mean final margin **+86.09**, mean gain **+18.62**. The gain survives **4/4**
continuations. Native continuation time totals **44.9 seconds**, plus **10.4
seconds** of root-search probes and process/model-loading overhead. All use CUDA,
matching the campaign. No numeric validation was relaxed.

An independent native replay verifies all five SGFs: source prefix, forced move,
full legality, natural completion, result, lengths and Black-relative score.
Model-file hashes are unchanged. No unfinished position is treated as a final
result, and Python does not implement SGF parsing or game rules.

## Ordinary search and critic

Fresh seeded searches at the same position, without forcing the first move:

| Seed | Pass visits | Total visits to eight-stone captures | Number of such root children | Largest single capture visits |
|---|---:|---:|---:|---:|
| 7101 | 6/32 | 8/32 | 4 | 2 |
| 7102 | 6/32 | 6/32 | 4 | 3 |
| 7103 | 5/32 | 10/32 | 4 | 3 |
| 7104 | 5/32 | 11/32 | 5 | 3 |

**Pass is the maximum-visit action in all four searches**, even though capture
alternatives are present every time. Temperature-one actual play samples pass
with probability 5/32–6/32 in these roots; maximum-visit selection and sampled
play should not be confused. In the original source, pass had 6/32 visits and
the selected exact capture had 3/32.

Eager-CPU inference with frozen native features cross-checks CUDA outputs:

- Root coarse pass probability **24.47%**.
- Exact terminal-pass normalized score **+0.64118**, outcome +1,
  utility **0.82059**.
- Immediate capture child, converted back to Black's perspective: predicted
  score **+0.93483**, outcome +1, utility **0.96741**.

The critic therefore already rates the immediate capture child above the exact
terminal pass under the actual objective. This is not a case where the head
cannot recognize the capture at all.

## Conclusion and limits

This is substantially clearer than the first pilot: an exact pass baseline,
four short continuations all preserving a 15–22 unit gain, and ordinary search
still assigning the largest individual visit count to pass. Capture visits are
spread among multiple near-equivalent coordinates while pass has a single
bucket. Low-budget root allocation/selection is a justified next investigation,
not another feature reset.

This does not establish a universal root cause, a perfect-play advantage, rare
reversal risk or the strength of any proposed fix. Four continuations from one
selected board are not a population/strength test. Immediate child estimates
are not instrumented backed-up root-child Q; deeper evaluation effects, pass
prior influence, PUCT exploration and visit fragmentation remain unisolated.
No claim is made that this explains early-game passes or all eye filling.
Continuation budgets are fixed at 32, unlike the campaign's randomized 32/256
self-play schedule. The full original source history is preserved, but new seeds
do not reproduce the original game RNG trajectory.

The subsequent [spatial-search design discussion](2026-11-09-spatial-search-and-pass-fragmentation.md)
records how coarse/fine policy, sampled actions and progressive widening became
combined, why the result raises a structural concern, and possible remedies.
