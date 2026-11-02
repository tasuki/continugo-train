# Cold-start search and early passing

All previous generated `runs/` and `models/` contents, including `cold1`, were
explicitly deleted at the user's request on 2026-11-02. No checkpoint or backup
archive was retained for reuse. The next campaign starts from scratch.

## Exact terminal pass checks

Both neural and unguided MCTS examine legal pass continuations before leaf-value
estimates or progressive widening can obscure them:

- A **second consecutive pass** that ends the game with a win is chosen regardless
  of prior/value bias. This is an exact terminal outcome, not a heuristic.
- Passing that loses immediately, or lets the opponent win immediately by passing,
  is a proven loss. Prefer an available alternative, without claiming it must win.
- Being ahead does **not** prove a first pass wins: the opponent need not pass back.
  Tests explicitly reject that cooperative-opponent assumption.
- Draws, including opening-pass draws, remain legal. Incomplete sampled placement
  sets never prove that the whole position is lost.

Only terminal hypothetical games are scored through shared rules. There is no
unfinished-territory bonus, pass ban, eye filter, feature-schema change, network
resizing, or learning-rate change. Full-policy normalization and independent
coordinate exploration remain intact. Default widening remains 2.0.

## Experimental rollout leaf evaluator

Standalone self-play/arena support optional cold-start evaluations independent of
neural value predictions:

- `--rollouts N`: default zero retains neutral unguided / neural leaf values.
  Positive N uses N legal playouts per new leaf; a neural model supplies policy only.
- `--rollout-max-moves 512`: safety cap per hypothetical playout, never forced scoring.
- `--rollout-pass-probability 0.03125`: otherwise use the existing bounded
  full-domain legal coordinate proposals. Exhausting proposal attempts also passes;
  it does not claim there is no legal placement. Immediate winning passes are taken.

Rollouts do **not** use the reference bot's eye filter. Only natural terminal
outcomes supply observed values; capped continuations stay unknown, not draws.
Selection falls back to neutral lack of evidence when none complete. Censoring can
still bias completed-only estimates. SGF root comments retain the configuration and
`rollout_stats` work counters; these hypothetical games are not extra training rows.
Actual training targets remain real root visits and natural actual-game outcomes.

Rollouts are off by default and not exposed by the supervised loop/rating protocol.
They need stronger cost/strength evidence before adoption. `arena --unguided
--reference-random` can compare bootstrap MCTS against the unchanged direct reference
without loading ONNX. Those diagnostic matches use maximum visits; regular
self-play samples visits. Keep evaluation records out of replay.

## Findings motivating a fresh start

The original overnight pilot's gate had 94 empty-board, two-opening-pass draws
and six four-move losses; its initial network lost all 10 reference matches. Weak
values, pass-heavy supervision and shallow/wide search were not a sound bootstrap.
Narrowing widening alone did not repair the already pass-biased network.

A same-seed 100-game bootstrap comparison at 32 simulations/move and 40/40/20 sizes:

| Search | Mean moves | Completed | Opening-pass draws |
|---|---:|---:|---:|
| Original neutral leaves | 23.03 | 100 | 0 |
| Neutral leaves + exact terminal checks | 127.12 | 100 | 1 |

A fresh small network trained on the corrected corpus preferred placements in
all ten opening-search probes. Its reference batch finished with 1 win, 7 losses,
no draws and two size-19 games censored at 512 moves. Validation value MSE remained
around 1: neither strong play nor trustworthy neural values were established.

In a tiny maximum-visit size-9 reference comparison, exact checks alone and two
rollouts/leaf each scored 2 wins / 8 losses, while rollouts cost roughly 8 times
more. This does not prove equal strength; it did not justify enabling rollouts.
Longer games or lower loss alone are not evidence of improvement.

The fixes and regression tests remain; the experiment artifacts do not. Continue
under the [manual-approval policy](training-policy.md) and use the [v5 loop](loop.md)
for future experiments, not old run-specific scripts or automatic-promotion loops.
