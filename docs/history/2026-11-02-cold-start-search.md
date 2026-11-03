# 2026-11-02 cold-start search experiments

Historical diagnostic snapshot following the [overnight pilot](2026-11-01-overnight-pilot.md).
The generated artifacts, including `cold1`, were deleted at the user's request on
2026-11-02; no checkpoint archive was retained for reuse. The surviving
implementation is documented in [current search](../current/inference.md#search-semantics).

## Failure investigated

The overnight pilot's gate had 94 empty-board, two-opening-pass draws and six
four-move losses; its initial network lost all ten reference matches. Weak values,
pass-heavy supervision and shallow/wide search were implicated in the bootstrap
failure. Narrowing widening alone did not repair the already pass-biased network.
These observations did not isolate every cause.

## Exact terminal-pass checks

The search change probed terminal pass continuations before leaf-value estimates
or progressive widening could hide them. It prioritized a winning second pass and
avoided a pass proven to lose immediately or to an opponent's immediate winning
pass when an alternative was available. It did not assume that an opponent would
cooperatively answer a first pass, ban opening draws, score unfinished positions,
change features, or add an eye filter. Widening remained 2.0.

A same-seed 100-game bootstrap comparison used 32 simulations/move and a 40/40/20
size-9/13/19 mix:

| Search | Mean moves | Completed | Opening-pass draws |
|---|---:|---:|---:|
| Original neutral leaves | 23.03 | 100 | 0 |
| Neutral leaves + exact terminal checks | 127.12 | 100 | 1 |

A fresh small network trained on the corrected corpus preferred placements in
all ten opening-search probes. Its reference batch finished with **1 win / 7
losses / 0 draws / 2 size-19 truncations** at 512 moves. Validation value MSE
remained around one. Neither strong play nor trustworthy values were established.

## Optional rollout comparison

An experimental evaluator used legal playouts instead of neural leaf values,
without the reference bot's eye filter. Only naturally completed playouts supplied
values; capped continuations stayed unknown. SGF comments recorded configuration
and work counters, not extra training rows.

In a tiny maximum-visit size-9 reference comparison, exact checks alone and two
rollouts/leaf each scored **2 wins / 8 losses**, while rollouts cost roughly
**eight times more**. This did not prove equal strength and did not justify making
rollouts the default. The standalone option was retained, disabled by default;
it was not added to the supervised loop/rating protocol.

## Decisions at that snapshot

The old generated artifacts were discarded and a fresh campaign was proposed.
Longer games and lower loss were not treated as evidence of improvement. The
operating intent was initially a resumable week, with a first-day review and a
possible second-week extension; short diagnostic timeouts were not to become an
implicit learning deadline. Network-size changes were to remain manual, with
40/40/20 board coverage and ongoing fresh self-play rather than prolonged fitting
of one bootstrap corpus.

Accepted publication was separated from experimental actor advancement under
explicit hash-pinned user approval. Automatic gating or a daily promotion limit
alone was rejected as authorization. No numeric definition of "convincingly much
stronger than the reference" was chosen. Training/optimizer progress was to survive
failed or inconclusive evaluations rather than roll back with the actor.

Those dated campaign decisions are context, not instructions to reset future runs
or a standing launch plan. Use the [current approval procedure](../current/model-store.md#approval-operation),
[loop](../current/loop.md), and [open work](../todo.md) for contracts and plans.
