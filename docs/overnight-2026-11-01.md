# Overnight pilot — historical results

**Artifacts were deleted at the user's request on 2026-11-02**, together with all
other old `runs/` and model stores. No checkpoint/history archive was retained.
This page is a measurement summary, not instructions for resuming an existing job.

The pilot ran from 2026-11-01 22:39:24 UTC to 2026-11-02 01:55:02 UTC:
**3h 15m 38s**, completing its work limit before a six-hour deadline.

- Existing 32-channel/4-block network, CUDA, SGD lr 0.01 / momentum 0.9.
- 40/40/20 size-9/13/19 coverage, real MCTS visits, natural outcomes only.
- Bootstrap: 100 unguided games, 32 simulations/move, all naturally completed;
  2,371 eligible rows, 80 training / 20 validation games.
- Initial training: three epochs, 900 optimized rows each; 11 minutes total.
- Neural self-play: 300 games / 6,934 moves in 2h 18m 41s, all naturally completed,
  including 55 draws. Three further epochs optimized 2,000 rows each.
- Reference: initial network lost all 10 games to the random-but-not-eye-filling bot.
- Gate: **0 wins / 6 losses / 94 draws**, no truncations; candidate rejected at
  the provisional 60% planned-score requirement. No candidate promotion occurred.

Rust replay of all 510 SGFs confirmed every draw was an empty-board game consisting
of Black pass, White pass. All six gate losses had four moves; an inspected example
was Black pass, White placement, Black pass, White pass. There were no observed
draws after stone placement.

The pilot demonstrated plumbing/recovery, not useful strength. See
[cold-start search](cold-start.md) for the fixes and follow-up measurements. The
[manual-approval policy](training-policy.md) now supersedes this pilot's automatic
promotion mechanics. Future learning starts from scratch with fresh games; use
the current [loop](loop.md) and [model-store](model-store.md) instructions.
