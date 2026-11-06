# Developed-board pass statistics, cg2 versus cg4 — 2026-11-06

Read-only user-requested audit; no active campaign changes. Artifacts:
`runs/20261106-193548-midgame-passes/` (input manifest, Rust source, per-position
records, per-round CSV/JSON, summaries, closing-pass counts and examples).

## Scope and definitions

Snapshot through completed round **14**: 50 games/round from each campaign,
**700 old + 700 new games**. The new round 15 was active and excluded. Old records
are read from recoverable trash, not restored into the new run or ratings.

The diagnostic uses the shared SGF decoder and legal game replay, checks player
turns/natural completion, and verifies every scored RE against authoritative
territory. Old numeric-action and new coordinate policy comments are decoded in
this diagnostic only; no training data or records are rewritten.

Main operational definition: at least **20 stones on the board**, not either of
the final two ending passes, and **at least five subsequent placements** in the
actual game. Apply the same board/future-play condition to turns in the rate's
denominator. This is a measurable developed-board/followed-by-play category,
not an oracle proving every pass strategically bad. Thresholds 10 and 30 are
also reported. Raw board lead is diagnostic only, not a nonterminal value label.

## Most recent five completed rounds (10–14), 250 games each

| Statistic, threshold 20 | Old cg2 | Current cg4 |
|---|---:|---:|
| Qualifying developed-board turns | 16,719 | 4,905 |
| Nonterminal passes followed by ≥5 placements | 2,143 | **417** |
| Passes / qualifying turns | 12.82% | **8.50%** |
| Games containing these passes | 190/250 | **78/250** |
| Pass is most-visited action | 93.2% | **83.5%** |
| Mean pass visit fraction at those events | 49.8% | **39.5%** |
| Verified legal positive-visit placement alternative | 2,128 | **417/417** |
| Currently behind on raw board area | 0 | 39 |

Every one of these 417 current pass events is followed by a later placement by
that same player. They are not simply positions with no sampled legal moves.
114 occur at **256 simulations**; pass is the most-visited action in **87.7%** of
those, with mean pass mass **50.4%**. So neither visit sampling nor only the cheap
32-simulation moves explain the symptom. In 230/417 events placements have a
majority of total visits, spread over separate actions; argmax can actually favor
pass more often than proportional sampling. Aggregating the largest coarse cell
beats pass in 127/417 events (diagnostic only, not an alternative selection rule).

Sensitivity: threshold 10 yields **516/6,198 = 8.33%** in 100/250 current games;
threshold 30 yields **363/3,817 = 9.51%** in 69/250. Round-by-round counts are in
`rounds.csv`, not a claim of monotonically increasing pass frequency.

## Strong current example

Round 14, game 43, seed 10,100,692, **move 50** (zero-based ply 49):

- **35 stones** already on the board.
- Pass receives **224/256 visits (87.5%)**.
- **32** positive-visit legal placement alternatives, each visited once.
- The game subsequently has **52 placements**, 22 by the player who passed.
- The passer is currently ahead by normalized board margin ~0.250; final RE is
  W+14.0. These facts do not establish which unplayed continuation is optimal.

Path:
`runs/20261106-142353-scratch-score/loop/iteration-0014/selfplay/attempt-0001/game-000043-size9-seed10100692.sgf`.
Other recent examples include move 34 with 26 stones and 82 later placements,
and move 30 with 24 stones while behind and 42 later placements. `examples.json`
retains their policies, legal alternatives and outcomes.

## Important correction: frequency versus premature termination

The current run does **not** have a higher developed-board pass rate than the
old run under this definition. Old games were much longer (recent median **92**
plies versus **19.5**), with many repeated passes while ahead. Claiming simply
"the new run passes more" would be unsupported.

The sharper regression is **ending games much earlier, including conceding while
legal continuations remain**:

- Nonempty games ending within 20 plies: **82/250 current versus 0/250 old**.
  Empty-board draws are counted separately: 44 current / 56 old.
- With at least ten stones, **120 current games** start their final pass pair
  with a player behind on board area who still has a verified positive-visit
  legal placement alternative. The opponent then ends the game. Old: **zero**.
- Another **14 current games** have a losing second pass with a positive-visit
  legal alternative. Old: **zero**.
- At threshold 20, those closing figures are **73** and **9** current versus zero
  old. The old zeros are expected consequences of its hard-coded losing-pass
  filter, not evidence that it learned to reject those passes.

A losing pass can be optimal under a score-sensitive objective if continuation
is truly worse; these records alone cannot prove all such passes mistakes. But
premature termination plus strongly pass-preferring search and weak measured
midgame critics is a concrete pattern deserving controlled investigation. Do not
restore the old incorrect filters or ban passes merely to change these counts.
