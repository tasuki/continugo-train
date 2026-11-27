# Game contexts

Defaults for **new** runs; resumed runs retain recorded settings. Python's
[`SelfplayConfig`](../../python/continugo_train/selfplay.py) owns common playing
defaults, shared by the loop and direct self-play commands. `EvaluationConfig`
owns evaluation budgets, shared by the evaluator and arena command. Native workers require these values. Use the `mise` launchers for defaulted
commands. Rated komi is Rust-owned: ordinary Python callers omit it.

| Context | Simulations per search turn | Played move | Komi | Move cap |
|---|---|---|---|---|
| Training self-play (loop or direct command) | 100 on 75% of moves, 500 on 25% | Temperature-sampled visits | Gaussian mean 7, σ 2, half-point rounding, clipped [0,14] | 512 |
| Bootstrap (untrained MCTS with rollout leaves) | 32, fixed | Temperature-sampled visits | Same randomized komi | 512 |
| Evaluation (evaluator or arena command) | 128, fixed | Maximum visits | Fixed 7 | 512 for both neural and random-reference matches |
| Cross-lineage comparison | Inherits lineage A's evaluator | Maximum visits | Fixed 0 by default; configurable | Inherits lineage A's evaluator |
| Interactive browser play | 128; per-request override 1–1024 | Maximum visits | From game/SGF | No worker-enforced cap |

Launching a command directly does not create another playing context. All modes
share widening 3 and independent exploration 0.5. The evaluator inherits its source
loop's search settings, then freezes them. Other common defaults are
`cpuct=1.5`, widening exponent 0.5, 128 proposal attempts and equal score/outcome
utility weighting. Maximum-visits selection breaks ties randomly.

Self-play temperatures decay from 0.75 toward 0.15 with half-life 19 plies on size
19, scaled by board width. Only expensive training searches receive policy
supervision; naturally completed cheap positions retain outcome/score/ownership
labels. Bootstrap explicitly uses a cheap fixed budget and completed random
rollouts for leaf evaluation; it has no learned policy to improve.

The size list is 9/13/19, but current weights select **size 9 only**. Rated games
use same-seed, same-size, color-swapped pairs. Direct random play never searches
and avoids own-eye filling. Move caps truncate rather than force scores;
truncations do not update ratings. Ratings never select the training actor or
publish an accepted model. See [evaluation](evaluation.md), [loop](loop.md) and
[constants](constants.md) for details.
