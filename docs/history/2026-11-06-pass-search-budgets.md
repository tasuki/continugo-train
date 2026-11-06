# Pass behavior at 32 versus 256 simulations — 2026-11-06

User asked whether more search would help premature termination and requested
an actual split of current midgame passes by randomized search budget. Read-only
audit: **no active campaign changes**. Artifacts and reproducible diagnostic
source are in `runs/20261106-200313-playout-audit/`.

## Observed games

Snapshot through completed round **15**; round 16 was active at collection.
750 complete cg4 games are replayed with the shared SGF decoder/rules and
verified authoritative RE. The main comparison is the most recent five completed
rounds **11–15**, 250 games. Every recorded cap is checked as 32 or 256, with
`train_policy` enabled exactly at 256. Assignment occurs randomly before search
and uses the same frozen actor within each round.

Rates use turns at each budget as denominators, not raw pass counts (roughly
three quarters of turns are cheap). Mean pass visit mass is also recorded.

### Continuing developed-board category

At least 20 stones, excluding the final two ending passes and requiring at least
five later placements, matching the prior audit's definition. Conditioning on
future play can itself select away terminating passes, so this category is
**descriptive**, not a complete causal estimate of game-ending behavior.

| Condition | 32 simulations | 256 simulations |
|---|---:|---:|
| All qualifying turns | **348/4,311 = 8.1%** pass | **137/1,437 = 9.5%** pass |
| Currently ahead on board area | 309/1,757 = **17.6%** | 129/571 = **22.6%** |
| Currently behind on board area | 39/2,554 = **1.5%** | 8/866 = **0.9%** |
| Mean pass visit mass, all qualifying turns | 10.1% | 11.4% |
| Most-visited action is pass, all qualifying turns | 27.7% | 20.2% |

The observed overall +1.46 percentage-point difference has a game-clustered
bootstrap 95% interval of approximately **-0.58 to +3.56 pp**, so do not claim an
overall significant increase or improvement. The ahead difference is +5.0 pp
(interval +0.27 to +9.99 pp); behind -0.60 pp (interval -1.42 to +0.28 pp).
These are exploratory intervals across several descriptive subsets, not a
pre-registered independent-trial test.

### Including closing passes (pre-move board condition only)

All turns with ≥20 stones, without filtering on future play:

| Condition | 32 simulations | 256 simulations |
|---|---:|---:|
| All developed turns | 573/4,957 = **11.6%** | 228/1,658 = **13.8%** |
| Ahead | 430/2,028 = **21.2%** | 205/681 = **30.1%** |
| Behind | 143/2,929 = **4.9%** | 23/977 = **2.4%** |

Ahead/behind refers to static current territory, not proof of eventual optimal
win/loss. Includes legitimate winning game-ending passes; a higher pass rate is
not automatically worse play. On developed replies to an opponent pass, the
ending-pass rate is 14.7% at 32 and 23.4% at 256, which often reflects correctly
taking an available win rather than an error.

A more specifically undesirable-looking event: after an opponent pass, ≥10 stones,
currently behind, and a positive-visit legal placement alternative exists. The
player accepts the exact losing second pass **11/454 times (2.4%) at 32**, versus
**0/149 at 256**. Zero observed is not a guarantee. More search does appear useful
for this category. It does not establish that every continuation beats the
known loss under the blended score-sensitive objective.

## Matched-position searches

Frozen **round-14 actor** (the model that generated round 15), hash
`c7d3ad83ca76cadaea9b2d8acb48e56e10e94c9a9cbc136f95afebc5244ee1bb`.
Eleven failure-conditioned real positions from round 15: three continuing passes
while ahead, three while behind, three behind passes starting the ending pair,
and two losing second-pass events. Prefer distinct games within each category;
this is not an unbiased position sample.

**66 searches:** each position at both budgets, paired seeds 73/74/75, CPU ONNX,
one inference/feature thread and fresh context cache per query. No active GPU
model, training target or network weight is changed. Search uses the campaign's
normal 50/50 utility and no experimental leaf bounds or action filters. Budget
also changes root proposal-pool size, so this is the actual 32/256 configuration
comparison rather than an identical tree extended by 224 simulations. CPU and
new per-position seeds do not reproduce original CUDA/self-play RNG trajectories.

| Selected category | Positions | Mean pass visit mass, 32→256 | Max-visit pass choices, 32→256 |
|---|---:|---:|---:|
| Continuing, ahead | 3 | **26.7%→43.2%** | 8/9→6/9 |
| Continuing, behind | 3 | 13.2%→10.3% | 3/9→3/9 |
| Closing first pass, behind | 3 | 13.9%→13.5% | 4/9→3/9 |
| Losing second pass | 2 | 3.1%→2.5% | 0/6→0/6 |

Individual ahead positions go 39.6%→68.0%, 16.7%→6.5%, and 24.0%→55.1%.
Thus more search finds alternatives in some positions but **sharpens pass
preference in others**. The first portion timed out after 56 completed queries;
all complete JSON-line results were retained, verified against the actor hash,
and the remaining ten queries resumed without repeating completed pairs.
`paired-search.json` contains all 66 results; `cases.sha256` verifies unchanged
case ordering. Repeated seeds are not independent board positions.

## Practical interpretation

More visits are **not a blanket cure** for excessive passing or premature endings.
They reduce some passes while behind and some exact losing concessions. They
can also concentrate more visit mass on pass when the critic/prior already
prefers it. With current temperature-one visit sampling, that can increase actual
pass probability, even when the fraction of positions whose single most-visited
action is pass decreases.

Neither automatically raising every move to 256 nor globally switching to max
visits is established as a cure. More-search full-game trials should use a frozen
actor, paired seeds/colors, and measure outcomes/margins and premature endings,
not pass counts alone. The prior fixed-data critic audit remains relevant: a
controlled critic-only fit/calibration test followed by rerunning these exact
positions can isolate whether better estimates reduce the problematic preferences.
Do not introduce a pass ban or restore the old incorrect pass filters merely to
improve the statistic. No new training or search setting has been applied.
