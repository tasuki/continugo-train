# Candidate acceptance matches

**Current operating policy:** [every promotion requires explicit user approval](training-policy.md),
and none are authorized until convincingly much stronger than the fixed reference.
A favorable gate verdict is evidence, not approval. Experimental self-play actors
may advance without becoming accepted. This separation is not implemented in the
supervisor yet; the automatic gate/promotion mechanics below describe existing
code, not the newly agreed publication policy.

The loop gates a trained/exported candidate against the **last accepted model**.
It does not use training or validation loss for promotion. This follows KataGo's
optional [gatekeeper](https://github.com/lightvector/KataGo/blob/master/cpp/command/gatekeeper.cpp)
pattern: head-to-head win points, with a configurable required proportion (default
50%, candidate wins ties). A finite match is a regression guard, not statistical
proof that the network improved.

## Budget and board mix

Loop controls:

- `--gate-games 20`: even number of games, paired by candidate color. Zero explicitly
  disables gating. Defaults give ten pairs, split **40% size 9 / 40% size 13 / 20%
  size 19**: 8/8/4 games. Ten games give 4/4/2.
- `--gate-simulations 32`: same MCTS simulation budget for each competitor per move.
- `--gate-max-moves 512`: a safety cap, **not a forced-scoring rule**.
- `--acceptance-rate 0.5`: required win-point proportion; configurable in 0.5..1.
- `--sizes` and `--size-weights 9:2,13:2,19:1` share the generation/training policy.
  Per-size counts/outcomes are always reported. No separate size-19 acceptance
  requirement overrides its intended 20% contribution.

Pairs share board size and RNG seed and swap candidate/incumbent colors. They start
from the empty authoritative board; randomized opening positions are not implemented.
The native `arena` binary selects maximum root visits with random tie breaks, rather
than training's temperature-one move sampling. Both use the existing PUCT,
progressive widening, independent exploration, exact legality and full-domain
policy normalization. Each network evaluates its own search; neither uses a
handcrafted value function or the random bot's eye filter.

Matching is currently sequential within one native process, with two frozen ONNX
evaluators, bounded feature pools and context caches. It does not yet batch across
independent evaluation games. Equal simulations are not necessarily equal wall time
when comparing different architectures or board sizes.

## Outcomes and decisions

- Natural win: **1** point; loss: **0**; natural draw: **½**.
- A draw is the shared scorer's actual `RE[0]`, not a small rounded score margin.
  It emits **WARNING DRAW**, with the SGF retained for inspection.
- A capped game has **no RE/outcome** and emits **WARNING TRUNCATED**. It is never
  converted to a draw, never force-scored and never used as a training label.

Let `P` be candidate points from completed games, `T` the number of truncated games,
and `N` the planned match count. The possible final match score is bounded by
`P/N .. (P+T)/N`:

- Accept only if the **lower** bound reaches the configured threshold.
- Reject if even the **upper** bound cannot reach it.
- Otherwise mark the test **inconclusive** and retain the incumbent.

Thus missing outcomes cannot manufacture acceptance or rejection. A test may still
be decided when it is mathematically guaranteed regardless of the unknown games,
similar to KataGo's mathematically-decided early-stop logic. This implementation
still plays the full planned match rather than stopping early.

The report includes wins/losses/draws/truncations, per-size breakdown, completed-only
score, planned-score bounds, threshold and decision. With no completed games, the
completed-only score is `None`, not 50%. Reports and warnings live in `loop.log`,
`loop.pt`, and native arena logs. All SGFs have standard `SZ`, `PB`/`PW`, and RE only
when naturally scored; model/search/seed metadata and actual visits are comments.

## Training is not rolled back

Accepted model and latest training checkpoint are separate pointers. Rejected or
inconclusive candidates keep their weights and optimizer progress for the next
training phase. Self-play remains on the accepted model until a later candidate
passes. Every candidate checkpoint/replay corpus is also backed up in the durable
[model store](model-store.md), not just the accepted ones. Gate SGFs are excluded
from the training replay selection.

## Manual matches

```sh
mise run arena -- --candidate PATH/TO/CANDIDATE --incumbent PATH/TO/ACCEPTED --games 20 --sizes 9,13,19 --size-weights 9:2,13:2,19:1 --simulations 32 --max-moves 512 --output runs/arena-001
```

Use a fresh output directory, or omit `--output` in this launcher for
`runs/YYYYMMDD-HHMMSS-arena` here automatically (UTC). Explicit paths are literal;
add `--device cpu` for CPU inference. Inputs can be
plain ONNX bundles or durable accepted bundles containing the two ONNX graphs.
`--game-offset` exists for the supervisor's missing-game recovery; even global
ordinals put the candidate on black. Each pair advances the seed once.
The standalone binary writes records and raw outcome totals; it does not move
models or update any accepted pointer by itself.

## Baseline-anchored static Glicko

The reference is the existing **direct random-but-not-eye-filling bot**, versioned
as `random-no-eye-fill-v1`, fixed at **100 with zero RD**. Its turns use the same
random policy and narrow eye filter as random diagnostics, not MCTS. Only neural
turns have search-policy targets. Both colors are exercised; reference matches
use maximum-visit neural play and the same `--gate-simulations` budget per move.

The loop plays `--reference-games 10` (4/4/2 by board size) for the initial accepted
network and each newly accepted candidate. `--reference-max-moves 2048` allows
longer natural completion than the default gate cap; zero reference games disables
calibration, not gating. These can be expensive: use zero explicitly for plumbing
runs. Reference results **do not affect promotion**. Gate matches, including rejected
and inconclusive candidates, also contribute completed outcomes to ratings.
All evaluation records stay out of training replay and the training data counter.

`../continugo/backend/rating` supplies the original Glicko update with **no time
decay, inactivity inflation, or active volatility/process-noise inflation**.
The native `rating` adapter applies a completed batch simultaneously from pre-batch
opponent ratings. Nothing changes just because a bot gets older. Draws contribute
half-points; truncations and self-matches do not update skill. A new network has
a broad RD=350 prior centered on its calibrated training parent when available,
otherwise 100. That prior is **not** a measured strength: untested networks and
networks without completed evidence connecting them to the reference show
**unrated**, with blank rating/RD cells. RD is not a 95% confidence interval.

Pools separate different simulation/search settings, board mixes, move caps, and
native evaluation/rating executable fingerprints. The arena stamps its fingerprint
in each SGF root comment. A running loop freezes these fingerprints and refuses
to resume after an executable change; fork a new loop instead. Equal simulations
are not equal wall-clock compute across architectures. A rating is relative to
this reference/protocol, not a human rank or an external strength scale.

### Two CSVs, no database or rating JSON files

Inside the durable model store:

```text
ratings/
  networks.csv   all archived/evaluated networks, one row per network/pool
  matches.csv    one row per processed SGF, batch history and post-batch ratings
  games/*.sgf    authoritative immutable match evidence (store-owned hardlinks)
  .lock
```

`networks.csv` contains readable names, full IDs, parent IDs, progress counters,
`accepted` (1/0), acceptance dates/bundle paths, status, rating, deviation, `rating ± RD`,
and completed W/D/L/truncation counts. **Filter `accepted=1` for all accepted networks**;
rejected candidates remain in the same table. The reference has `accepted=0`.
Numbers may be negative for bots weaker than the 100 reference.

`matches.csv` records raw SGF hashes, players, actual results (blank means unknown),
board size, protocol columns, audit dates, batch IDs and each player's **post-batch**
rating/RD/anchor state. Snapshots repeat across a batch's rows, not per-game updates.
It indexes evidence/history, not moves or training targets: SGFs remain the game
records. One `(pool, SGF hash)` is counted at most once, even across overlapping
imports or a crash after the CSV commit but before supervisor acknowledgement.

A writer lock serializes updates. CSVs are fsynced and atomically replaced, not
appended in place. Metadata is saved before committing match rows. On restart or
refresh, current numeric ratings/counts are rebuilt from **committed match rows**,
repairing an ahead/behind network-table cache without replaying the math or
inflating uncertainty. Keep both CSVs; a missing match ledger is an error,
not permission to reset ratings. Back up the whole store and quiesce writers while
copying the two CSVs. Do not edit rating snapshots or processed hashes by hand.

```sh
mise run ratings
mise run ratings -- --accepted-only
# Additional arena SGFs must match the chosen existing pool's full protocol:
mise run ratings -- --pool FULL_POOL_ID --ingest runs/arena-001 --checkpoints PATH/TO/CANDIDATE/checkpoint.pt
# Standalone neural-vs-reference matches (ingest separately to update the table):
mise run arena -- --candidate PATH/TO/NETWORK --reference-random --games 10 --simulations 32 --max-moves 2048 --output runs/reference-001
```

The stateless Rust adapter uses JSON **only as stdin/stdout IPC**, not persistent
rating storage or SGF sidecars. Python validates/replays SGFs through Rust, pins
players/search/budgets, retains store-owned evidence, and manages the two CSVs.
Completed-only ratings can be biased if truncation censors particular boards or
outcomes. Inspect per-size reports and truncation counts; a small RD does not
account for that bias or prove improvement. The budgets still need calibration.

## Limits

Twenty games are cheap enough for initial wiring but noisy. The threshold, visit
budget, number of games, model quality and sample diversity need calibration before
scaling or trusting fine strength differences. There is no SPRT/confidence-bound
promotion test, resignation, opening suite or per-size minimum-strength constraint
in this first gate. The conservative bounds above account for **missing outcomes**,
not statistical uncertainty. Diagnostics/parity tests still cover all board sizes.
