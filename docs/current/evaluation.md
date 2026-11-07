# Evaluation and ratings

Match results measure progress; accepted publication requires the separate
[approval operation](model-store.md#approval-operation).

Head-to-head matches compare a trained candidate against its **frozen experimental
self-play parent**, independently of the official accepted network. This adapts
KataGo's optional [gatekeeper](https://github.com/lightvector/KataGo/blob/master/cpp/command/gatekeeper.cpp)
pattern: configurable win-point proportion (default 50%, candidate wins ties).
Verdicts are `passed`, `failed`, `inconclusive` or `disabled`, not accepted models.
Actor selection and scheduling are defined in [loop](loop.md); a failed match
never rewinds training. Evaluation SGFs stay outside training replay.

New v2 evaluation protocols record `score_weight` (default 0.5, a linear score fraction) alongside the
other search settings. Existing v1 rating ledgers are not migrated; use a fresh
model store for the new score/ownership campaign. Historical models require their
historical runtime. See [auxiliary supervision](auxiliary.md).

## Budget and board mix

Loop controls:

- `--gate-games 20`: even number of games, paired by candidate color. Zero explicitly
  disables the experimental actor match. Defaults give ten pairs, split **40% size 9 / 40% size 13 / 20%
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
progressive widening, independent exploration, exact legality and pass-preserving
selective policy normalization. Each network evaluates its own search; neither
uses a handcrafted value function or the random bot's eye filter.

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

- The match **passes** only if the lower bound reaches the configured threshold.
- It **fails** if even the upper bound cannot reach it.
- Otherwise the match is **inconclusive**. Optional gated actor selection holds
  its actor on failure/inconclusiveness; default latest-actor selection may advance.
- No result authorizes an accepted publication.

Thus missing outcomes cannot manufacture a decisive match verdict. A test may still
be decided when it is mathematically guaranteed regardless of the unknown games,
similar to KataGo's mathematically-decided early-stop logic. This implementation
still plays the full planned match rather than stopping early.

The report includes wins/losses/draws/truncations, per-size breakdown, completed-only
score, planned-score bounds, threshold and decision. With no completed games, the
completed-only score is `None`, not 50%. Reports and warnings live in `loop.log`,
`loop.pt`, and native arena logs. All SGFs have standard `SZ`, `PB`/`PW`, and RE only
when naturally scored; model/search/seed metadata and actual visits are comments.

## Manual matches

```sh
mise run arena -- --candidate PATH/TO/CANDIDATE --incumbent PATH/TO/ACCEPTED --games 20 --sizes 9,13,19 --size-weights 9:2,13:2,19:1 --simulations 32 --max-moves 512 --output scratch/arena-001
```

Use a fresh output directory, or omit `--output` in this launcher for
`scratch/YYYYMMDD-HHMMSS-arena` here automatically (UTC). Explicit paths are literal;
add `--device cpu` for CPU inference. Inputs can be
run-owned ONNX exports or deliberately retained/accepted bundles containing both graphs.
`--game-offset` exists for the supervisor's missing-game recovery; even global
ordinals put the candidate on black. Each pair advances the seed once.
The standalone binary writes records and raw outcome totals; it does not move
models or update any accepted pointer by itself.

## Adaptive rated matching

Random is a calibration baseline, not a sufficient opponent pool after models
consistently beat it. **Prefer adaptive near-strength matching**, not a fixed
round-robin, when the purpose is rating precision:

```sh
mise run rate-models -- --loop PATH/TO/LOOP --latest 3 --target-rd 75 --min-pairs 20 --min-opponents 2 --min-opponent-pairs 4 --batch-games 4 --max-games 360 --seed 9400000 --output scratch/adaptive-ratings
mise run rate-models -- --output scratch/adaptive-ratings --resume
```

The newest three **available registered** neural checkpoints are the precision targets;
`--targets FULL_CHECKPOINT_ID...` instead pins exact available targets independently of
registration order. Run-owned exports, saved milestone snapshots and retained/accepted bundles
form the opponent pool (or restrict it with `--pool-latest N`). Deleted unretained
models remain in rating history but are not available opponents. No random opponents are
scheduled. After every four games (two same-seed color pairs), ingest outcomes
and reread current ratings. Choose the matchup with greatest expected reduction
in unfinished targets' rating variance, using Glicko's per-game Fisher information.
Near-equal strengths are most informative; opponent uncertainty and observed
truncation rates reduce the expected information. Until opponent diversity is
covered, substantially discount already-covered matchups. Selection, pre-match
ratings and expected score are saved **before** launching the batch. Unexpected
results therefore change subsequent opponents, not a precomputed bracket.

Each target must meet **all** of:

- Global Glicko RD ≤ 75, anchored to the reference through actual evidence.
- At least **20 complete, distinct color-swapped seed pairs** (40 games) of
  compatible head-to-head evidence; prior ledger evidence counts too.
- At least **two neural opponents with four complete pairs each**.

This is moderate confidence with paired evidence and opponent diversity, not a
high-precision ranking. Request a smaller `--target-rd` explicitly for a tighter study. **RD is model-based uncertainty, not a guaranteed 95% confidence
interval**; paired-game correlation, model misspecification and adaptive sampling
limit such interpretations. Close ratings may remain statistically indistinguishable;
meeting the policy does not force a ranking or imply a particular win probability.
The default hard cap is **360 additional games**, counting truncations against the budget.
A budget stop explicitly says the target was **not met**. Truncations, incomplete
pairs, repeated seeds and repeated colors cannot satisfy the paired-evidence gate.
Paired evidence is filtered to the source loop's exact evaluation protocol.

`progress.json` records each target's rating/RD, complete pair count, opponent
coverage and stop reason. Graph/checkpoint snapshots, source-loop settings,
policy and fresh seed stream are frozen in the evaluation control file. Resume
finishes its pending batch before choosing another opponent and does not retrain,
republish models, change actors, or double-count committed games. Training stays
separate. The loop's ordinary 20-game parent match remains a progress diagnostic,
not a claim of meeting this stronger precision policy. New campaigns enable the
loop's [automatic adaptive phase](loop.md#automatic-adaptive-rating-precision) by
default, applying the stronger policy to scheduled candidates without manual runs.
Older saved loops retain their frozen settings and can be explicitly upgraded.

## Saved milestones and fresh background ratings

New campaigns automatically save, rate and plot each tenth round. Milestone
snapshots use `models/<campaign>-rNNNN/`; the graph assigns a distinct colour to
each campaign and connects its rated rounds in order, without joining different
campaigns. Legend entries include the network-family prefixes used in game
filenames, e.g. `cg4 · 20261106-142353-scratch-score`. A family prefix can be shared
by several campaigns; the full campaign name still distinguishes the curves.
The x-axis remains cumulative unique training games; error bars are
± rating deviation, not 95% confidence intervals.

Backfill snapshots with `mise run save-models -- --runs runs`, then evaluate them
with `mise run rate-saved -- --output scratch/YYYYMMDD-HHMMSS-ratings`. The saved-model
runner calibrates all models with ten reference games each, then adaptively
chooses similarly rated neural opponents. It is **not** an all-pairs schedule.
The default extra budget is 360 games per compatible cohort; unmet RD/evidence
requirements are explicitly reported. Use the same output with `--resume` after
interruption; native fingerprints, seeds, completed SGFs and ledger commits are
verified, and finished games are not replayed.

Historical model schemas use compatible frozen evaluation runtimes and are not
matched against incompatible schemas. The shared reference anchors them, but
cross-protocol ratings should still be compared cautiously. No historical ledger
is imported by this command. Resetting old ratings is a separate deliberate
operation; the current background job started from an empty ledger.

## Rated round-robin

An explicit fixed round-robin is still available for deliberate all-pairs studies;
it is **not** the precision-driven default:

```sh
mise run round-robin -- --loop PATH/TO/LOOP --latest 4 --games 20 --seed 9200000 --output scratch/rated-comparison
# Interrupted evaluation resumes missing games, without retraining or double rating:
mise run round-robin -- --output scratch/rated-comparison --resume
```

The source loop is read-only and supplies frozen binaries, board mix, search
budget, device and neural move cap. `--latest 4` selects the four newest available
registered neural checkpoints (not the four highest ratings); `--bundles PATH...`
instead chooses explicit retained/accepted recovery bundles. Each unordered pair gets
20 games, so four models schedule six matchups / 120 games. Both colors share each
pair's seed. Use a fresh seed for a genuinely new comparison.

Graphs/checkpoints are verified and snapshotted into the evaluation directory.
The existing arena's SGF validation, partial-game recovery and inherited process
lock are reused. Each completed matchup is ingested into the shared ratings ledger
and refreshes its plot; rerunning `--resume` does not duplicate evidence. Truncations
remain unknown outcomes. Neither training state, actor pointers nor accepted
publication is changed. Keep evaluation evidence out of training replay.

To retire an interrupted round-robin while preserving finished games without
launching the rest of its schedule:

```sh
mise run round-robin -- --output scratch/rated-comparison --resume --ingest-only
```

Stop its supervisor/child cleanly first. This validates/adopts partial-attempt
SGFs and ingests only those games; the remaining schedule is not played.

## Baseline-anchored static Glicko

The reference is the existing **direct random-but-not-eye-filling bot**, versioned
as `random-no-eye-fill-v1`, fixed at **100 with zero RD**. Its turns use the same
random policy and narrow eye filter as random diagnostics, not MCTS. Only neural
turns have search-policy targets. Both colors are exercised; reference matches
use maximum-visit neural play and the same `--gate-simulations` budget per move.

The loop plays `--reference-games 10` (4/4/2 by board size) for the initial
experimental actor and each candidate scheduled by `--evaluation-interval`,
including candidates failing the head-to-head match. `--reference-max-moves 2048` allows
longer natural completion than the default gate cap; zero reference games disables
calibration, not gating. These can be expensive: use zero explicitly for plumbing
runs. Actor matches, including failed and inconclusive candidates, also contribute
completed outcomes to ratings.
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

There is **one shared rating table**, with one rating per checkpoint and one fixed
random reference. Training runs, board mixes, search budgets and executable
versions do not create separate leaderboards. Each match retains its complete
evaluation settings and arena fingerprint for audit. A running loop still freezes
its executables and refuses to resume after a binary change; fork a new loop in
that case, using the same model store. Ratings summarize the recorded playing
conditions, not a human rank; equal simulations are not equal compute.

## Ratings storage

Inside the durable model store:

```text
ratings/
  networks.csv   scheduled/evaluated networks, exactly one row per network
  matches.csv    one row per processed SGF, batch history and post-batch ratings
  games/*.sgf    dated, readable immutable evidence (independent copies)
  inference/     hash-pinned run-owned model locators and cached training-game counts
  .lock
```

Only networks evaluated or explicitly scheduled for evaluation enter the registry.
Explicitly retaining a model alone does not register it. Interrupted/truncated
evaluations retain their entries for later rating; approved publications also
enter the registry. Loops do not make automatic recovery copies. Rating history,
plot metadata and match evidence survive deletion of unretained campaign models.

`networks.csv` contains readable names, full IDs, first-registration timestamps
(`registered_at`, UTC), parent IDs, progress counters,
`accepted` (1/0), acceptance dates/bundle paths, status, rating, deviation, `rating ± RD`,
and completed W/D/L/truncation counts. **Filter `accepted=1` for all accepted networks**;
rejected candidates remain in the same table. The first data row is always
`random-no-eye-fill-v1`, fixed at rating **100**, deviation **0**, status `fixed`
and `accepted=0`, even before any network is evaluated or registered.
Numbers may be negative for bots weaker than the 100 reference. Newly evaluated
or selected experimental actors remain `accepted=0` until explicit approval;
supported legacy accepted entries are preserved. Match `decision` values describe
experimental verdicts; migration preserves their historical values.

`matches.csv` records full SGF hashes and player IDs, actual results (blank means
unknown), board size, protocol columns, audit dates, batch IDs and each player's
**post-batch** rating/RD/anchor state. Snapshots repeat across a batch's rows, not
per-game updates. It indexes evidence/history, not moves or training targets:
SGFs remain the game records. Each SGF hash is counted at most once globally, even
across overlapping imports or a crash after the CSV commit but before supervisor
acknowledgement.

Match rows are **newest committed batch first**, with games in ascending
`game_number` within each batch. `batch_number` is a store-wide monotonic commit
sequence, independent of timestamps (including clock ties or rollback).
`game_number` is a stable one-based ordinal within the committed batch, assigned
by full SGF hash order on ingestion, not the source arena's game ordinal.
Recovery applies snapshots in ascending batch/game order, regardless of CSV row
order. Do not change these ordering keys by hand.

Each row's `sgf_path` explicitly locates its evidence relative to `ratings/`:

```text
games/YYYYMMDD-HHMMSS-BLACK-vs-WHITE-size9-game000001-FULL_SGF_SHA.sgf
```

The date is the batch's UTC recording time. Player labels are sanitized, bounded
network names with short checkpoint IDs (or the unmistakable reference name).
Names are for browsing; full player IDs and SGF hashes in the ledger remain the
identities. The full SGF hash prevents filename collisions even when readable
labels, dates, or short hashes coincide. Distinct links for the same SGF share a
single store-owned blob/inode.

A writer lock serializes updates. CSVs are fsynced and atomically replaced, not
appended in place. Metadata is saved before committing match rows. On restart or
refresh, current numeric ratings/counts are rebuilt from **committed match rows**,
repairing an ahead/behind network-table cache without replaying the math or
inflating uncertainty. The fixed reference is always first, identified by its
full name and `fixed` status. Neural network rows follow newest registrations
first, with full network ID ascending to break timestamp ties. Registration timestamps never change on re-registration, acceptance, or
refresh.

Old split rating tables migrate automatically on refresh: original CSVs are backed
up under `ratings/backups/before-global-*/`, SGF hashes are verified and deduplicated,
and outcomes are replayed in commit order to produce global batch snapshots.
Unplayed intermediate registry entries are removed; archived recovery packs remain.
This one-time replay changes ratings that previously had separate histories.
Subsequent refreshes restore snapshots without replaying rating math. Do not run
old and new software concurrently against a store during migration.

Existing network CSVs without `registered_at` migrate automatically on the next
successful refresh/update (`mise run ratings` also refreshes). Their unknown dates
remain blank and sort after dated neural networks, by full ID;
no registration history is invented from acceptance or checkpoint dates. The
migration preserves metadata, match history, and SGF evidence, and numeric ratings
are rebuilt from committed snapshots as usual.

Legacy match CSVs also migrate automatically: batch sequences come from their
original ledger insertion order, and per-batch game order is preserved. Dates,
protocols, results, decisions, hashes, and rating snapshots are unchanged. Readable
Independent SGF copies are verified and synced **before** atomically publishing their ledger
paths; original `games/<full-SGF-SHA>.sgf` links are retained for old backups and
safe retry after interruption. No files are renamed in place or deleted, and
no rating math is replayed. Back up the store before migration; older software
cannot read the expanded CSV schema. Subsequent refreshes are stable and do not
renumber batches/games. Uncommitted evidence links after a crash are harmless;
the CSV ledger still controls which games count.

Keep both CSVs; a missing match ledger is an error, not permission to reset
ratings. Back up the whole store and quiesce writers while copying the CSVs.
Do not edit rating snapshots or processed hashes by hand.

```sh
mise run ratings
mise run ratings -- --accepted-only
# Read-only plot; default output models/ratings.png:
mise run plot-ratings
mise run plot-ratings -- --output scratch/ratings.svg --labels
# X is cumulative unique training games from ancestral checkpoint split histories,
# excluding validation and deduplicating replay/epochs. It is NOT arena games.
# The x axis is linear, with the random-bot star at zero and no axis break.
# Error bars are ±RD, not 95% confidence.
# Missing checkpoints/history are errors, never replaced by evaluation counts.
mise run ratings -- --matches
# Resolve and hash-verify exact evidence:
mise run ratings -- --sgf FULL_SGF_SHA
# Ingest using the latest committed match's evaluation settings, or explicitly
# select settings with --protocol-from EXISTING_SGF_SHA:
mise run ratings -- --ingest scratch/arena-001 --checkpoints PATH/TO/CANDIDATE/checkpoint.pt
# Standalone neural-vs-reference matches (ingest separately to update the table):
mise run arena -- --candidate PATH/TO/NETWORK --reference-random --games 10 --simulations 32 --max-moves 2048 --output scratch/reference-001
```

Every newly committed arena batch automatically refreshes `models/ratings.png`.
Updates happen after the CSV commit and use atomic image replacement. A missing
plot dependency, checkpoint history, or rendering failure logs a warning without
rolling back results or stopping training. `mise run setup` includes plotting
dependencies; manual environments can install `.[plot]`.

The stateless Rust adapter uses JSON **only as stdin/stdout IPC**, not persistent
rating storage or SGF sidecars. Python validates/replays SGFs through Rust, pins
players/search/budgets, retains store-owned evidence, and manages the two CSVs.
Completed-only ratings can be biased if truncation censors particular boards or
outcomes. Inspect per-size reports and truncation counts; a small RD does not
account for that bias or prove improvement.

## Limits

Default match budgets and thresholds are configuration defaults, not statistically
calibrated strength guarantees. There is no SPRT/confidence-bound promotion test,
resignation, opening suite, or per-size minimum-strength constraint. The conservative
bounds above account for **missing outcomes**, not statistical uncertainty.
Diagnostics/parity tests cover all board sizes; tuning work belongs in
[todo.md](../todo.md).
