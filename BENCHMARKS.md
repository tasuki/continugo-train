# Random self-play timing — 2026-10-30

The first whole-game table describes the **original full-coordinate sampler**,
before the incremental coverage-count index. Switching to the free-list sampler
changes seeded trajectories, so that switch cannot be judged by whole-game times
alone. Later sections isolate the index on fixed positions, then compare identical
games before/after the liberty and eye-analysis optimizations.

Measured sequentially on an Intel Core i5-8400 @ 2.80 GHz, one self-play worker,
optimized Rust release build. Default 512-move limit; all three profiled games
finished naturally. Shared backend revision: `26722b7` (backend clean).
These are individual diagnostic runs, not multi-seed benchmark averages.

## Whole games and stages

Times below are profiled game-generation wall times, excluding compilation,
process startup, and file output. Stage totals include all rejected candidates,
not just the moves ultimately played.

| Equivalent size | Seed | Moves | Total | Rules / move application | Small-eye detection | Coordinate proposals | Full scans before passing |
|---|---:|---:|---:|---:|---:|---:|---:|
| 9 | 0 | 106 | 19.45 s | 15.11 s (77.7%) | 4.23 s (21.8%) | 3,110,456 | 3 |
| 13 | 1 | 183 | 124.31 s | 116.44 s (93.7%) | 7.70 s (6.2%) | 4,891,516 | 4 |
| 19 | 2 | 397 | 232.77 s | 215.60 s (92.6%) | 16.89 s (7.3%) | 7,727,176 | 5 |

Capture-count bookkeeping took 0.0005 / 0.0015 / 0.0045 s respectively.
Proposal iteration, remaining bookkeeping, and profiling overhead accounted for
0.107 / 0.168 / 0.267 s. Final scoring took 0.0002 / 0.0009 / 0.0044 s;
SGF export took under 0.00003 s per game. Metadata serialization and file output
took under 0.001 s per game, even with the detailed profiles included.

The whole rules stage includes geometry checks, group/liberty/capture work,
position/history cloning, and construction of illegal-move errors. This original
measurement did not split those operations. Subsequent internal profiling below
identifies exact liberty checking—not group construction—as the dominant cost.

| Size | Rules time on legal candidates | Rules time on illegal candidates | Legal candidates tested | Eye rejections |
|---|---:|---:|---:|---:|
| 9 | 1.27 s | 13.84 s | 9,531 | 9,428 |
| 13 | 40.97 s | 75.47 s | 12,663 | 12,484 |
| 19 | 97.75 s | 117.85 s | 25,430 | 25,038 |

## Where the time goes

- Size 9: moves 1–64 took **0.21 s**. The last four turns took **17.74 s**,
  including three full scans. The final ten moves took 18.43 s.
- Size 13: the first 173 turns took **1.94 s**; the last ten took **122.37 s**
  (98.4% of the game). Individual pass scans took 23.5–26.6 s.
- Size 19: pass scans took **20.7–26.1 s** each. A late placement also required
  925,432 proposals and 21.70 s. So unsuccessful scans are not the only cost:
  finding the last few acceptable moves can be nearly as expensive.

The main bottleneck is repeated move validation in dense positions, amplified by
exhaustive coordinate searches. The rules currently rebuild groups and search
exact liberty rings for each geometrically available trial placement. Small-eye
flood filling is a secondary cost and is also recomputed per candidate.

The next optimization targets are reusing geometry/group/liberty work across
candidates and caching pocket analysis per position. This need not change the
rules, eye filter, or exhaustive-before-passing behavior. No such optimizations
were made as part of this measurement.

## Overhead and correctness checks

Ordinary runs of the same size/seed took 19.45 s (9) and 124.38 s (13), closely
matching the profiled timings. The ordinary size-19 run was stopped at a
180-second measurement limit; the subsequent profiled run completed in 232.77 s.
Do not interpret that timeout as a completed game or a full timing.

All three profiled SGFs match the existing `runs/random-new` records byte-for-byte;
the completed ordinary 9/13 runs also match their profiled counterparts.
Instrumentation did not change move selection. Timing noise and instrumentation
cost remain possible; these numbers are diagnostic, not precision microbenchmarks.

## Incremental geometric availability: fixed-position comparison

The current index keeps a coverage count per in-bounds coordinate and a dense
free-coordinate list. Only disks of added/removed stones are updated. It is not
a cache of suicide, captures, or eye decisions; those still depend on the position.

On the exact same size-9 position before move 103 (48 stones, a forced pass),
both samplers exhausted their domains and agreed no acceptable move existed:

| Sampler | Coordinates tried | Total scan | Rules | Eyes |
|---|---:|---:|---:|---:|
| Original full domain | 1,000,000 | 7.275 s | 7.239 s | 0 s |
| Indexed geometric free list | 7,732 | 7.134 s | 7.134 s | 0 s |

Building the index from that existing position took **0.0198 s**. During normal
play it is built once for the empty board and updated incrementally.
The fixed-position comparison does not depend on candidate order because both
scans exhausted all acceptable candidates; their legal/eye-rejection counts also
matched. These are single diagnostic measurements, not averaged microbenchmarks.

This eliminates **99.23% of candidate rule calls**, but only about **2% of scan
time** in this case: nearly all cost is in the remaining geometrically possible
but suicidal placements, not cheap overlap/border rejections. Group/liberty reuse
is still needed for a large speedup. No rules or passing criteria were weakened.

A 64-move smoke run at each size verified the integrated index. Initialization
cost 7.24 / 7.27 / 4.50 ms for 9/13/19; all 64 incremental updates together cost
17.48 / 9.28 / 4.15 ms. These trajectories differ from the old sampler's games;
these numbers characterize index overhead, not a before/after playing-speed claim.

A subsequent complete size-13 run (seed 1) finished naturally in **116.78 s**,
with **96.39 s** in its last ten turns. The earlier full-coordinate run took
124.31 s total and 122.37 s in its last ten turns. The new sampler produced
202 moves rather than 183, so these are different trajectories, not a controlled
speedup measurement. The new report is in `runs/availability-full13`.

To run the same fixed-position benchmark using a saved full SGF with a forced pass:

```sh
CONTINUGO_BENCH_SGF=runs/random/game-000001-size9-seed0.sgf \
  mise exec -- cargo test --locked --release action::tests::benchmark_cached_endgame_scan -- --ignored --nocapture
```

The test reads board size from SGF, replays to immediately before its first pass,
then runs both candidate sources through the same policy. It requires that the
saved pass was forced by the random baseline, rather than chosen by MCTS.

## Exact row-interval liberty search

Internal `Position::play_profiled` timings found **99.3–99.4%** of move-validation
time in liberty checks on both fixed endgame positions; group construction was
under 1%. Caching groups was therefore not the first worthwhile target.

Rust now follows the frontend's `Rules.Liberties` exact integer scanline method:
precompute strict-disc row widths once per stone radius, then check gaps between
blocked integer intervals instead of testing each coordinate in the ring against
every stone. Nearby blockers are filtered once per query. Exact touching points,
board clipping, captures and suicide rules are unchanged. No approximate liberty
samples or new gameplay heuristics were introduced.

The original exhaustive liberty routine remains available only in tests or the
optional `reference` feature of the rules crate. The fixed-position benchmark
compares **every candidate's full resulting position**, not just its legality.

| Fixed position | Candidates | Original rules path | Scanline rules path | Speedup |
|---|---:|---:|---:|---:|
| Size 9, before move 103, 48 stones | 7,732 | 7.179 s | 0.172 s | 41.7× |
| Size 13, before move 199, 110 stones | 8,670 | 45.504 s | 1.057 s | 43.0× |

All **16,402** candidate results matched the original path, including surviving
stone colors and coordinates. Standard tests also compare exhaustive/scanline
liberties at boundaries, enclosures and tiny gaps, randomized placements, and a
known multi-stone capture. Ordinary `play` does not read clocks or accumulate
timing counters.

### Controlled whole-game comparison

This comparison uses the **same indexed sampler and seed** on both sides; unlike
the earlier full-domain/free-list comparison, the game did not change:

| Size 13, seed 1 | Before liberty optimization | After | Speedup |
|---|---:|---:|---:|
| Whole 202-move game | 116.775 s | 17.163 s | 6.80× |
| Last ten turns | 96.393 s | 15.697 s | 6.14× |
| All move-validation work | 102.169 s | 2.668 s | 38.30× |
| All eye analysis | 14.556 s | 14.450 s | essentially unchanged |

The SGFs are byte-for-byte identical. Proposal, legality, and eye-rejection counts
also match. After this optimization, the bottleneck was eye analysis (84.2% of this game's
runtime), not move validation. Eye analysis was still unchanged at this stage;
the next section measures its subsequent optimization.

The geometric index is still useful. Repeating the size-9 fixed forced-pass scan
with fast liberties gave **0.338 s** over all million coordinates versus **0.179 s**
over the cached 7,732 coordinates: about **1.89×** faster for that scan. Removing
the main liberty bottleneck made skipping cheap geometry rejections more relevant.
These remain individual diagnostic runs, not averaged benchmark estimates.

To reproduce disjoint internal timings and compare both rule paths:

```sh
CONTINUGO_BENCH_SGF=runs/availability-full13/game-000001-size13-seed1.sgf \
  mise exec -- cargo test --locked --release --features reference-rules \
  action::tests::benchmark_rule_internals -- --ignored --nocapture
```

The benchmark defaults to immediately before the SGF's first pass. Set
`CONTINUGO_BENCH_PLY` to a 1-based turn to inspect another fixed position.
Without `--features reference-rules`, it times only the optimized path. The
`reference-rules` feature enables a test oracle, not a different production policy.
Internal logs are saved in `runs/liberty-benchmarks`; the optimized whole-game
profile is in `runs/liberties-full13`.

## Per-position eye reuse

`continugo-analysis::SmallEyeAnalysis` now lazily caches complete free-space
pockets and their boundary group for one immutable position. It reuses the existing
geometric-availability index during flood fills and labels friendly groups once.
A pocket's actual left/right extrema on each integer row suffice to test exactly
whether a candidate stone overlaps every point. Color and connections to other
friendly groups are checked for each candidate; this is not a cached rejection
boolean for the whole component.

A flood fill that escapes the proposed stone's exclusion disc returns false
without leaving partial component labels. Thus early-game open space does not
require a board-wide analysis. Complete pockets are reused after rejected eye
fillings. The random sampler drops the analysis after each turn; no eye decisions
or pocket labels survive placements/captures. Captures still bypass eye avoidance.
The one-shot `fills_small_eye` wrapper uses the same implementation without a
persistent cache. No rule, filter criterion, candidate order, or self-play RNG
consumption changed.

### Fixed-position differential checks

Every geometrically available coordinate was checked for **both colors**, in the
same order through cached analysis and the original per-candidate detector:

| Position | Queries | Original flood fills | Cached analysis |
|---|---:|---:|---:|
| Size 9, before move 103 | 15,464 | 3.424 s | 0.00754 s |
| Size 13, before move 199 | 17,340 | 7.454 s | 0.29462 s |
| Size 13, before move 202 | 14,964 | 6.566 s | 0.00881 s |

All **47,768** results matched. These scans deliberately include suicidal
coordinates, unlike actual self-play, which checks legality before eye analysis.
Incomplete/non-filling pockets are not cached, so including those candidates and
changing query order can substantially affect cached scan time. These are single
diagnostic timings, not averaged estimates or a universal speedup claim.

Tests also exercise repeated queries, alternating colors, early escape followed
by a complete flood fill, a cached pocket that one candidate fills but another
leaves open at an exact touching point, separate-group connections, and stale
availability-index rejection.

### Controlled whole-game comparison

Same size 13, seed 1, indexed sampler, **202 moves**, naturally finished:

| Stage | Before eye reuse | After | Speedup |
|---|---:|---:|---:|
| Whole game | 17.163 s | 2.736 s | 6.27× |
| Last ten turns | 15.697 s | 2.272 s | 6.91× |
| All eye checks | 14.450 s | 0.0504 s | 287× |
| All move validation | 2.668 s | 2.640 s | essentially unchanged |

The SGF is byte-for-byte identical. All 202 turns have the same geometric-candidate,
proposal, legality, eye-call/rejection and pass counts; the outcome also matches.
The bottleneck is now move validation again (96.5% of this game's runtime); eye
analysis accounts for 1.8%. Overall, the two optimizations reduced the original
indexed 116.775-second game to 2.736 seconds, about **42.7×** faster on this trajectory.

To reproduce an eye comparison against the original test oracle:

```sh
CONTINUGO_BENCH_SGF=runs/liberties-full13/game-000001-size13-seed1.sgf \
  mise exec -- cargo test --locked --release --features reference-eyes \
  action::tests::benchmark_eye_cache -- --ignored --nocapture
```

Set `CONTINUGO_BENCH_PLY=202` for the second size-13 position; the default is before
the first pass. The `reference-eyes` feature only compiles the old detector for
tests/benchmarks, not a different production policy. Logs are in
`runs/eye-benchmarks`; the whole-game record and profile are in `runs/eyes-full13`.

## Profile the current implementation

```sh
mise run selfplay -- --engine random --profile --games 3 --output runs/profile-001
```

Defaults cycle sizes 9, 13, 19 with seeds 0, 1, 2. Every game logs generation and
output time. `--profile` prints stage totals and slow moves, and writes per-move
counters/timings into the SGF root and move comments as JSON. Full
generation/export timings are stderr-only, since they finish after the comments
are serialized. Omit `--profile` for ordinary timings. Older records used JSON
sidecars; current output is a single self-contained SGF. Compare decoded moves,
not whole-file bytes, when comparing runs with different profiling annotations.

Raw reports for this measurement are in `runs/timing-profile` and
`runs/timing-baseline`. As with everything under `runs/`, those are temporary
artifacts; this document retains the results if they are deleted.
