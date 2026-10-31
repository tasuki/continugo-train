# ContinuGo training benchmarks

Initial self-play and network measurements: 2026-10-30.

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

## Per-position groups and exact liberty witnesses

The shared rules now expose `Position::prepare` and `Game::prepare`. They build
same-color connected groups once and retain up to two separated, exact liberty
coordinates per group. A witness not overlapped by the candidate proves that
its group still has a liberty. If both witnesses are blocked, the validator
rechecks the group with the existing exact scanline search. These are existence
witnesses, not approximate liberty counts or tactical heuristics.

Enemy groups are captured simultaneously before checking the new friendly
merged group and other friendly groups. Original stone order, captures, suicide,
strict touching boundaries, and board bounds are preserved. `Position::play`
remains the uncached differential baseline. Prepared positions borrow their
immutable source; prepared games own a snapshot, preventing stale analysis.
Random self-play prepares once per turn. MCTS prepares lazily once per expanded
node and reuses that snapshot across progressive-widening proposals; its policy,
RNG consumption, pass handling, and lack of eye avoidance are unchanged.

### Fixed-position differential checks

Every geometrically available coordinate was checked for **both colors** against
the uncached validator, comparing full resulting positions, including stone order:

| Position before turn | Queries | Uncached | Prepared | Preparation |
|---|---:|---:|---:|---:|
| Size 9, turn 80, 51 stones | 18,614 | 3.4681 s | 0.09094 s | 0.000289 s |
| Size 13, turn 199, 110 stones | 17,340 | 2.3647 s | 0.09834 s | 0.000276 s |
| Size 19, turn 409, 241 stones | 10,558 | 4.5403 s | 0.07461 s | 0.000850 s |

All **46,512** candidate results matched. Scan timings exclude comparison checks;
preparation is reported separately. Ordinary tests additionally cover both colors
across sizes 1/2/3/9/13/19/500, exact touching, source immutability, game history and
completion, capture/suicide fixtures, and dense setup positions with simultaneous
captures, connections, and pre-existing dead groups.

### Controlled whole-game comparison

Same machine and release configuration as above, one worker, same indexed sampler
and seeds. All games completed naturally; generation times include preparation,
profiling, scoring and SGF serialization, but exclude compilation and file output.

| Size / seed | Moves | Before | Prepared | Speedup | All preparation | Candidate rules |
|---|---:|---:|---:|---:|---:|---:|
| 9 / 0 | 81 | 3.626 s | 0.161 s | 22.5× | 0.0122 s | 0.0963 s |
| 13 / 1 | 202 | 2.703 s | 0.270 s | 10.0× | 0.0499 s | 0.1326 s |
| 19 / 2 | 432 | 13.316 s | 0.570 s | 23.4× | 0.2263 s | 0.1908 s |

Decoded moves, player names, scored results, and every turn's proposal, legality,
eye-rejection and stage-call counts match. Profiled SGFs differ in timings and the
new preparation counters, so whole-file bytes are not compared. These are individual
diagnostic runs, not multi-seed averages or MCTS throughput measurements.
Preparation is now a significant cost (39.7% of the size-19 game); the optimization
is not free for positions requiring very few candidate checks.

Keeping just one witness left the size-9 game at about 2.6 s: many candidates
blocked that witness even when the group had another surviving liberty. Retaining
a second separated witness reduced unnecessary exact rechecks without changing
any candidate results.

To reproduce a fixed-position comparison (defaults to before the first pass):

```sh
CONTINUGO_BENCH_SGF=runs/reuse-before/game-000002-size13-seed1.sgf \
  mise exec -- cargo test --locked --release action::tests::benchmark_prepared_rules -- --ignored --nocapture
```

Set `CONTINUGO_BENCH_PLY` to select another turn. No reference feature is needed:
the current uncached scanline validator is the comparison baseline. Whole-game
profiles are in `runs/reuse-before` and `runs/reuse-two-witnesses`; differential
logs are in `runs/reuse-benchmarks`.

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

## Initial PyTorch policy/value network

Default architecture: **81,283 parameters**, 32 channels, 4 residual blocks,
16 coarse input planes, 12 exact-feature input columns, 16-wide shared policy
scorer, 32-wide pooled value head. Coarse grids are 38/54/78 for Rust stone
diameters 110/76/52. Policy chunks contain at most 65,536 coordinates, with
**all 1,000,000 coordinates plus pass** scored per position.

These are **synthetic FP32 eager** measurements, not neural self-play or trained
playing strength. Inputs/features are random, with a synthetic approximately 90%
legal mask; no Python rules or feature extraction is substituted for Rust. Both
inputs and sparse visit/outcome targets are preallocated on-device. Real Rust
features and SGF loading were added afterward and are measured below; these older
synthetic timings still exclude extraction and transfer. ONNX and neural MCTS
integration remain future work.

PyTorch **2.7.1+cu118**, Python 3.13.5, seed 0, 6 CPU threads, two warmup calls per
stage followed by the **median of five** timed calls. GPU: GTX 1080 (8 GiB), driver
550.163.01. CPU: same i5-8400 as above. CPU and GPU runs were sequential. Timings
include Python dispatch; CUDA is synchronized at timing boundaries. Training uses
forward/backward and SGD (lr 0.01, momentum 0.9, weight decay 1e-4). The coarse-only
step uses a synthetic squared-output objective; the full step uses visit-count
cross-entropy plus outcome MSE. Times are **milliseconds per batch**.

### GTX 1080

| Size / grid / batch | Coarse infer | Dense policy infer | Full infer | Coarse step | Full step | Full-step peak MiB |
|---|---:|---:|---:|---:|---:|---:|
| 9 / 38 / 1 | 0.901 | 6.759 | 7.848 | 3.694 | 23.641 | 160.6 |
| 9 / 38 / 4 | 0.875 | 29.579 | 30.551 | 3.673 | 85.972 | 971.3 |
| 13 / 54 / 1 | 0.921 | 6.390 | 7.288 | 3.727 | 21.616 | 164.3 |
| 13 / 54 / 4 | 0.916 | 29.639 | 30.696 | 4.178 | 86.737 | 986.6 |
| 19 / 78 / 1 | 0.918 | 6.354 | 7.305 | 3.993 | 20.953 | 171.9 |
| 19 / 78 / 4 | 1.488 | 29.587 | 31.091 | 6.064 | 82.534 | 1015.1 |

Peak memory is total live PyTorch allocation, including inputs, gradients and
optimizer state; it excludes reserved allocator memory, CUDA context and desktop
processes. Batch-one full inference peaked at 80.6–89.5 MiB. Chunking bounds
inference intermediates, but autograd retains all chunks until backward.

### i5-8400 CPU, six threads

| Size / grid / batch | Coarse infer | Dense policy infer | Full infer | Coarse step | Full step |
|---|---:|---:|---:|---:|---:|
| 9 / 38 / 1 | 1.784 | 153.891 | 154.964 | 6.771 | 272.562 |
| 9 / 38 / 4 | 3.102 | 300.192 | 297.495 | 13.846 | 741.158 |
| 13 / 54 / 1 | 2.340 | 100.842 | 100.369 | 9.664 | 274.579 |
| 13 / 54 / 4 | 5.849 | 286.445 | 294.391 | 25.781 | 743.998 |
| 19 / 78 / 1 | 3.140 | 98.672 | 103.042 | 15.219 | 278.858 |
| 19 / 78 / 4 | 14.527 | 288.343 | 306.536 | 57.583 | 782.695 |

The cheap dense head still dominates inference; increasing the coarse ResNet is
not the immediate throughput bottleneck. Full GPU inference delivers about
127–137 positions/s, while full training delivers about 42–48 positions/s. Batching
four positions did not improve dense GPU throughput here. CPU timings are noisier;
these five-call medians are diagnostics, not broad statistical comparisons.

Tests cover million-coordinate output, resolution changes using one model,
legality/pass masking, exact neighboring-coordinate distinctions, interpolation
axes/endpoints, chunked versus unchunked logits/gradients, sparse versus dense
cross-entropy, zero-count illegal padding, checkpoint round trips, and a tiny
fixed batch whose training loss falls from 3.07 to 0.41 in 200 updates. That last
check verifies learnability/wiring, not game strength.

Reproduce or compare architectures:

```sh
mise run model-bench -- --device cuda
mise run model-bench -- --device cpu
mise run model-bench -- --channels 64 --blocks 6 --policy-hidden 32
```

Raw logs: `runs/network/cuda.txt` and `runs/network/cpu.txt`. See
[python/README.md](python/README.md) for the input contract and benchmark options.

## Real Rust feature extraction — 2026-10-31

These measurements describe schema 1 (16 planes / 12 move fields). Schema 2
subsequently removed all four history planes and the exactly-one-liberty flags
(10 planes / 11 fields); it was not separately rebenchmarked here. Schema 3
implements UI-style area; its measurements are below. See FEATURES.md.

Version-1 inputs contained real stone/group/liberty/history/border planes and
exact move legality, connections, captures, resulting group size and capped exact
liberties. Rule facts use the shared validator and shared integer liberty search;
normalization/rasterization remain experimental code in `continugo-train`. Schema,
label semantics and the binary bridge are documented in [FEATURES.md](FEATURES.md).

The following are sequential, single-position diagnostics from the same three
completed random SGFs (sizes 9/13/19, seeds 0/1/2), using the i5-8400 and Rust release
build, one thread. Every extraction computes **all million coordinate rows**.
`ply` counts moves already played; these are pre-move positions. Export writes the
roughly 53 MB tensor packet to `/dev/null`; it does not measure disk persistence
or Python decoding. Process startup and action-ID vector construction are also
outside these stages. Source SGFs are replayed to completion to validate scored RE,
so replay time is nearly independent of the requested ply.

| Size | Ply | Legal placements | Replay | Coarse + preparation | Exact rows | Binary output |
|---|---:|---:|---:|---:|---:|---:|
| 9 | 0 | 793,881 | 0.00620 s | 0.00004 s | 0.16522 s | 0.02975 s |
| 9 | 40 | 19,444 | 0.00612 s | 0.00020 s | 0.29121 s | 0.03120 s |
| 9 | 79 | 3,140 | 0.00656 s | 0.00059 s | 0.45788 s | 0.03102 s |
| 13 | 0 | 855,625 | 0.02433 s | 0.00005 s | 0.17635 s | 0.03142 s |
| 13 | 100 | 6,127 | 0.02376 s | 0.00085 s | 0.30787 s | 0.03178 s |
| 13 | 198 | 6,285 | 0.02353 s | 0.00062 s | 0.23624 s | 0.03184 s |
| 19 | 0 | 900,601 | 0.12643 s | 0.00009 s | 0.17890 s | 0.02951 s |
| 19 | 200 | 12,028 | 0.12516 s | 0.00131 s | 0.26380 s | 0.02884 s |
| 19 | 408 | 3,903 | 0.12365 s | 0.00146 s | 0.52424 s | 0.02949 s |

Legality is not the random bot's acceptability filter: these late positions can
have thousands of legal moves even when random self-play must pass because all
such placements fill excluded small eyes. Tests explicitly preserve that distinction.

One complete size-13 ply-198 sample loaded through the Python subprocess bridge
in **0.361 s**, including full replay, extraction, binary transport and decoding
(a separate single run, not the sum of table medians). It fed the default network
on the GTX 1080 and produced the expected `[1, 1_000_001]` logits with finite pass.
The initial one-request-per-position bridge is not a batched data-loader service;
feature extraction/transport now costs much more than the approximately 7 ms
GPU inference measured above. Efficient offline loading is the next priority.

A separate real-data GPU smoke test used a naturally completed **9-move MCTS SGF**
(size 9, seed 41, 8 simulations). Its actual root policy had total visit mass 8 and
7 positive actions. Three default-network SGD updates (initialization seed 0,
lr 0.01, momentum 0.9) produced finite gradients and losses **14.486 → 13.590 →
13.140**. This proves the real SGF → Rust features → PyTorch loss path works;
it is not a strength evaluation or a trained model release.

The existing three seeded random games remain **byte-for-byte identical** after
the shared liberty API refactor. Tests also cover exact capped liberty counts
against an exhaustive one-liberty enclosure, resulting-group analysis against
normal validation, touching/capture/suicide distinctions, side-to-move/history
alignment, RE/target validation, terminal/truncated samples, corrupt binary data,
and a full million-coordinate real-feature training step.

```sh
mise run build-features
# Diagnostic tensor output; creates no alternate game record.
mkdir -p runs
mise run features -- --sgf runs/random-new/game-000002-size13-seed1.sgf \
  --ply 198 --output runs/position.cfeat
```

Raw diagnostics: `runs/features/timing.txt`, `runs/features/training-smoke.txt`,
`runs/features/selfplay.txt`; the MCTS source SGF is under `runs/feature-mcts`.

## UI-style liberty area — schema 3

The group feature is now the **union area of the UI's stone-radius swept liberty
spans**, divided by physical stone-disc area, without capping or clipping. Exact
integer centers come from the shared rule scanlines; only continuous area
integration is numerical. No history, exactly-one-liberty flags or capped liberty
counts remain in the network input. A single isolated center has exactly one
stone-disc of highlighted area, not a negligible coordinate-count fraction.

Area is calculated once per existing group, not for every hypothetical move. The
old resulting-liberty-count column was removed instead of replacing it with a
million expensive area calculations. Thus schema 3 has **10 coarse planes and
10 move fields**, 79,523 parameters with the default 32-channel / 4-block network.
A child position gets its own fresh group areas when extracted. Captures,
connections, legality and resulting group sizes remain exact per-move features.

Sequential release diagnostics on the same SGFs/CPU as above, one thread, all
million coordinate rows (individual runs, not statistical speedup estimates):

| Size | Ply | Legal placements | Coarse + preparation, including area | Exact rows |
|---|---:|---:|---:|---:|
| 9 | 0 | 793,881 | 0.00004 s | 0.11288 s |
| 9 | 40 | 19,444 | 0.00604 s | 0.14778 s |
| 9 | 79 | 3,140 | 0.00140 s | 0.08800 s |
| 13 | 0 | 855,625 | 0.00008 s | 0.11929 s |
| 13 | 100 | 6,127 | 0.00474 s | 0.21907 s |
| 13 | 198 | 6,285 | 0.00206 s | 0.08539 s |
| 19 | 0 | 900,601 | 0.00009 s | 0.12269 s |
| 19 | 200 | 12,028 | 0.01243 s | 0.18354 s |
| 19 | 408 | 3,903 | 0.00445 s | 0.12136 s |

Exact-row extraction is cheaper than schema 1 because it no longer searches
resulting-group liberties per candidate. Replay is still about 0.006/0.024/0.125 s
for the three complete source games; output is about 0.025–0.027 s for a roughly
45 MB packet to `/dev/null`. Kernel timings exclude replay, transport, Python
loading and inference. The largest populated coarse/preparation stage above was
12.4 ms, so area computation is not the main full-domain cost here.

### Area accuracy checks

Union lengths are integrated with midpoint slices no higher than one board unit
or r/32, aligned with integer capsule activation boundaries. Single capsules use
the analytic area directly. Tests check overlapping/disjoint disk unions, long
capsules, an isolated touching center, translation, and refinement at half the
slice height. Center spans are independently checked pointwise against the exact
shared geometry on boards 3/9/13/19, including borders and exact touching.

A separate half-step audit over the six populated positions above covered **131
groups**. Largest coarse/fine difference: **0.003086 stone-disc units**; largest
relative difference: **0.0283%**. These are observed convergence differences, not a
certified worst-case error bound or a claim of exact continuous integration.
Exact legal/capture decisions do not depend on numerical area.

The real SGF → features → default-network CUDA loss/backward smoke test also
passes on a nonempty MCTS position (schema 3, group area reaching 9.79 stone-discs),
with finite gradients. This checks wiring, not game strength. Source SGFs stay
compatible; old derived feature packets/checkpoints must be discarded or migrated.

```sh
CONTINUGO_BENCH_SGF=runs/random-new/game-000003-size19-seed2.sgf CONTINUGO_BENCH_PLY=409 mise exec -- cargo test --locked --release liberty_area::tests::audit_record_areas -- --ignored --nocapture
```

The audit's `CONTINUGO_BENCH_PLY` is a **one-based turn**, following the older
benchmark helpers; the feature CLI's `--ply` is zero-based. Raw logs are in
`runs/area/timing.txt` and `runs/area/accuracy.txt`.

## Exact one-move-capturable planes — schema 4

Two float32 0/1 planes now mark friendly/enemy groups with a verified legal
opposing placement that captures the whole group on the unchanged position.
The attacker is the group's opposite color, irrespective of the actual turn;
this is a threat flag, not a forced-capture or capture-depth prediction. Values
are painted on group stones. There are **12 coarse planes / 10 move fields**
(default network: 80,099 parameters). Source SGFs are unchanged; derived packets
and checkpoints from earlier feature schemas are incompatible.

The query reuses exact liberty spans already needed for area. Wide-spread
liberties reject via bounds. For each remaining y, intersect strict exclusion
constraints for span endpoints, subtract occupied geometric intervals, then
validate candidates with the shared prepared capture/suicide rules. This is an
exhaustive exact filter, not a sampled candidate budget or an area threshold.
Nothing is added to the per-million-coordinate feature loop.

One-thread release diagnostics on the same nine positions as schema 3:

| Size | Ply | Coarse + preparation, including flags | Exact rows |
|---|---:|---:|---:|
| 9 | 0 | 0.00005 s | 0.11279 s |
| 9 | 40 | 0.00585 s | 0.14927 s |
| 9 | 79 | 0.00144 s | 0.08873 s |
| 13 | 0 | 0.00007 s | 0.11926 s |
| 13 | 100 | 0.00581 s | 0.21670 s |
| 13 | 198 | 0.00223 s | 0.08765 s |
| 19 | 0 | 0.00013 s | 0.12184 s |
| 19 | 200 | 0.01457 s | 0.18666 s |
| 19 | 408 | 0.00448 s | 0.11955 s |

Legal-coordinate counts match schema 3. Output remains about 0.025–0.027 s to
`/dev/null` for roughly 45 MB. These are individual diagnostics, not paired
averages isolating flag overhead; the largest coarse/preparation stage is now
14.6 ms. Full-domain exact-row extraction still dominates this CPU path.

Tests independently compare the candidate filter against exhaustive pointwise
geometry (including strict touching), and capturability against exhaustive legal
placements in dense mixed-color setups. They cover corner captures despite many
liberties, protected separated pockets, whole-group removal, both colors, turn
changes, float32 0/1 encoding and painting on group stones. Normal search/self-play
receives no handcrafted value bonus or policy filter from these features.

Raw diagnostics: `runs/capture-flags/timing.txt`.
