# ContinuGo training benchmarks

Historical measurements, not current defaults or a roadmap. See
[current docs](current/) for supported behavior and [todo.md](todo.md) for open work.

Initial self-play and network measurements: 2026-10-30.

**Artifact reset:** the generated `runs/` contents and the old sibling model store
were explicitly deleted at the user's request after these measurements. Paths and
hashes below are historical diagnostics, not currently available models. New
outputs stay in this repository: UTC-dated `runs/` directories and dated recovery
bundles under `models/`. These written measurements are retained.

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
[network architecture](current/network.md) for the input contract and benchmark options.

## Real Rust feature extraction — 2026-10-31

These measurements describe schema 1 (16 planes / 12 move fields). Schema 2
subsequently removed all four history planes and the exactly-one-liberty flags
(10 planes / 11 fields); it was not separately rebenchmarked here. Schema 3
implements UI-style area; its measurements are below. See features.md.

Version-1 inputs contained real stone/group/liberty/history/border planes and
exact move legality, connections, captures, resulting group size and capped exact
liberties. Rule facts use the shared validator and shared integer liberty search;
normalization/rasterization remain experimental code in `continugo-train`. Schema,
label semantics and the binary bridge are documented in [features.md](current/features.md).

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

## First offline training job — 2026-10-31

Real completed MCTS SGFs, not synthetic inputs: **24 games**, sizes 9/13/19,
seeds 100–123, 32 simulations per move, **630 eligible pre-move positions**.
All finished naturally. The canonical-game split held out six games (226 eligible
positions); the other 18 games supplied 404 eligible positions. No duplicates or
unlabeled games were skipped. SGFs remain the only primary game records.

Same i5-8400 / GTX 1080 / PyTorch 2.7.1+cu118 stack. Current 12-plane / 10-column
features, **80,099 parameters**, FP32, batch size 2, two persistent Rust feature
workers, shuffle buffer 8, four PyTorch CPU threads. SGD lr 0.01, momentum 0.9,
weight decay 1e-4, gradient clipping at norm 5. Every forward/loss normalizes over
**all legal coordinates plus pass**, not just visited/proposed actions.

This short sanity run capped training at eight sampled positions per game and
20 batches per epoch (**40 optimized positions**). Validation used a fixed seeded
four-position subset from each held-out game (**24 positions**). Three epochs
therefore performed 60 optimizer updates; this was not a full-corpus fitting run.

| Epoch | Train policy CE | Train value MSE | Train total | Validation policy CE | Validation value MSE | Validation total |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 12.373528 | 1.194239 | 13.567767 | 12.064493 | 1.184399 | 13.248892 |
| 2 | 11.996482 | 1.080416 | 13.076898 | 12.077291 | 0.984017 | **13.061308** |
| 3 | 11.490949 | 1.000858 | 12.491807 | 12.208189 | 0.985361 | 13.193550 |

Training loss falls, but held-out loss is **not monotonic**. `best.pt` correctly
selects epoch 2 rather than the latest epoch. These noisy, weak-teacher data and
small validation subsets verify the learning/checkpoint pipeline, **not playing
strength**, generalization, or a reason to prolong baseline fitting.

End-to-end training throughput was **4.01–4.30 positions/s**; validation was
**4.79–4.81 positions/s**. The six phases totaled about 44 seconds, excluding
initial process/model/corpus setup and checkpoint writes. Training loader wait
was 7.41–8.27 seconds per epoch, versus 1.25–1.59 seconds for compute plus transfer.
Full-domain extraction/decoding/transport dominates; these are not neural-only
throughput measurements. Timings include shuffle fill and worker shutdown.

Peak live CUDA allocation was **495.1 MiB training / 145.0 MiB validation**;
parent peak RSS reached **1,793.5 MiB**, stable over the final two epochs. CUDA
figures exclude allocator reservation/context/desktop use; parent RSS excludes
Rust child processes. Each checkpoint was about 683 KiB, including optimizer,
RNG, architecture, feature compatibility and embedded corpus/split metadata.

A separate short GPU job resumed successfully from epoch 1 to epoch 2, preserving
optimizer progress. Tests compare unchanged single-thread CPU resume against an
uninterrupted run **bit-for-bit**, including BatchNorm and SGD momentum buffers.
GPU backward is not promised bitwise deterministic. Tests also cover changed-data
and feature rejection, duplicate-game holdout, sparse target padding, resolution
bucket leftovers, ordered worker errors/cleanup, and runtime resume overrides.

```sh
mise run selfplay -- --engine mcts --games 24 --sizes 9,13,19 --seed 100 --simulations 32 --output runs/offline-bootstrap
mise run train -- --data runs/offline-bootstrap --output runs/offline-training --epochs 3 --steps-per-epoch 20 --positions-per-game 8 --validation-positions-per-game 4 --device cuda
```

Use fresh output directories to reproduce. Artifacts: `runs/offline-bootstrap/`,
`runs/offline-training/training.log`, `runs/offline-training/best.pt` (epoch 2),
and `runs/offline-training/latest.pt` (epoch 3). The separate resume check is in
`runs/offline-smoke/`. See [training.md](current/training.md) for normal uncapped training,
loading and resume. No feature-schema revision was introduced for this command.

## Native ONNX neural self-play — 2026-11-01

Exported the previous job's **epoch-2 best checkpoint**, unchanged 80,099-parameter
network and feature definitions, as separate encoder and coordinate-scorer graphs.
ONNX opset 17, Python onnx 1.18.0, native ONNX Runtime **1.22.1 CPU**, Rust ort
2.0.0-rc.10. Export verification varied resolution, batch and candidate count;
maximum absolute difference from eager PyTorch was **1.1921e-6**. Real-feature
native/PyTorch parity tests also cover all three sizes, neighboring coordinates,
legality, chunk boundaries, pass, value-only inference and incompatible models.

The following are **neural-guided games**, not unguided MCTS or direct random
play. Same i5-8400; four ONNX intra-op threads, one synchronous query at a time,
thread spinning disabled. No GPU inference or cross-game batching. Search used
8 simulations/move, PUCT 1.5, widening 2*sqrt(visits), independent uniform-legal
proposal mixture 0.25, and visit-count move sampling. All games finished naturally
before the 64-move cap. Shared SGF replay validated every move, policy target and
scored RE; **all 59 pre-move positions** have eligible training labels.

| Equivalent size | Seed | Moves | Generation | Moves/s |
|---|---:|---:|---:|---:|
| 9 | 200 | 13 | 11.912 s | 1.09 |
| 13 | 201 | 18 | 21.231 s | 0.85 |
| 19 | 202 | 28 | 46.469 s | 0.60 |

Together: **79.613 s / 59 moves = 0.74 moves/s**, excluding model/runtime loading,
process startup, compilation and SGF file output. These are individual diagnostic
games, not a strength evaluation, multi-seed throughput average or a before/after
speedup comparison. Eight simulations and a weak initial network are only a
pipeline smoke test. Early passing does not establish sensible play.

Native stage totals over the three games:

- **133 full-policy queries**, deferred until node expansion.
- **464 value-only leaf queries**, requiring no million-coordinate extraction or
  dense scoring; terminal nodes use authoritative outcomes without network calls.
- Rust features: **48.496 s** (60.9% of total), including coarse-only queries.
- Encoder: **2.005 s** (2.5%); dense scorer: **25.857 s** (32.5%).
- Remaining time includes sampling/normalizing full policies, tree work and scoring.

Full-policy logits are normalized over **all legal coordinates plus pass**.
Progressive widening then uses a bounded sampled candidate reservoir with
importance-corrected, pass-stratified priors. This is an approximate sampled
search, not a conditional visited-action training softmax. Nodes do not retain
million-element policies. See [inference.md](current/inference.md) for the proposal law,
why duplicates retain multiplicity, and independent exploration behavior.

The measured `mise run neural-selfplay` launcher reached **373.8 MiB peak RSS**
(`/usr/bin/time -v`, 382,764 KiB). This includes child processes, including the
short-lived Python/Torch runtime-library locator, so it is **not native-only RSS**.
No CUDA memory was used by this inference backend. Full feature extraction and
CPU scoring remain the throughput bottlenecks; GPU inference alone cannot remove
the Rust extraction cost. Batching/parallel feature work and reducing unnecessary
coordinate work are follow-ups before large runs.

```sh
mise run export -- --checkpoint runs/offline-training/best.pt --output runs/onnx-initial
mise run neural-selfplay -- --device cpu --model runs/onnx-initial --games 3 --sizes 9,13,19 --seed 200 --simulations 8 --max-moves 64 --inference-threads 4 --output runs/neural-initial
```

Use fresh directories. Artifacts: `runs/onnx-initial/encoder.onnx`,
`runs/onnx-initial/policy.onnx`, and three self-contained SGFs plus diagnostic text
log under `runs/neural-initial/`. The source checkpoint SHA-256 is
`d4d466ccee0ecfa0c2bf0862a87229e29440eaf1eceb07799c995ac095afcad0`.
The iterative training loop, warm-start on changing replay data, GPU inference,
batching and playing-strength evaluation are still pending at this measurement.
GPU inference was added next, as measured below.

## CUDA inference on GTX 1080 — 2026-11-01

Both CPU and CUDA now use the same Rust evaluator interface and the same exported
model. CUDA is the default; `--device cpu` selects CPU explicitly. CUDA requests
fail if provider setup fails, rather than silently running the network on CPU.

The Pascal-compatible GPU runtime is **ONNX Runtime 1.20.1 / CUDA 11.8 / cuDNN
8.9.6**, installed in `.venv/ort-gpu`. The Rust binding targets API 1.20, also
compatible with the existing CPU 1.22.1 runtime. PyTorch remains **2.7.1+cu118 /
cuDNN 9.1**, unchanged. No driver upgrade or training-stack replacement was needed.
Native CUDA uses FP32, TF32 disabled, heuristic convolution selection and bounded
workspace. GPU kernel profiling tests require Conv/Gemm/MatMul/GridSample on CUDA;
small shape/control operators may remain on CPU. All 31 Python tests (including
the GPU tests), Rust suites and checks pass on the local machine.

### Fixed-position CPU/GPU comparison

Same checkpoint, same nonempty source positions, **same runtime 1.20.1**, four CPU
threads, one warmup followed by the mean of three full-domain queries per device.
Each query computes all million placement logits plus pass and applies the
legality mask. The diagnostic returns raw logits, not a visited-action softmax;
MCTS normalization/sampling is outside these timings. Times below are milliseconds per query. GPU timers include transfers and wait for
host outputs; model loading, replay and warmup are excluded.

| Size / ply | CPU features | CUDA-run features (CPU) | CPU encoder | CUDA encoder | CPU dense scorer | CUDA dense scorer |
|---|---:|---:|---:|---:|---:|---:|
| 9 / 6 | 237.295 | 236.175 | 1.115 | 0.818 | 166.701 | 21.587 |
| 13 / 9 | 333.999 | 351.011 | 1.810 | 0.864 | 166.444 | 21.875 |
| 19 / 14 | 426.045 | 488.784 | 3.199 | 0.938 | 167.465 | 22.363 |

The dense stage is about **7.5–7.7x faster**, but extraction is unchanged CPU work
and has timing noise. Thus the whole query does not improve by that factor.
Maximum selected real-position logit difference was **4.77e-7**. Tests additionally
check values, value-only queries, legality and chunk boundaries at all three sizes.

### Neural self-play

Same search settings and seeds as the CPU run above: eight simulations, widening
2*sqrt(visits), independent exploration 0.25, 64-move cap, one game/query at a time.

| Size / seed | CPU moves | CUDA moves | CPU generation | CUDA generation | Identical decoded plays? |
|---|---:|---:|---:|---:|---|
| 9 / 200 | 13 | 13 | 11.912 s | 7.909 s | yes |
| 13 / 201 | 18 | 14 | 21.231 s | 10.481 s | **no** |
| 19 / 202 | 28 | 28 | 46.469 s | 29.011 s | yes |

All three CUDA games completed naturally and passed authoritative replay,
policy-target and scored-RE validation: **55 eligible positions in 47.402 s**
(**1.16 moves/s**). Size 13 changed trajectory, so combined totals are **not a
controlled whole-game speedup claim**. Floating-point backend differences can
change search choices even though network outputs agree closely. These remain
individual diagnostics and weak-network smoke tests, not strength results.

CUDA-run totals: **125 full-policy queries / 432 value-only queries**. Rust feature
extraction took **42.302 s (89.2%)**, encoder **0.546 s**, dense scorer **2.748 s**.
Feature extraction is now overwhelmingly dominant. This is why GPU inference is
useful but not sufficient: next throughput work should reduce/parallelize feature
work and batch queries, not merely enlarge the network.

Once-per-second `nvidia-smi` samples of the native process ranged from **146 to
258 MiB**, including its CUDA context/allocations but excluding desktop processes.
258 MiB is the **largest sampled value**, not a guaranteed instantaneous peak.
The configured CUDA arena limit is per session and does not bound total GPU use.

```sh
mise run setup-gpu
mise run infer -- --model runs/onnx-initial --sgf FILE --ply 1 --warmup 1 --iterations 3
mise run infer -- --device cpu --model runs/onnx-initial --sgf FILE --ply 1 --warmup 1 --iterations 3
mise run neural-selfplay -- --model runs/onnx-initial --games 3 --sizes 9,13,19 --seed 200 --simulations 8 --max-moves 64 --inference-threads 4 --output runs/neural-cuda
```

Fresh output directories are required. Raw measurements are in
`runs/cuda-inference/timing.txt`, `runs/cuda-inference/gpu-memory.txt`, and
`runs/neural-cuda/run.log`; neural games are self-contained SGFs in
`runs/neural-cuda/`. Batched inference and the iterative training loop remain next.

## Cached contexts and parallel exact features — 2026-11-01

Neural inference now has a bounded host LRU for projected coarse context/pass/value,
and a four-worker Rayon pool for exact move rows. Raw million-row features and
logits are **not cached**. Both native self-play and `infer` use mimalloc for Rust
allocations. Initial parallel whole-game tests regressed despite fast isolated
queries: short-lived rule-validation vectors contended under the system allocator.
A diagnostic allocator swap identified the issue; the scalable allocator restores
whole-game scaling. Shared rules and feature calculations are unchanged.

### Fixed positions, same allocator

Same nonempty inputs/checkpoint/CUDA runtime as above, one ONNX CPU helper thread,
cache disabled, one warmup followed by the mean of three queries. Both serial and
parallel cases use mimalloc. Times are milliseconds per position.

| Size / ply | Serial coarse | Parallel-run coarse (serial) | Serial exact rows | 4-worker exact rows | Row speedup |
|---|---:|---:|---:|---:|---:|
| 9 / 6 | 5.493 | 5.518 | 172.548 | 53.208 | 3.24x |
| 13 / 9 | 4.275 | 4.278 | 236.416 | 85.861 | 2.75x |
| 19 / 14 | 6.803 | 6.829 | 319.466 | 97.556 | 3.27x |

Serial/parallel diagnostic outputs match bit-for-bit. Tests independently compare
all million legality/feature rows on a nonempty size-9 position and noncanonical,
duplicated, uneven action chunks on all three sizes, including a capture. No
approximate validation, feature-schema change or altered normalization was used.

### Identical whole games

Same GPU, model, seeds, search settings and four ONNX CPU helper threads as the
previous CUDA measurement. New feature workers: 4; cache: 64 contexts. All three
SGFs are **byte-for-byte identical** to the previous CUDA games, including policy
visits and scored RE; all 55 positions remain valid labels.

| Size / seed | Moves | Previous CUDA pipeline | Cache + parallel rows + scalable allocator |
|---|---:|---:|---:|
| 9 / 200 | 13 | 7.909 s | 2.806 s |
| 13 / 201 | 14 | 10.481 s | 3.562 s |
| 19 / 202 | 28 | 29.011 s | 9.565 s |
| Total | 55 | **47.402 s** | **15.933 s** |

About **2.98x faster overall**, or **3.45 moves/s** rather than 1.16. This measures
the combined changes, not caching alone. These are individual diagnostic runs,
not statistical averages or a playing-strength improvement. The default one-ONNX-
thread configuration separately finished the same games in 16.203 s.

Final totals: 125 full-policy requests, 432 value-only requests; **435 actual
encoder executions**, **122 cache hits**, 371 LRU evictions. The previous pipeline
encoded all 557 requests, so caching eliminated about 22% of coarse extraction /
encoding. Final cache: 64 entries, **24,920,064 bytes (23.77 MiB)** of spatial data,
plus bounded Game keys/overhead. The hard spatial budget is 32 MiB. This is a host
cache, not retained GPU tensors or 64 copies of 40-MB move matrices.

Feature time: **11.193 s** (coarse 2.454 s, exact rows 8.739 s); encoder **0.486 s**;
dense scorer **2.653 s**. Feature work is still dominant, although much reduced.
Queries remain synchronous; cross-game inference batching is still future work.

```sh
# Defaults are already four feature workers and 64 cached contexts.
mise run neural-selfplay -- --model runs/onnx-initial --games 3 --sizes 9,13,19 --seed 200 --simulations 8 --max-moves 64 --inference-threads 4 --feature-threads 4 --cache-positions 64 --output runs/neural-cached-parallel-ort4
# Uncached serial control; add warmup/iterations for repeated diagnostics.
mise run infer -- --model runs/onnx-initial --sgf FILE --ply 1 --feature-threads 1 --cache-positions 0
```

Use fresh output directories. Raw results: `runs/parallel-features/timing-mimalloc.txt`,
`runs/neural-cached-parallel-ort4/run.log`, and the corresponding SGFs. The default
ONNX-thread run is under `runs/neural-cached-parallel/`. Earlier benchmark sections
are historical pipeline snapshots; allocator/cache defaults now differ.

## Bounded cross-game inference batching — 2026-11-01

Neural self-play now supports concurrent independent games through one bounded
model-owning worker. Both ONNX graphs batch positions of matching resolution;
value-only queries remain encoder-only. Exact move rows are streamed in chunks
(at most 262,144 rows across a batch), instead of allocating one full-domain move
matrix per game. The shared four-thread feature pool handles positions/chunks;
CPU feature extraction and GPU execution still alternate, without pipelining.
See [inference batching](current/inference.md#cross-game-batching) for controls, limits and failure handling.

Same i5-8400 / GTX 1080, CUDA ONNX Runtime 1.20.1, unchanged epoch-2 model and
features. FP32, one ONNX CPU helper thread, four feature workers, 64 cached contexts,
and mimalloc. Eight simulations per move, seeds 200..205, unchanged search settings.
Batch collection waits at most 2 ms; batch-one controls do not wait. All cases use
six games, with either one active game or six. Timings are the CLI's whole-run
wall timer, including thread startup and SGF output, excluding model/runtime load
and compilation. Per-game concurrent latencies are not summed.

| Corpus / move cap | Active games | Batch limit | Moves | Whole run | Moves/s | Native peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|
| Sizes 9/13/19 / 64 | 1 | 1 | 165 | 57.333 s | 2.88 | 536.0 |
| Sizes 9/13/19 / 64 | 6 | 1 | 165 | 54.121 s | 3.05 | 644.7 |
| Sizes 9/13/19 / 64 | 6 | 6 | 165 | 52.678 s | 3.13 | 684.7 |
| Size 9 / 16 | 1 | 1 | 63 | 13.399 s | 4.70 | 518.7 |
| Size 9 / 16 | 6 | 1 | 63 | 11.973 s | 5.26 | 587.2 |
| Size 9 / 16 | 6 | 6 | 63 | 10.753 s | 5.86 | 716.8 |

Within each corpus, **all six SGFs are byte-for-byte identical across all three
configurations**, including visits and RE. Authoritative inspection validated all
36 files. Five games per corpus finished naturally. The mixed-size seed-205 game
hit 64 moves; the size-9 seed-201 game hit 16. Those truncated games have policies
but **no outcome labels**; the completed games supply 101 and 47 eligible positions
respectively. These remain weak-network pipeline/throughput diagnostics, not
playing-strength results.

These individual measurements show about **1.09x mixed-size / 1.25x size-9**
throughput versus sequential execution. Concurrent batch one isolates some benefit
from overlapping independent tree/policy-sampling work with the inference worker.
The additional batching benefit over that concurrent control was only about
**1.03x / 1.11x**, not a multiple-fold gain. Runs were sequential, not randomized
multi-seed repetitions; small differences may be timing noise. Concurrency remains
opt-in, rather than assuming one configuration is universally faster.

The mixed-size batched run reached actual encoder/scorer batch sizes of only **2**:
there were two game slots per resolution and many queries were value-only. It
encoded 1,288 positions in 1,016 encoder calls, versus 1,288 calls for concurrent
batch one. Full policies: 347; value requests: 1,285. Feature time was **43.122 s**
(coarse 10.028 s, exact rows 33.095 s), about **82%** of runtime; encoder 1.103 s,
scorer 6.294 s. Mean worker request latency was **97.4 ms**, including queueing,
collection and execution. It is not the 2-ms collection bound.

The size-9 batched run reached encoder/scorer batches of **6**. It encoded 491
positions in 182 encoder calls; concurrent batch one used 491 calls. Full policies:
123; value requests: 490. Coarse extraction fell from **2.845 s** in the concurrent
control to **1.450 s** with batching, while exact rows stayed essentially unchanged
(**6.402 / 6.519 s**). The shared pool can parallelize coarse extraction across
positions; that, not just GPU batching, contributes to the measured gain. Batched
encoder/scorer time was **0.299 / 2.050 s**. Mean worker request latency: **81.1 ms**.
Features still account for about 74% of total runtime.

RSS is `/usr/bin/time -v` on the **native executable directly**, not the Python/Cargo
launcher. It includes runtime/model initialization and host allocations, but not
GPU VRAM. No instantaneous CUDA peak is claimed here. More game state, batched
contexts and allocator/runtime buffers raise RSS even though exact rows are now
streamed. The 32-MiB cache payload ceiling is not a total process-memory limit.

```sh
mise run neural-selfplay -- --model runs/onnx-initial --games 6 --sizes 9,13,19 --seed 200 --simulations 8 --max-moves 64 --concurrent-games 6 --inference-batch-size 6 --inference-batch-wait-ms 2 --output runs/batch-measure-001
# Concurrent control: --inference-batch-size 1
# Sequential control: --concurrent-games 1 --inference-batch-size 1
# Single-resolution experiment: --sizes 9 --max-moves 16
```

Use fresh output directories for each case. Final measurements and SGFs are under
`runs/batched-inference-final/{mixed,same}-{serial,concurrent-single,batch6}/`, with
native resource/timing reports in each `run.log`. CPU/CUDA integration tests compare
all logits, values and legality rows against single-position execution on nonempty
sizes 9/13/19, including a six-policy batch that exercises the aggregate row cap.
They also cover duplicate contexts, disabled caching, encoder-only batches,
partial batches, terminal errors, worker disconnect/shutdown, stable seeds and
output-error cleanup. All 35 Python tests, shared/local Rust tests and checks pass.

## Replay-corpus warm-start smoke — 2026-11-01

The trainer now has a separate `--warm-start` contract for changed replay corpora;
strict fixed-corpus `--resume` is unchanged. Warm-start copies weights/BatchNorm,
resets SGD momentum by default (optional explicit `keep`), and starts fresh
run/epoch/step/best-loss/RNG state. Checkpoints retain source SHA-256 provenance
and ancestral game-level split assignments, including games absent from the current
window. See [training.md](current/training.md#warm-start-on-a-changing-replay-corpus).

A real CUDA smoke run used the earlier **epoch-2 / step-40** offline checkpoint
as source, with the unchanged 80,099-parameter network and schema-4 features.
Its new corpus was `runs/batched-inference-final/mixed-batch6`: five completed
neural games across sizes 9/13/19, **101 eligible positions**, plus one truncated
SGF skipped for missing outcome. These are different games from the source's
24-game bootstrap corpus, not fixed-corpus continuation.

The new split had three training games (79 eligible positions) and two validation
games (22). Seed 303, batch one, one feature worker, no shuffle buffer, one sampled
position per game, two training steps and one epoch. Thus it optimized only **two
positions** and validated two; this is deliberately a wiring check, not a training
throughput benchmark, strength test or improvement claim.

| Phase | Policy CE | Value MSE | Total | Samples |
|---|---:|---:|---:|---:|
| Train | 11.402985 | 0.728577 | 12.131562 | 2 |
| Validation | 12.793375 | 1.060911 | 13.854285 | 2 |

The result is a **new run at epoch 1 / step 2**, with a fresh `best.pt`; the source
checkpoint's SHA-256 remains
`d4d466ccee0ecfa0c2bf0862a87229e29440eaf1eceb07799c995ac095afcad0`.
The new manifest describes five current games, while `split_history` retains
**29 IDs**: all 24 source assignments plus the five new games. Source run/epoch/
step/hash and optimizer reset policy are embedded in the checkpoint, not sidecars.

The resulting checkpoint exported successfully, with dynamic ONNX/PyTorch maximum
absolute error **1.1920929e-6** across the standard verification shapes. Native
CUDA value inference on a nonempty size-9 position also succeeded. This checks
that warm-start output remains usable by the existing export/inference pipeline.

```sh
mise run train -- --data runs/batched-inference-final/mixed-batch6 --warm-start runs/offline-training/best.pt --output runs/warm-start-smoke --epochs 1 --device cuda --seed 303 --batch-size 1 --workers 1 --shuffle-buffer 0 --steps-per-epoch 2 --positions-per-game 1 --validation-positions-per-game 1
mise run export -- --checkpoint runs/warm-start-smoke/best.pt --output runs/warm-start-smoke-onnx
```

Use fresh output directories. Artifacts are in `runs/warm-start-smoke/` (including
`training.log`) and `runs/warm-start-smoke-onnx/`. CPU tests verify exact initial
weight/BatchNorm copying, empty reset state, retained momentum and hyperparameter
overrides in `keep` mode, new RNG/counters/best tracking, compatibility checks,
legacy checkpoint support, persistent holdouts across replay windows, strict
changed-data resume rejection, and warm-run resume versus uninterrupted training
bit-for-bit. All **40 Python tests**, shared/local Rust tests and checks pass.
Restartable multi-iteration orchestration remains the next milestone at this snapshot.

## Restartable two-iteration neural loop — 2026-11-01

The [synchronous loop supervisor](current/loop.md) now connects native self-play, bounded
SGF replay, warm-start training and verified ONNX export. This is the small-machine
synchronous pattern from KataGo's training guide, without a strength gate: latest
completed epoch weights are promoted automatically. No rules, network feature
schema or policy normalization changes were needed. Rust SGF inspection additionally
exposes root generation metadata and per-move visit totals for handoff validation.

A real GTX 1080/CUDA run started from the same offline epoch-2 best checkpoint
(`d4d466...cad0`, 80,099 parameters) and snapshotted the 24-game bootstrap corpus.
Configuration: six games per iteration, sizes 9/13/19, eight simulations, 64-move
cap, six concurrent games/batch limit six, four exact-feature workers and one ONNX
helper thread. Replay: current/previous iteration plus older bootstrap candidates,
round-robin by size, at most **32 unique labeled games**.

Each iteration used one FP32 epoch, batch one, one feature worker, no shuffle
buffer, at most two training steps, one sampled training/validation position per
game, inherited SGD settings and **kept momentum**. Thus only **four new SGD updates**
were performed across both iterations. Local epoch/step counters restart per warm
run; source lineage and optimizer state are retained separately.

| Iteration | Game seeds | Games / naturally completed | Generated moves | Selected replay games | Train / validation games | Native generation |
|---|---|---:|---:|---:|---:|---:|
| 1 | 200..205 | 6 / 5 | 165 | 29 | 23 / 6 | 53.673 s |
| 2 | 206..211 | 6 / 5 | 258 | 32 | 26 / 6 | 71.294 s |

All 12 generated SGFs were authoritatively validated. One game per iteration hit
the 64-move cap and supplied **no value labels**; the other ten supplied 295 eligible
positions in total. Truncated games remain SGF diagnostics and are excluded from
replay. Archive bootstrap copies are exact snapshots; no game JSON sidecars exist.
These native generation times exclude model loading, training/export and supervisor
setup; trajectories/models differ between iterations, so they are not a controlled
speedup comparison or an end-to-end training throughput benchmark.

The model handoff was checked by immutable checkpoint hashes:

```text
initial:     d4d466ccee0ecfa0c2bf0862a87229e29440eaf1eceb07799c995ac095afcad0
iteration 1: 8bff852e17ada6b6abf129b5d1d94f4b4bd0681e40bc9f01731c05bd2ab3d3f1
iteration 2: 328c0402e366016c2642b5a3544e799c6cf4b004b50d612f28d95b6d38b64f84
```

Iteration-2 SGFs identify the iteration-1 checkpoint, not the initial model. Both
new checkpoints are local epoch 1 / step 2; their provenance points to the correct
parent. Each encoder/scorer bundle passed export parity and metadata validation.
The holdout ledger grew from 24 to **29 then 34** canonical IDs even though the
active replay corpus was capped at 32. Known training games were never reassigned
to validation. Loss values remain weak-data diagnostics, not evidence of stronger
play or an optimal training/data ratio.

The run was first stopped normally after iteration 1, then restarted with
`--resume --iterations 2`. A further idempotent `--resume` verified and reused all
committed games/checkpoints/models without launching another stage. Artifacts and
stage logs are under `runs/loop-smoke/`; reproduce with the small command in
[loop.md](current/loop.md), using a fresh output directory.

Fault-injection tests additionally cover partial self-play, publication gaps after
completed epochs and atomic export renames, failed first epochs, input snapshot
recovery, immutable artifact checks, configuration conflicts and inherited writer
locks. A **real CPU integration test** sends SIGINT to an active supervisor after
native SGFs appear, verifies child cleanup, resumes without overwriting partial or
completed files, and completes two iterations across 9/13/19. That test uses a tiny
pass-biased network/reduced widening for fast plumbing verification, not strength.
All **53 Python tests**, shared/local Rust tests and checks pass. The next milestone
was fixed-budget playing-strength evaluation at this snapshot, not larger loop runs or networks.

## 40/40/20 sampling, acceptance gate and durable recovery — 2026-11-01

New defaults emphasize size 9 and 13: `--size-weights 9:2,13:2,19:1` drives
self-play, active replay selection, actual training/validation positions and
paired-color evaluation. This reduces expensive size-19 **extraction work**, not
merely its loss coefficient. Historical sections used literal/equal size cycling;
add `--size-weights uniform` to reproduce those older schedules. Shape/parity tests
still cover all resolutions.

A real CUDA smoke forked from the previous loop's latest checkpoint (`328c0402...`),
preserving its optimizer state, rather than restarting from random or rolling back
to the original bootstrap model. Configuration: five generation games, seeds
400..404, sizes 9/13/19, eight simulations, 32-move generation cap, five concurrent
games/batch limit five, 20-game replay cap, one epoch, five training batches of one,
one feature worker, one sampled position per game, and kept SGD momentum.

| Work | Size 9 | Size 13 | Size 19 |
|---|---:|---:|---:|
| Generated games | 2 | 2 | 1 |
| Actually optimized positions | 2 | 2 | 1 |
| Actually validated positions | 2 | 2 | 1 |
| Evaluation games (color-paired) | 4 | 4 | 2 |

Generation produced **110 moves**: three naturally completed games (46 eligible
positions) and **two truncated size-13 games** at the 32-move cap. All truncations
were explicitly reported and excluded from replay. The frozen replay corpus had
20 unique labeled games (15 train / 5 validation). Train policy CE/value MSE were
12.725578 / 1.075501; validation 12.536001 / 1.125330. These are five-update wiring
measurements with changed data/weights, not evidence of strength or a controlled
throughput comparison.

The acceptance smoke deliberately used only **eight moves per game** and eight
simulations, to exercise two-model native CUDA matching cheaply. **All ten matches
truncated** (zero draws, zero completed outcomes), and warnings named every SGF.
The result was correctly **inconclusive**: no invented draw scores and no promotion.
The accepted model stayed at `328c0402...`; the trained candidate
`2996db3124c7ff7fcfc0629b528a2cd4e5e9067fe1933a37b9a6ce520c912173` was retained
with full optimizer/RNG/corpus state. Production defaults allow 512 moves, but
budgets still need calibration rather than claiming this smoke is an acceptance win.

Artifacts are under `runs/weighted-gate-smoke/`. Durable bundles are outside the
repository at `../continugo-models/`: accepted ONNX + full checkpoint/replay packs
and a separate latest-training pack, including this inconclusive candidate. The
three previously auto-accepted models (`d4d466...`, `8bff85...`, `328c04...`) were
also preserved as historical baselines; they were not retroactively gate-qualified.
SGF blobs are independent copies, shared only between store-owned recovery packs.
Bundles are hashed, atomically published, read-only and never automatically pruned.

Tests exercise default native/Python schedules, early-step training sample ratios,
without-replacement downsampling of surplus size-19 positions, pairing/color/model
identity, draw warnings, unknown-outcome bounds, rejection/inconclusive separation
of training and self-play state, partial gate recovery without retraining, foreign
store rejection, corruption checks and lost-run recovery. A real CPU gate uses a
deliberately pass-biased tiny network: its ten genuine draws are loudly reported,
not treated as a strength result. Its archived checkpoint and relocated SGFs then
continue training **bit-for-bit against the original corpus after the original run
is deleted**, including SGD momentum. See [evaluation.md](current/evaluation.md),
[model-store.md](current/model-store.md) and [loop.md](current/loop.md) for controls and limitations.
All **66 Python tests**, shared/local Rust tests and checks pass, including CPU/CUDA
inference parity, paired-gate recovery and bit-for-bit archive continuation.
