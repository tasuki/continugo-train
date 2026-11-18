# Exact liberty-footprint optimization — 2026-11-18

The user asked to speed up footprint rasterization after area8 made it the
largest coarse-worker component. Default implementation now uses two exact
optimizations; no schema, sampling, model, move-rule or target changes:

1. Update the endpoints directly when an incoming interval overlaps the current
   single disjoint interval. Avoid repeated partitioning/Vec draining in this
   common case, as the scalar integrator already does.
2. Precompute circle extensions for the three default density-4 grids
   (radius/resolution/samples: 55/38/8, 38/54/8, 26/78/8). Three immutable tables
   together use less than 1 MiB. Actual sample-axis equality is checked; other
   axes, dimensions, densities, radii or source-coordinate ranges retain the
   original square-root expression. Tables are initialized from the canonical
   raster, not from arbitrary first-caller geometry.

These are **not per-group/per-position memoization caches**. There is no cache
invalidation bookkeeping and no option/alternate runtime scheduler. Source spans,
8x8 footprint midpoint samples, interval-union meaning and final normalization
are unchanged. Schema-6 scalar area keeps the previously approved area8 precision.

Code: `src/features/spatial/footprint_raster.rs`. A frozen original streaming
rasterizer/union is retained only under cfg(test), plus the independent original
canvas oracle. Tests check every lookup-table value's f64 bits, table-storage
bounds, altered axes, exterior source coordinates, fragmented/empty/singleton
spans, exact touching/boundary samples, duplicate axes/zero-width pixels, and
ordinary through 2048-wide grids. Legacy schema/history hashes and approved
schema-6 area8 hashes remain unchanged.

## Scratch experiments

Artifacts: `scratch/20261118-footprint-exact/`. Base revision `6b8a42c` (area8).
The paired microbenchmark reuses the area-approximation corpus: 240 positions
from 48 SGFs, sizes 9/13/19; 12 alternating-order paired repetitions per position,
real per-colour unioned liberty spans, same grids, bit-equal outputs on every pair.
Larger-board samples remain early-game positions, not dense-endgame coverage.

A conservative hierarchical source-envelope index was tried first. It preserved
bits but was slower: speedups .84x/.74x/.73x at sizes 9/13/19. It was rejected;
source/results are archived but the index is not in production.

| Candidate | Size 9 speedup | Size 13 | Size 19 |
|---|---:|---:|---:|
| Single-interval fast path only | 1.15x | 1.08x | 1.09x |
| Fast path + immutable geometry tables | 1.32x | 1.22x | 1.21x |

Combined size-9 footprint time: .485 -> .367 ms per position (both colours).
This is footprint-only timing, not whole feature extraction or self-play.

## End-to-end matched self-play

Same frozen cg10 round-22 model, GTX 1080, six-core i5-8400, 32 active games /
batch cap 32, six feature workers, one ONNX host thread, 64 context-cache entries,
2 ms collection wait. Before uses source area8; after also has the exact footprint
optimizations. Sequence before/after/after/before, no concurrent compilation,
training or evaluation. Each case: 64 size-9 games, seed 221000, 32-move cap,
32/128 search caps with 25% expensive searches, profiling on, no training-cache
creation. Both variants play 2,019 moves and complete one game before the cap.

| Case | Wall s | Moves/s |
|---|---:|---:|
| Before, first | 120.316 | 16.781 |
| After, first | 118.735 | 17.004 |
| After, repeat | 118.872 | 16.985 |
| Before, repeat | 121.426 | 16.627 |

Means: **16.704 -> 16.995 moves/s (+1.7%)**, 120.871 -> 118.803 s (-1.7%).
Control drift is about .9%, so describe this as a **roughly 1–2% pilot gain**, not
a precise full-campaign guarantee. Encoded-position throughput also rises 1.7%.
Both after runs' SGFs are 64/64 byte-identical to the first before run; the last
before run differs in one game (existing batch-timing/floating-point sensitivity).

| Mean cost | Before | After |
|---|---:|---:|
| Footprint summed worker clocks | 103.770 s | 76.876 s |
| Coarse feature elapsed bucket | 74.363 s | 67.664 s |
| Encoder calls/host handling | 64.667 s | 64.619 s |
| Exact move features | 16.172 s | 14.849 s |
| Geometry proposals | 14.265 s | 14.546 s |
| Policy scorer | 3.278 s | 3.256 s |
| Average busy CPU cores | 3.32 | 3.17 |
| Sampled GPU utilization | 47.6% | 48.3% |
| Mean peak RSS | 609.8 MiB | 621.1 MiB |

Worker footprint work is **1.35x faster / 26% less worker elapsed time**, and the
coarse bucket falls 9%, but most preparation overlaps the execution stage. Stage
clocks are not additive wall shares. The small whole-run gain despite a clear
component speedup shows that further coarse optimization is not the main lever
for this workload. The serial model-owning execution stage still combines
encoder/scorer calls with CPU proposal/exact-feature work and bookkeeping.
Remaining major opportunities should be profiled there rather than inferred
from coarse-worker percentages alone. Lower CPU usage reflects work removed.

This remains a reduced-budget early-game throughput pilot, not playing-strength
or full-campaign evidence. Existing frozen cg10 runtime/data are untouched and
its campaign remains paused after round 22.

## Validation / provenance

- Complete Rust suite passed after adoption; immutable table checks also enforce
  less than 1 MiB total geometry-table storage.
- 41 Python schema compatibility, CPU/CUDA inference and loop tests passed,
  including real interruption/resume and output-failure lifecycle coverage.
- Native mixed-resolution CUDA parity/lifecycle passed on real cg10/schema 6.
- 240-position microbenchmark and randomized/boundary tests retain exact bits;
  no numerical-precision fixture was loosened or regenerated.

Production canonical table construction adds only a one-time canonical-axis
build to the measured prototype, whose actual default input axes were identical;
hot-path table contents and sampled outputs are unchanged. Before/after archived
executables, commands, source patches/fingerprints, GPU samples, time/RSS files,
SGFs, raw microbenchmarks and test logs are in the scratch directory. Scratch
worktree Cargo path overrides and the rejected index are not production changes.
