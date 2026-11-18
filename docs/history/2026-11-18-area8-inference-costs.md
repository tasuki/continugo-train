# Schema-6 area integration at 8x coarser precision — 2026-11-18

Explicit user approval: adopt 8x coarser midpoint integration without changing
schema 6, then measure new self-play costs. No per-group cache or runtime option
was added. Investigation artifacts: `scratch/20261118-area-approximation/`;
adoption/benchmark artifacts: `scratch/20261118-area8-selfplay/`.

## Scope / compatibility

Only schema-6 coarse group-area inputs use the new precision. The capsule union,
log-scaling/cap, analytic empty/single-span cases, channel layout and field names
are unchanged. Each sample uses the existing lower anchor and eight times the
original spacing; the final partial slice uses its own midpoint and height.
Legacy schema 4, schema 5, diagnostic/display area, exact spans, move legality,
exact move fields and final-game targets are unchanged. No new schema number,
checkpoint conversion, retraining or model export is required by this approved
numerical-precision change. Future extraction and inference use the same schema-6
implementation. Matching derived feature sidecars must be rebuilt with the new
extractor where needed; extractor hashes prevent silently accepting stale caches.

The original precise integral is retained for legacy/accuracy use, not as a
schema-6 opt-in alternative. Both precisions share the same union implementation
with compile-time specialization. Frozen cg10 binaries/caches/loop state were
not modified or resumed; the campaign remains paused after round 22.

## Regression coverage

`tests/fixtures/liberty-area8.json` freezes 54 scalar cases from the measured
scratch experiment (empty, isolated, overlapping/disjoint/touching and separated
pockets across radii 26/38/55). Tests verify exact approved 8x values and preserve
the old precise values bit-for-bit, analytic cases and integer translations.

The original historical-schema fixture is unchanged. Fifteen schema-6 packet
cases explicitly record approved coarse hashes, their old precise hashes, and
hashes of all twelve unaffected planes. Exact fields/legality/ownership remain
checked against the historical hashes. Schemas 4/5 remain fully bit-exact against
Git history. The per-group-versus-colour-union integration test permits the
measured normalized approximation difference instead of incorrectly requiring
equality to the legacy precise scalar.

Validation: complete Rust suite passed (including new scalar regression); 41
Python tests in `test_schema_compatibility`, `test_inference`, and `test_loop`
passed; native CUDA batching/parity/lifecycle also passed on the actual cg10
schema-6 model. Commands/logs and fixture-generation provenance are in scratch.

## Matched self-play benchmark

Before binary: revision `1590091` (pipelining plus direct ready-buffer gathering),
precise scalar integration. After: same code plus the schema-6 8x input precision.
Same frozen cg10 round-22 model, GTX 1080, i5-8400 (six cores), FP32 CUDA,
32 active games / batch cap 32, six shared feature workers, one ONNX host thread,
64 context-cache entries, 2 ms collection wait. Sequence: before, after, after,
before; no concurrent training/evaluation/compilation during timed runs.

Each case requests 64 size-9 games, seed 221000, max 32 moves, 32/128 search caps
with 25% expensive searches, detailed profiling on, training-cache creation off.
This is a reduced-budget early-game pilot, **not full-campaign throughput or
playing-strength evidence**.

| Case | Played moves | Naturally completed games | Wall s | Moves/s |
|---|---:|---:|---:|---:|
| Before, first | 2048 | 0 | 145.561 | 14.070 |
| After, first | 2019 | 1 | 120.160 | 16.803 |
| After, repeat | 2019 | 1 | 120.169 | 16.801 |
| Before, repeat | 2048 | 0 | 145.442 | 14.081 |

Mean throughput: **14.075 -> 16.802 played moves/s (+19.4%)**. Mean whole-job
wall time: **145.502 -> 120.164 s (-17.4%)**. Input precision intentionally changes
trajectories (all 64 move sequences differ); the after runs include one game that
ends before the cap. Therefore these are not identical-position executions.
Encoded-position throughput also improves **19.2%** (937.8 -> 1117.7/s), and
policy evaluations/s improve about 19%, corroborating that the gain is not merely
fewer played moves. After repeats have 63/64 identical move sequences.

## New costs, means of two runs

| Counter | Before | Area8 |
|---|---:|---:|
| Coarse feature elapsed bucket | 126.661 s | 73.987 s |
| Encoder calls, including host output handling | 66.483 s | 64.513 s |
| Exact move-feature work | 22.291 s | 15.949 s |
| Geometry proposals | 15.107 s | 14.214 s |
| Policy scorer calls/handling | 3.273 s | 3.241 s |
| Scalar-area summed worker clocks | 278.309 s | 33.904 s |
| Area cost per group | 689.0 us | 85.3 us |
| Average busy CPU cores | 4.235 | 3.335 |
| Mean sampled GPU utilization | 39.7% | 48.0% |
| Peak RSS, mean | 617.4 MiB | 624.0 MiB |

Stage elapsed times overlap and **must not be summed as wall-time shares**.
Scalar-area clocks are summed across workers, not an additional elapsed stage.
Per-group area work is **8.08x faster**. Exact move work also benefits from less
feature-pool contention; encoder/scorer work is essentially unchanged.

The largest remaining coarse-worker costs are now:

| Component | Share of summed top-level coarse-worker clocks |
|---|---:|
| Liberty-footprint rasterization | 37.5% |
| Group-size/group-area canvas reductions | 19.9% |
| Scalar area integration | 12.3% (formerly 51.5%) |
| Shared shape painting | 8.8% |
| Exact liberty spans | 7.5% |

The integration bottleneck has largely moved out of the way. More time is now
paced by the encoder/execution stage, which still performs CPU policy work and
bookkeeping between GPU calls; higher GPU duty cycle does not mean full GPU
saturation. Lower CPU occupancy reflects work removed, not a performance loss.

## Reproduction / provenance

`benchmark.py`, `summary.json`, per-case commands/stats, GPU samples, `time.txt`,
SGFs, before/after executables, test logs, source and binary fingerprints,
`freeze_regressions.py` are preserved under the adoption scratch directory.
`summary.json` includes per-stage and per-group costs and the individual runs.
The prior approximation reports contain raw/normalized input errors, small-pocket
stress cases, fixed-checkpoint perturbations and the 16x/32x comparison.
