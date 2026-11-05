# Benson versus cached eye analysis — 2026-11-05

Added an opt-in benchmark in `src/action/analysis_bench.rs` in **continugo-train**.
It calls the shared Benson implementation without changing the sibling repository.
No feature planes, rules, eye filters, or selection behavior were changed.

## Method

Intel Core i5-8400, Rust 1.99.1, release libtest build, system allocator, one test
thread. This does **not** use the production executable's mimalloc. No GPU/model
or competing benchmark. Reused the dense, naturally completed random games from
`runs/20261104-representative-positions/`, at the same nine early/middle/late
snapshots used by the pipeline profiler. These phase names are move-count proxies,
not a claim of tactical diversity. There is only one source game per size.

Five samples per metric; measured batches last at least 100 ms. Each batch
reports elapsed time / iterations. Calibration batches and warmups are excluded.
All analyses are freshly constructed and dropped for each operation. Repeated
positions do not retain pocket or Benson results. Source replay, availability
index construction, eye-trace recovery, query-vector construction, and JSON
serialization are outside the timers. Availability is rebuilt once from each
snapshot; list order is not reconstructed from the original game's incremental
index. The fixed RNG seeds below refer to this rebuilt index, not the source
game's original random decisions.

Four distinct workloads:

1. **Benson:** complete extraction for both colors, including allocation, groups,
   geometry, pruning and destruction. Excludes feature-plane painting.
2. **Eye trace:** only the cached eye checks actually requested by the production
   random selector, for the side to move, after legal/capture filtering. Recover
   each trace from the profiled selector's proposal prefix; assert that query and
   rejection counts agree. Time the trace again without rules or per-query clocks.
3. **Random selector:** actual `random_move::<false>`, including rules preparation,
   candidate validation, capture exemption and eye checks. Reuses the existing
   geometric index; excludes index updates, committing the chosen move and passing.
4. **Full eye scan:** cached eye checks over every geometrically available center,
   for both colors, including suicidal moves. Same workload as the earlier
   `benchmark_eye_cache` diagnostic, **not ordinary random selection**.

Eye-trace and random-selector samples use seeds 0–4, one fixed seed per sample.
Thus their sample variation includes candidate-order differences, not just clock
noise. Full scans use fixed availability-list order and one fresh shared pocket
cache per complete two-color scan. Eye-trace time is inside random-selector time;
these columns must not be added. Medians need not refer to the same seed.

## Results

Medians. Benson, selector and scan columns are **milliseconds**; eye traces are
**microseconds**. Full raw samples and iteration counts are retained in JSON.

| Size | Phase / ply | Stones | Benson, both colors (ms) | Actual eye trace (µs) | Random selector (ms) | Full eye scan (ms) |
|---|---|---:|---:|---:|---:|---:|
| 9 | early / 7 | 7 | 5.175 | 0.032 | 0.0016 | 5252.643 |
| 9 | middle / 39 | 38 | 6.681 | 4.180 | 0.0699 | 779.208 |
| 9 | late / 79 | 51 | 4.713 | 2449.827 | 51.5005 | 9.386 |
| 13 | early / 20 | 20 | 4.786 | 0.074 | 0.0041 | 3732.431 |
| 13 | middle / 100 | 94 | 6.876 | 478.730 | 0.9276 | 640.321 |
| 13 | late / 200 | 111 | 4.616 | 5892.414 | 23.0172 | 7.406 |
| 19 | early / 43 | 43 | 4.304 | 0.136 | 0.0120 | 3591.120 |
| 19 | middle / 215 | 200 | 7.988 | 0.597 | 0.7602 | 1084.426 |
| 19 | late / 430 | 239 | 4.961 | 5377.748 | 32.5862 | 8.750 |

The five Benson samples have min–max ranges of 5.148–5.262, 6.627–6.738,
4.683–4.719, 4.781–4.795, 6.853–7.119, 4.608–4.650, 4.284–4.346,
7.961–8.117, and 4.881–5.011 ms respectively, in table order.

### Work actually requested

| Size | Phase | Geometric centers | Eye queries per random decision | Result |
|---|---|---:|---:|---|
| 9 | early | 547,576 | 1 | placement |
| 9 | middle | 16,473 | 1 | placement |
| 9 | late | 9,307 | 3,140 | exhausted; every legal candidate rejected as an eye fill |
| 13 | early | 554,799 | 1 | placement |
| 13 | middle | 9,369 | 1–3 | placement |
| 13 | late | 7,482 | 6,285 | exhausted; every legal candidate rejected as an eye fill |
| 19 | early | 574,694 | 1 | placement |
| 19 | middle | 11,018 | 1 | placement |
| 19 | late | 5,681 | 3,903 | exhausted; every legal candidate rejected as an eye fill |

Full scans ask twice the geometric-center count, including illegal coordinates.
One eye question can still vary greatly by candidate: the size-19 middle trace
ranges from 0.591 to 214.237 µs across the five seeds despite always asking one
question. Use the raw samples, not just the median, for such order-sensitive work.

Benson certifies no stones on these early/middle snapshots; this means **not
certified**, not dead. In the late snapshots it certifies all surviving stones:
Black/White counts are 22/29, 56/55, and 123/116. These dense late positions are
especially favorable examples of unconditional-life detection.

## Interpretation

- **A both-color alive feature currently costs about 4–8 ms per position**, before
  raster painting. Cost is not monotonic in stone count: the middle positions
  are the slowest in this suite. This is a meaningful CPU feature-extraction cost,
  not a free extra plane.
- Ordinary early/middle random decisions ask only 1–3 eye questions here. Their
  eye work is generally much cheaper than computing Benson for the whole board.
- At exhausted endgames, actual eye-only work costs about **2.45–5.89 ms**; Benson
  is in the same range at **4.62–4.96 ms**, while full move selection takes
  **23–52 ms**, largely beyond the eye checks themselves.
- Benson beats exhaustive two-color eye scans, sometimes by hundreds of times,
  but that is **not a demonstrated self-play speedup**. The two analyses have
  different outputs, and full scanning is not the normal selector workload.
- No strength, policy-equivalence, neural end-to-end throughput or allocator
  improvement is claimed. Recheck more games and real inference integration
  before deciding whether any optimization is worthwhile.

## Reproduction and provenance

```sh
export CONTINUGO_BENCH_SGFS="runs/20261104-representative-positions/game-000001-size9-seed0.sgf:runs/20261104-representative-positions/game-000002-size13-seed1.sgf:runs/20261104-representative-positions/game-000003-size19-seed2.sgf"
CONTINUGO_BENCH_OUTPUT="runs/$(date -u +%Y%m%d-%H%M%S)-benson/results.json" \
  CONTINUGO_BENCH_SAMPLES=5 CONTINUGO_BENCH_MIN_MS=100 \
  CONTINUGO_BENCH_FULL_SCAN=1 mise run analysis-bench
```

Omit `CONTINUGO_BENCH_FULL_SCAN=1` for a faster practical-workload benchmark.
The implementation, normal median-summary test, task and documentation all live
in this repo. The sibling was only read/built; no benchmark code was added there.

Raw artifacts: `runs/20261105-122746-benson/{results.json,benchmark.log,environment.txt}`.
The complete measured test took 211.04 s, excluding compilation. An earlier
one-sample smoke run is diagnostic only and is not included in this table.

Train base revision at measurement: `e40fe99e2f1e33760f2df559cb54e460a86c16d3`.
Shared base revision: `4fc0653f52e4dadc53b06ca313e3b7d39e84de16`, with the staged
Benson implementation recorded by source hashes and dirty status in the JSON.
The revision alone therefore does not identify the measured shared code.

Rust source-tree SHA-256 (sorted relative `.rs` paths and contents, NUL-delimited):

- Train: `a5f7c2942e8b8d3c67180e91d0d8dff3957f7a1b68cd662b7e52a4126c37d36a`
- Analysis: `c5ce62039d4f21d7d2b34ed1691106e267dfd113d902d8753403cfc2c879286e`
- Rules: `3b79374ef783248b19a80ba2d5617bb281bbaa7cd9f556ae51a2567516208a6a`

SGF SHA-256, sizes 9/13/19 respectively:

- `3d69b3a6b2b8efed454d462b09be1b8c0dfa27448ca1893454e1d2cb4f574001`
- `30074d2fac4d63c592d4d017d222675229723ed7353c150ea1d6e3185dce4d7f`
- `09a0584b6943c70b8c4bb361587d9cef47931c49abcf86aa7ee11df3f27683f0`

Validation: `cargo test --locked --release`, `cargo fmt --check`, and
`cargo clippy --locked --all-targets -- -D warnings` pass.
