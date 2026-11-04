# Coarse-feature profiling and area-union optimization — 2026-11-04

Follow-up to the [representative pipeline diagnostic](2026-11-04-representative-pipeline.md).
This is a throughput diagnostic with **random weights**, not strength evidence.

## Controlled change

Added opt-in coarse stage clocks to native inference, including value-only cache
misses. Preparation, additional group construction, shared group analysis, area
integration, and painting are disjoint subtotals of coarse extraction. The shared
rules API combines exact spans and one-move capturability; that split is **still
open**, rather than estimated by subtracting separate benchmark runs.

The baseline showed area integration taking **54.32 of 58.69 seconds** of coarse
features in six actual neural games. Preparation took 0.123 s, additional group
construction 0.045 s, spans + capturability 3.493 s, and painting 0.351 s. Unlisted
coarse time includes allocations, flattening, bookkeeping and instrumentation.

Changed only numerical area union construction: maintain a sorted disjoint union
of active capsule intervals instead of allocating an interval for every active
span and sorting all of them on every integration slice. Reuse the interval vector
between slices and calculate each row's round-end extension once per slice. The
union usually has very few components despite many overlapping source spans.
No sampling-density reduction, precision change, rules fork, move-analysis change,
context-cache change, cell-budget change, or additional concurrency was introduced.

Remaining repeated work: preparation and raster extraction independently build
groups; exact span construction scans/sorts stone blockers on each row for each
group. Capturability also scans blockers and validates candidate captures. The
shared group-analysis stage still allocates spans per group. Those are candidates
for subsequent measurement, not reasons to change shared rules speculatively.

## Fixed populated positions

Same size-9/13/19 source games and early/middle/late plies as the preceding report.
CPU area-only measurements used two warmups and 20 measured queries. Native CUDA
pipeline measurements below used one warmup and 20 queries, with context caching
disabled, six feature threads and one ONNX CPU helper thread. Runs were sequential;
compilation, runtime loading and SGF replay are outside query times.

Mean milliseconds; coarse columns are policy queries. Value-only misses perform
the same coarse work; their **whole-query** times are shown separately.

| Size | Phase / ply | Coarse before | Coarse after | Value query before | Value query after |
|---|---|---:|---:|---:|---:|
| 9 | early / 7 | 8.67 | 4.44 | 9.28 | 5.11 |
| 9 | middle / 39 | 6.47 | 4.29 | 7.05 | 4.94 |
| 9 | late / 79 | 1.46 | 1.25 | 2.14 | 1.92 |
| 13 | early / 20 | 8.27 | 4.66 | 8.96 | 5.45 |
| 13 | middle / 100 | 5.70 | 5.14 | 6.35 | 5.80 |
| 13 | late / 200 | 1.84 | 1.73 | 2.52 | 2.44 |
| 19 | early / 43 | 18.66 | 11.27 | 19.46 | 12.00 |
| 19 | middle / 215 | 12.94 | 12.06 | 13.84 | 12.85 |
| 19 | late / 430 | 3.80 | 3.62 | 4.61 | 4.45 |

For the CPU size-19 early snapshot, area alone fell from 17.71 to 10.40 ms.
Middle/late boards have a different mix: size-19 middle shared spans +
capturability cost about 7.7 ms before the change. This optimization does not
eliminate that cost or justify assuming every position gets the opening speedup.

## Actual self-play

Same i5-8400 / GTX 1080, CUDA ONNX Runtime 1.20.1, FP32, unchanged checkpoint
`30d363f9ac21bfa5dcba9de410b8497be4b70ec7cbaa4806b2eb27c4f696325a`.
Same six games, seeds 800–805, eight simulations, cap 512, one active game, default
64-position / 32 MiB context cache, top 32 + eight exploration cells. Both builds
include the same coarse instrumentation; the baseline retains the old area union.

| Metric | Before | After |
|---|---:|---:|
| Generation/output wall time | 66.400 s | 44.571 s |
| Moves/s | 7.44 | 11.08 |
| Coarse features | 58.690 s | 37.236 s |
| Area integration subtotal | 54.319 s | 32.185 s |
| Move features/selection | 0.915 s | 0.909 s |
| Encoder | 3.626 s | 3.505 s |
| Scorer | 0.331 s | 0.364 s |
| Native lifetime peak RSS | 517.9 MiB | 520.0 MiB |

**All six SGFs are byte-identical**, including decoded moves, visits and scored
results: 494 moves, six natural completions, no truncations. Both runs have 993
policy queries, 3,877 value queries, 3,883 extracted/encoded positions, 987 cache
hits and 9,369,306 exact coordinate rows. This is **1.49× observed throughput**
for this fixed workload, not a general speedup or a strength result. No memory
improvement is claimed; lifetime RSS includes model load/replay/warmup, and does
not account for native GPU peak memory.

## Training and differential checks

Fixed original six-game genuine-policy corpus, unchanged game holdout, sample
plan, weights and optimizer configuration. Two feature workers, batch two, one
PyTorch CPU thread, CUDA FP32. Each ordinary run uses 20 untimed warmup steps,
then restores weights, BatchNorm buffers, RNG and a fresh optimizer. A phase still
opens new workers: worker startup is included, not fully persistent-loader time.

Three additional alternating before/after measurements, each 120 SGD steps and
237 positions (47/67/123 at sizes 9/13/19):

| Median ordinary training metric | Before | After |
|---|---:|---:|
| Phase wall time | 3.052 s | 2.097 s |
| Positions/s | 77.66 | 113.00 |
| Exposed loader wait | 1.991 s | 1.013 s |
| Compute + transfer | 1.055 s | 1.069 s |

Wall-time ranges were 3.051–3.074 s before and 2.094–2.111 s after. CUDA loss
trajectories differ between repeats **even within the same build** (before losses
13.208–13.270; after 13.213–13.568). Do not interpret these timing diagnostics as
identical optimizer trajectories or improved learning. No checkpoints were saved.
The separate synchronized stage runs also retained the same board mix.

Correctness evidence:

- All fields/tensors for **all 494 eligible policy positions** in the fixed corpus
  match exactly between the two feature-server binaries, including pass-only
  targets, coarse planes, exact facts, borders, legality, visits and outcomes.
- Complete binary feature packets on all nine populated fixed snapshots match
  byte-for-byte; diagnostic native values, selected cells and queried logits match.
- A frozen independent sort-and-merge area oracle matches the new algorithm's
  **f64 bits** on randomized fragmented, nested, duplicate-row and bridging spans,
  eight radii, and both base/doubled slice density. Real group areas on all nine
  snapshots also match that oracle's bits.
- Doubled-density numerical audit over those snapshots: maximum absolute change
  0.003086 stone-discs, maximum relative change 0.000283, within the existing
  accuracy criterion. Integration density and f64 precision are unchanged.
- Existing capture, suicide, touching-connection, side-to-move and border tests
  pass. New tests cover profiling parity/counts, warmup exclusion and cache hits
  doing no coarse work, for both policy and value queries. Parallel batched
  profiling counts unique extractions, not duplicate requests.
- Validation: 58 release Rust tests and all 124 Python tests passed;
  `cargo clippy --all-targets --locked -- -D warnings` and Ruff checks passed.

The round trip is still not split into extraction/serialization/transport/decode;
Rust child RSS and native GPU memory remain unaccounted for. These 2–3 second
training windows are longer than the previous sub-second diagnostic but not a
sustained campaign. Broader seeds/tactics and a trained actor remain necessary.

## Artifacts and reproduction

Git-ignored artifacts:

- `runs/20261104-coarse-split/`: saved before/after release binaries, original area
  source, CPU fixed-position JSON, packet hashes, numerical audits,
  `repeat_training.py`, and three-repeat training results.
- `runs/20261104-coarse-before-cuda/` and `runs/20261104-coarse-after-cuda/`:
  full pipeline summaries, binary/model/source hashes, self-play SGFs and raw logs.

Each full pipeline run used the corresponding saved release binaries and:

```sh
.venv/bin/python -m continugo_train.pipeline_profile \
  --positions runs/20261104-representative-positions \
  --model runs/20261104-000500-cell-policy-bench/selective \
  --checkpoint runs/20261104-000500-cell-policy-bench/selective.pt \
  --data runs/20261104-representative-cpu/selfplay --device cuda \
  --iterations 20 --steps 120 --positions-per-game 0 --warmup-steps 20 \
  --output runs/NEW-OUTPUT
```

Build release binaries first; do not run concurrent benchmarks. The ordinary
production build was restored after baseline measurement. The reusable
[profiling guide](../current/profiling.md) explains stage overlap and memory limits.
