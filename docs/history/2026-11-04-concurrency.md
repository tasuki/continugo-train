# Cross-game concurrency and batching — 2026-11-04

Settings-only experiment following [coarse extraction optimization](2026-11-04-group-geometry.md).
No inference/search algorithm, model, search budget or cache limit changed.
Measurements used the existing scheduler and original defaults; the user
subsequently requested adoption of the selected settings as defaults.

## Result

The measured recipe, subsequently selected as the default, is
**12 concurrent games, maximum batch 12, wait 2 ms**.
On the fixed workload, repeated median throughput is **31.42 moves/s**, versus:

- **13.92 moves/s sequentially: 2.26× throughput**;
- **22.22 moves/s with six games / batch eight / wait two: 1.41× throughput**.

The latter uses the loop's existing concurrency settings, but the *same eight-
simulation benchmark workload*, not the loop's default 32-simulation budget.
This is a self-play generation gain, **not** a measured 2.26× improvement to SGD
or to the complete generate/train/evaluate loop. No playing-strength claim.

Every one of the **14 measured runs** generated the same **18 byte-identical
SGFs / 1,507 moves**, including root visits and scored results. All 18 games
naturally completed, none truncated. Game indices/seeds/filenames stayed fixed;
only completion order changed. Rust inspection validated every generated record,
including policy legality, visit totals and authoritative scored completion.
The 11 existing native inference integration tests also pass, including CPU/CUDA
batch parity, concurrent-game replay and worker lifecycle checks.

## Controls and workload

Sequential measurements on the same i5-8400 / GTX 1080, CUDA ONNX Runtime 1.20.1,
FP32, random-weight actor. Unchanged checkpoint SHA-256:
`30d363f9ac21bfa5dcba9de410b8497be4b70ec7cbaa4806b2eb27c4f696325a`.
Same 32-channel/four-block model, top 32 + eight exploration cells, seeds
800–817, uniform 9/13/19 size cycle, eight MCTS simulations, cap 512, six native
feature threads, one ONNX CPU helper thread, 64-position / 32 MiB context cache.
Training feature workers are unrelated and unchanged.

Native binary SHA-256, identical across all runs:
`611143c2140eadcf455f00b43c4454609c36b91267c173f1ab0238f5343ae43d`.
Source revision `1b50353d5d4f7799d7387440660cd1d30c6dab7d`, with the shared-rule
changes from the preceding report present. Reports retain both repositories'
revision/dirty status, model hashes, native hash and complete native statistics.

Times below are game generation/output wall time, excluding model load and record
inspection, including initial inference warmup. Native process wall/RSS and
sampled driver GPU memory were measured separately using the existing profiler
helper. All benchmarks ran sequentially, not alongside training/other benchmarks.

## 1. Concurrency sweep

Only active-game count varied; batch limit eight and wait two ms stayed fixed.
The scheduler caps the effective batch limit by active slots. A single game uses
the sequential path without a queue/collection wait.

| Active games | Wall time | Moves/s | Native lifetime peak RSS | Sampled GPU peak |
|---:|---:|---:|---:|---:|
| 1 | 108.596 s | 13.88 | 523.3 MiB | 204 MiB |
| 2 | 106.249 s | 14.18 | 527.1 MiB | 248 MiB |
| 3 | 96.834 s | 15.56 | 537.0 MiB | 248 MiB |
| 6 | 67.751 s | 22.24 | 582.4 MiB | 280 MiB |
| 12 | 50.851 s | 29.64 | 598.5 MiB | 314 MiB |

This is not linear CPU-core scaling. Different resolutions run in separate
encoder buckets; multiple active games make useful same-resolution batches more
likely. The pool can then extract multiple positions in parallel. Cache behavior
also changes because the same bounded cache is now shared by interleaved games.

## 2. Batch-size sweep

Active games fixed at twelve, wait fixed at two ms; only batch limit varied.

| Batch limit | Wall time | Moves/s | Mean actual encoder batch | Maximum actual encoder batch |
|---:|---:|---:|---:|---:|
| 1 | 105.682 s | 14.26 | 1.000 | 1 |
| 4 | 64.317 s | 23.43 | 1.784 | 4 |
| 8 | 50.851 s | 29.64 | 2.527 | 6 |
| 12 | 48.031 s | 31.38 | 3.024 | 6 |

Batch-one concurrency is the important control: merely running many trees does
not create the gain. One model-owning worker still handles their queries, and
batch one cannot parallelize coarse extraction across positions. Gathering
requests enables both CPU feature parallelism and fewer encoder calls.

The requested batch limit is **not** the actual GPU batch size. Resolution
bucketing, cache hits and partial collections give smaller batches here.

## 3. Collection-wait sweep

Active games and batch limit fixed at twelve; only collection wait varied.

| Wait | Wall time | Moves/s | Mean actual encoder batch |
|---:|---:|---:|---:|
| 0 ms | 55.233 s | 27.28 | 2.297 |
| 2 ms | 48.031 s | 31.38 | 3.024 |
| 5 ms | 51.000 s | 29.55 | 3.127 |

Zero drains currently queued requests but often forms smaller batches. Five ms
collects slightly larger batches but loses enough time waiting to reduce total
throughput. Retain two ms for this recipe; this does not establish a universal
optimal wait or an upper bound on request latency.

## Repeats, parity and memory

Additional sequential/six-game controls and two further selected-setting runs:

| Configuration (games / batch / ms) | Runs | Median wall | Wall range | Median moves/s |
|---|---:|---:|---:|---:|
| 1 / 8 / 2 | 2 | 108.279 s | 107.963–108.596 s | 13.92 |
| 6 / 8 / 2 | 2 | 67.830 s | 67.751–67.909 s | 22.22 |
| 12 / 12 / 2 | 3 | 47.964 s | 47.961–48.031 s | 31.42 |

All records match byte-for-byte against the first sequential run, not just move
counts. Network batching can change floating-point outputs/trajectories in general;
that did not happen on these 18 games. Original baseline corpus SGFs remain
unchanged, so there is no new SGD comparison: this experiment changes generation
scheduling, not the fixed-corpus training path.

Representative stage times, sequential → selected: coarse extraction wall
**85.470 → 31.362 s**, encoder **10.785 → 6.704 s**, exact selection/features
**2.714 → 2.206 s**, scorer **1.048 → 0.936 s**. Coarse per-group clocks sum
worker times and are not additive with batch elapsed time. Queue/request clocks
sum client latency and can exceed run wall time.

Inference work is not identical despite identical SGFs: sequential encoding has
11,805 positions / 2,993 cache hits. Selected repeats encode 12,052–12,064
positions / 2,719–2,730 hits, with some duplicate requests coalesced. Holding cache
limits fixed causes extra recomputation at higher concurrency. This experiment
reports end-to-end benefit under that tradeoff, not a pure encoder-kernel speedup;
cache tuning is deliberately deferred.

Maximum observed native lifetime RSS across selected repeats was **645.6 MiB**,
versus **523.3 MiB** sequential and **588.0 MiB** at the six-game setting.
Selected driver GPU memory sampled **318 MiB**, versus 204 / 280 MiB respectively.
The cache cap remains 32 MiB, not a total-process bound: concurrent reply buffers,
search state, feature tensors and native/GPU arenas add memory. RSS includes
model load/warmup; driver polling has gaps of at least 100 ms and can miss peaks.
No exact allocation-peak or universal memory bound is claimed.

## Operating decision and limits

The experiment initially retained the recipe as opt-in. Following the results,
the user requested **12 concurrent games / batch limit 12 / wait 2 ms as defaults**
for neural self-play, new loops and profiling. Random/unguided engines remain
sequential; explicit flags still override the defaults. Active games are capped
by games requested: a 10-game iteration cannot use twelve active slots. Game
budgets were not increased to fill the slots. Resume keeps frozen settings.

No CPU-provider, larger/trained-model, size-9-only, default 40/40/20 mix, or
32-simulation scaling claim is made. Recheck on the learning-trial workload before
adopting it there; do not extend tuning indefinitely before starting that trial.
No counts above twelve active games or batch twelve were tested.

## Artifacts and reproduction

Git-ignored `runs/20261104-concurrency/` contains `sweep.py`, `summary.json`, all
14 SGF directories, raw run statistics/stdout/stderr/RSS, model/binary hashes,
validated per-game metadata and the concurrency/batch/wait/repeat logs.
All settings reuse the same binary/model/search/cache; vary one factor at a time.

```sh
mise run neural-selfplay -- \
  --model runs/20261104-000500-cell-policy-bench/selective \
  --device cuda --games 18 --sizes 9,13,19 --size-weights uniform \
  --seed 800 --simulations 8 --max-moves 512 \
  --feature-threads 6 --inference-threads 1 --cache-positions 64 \
  --concurrent-games 12 --inference-batch-size 12 --inference-batch-wait-ms 2 \
  --stats-json runs/NEW-STATS.json --output runs/NEW-GAMES
```

Use fresh paths. Repeat with `--concurrent-games 1` for the sequential control,
and six / batch eight / two ms for the existing loop-concurrency control.
The [inference guide](../current/inference.md#cross-game-batching) describes bounds
and semantics; the [profiling guide](../current/profiling.md) explains clock/memory
overlap. Search-cache and evaluation parallelism tasks are now at the top of the
conditional later section, not prerequisites for learning.
