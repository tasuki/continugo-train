# Representative pipeline diagnostics — 2026-11-04

Initial profiling, **not a learning trial or strength evidence**. This replaces
one-stone-only reasoning about the current selective-cell pipeline. Harness and
reproduction instructions: [profiling](../current/profiling.md).

## Setup and scope

Intel i5-8400 (six cores), GTX 1080, release Rust/mimalloc, FP32. Native ONNX CPU
Runtime 1.22.1, CUDA Runtime 1.20.1; PyTorch 2.7.1+cu118. Native feature pool six
threads, ONNX one CPU helper thread; training one PyTorch CPU thread, two persistent
feature workers, batch two, shuffle buffer eight. CPU and CUDA measurements ran
sequentially without concurrent tests/benchmarks.

Default `cells-v1` network, 80,116 parameters: 32 channels, four blocks, policy
hidden 16, value hidden 32, top 32 cells plus eight exploration cells. **Random
weights**, not a trained actor. Checkpoint SHA-256:
`30d363f9ac21bfa5dcba9de410b8497be4b70ec7cbaa4806b2eb27c4f696325a`.
Both ONNX graphs were verified against that checkpoint by the harness.

Representative source games: direct random baseline, sizes 9/13/19, seeds 0/1/2,
81/202/432 moves, all naturally completed. Early/middle/late snapshots are
move-count proxies, not assertions of tactical representativeness. Unlike the
previous one-stone check, **every measured source position is populated**. There
is only one selected source game per size; broader seed/tactical coverage remains
necessary. Policies use selected cells, not all million coordinates.

## Fixed positions

Mean milliseconds per query, one untimed warmup followed by three queries, cache
disabled. Feature/encoder/scorer columns are from the CUDA run; feature computation
still runs on CPU. Overall query time includes unlisted normalization/result work.
These are short diagnostics, not statistical confidence intervals.

| Size | Phase / ply | CPU total | CUDA total | Coarse/rules | Move features/selection | Encoder | Scorer |
|---|---|---:|---:|---:|---:|---:|---:|
| 9 | early / 7 | 20.56 | 12.18 | 8.62 | 0.97 | 0.79 | 0.42 |
| 9 | middle / 39 | 20.09 | 10.85 | 6.27 | 1.01 | 0.78 | 1.35 |
| 9 | late / 79 | 14.43 | 5.19 | 1.45 | 1.18 | 0.77 | 0.41 |
| 13 | early / 20 | 19.94 | 12.83 | 8.38 | 1.10 | 0.84 | 0.30 |
| 13 | middle / 100 | 16.99 | 9.02 | 5.17 | 1.02 | 0.80 | 0.29 |
| 13 | late / 200 | 13.77 | 5.35 | 1.84 | 0.70 | 0.77 | 0.31 |
| 19 | early / 43 | 32.76 | 22.45 | 18.53 | 0.83 | 0.88 | 0.32 |
| 19 | middle / 215 | 26.51 | 16.47 | 11.93 | 1.27 | 0.92 | 0.28 |
| 19 | late / 430 | 18.68 | 8.44 | 3.98 | 1.22 | 0.87 | 0.28 |

Dense late-game boards were **not** the slowest coarse-feature cases. The populated
early size-19 snapshot cost 18.53 ms just for coarse/rules features. Value-only
queries are also recorded in the raw summary; they avoid exact scoring but still
need this coarse analysis on cache misses.

Native lifetime RSS across these queries was approximately 50–54 MiB CPU and
472–477 MiB CUDA. This includes runtime loading, complete SGF validation/replay,
and warmup; it is not steady-state feature memory or native GPU peak allocation.

## Actual neural self-play

Six games, seeds 800–805, uniform 9/13/19 schedule, eight simulations per move,
512-move cap, one active game and default 64-context cache. All six finished
naturally, no draws/truncations. Game lengths: **37, 57, 123, 47, 67, 163**.
CPU/CUDA SGFs are **byte-for-byte identical**, including visits and scored results.
This agreement is observed here, not a general cross-backend determinism promise.

| Metric | CPU | CUDA |
|---|---:|---:|
| Moves | 494 | 494 |
| Generation/output wall time | 91.384 s | 66.260 s |
| Moves/s | 5.41 | 7.46 |
| Coarse/rules features | 58.584 s (64.1%) | 58.582 s (88.4%) |
| Move features/selection | 0.927 s (1.0%) | 0.918 s (1.4%) |
| Encoder | 25.313 s (27.7%) | 3.617 s (5.5%) |
| Scorer | 3.682 s (4.0%) | 0.332 s (0.5%) |
| Native lifetime peak RSS | 112.6 MiB | 510.8 MiB |

Both runs: 993 policy queries, 3,877 value queries, 3,883 encoder executions,
987 cache hits, 9,369,306 exact rows, and 26,477,568 final cache payload bytes.
GPU execution reduces neural compute, **not** the dominant coarse/rules CPU work.
This is two games per size with a weak random network, not campaign throughput.
The earlier un-warmed profiling passes gave the same 494 moves and approximately
91.35/66.23 seconds; that consistency is not a multi-seed statistical estimate.

## Real training and loader round trips

Training corpus: the original CPU six-game self-play set, validated by Rust,
with genuine search visits and naturally scored outcomes. Both device runs use
that fixed corpus and game-level holdout. No random-bot policies or outcome labels
were manufactured. Twelve measured SGD steps optimize **24 positions**, with an
actual size mix of **10/10/4**; the step cap cuts a partial sampling window. Fresh
optimizer state and original weights are restored after one untimed warmup step.
No checkpoint was saved, accepted, exported, or used for subsequent self-play.

| Ordinary training baseline | CPU | CUDA |
|---|---:|---:|
| Phase wall time | 0.716 s | 0.354 s |
| Positions/s | 33.50 | 67.71 |
| Loader wait | 0.150 s | 0.240 s |
| Compute + transfer | 0.545 s | 0.110 s |
| Peak live PyTorch CUDA allocation | — | 48.7 MiB |
| Parent process lifetime peak RSS | 548.6 MiB | 1098.4 MiB |

These **sub-second partial phases are noisy**; do not extrapolate their throughput
to a campaign. Ordinary CPU and synchronized diagnostic losses matched exactly;
small CUDA backward differences occurred. Synchronized diagnostic stage totals:

| Training stage | CPU | CUDA |
|---|---:|---:|
| Transfer | 0.00018 s | 0.00240 s |
| Encoder | 0.20032 s | 0.01881 s |
| Scorer/normalization | 0.01733 s | 0.01364 s |
| Loss/backward/optimizer | 0.32489 s | 0.06247 s |

CUDA's ordinary baseline spends about 68% of its phase in exposed loader wait.
The training parent RSS excludes Rust children; live CUDA allocation excludes
reservation/context. The original pass without an untimed training warmup was
strongly affected by first-use CUDA costs; do not compare it as steady-state SGD.

Warm persistent feature-server round trips on first/middle/last eligible target
positions cost about **1.1–1.3 ms opening**, **11.9–21.8 ms middle**, and **8.5–26.7 ms
late** across sizes. The last eligible position in each selected training game
has a pass-only target (**zero exact rows**), yet still needs coarse analysis.
These timings include Rust features/serialization, pipe wait, decoding and tensor
copies. They **do not isolate pure transport**. Training target cells also differ
from the model-selected cells in native inference, so subtracting those two
measurements would not isolate IPC overhead.

## What to prioritize next

1. **Split coarse/rules feature cost first**, including preparation, liberty-area
   integration and one-move capturability. Coarse extraction dominates actual CUDA
   self-play, including the many value-only queries. Do not assume the per-move
   `analyze_move` path is the principal end-to-end bottleneck: all move preparation,
   selection and exact rows together were only 1.4% of this self-play run.
2. Instrument feature-server extraction/serialization versus pipe/decode separately,
   then repeat longer training phases with controlled worker counts. Warm loader
   stall remains significant after moving neural compute to CUDA.
3. Only then tune cache/batching/concurrency and cell budgets against these costs,
   checking captures/saves and actual strength. No gameplay/rule/model tuning was
   performed in this profiling pass.

The first TODO remains open: broader tactical/seed/trained-model coverage, a true
transport split, and native GPU/child-memory accounting are not finished.

## Artifacts and reproduction

Temporary raw artifacts (under Git-ignored `runs/`):

- `20261104-representative-positions/`: three completed random source SGFs.
- `20261104-representative-cpu-warm/`, `20261104-representative-cuda-warm/`:
  `summary.json`, raw native diagnostics, self-play SGFs, timings/RSS.
- `20261104-representative-cpu/selfplay/`: fixed six-game training source.
- `20261104-000500-cell-policy-bench/selective.pt` and `selective/`: unchanged
  random checkpoint/export from the preceding selective-policy diagnostic.

```sh
mise run pipeline-profile -- --positions runs/20261104-representative-positions \
  --model runs/20261104-000500-cell-policy-bench/selective \
  --checkpoint runs/20261104-000500-cell-policy-bench/selective.pt \
  --data runs/20261104-representative-cpu/selfplay --device cpu
# Then repeat sequentially with --device cuda.
```

Omit `--data` to generate a fresh self-play training corpus in the new run. Use
fresh outputs; generated records are never overwritten. Raw summaries preserve
all source hashes and model identities if artifacts remain available; this dated
report retains the diagnostic results if temporary run directories are removed.
