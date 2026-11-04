# Feature transport, decoder and worker allocation — 2026-11-04

Follow-up to [coarse-feature optimization](2026-11-04-coarse-features.md).
Throughput diagnostics with **random weights**, not a learning/strength trial.
No search, legality, feature schema, cell budget, cache or shared-rule changes.
No checkpoints were saved/promoted. Measurements below preceded the user-approved
change to six feature workers by default.

## Instrumentation and controlled change

Successful opt-in feature-server requests now append a separate timing trailer;
ordinary CGFEAT01 packets/framing remain unchanged. Server clocks separate:

- validated-game loading/snapshot access and target-cell selection;
- coarse extraction, exact move extraction, and packet serialization;
- output writes, including pipe backpressure.

Client clocks separate request encoding/path resolution, writes, response reads,
timing-trailer reads, array copies/conversions, validation and Torch construction.
The IPC **residual** subtracts server work from client write/read wall time. It
includes dispatch, framing, allocation cleanup and scheduling, not pure bandwidth.
Server writes overlap client reads: those clocks must not be added together.
Synchronized training aggregates worker stages alongside exposed consumer wait;
worker sums include prefetch and are not disjoint training wall time.

Added Linux Rust-child lifetime RSS readings, separate Torch reservation peaks,
and optional process-tree driver GPU-memory sampling with `nvidia-smi`.
Sampling excludes unrelated GPU users, has gaps of at least 100 ms, can perturb
timings and can miss peaks. Missing support produces null, not invented memory.
Child RSS sums are sums of individual lifetime peaks, not simultaneous tree peaks.
See the [profiling guide](../current/profiling.md) for the exact field definitions.

The initial measurements found validation, not tensor construction, dominating
Python decoding. Changed only the decoder's implementation:

- `math.prod` instead of NumPy dispatch for small, bounded shape products;
- direct binary comparisons instead of general `np.isin` membership tests;
- unsigned legality bytes are checked with `> 1`.

All validation checks and writable tensor copies remain. No zero-copy writable
views of immutable packets, unchecked server fast path, or label changes.

## Workload

Same i5-8400 / GTX 1080, CUDA ONNX Runtime 1.20.1, PyTorch 2.7.1/cu118, FP32.
Unchanged checkpoint SHA-256:
`30d363f9ac21bfa5dcba9de410b8497be4b70ec7cbaa4806b2eb27c4f696325a`.
32 channels, four residual blocks, top 32 + eight exploration cells, six native
feature threads and one ONNX CPU helper thread. Measurements ran sequentially.

Retained the original populated size-9/13/19 early/middle/late source positions,
including value-only context-cache misses. Expanded genuine-policy data with
12 neural games, seeds 806–817, eight simulations, cap 512: **1,013 moves, all
12 naturally completed**, no truncations. Added to the original six games,
seeds 800–805, this gives 18 unique games / **1,507 eligible positions**.
No manufactured policy/outcome labels. The fixed game-level split has 12 training
games / **1,039 positions** and six validation games / 468 positions.
Corpus fingerprint:
`3c369ee04845199a64ab280b859afca8a45d3b5b46010ba5d4ef45f0a531d8ad`.

Training uses batch two, one Torch CPU thread, uniform size sampling, all eligible
positions, seed 800, SGD 0.01 / momentum 0.9 / decay 0.0001, clipping five.
Each run gets 20 untimed compute warmup steps, then restored weights/BatchNorm,
reset RNG and a fresh optimizer. The 600-step cap exceeds corpus exhaustion:
actual **520 steps, 1,039 positions**, sizes 9/13/19 = **158/298/583**.
Workers are reopened for each phase: these longer windows are **not a persistent,
fully warmed loader** and still do not constitute a sustained training campaign.

## Warm feature round trips

Original genuine-policy corpus, same first/middle/last eligible position per size,
one warmup and 100 requests per position on one persistent Rust server. These
are different SGFs/phase proxies from the dense fixed-position native queries.
Mean milliseconds; server/IPC columns are after the decoder change.

| Size | Phase | Coarse | Exact | Serialize | IPC residual | Decode before → after | Round trip before → after |
|---|---|---:|---:|---:|---:|---:|---:|
| 9 | early | 0.022 | 0.270 | 0.133 | 0.164 | 0.501 → 0.392 | 1.228 → 1.098 |
| 9 | middle | 5.574 | 2.054 | 0.188 | 0.179 | 0.689 → 0.571 | 8.875 → 8.717 |
| 9 | late | 4.927 | 0 | 0.032 | 0.133 | 0.194 → 0.110 | 5.390 → 5.310 |
| 13 | early | 0.054 | 0.259 | 0.189 | 0.125 | 0.469 → 0.362 | 1.249 → 1.116 |
| 13 | middle | 5.981 | 0.517 | 0.079 | 0.181 | 0.314 → 0.205 | 7.224 → 7.095 |
| 13 | late | 7.723 | 0 | 0.064 | 0.180 | 0.212 → 0.125 | 8.247 → 8.216 |
| 19 | early | 0.126 | 0.080 | 0.279 | 0.139 | 0.316 → 0.203 | 1.087 → 0.944 |
| 19 | middle | 12.213 | 0.289 | 0.254 | 0.200 | 0.326 → 0.199 | 13.533 → 13.328 |
| 19 | late | 16.467 | 0 | 0.242 | 0.213 | 0.271 → 0.166 | 17.382 → 17.263 |

Not all round-trip components are shown in this table. Dense/populated coarse
work still dominates; rewriting pipe transport or reducing tactical budgets is
not justified by these measurements.

## Actual training: decoder only

Five additional alternating before/after ordinary measurements at **two workers**,
fixed games, sample order, weights and optimizer setup:

| Median metric | Before | After |
|---|---:|---:|
| Phase wall time | 9.560 s | 9.452 s |
| Positions/s | 108.69 | 109.92 |
| Exposed loader wait | 4.817 s | 4.755 s |
| Compute + transfer | 4.681 s | 4.634 s |

Wall ranges: 9.555–9.572 s before, 9.447–9.454 s after. About **1.1% observed
throughput gain**, not a large CPU bottleneck fix. CUDA loss trajectories vary
even within the same build: before 12.9157–12.9291; after 12.9120–12.9288.
No learning improvement or identical SGD trajectories are claimed.

Separate synchronized diagnostics over 1,039 requests reduced aggregate decode
time from **0.490 to 0.340 s** (array stage 0.111 → 0.068 s, validation
0.355 → 0.249 s, Torch construction about 0.019 s). After-change server work:
coarse 9.577 s, exact 0.708 s, serialization 0.200 s, replay/selection 0.092 s;
IPC residual 0.264 s. These are summed worker wall times, not additive phase time.
The decoder optimization is retained for this small but repeated end-to-end gain.

## Worker allocation: one additional factor

After the decoder change, three sweeps with alternating ascending/descending worker
counts, identical corpus and initialization, full corpus consumption each time:

| Workers | Median phase | Positions/s | Exposed loader wait | Wall range |
|---|---:|---:|---:|---:|
| 0 (synchronous) | 16.137 s | 64.38 | 11.525 s | 16.119–16.144 s |
| 1 | 11.608 s | 89.51 | 6.941 s | 11.594–11.640 s |
| 2 | 9.454 s | 109.90 | 4.748 s | 9.447–9.482 s |
| 3 | 5.915 s | 175.65 | 1.155 s | 5.904–5.928 s |
| 4 | 6.988 s | 148.69 | 2.231 s | 6.979–6.999 s |
| 6 | 5.696 s | 182.42 | 0.910 s | 5.695–5.707 s |

**Worker counts change round-robin sample order and optimizer trajectories.**
These are observed throughputs over the same 1,039 real samples/board counts,
not a controlled same-trajectory speedup. Non-monotonic scaling is consistent
with shard imbalance and ordered-queue backpressure; do not infer a universal
CPU-core scaling law.

Three workers give about 1.60× the two-worker throughput; six uses all six CPU
cores for another ~4%. Following these measurements, the user selected
**`--workers 6` as the default** for fresh training, new loops and profiling.
This is the fastest tested count on this machine, not a universal optimum;
recheck on the trial corpus/trained actor. Existing saved configurations retain
their worker settings. Do not alter caches or cross-game concurrency in the same
comparison.

Separate synchronized memory diagnostics:

| Workers | Sum of Rust child lifetime RSS peaks | Sampled driver GPU peak |
|---|---:|---:|
| 2 | 9.06 MiB | 222 MiB |
| 3 | 13.66 MiB | 222 MiB |
| 6 | 26.86 MiB | 222 MiB |

Training live Torch peak was **48.76 MiB**, reservation **82 MiB**; parent lifetime
RSS about 1.1 GiB includes Torch/model/corpus inspection and earlier phases.
Do not add separate lifetime peaks and call that simultaneous memory consumption.

## Native self-play and populated snapshots

Before/after native binaries are identical; only Python decoding changed.
Reran actual six-game self-play at seeds 800–805 with original model/search/cache
settings. All six **SGFs are byte-identical**, including moves, visits and results:
494 moves, six natural completions, zero truncations. Generation/output times
44.817 vs 44.490 s, RSS 521.4 vs 517.4 MiB, sampled driver GPU peak **204 MiB**
in both. **No native speedup is claimed.** Additional seeds 806–817 took 90.624 s
for 1,013 moves, RSS 513.5 MiB and sampled driver peak 204 MiB.

All 18 fixed populated policy/value diagnostics have identical values, queried
logits, selected cells and masses. Twenty measured queries after one warmup,
context cache disabled. Whole-query mean milliseconds:

| Size | Phase / ply | Policy before → after | Value-only miss before → after |
|---|---|---:|---:|
| 9 | early / 7 | 7.148 → 7.098 | 5.175 → 5.175 |
| 9 | middle / 39 | 7.144 → 7.269 | 4.972 → 4.998 |
| 9 | late / 79 | 3.955 → 4.021 | 2.015 → 1.954 |
| 13 | early / 20 | 8.117 → 8.133 | 5.446 → 5.360 |
| 13 | middle / 100 | 8.036 → 8.048 | 5.794 → 5.862 |
| 13 | late / 200 | 4.363 → 4.369 | 2.381 → 2.395 |
| 19 | early / 43 | 14.438 → 14.491 | 12.218 → 12.036 |
| 19 | middle / 215 | 15.691 → 15.527 | 12.838 → 12.877 |
| 19 | late / 430 | 7.054 → 7.123 | 4.439 → 4.407 |

## Correctness and remaining shared-rule work

- Old/new decoders produce exactly equal fields/tensors on **all 1,507** genuine
  policy packets, including pass-only/no-coordinate cases. Packet hashes retained.
- Tests cover diagnostic/ordinary response parity, mixing request modes, errors
  without unread trailers, invalid capture flags/legality bytes, finite features,
  thread-safe aggregation and unavailable/sampled GPU-memory behavior.
- CPU synchronized/ordinary forward and optimizer parity still passes; existing
  capture, suicide, touching connection, eye-mask and pass tests pass unchanged.
- Validation: **58 release Rust tests**, **137 Python tests**, Clippy with warnings
  denied, Ruff checks/format and Rust formatting pass.

Read-only audit of the shared rules confirms that `liberty_spans` allocates
reachable/blocker/span vectors per group, reuses the first two across rows, but
rescans **all stones and sorts blockers per row per group**. Capture-cover scanning
also rebuilds/sorts blockers on candidate rows. `PreparedGroups` already builds
groups; raster extraction calls `Position::groups` again. Those are still shared
API optimization candidates. The exact span/capturability timing split remains
open: do it inside the shared implementation rather than copying capture rules
here. No changes were made in the main ContinuGo repository.

Broader tactical fixtures, a trained actor, sustained persistent-loader phases,
cache/concurrency experiments and a playing-strength trial remain open.

## Artifacts and reproduction

Git-ignored `runs/20261104-feature-transport/` contains:

- `features_before.py`, profiled feature binary and instrumentation patch;
- `transport-before.json`, `transport-after.json` (100-request warm snapshots);
- `extra-selfplay/`, raw logs/stats and sampled memory for seeds 806–817;
- `pipeline-before/`, `pipeline-after/`: all native outputs/SGFs, source/model/
  binary hashes, corpus manifest and ordinary/synchronized training metrics;
- `run_profile.py`, `repeat_training.py`, `worker_sweep.py`, their logs/JSON;
- `packet-parity.json`, `native-parity.json`, `worker-memory.json`, test log.

The saved scripts swap **only** the decoder function for before/after measurements;
Rust extraction is unchanged. Run from the repository root, sequentially. The
expanded fixed-corpus six-worker recipe is:

```sh
mise run pipeline-profile -- \
  --positions runs/20261104-representative-positions \
  --model runs/20261104-000500-cell-policy-bench/selective \
  --checkpoint runs/20261104-000500-cell-policy-bench/selective.pt \
  --data runs/20261104-representative-cpu/selfplay \
         runs/20261104-feature-transport/extra-selfplay \
  --device cuda --iterations 20 --workers 6 --steps 600 \
  --positions-per-game 0 --warmup-steps 20 --output runs/NEW-OUTPUT
```
