# Shared group geometry and area-union optimization — 2026-11-04

Follow-up to [feature transport and worker allocation](2026-11-04-feature-transport.md).
The first coarse-extraction todo item is implemented and removed from open work.
These are cost diagnostics with **random weights**, not playing-strength evidence.

## Changes, measured separately

1. Added opt-in clocks **inside shared** `PreparedPosition::analyze_group` for
   exact liberty spans and validated one-move capturability. Ordinary analysis
   reads no clocks. No capture/legality algorithm was copied into the trainer.
2. Added a lazy, position-bound blocker-row index to the shared prepared view.
   Build sorted exclusion intervals once per row, merge their integer union, and
   reuse it across groups for both liberty spans and capture-cover candidates.
   Previously both paths repeatedly scanned/sorted all stone blockers per group
   and row. The index is built only for nonempty group analysis, **not** ordinary
   move-only preparation. The first span timer includes index construction.
3. Exposed borrowed prepared group members and used them in raster extraction,
   removing the second group construction and its allocations. Its isolated
   wall-time effect is small; no meaningful independent speedup is claimed.
4. Optimized area integration's common single-component union case: update the
   existing interval directly rather than binary searching/draining its vector
   for each overlapping source capsule. The general fragmented-union path remains.

Exact integer geometry, candidate order, authoritative capture/suicide validation,
area equations, float64 arithmetic, integration density, float32 features, model,
search, cell budgets, context cache and worker settings are unchanged. The area
change is bookkeeping only; no coarser approximation or fast unchecked rule path.

The shared changes live in **`../continugo/backend/rules/src/`**; training changes
alone are insufficient. Keep both repository changes together when reproducing.
The pipeline profiler now records the sibling revision, dirty status and all Rust
rule-source hashes in addition to local/native/model provenance.

## Fixed workload

Same i5-8400 / GTX 1080, CUDA ONNX Runtime 1.20.1, PyTorch 2.7.1/cu118, FP32.
Unchanged checkpoint SHA-256:
`30d363f9ac21bfa5dcba9de410b8497be4b70ec7cbaa4806b2eb27c4f696325a`.
32 channels / four residual blocks, top 32 + eight exploration cells; six native
feature threads, one ONNX CPU helper thread. All benchmarks ran sequentially.

Native fixed-position queries retain the original dense source SGFs, all nine
populated early/middle/late size-9/13/19 snapshots, policy and value-only cache
misses, one warmup and 20 measured queries with context caching disabled.

Actual self-play uses unchanged seeds 800–817, eight simulations, cap 512,
one active game and the default 64-position / 32 MiB cache. The expanded fixed
training corpus is the same 18 genuine-policy games / 1,507 eligible positions
as the preceding report. Same game-level split and corpus fingerprint:
`3c369ee04845199a64ab280b859afca8a45d3b5b46010ba5d4ef45f0a531d8ad`.
Training consumes 1,039 positions / 520 batches, sizes 9/13/19 = 158/298/583.
Batch two, **six feature workers**, one Torch CPU thread, uniform sampling,
seed 800, SGD 0.01 / momentum 0.9 / decay 0.0001 / clipping five. Each run gets
20 untimed compute warmup steps, then restores weights/BatchNorm/RNG and uses a
fresh optimizer. Workers reopen per phase; this is not a fully warmed persistent
loader or a sustained learning campaign. No checkpoints were saved/promoted.

## Timing split and populated positions

The new baseline clocks show that span construction, not capture search, dominates
shared-group work on dense middle boards. Size-19 middle: spans **5.660 ms**,
capturability **1.009 ms**, area **3.344 ms** per query. After the blocker cache
alone: spans **0.763 ms**, capturability **0.546 ms**, area **3.301 ms**.
Sparse positions have a different mix: size-19 early area alone costs 10.716 ms
in the baseline. This motivated measuring the area bookkeeping change separately.

Mean milliseconds, baseline → all changes. Coarse columns are policy extraction;
value columns are complete value-only cache-miss queries.

| Size | Phase / ply | Coarse before → after | Value query before → after |
|---|---|---:|---:|
| 9 | early / 7 | 4.510 → 3.781 | 5.152 → 4.494 |
| 9 | middle / 39 | 4.219 → 3.002 | 4.875 → 3.643 |
| 9 | late / 79 | 1.230 → 1.012 | 1.933 → 1.700 |
| 13 | early / 20 | 4.649 → 3.813 | 5.382 → 4.518 |
| 13 | middle / 100 | 4.739 → 2.725 | 5.381 → 3.264 |
| 13 | late / 200 | 1.705 → 1.262 | 2.436 → 1.948 |
| 19 | early / 43 | 11.429 → 9.124 | 12.022 → 9.994 |
| 19 | middle / 215 | 11.082 → 4.880 | 11.862 → 5.651 |
| 19 | late / 430 | 3.563 → 2.500 | 4.430 → 3.329 |

## Actual self-play after each change

Same six reference games, seeds 800–805, 494 moves. Every stage produced
**byte-identical SGFs**, including moves, visits and scored outcomes; all six
naturally completed, none truncated.

| Build | Self-play/output wall | Moves/s | Coarse | Span subtotal | Area subtotal |
|---|---:|---:|---:|---:|---:|
| Split-clock baseline | 44.288 s | 11.15 | 36.599 s | 3.247 s | 32.390 s |
| + blocker-row cache | 41.375 s | 11.94 | 33.980 s | 0.915 s | 32.109 s |
| + borrowed groups | 41.281 s | 11.97 | 33.837 s | 0.875 s | 32.051 s |
| + single-union area path | 35.510 s | 13.91 | 27.931 s | 0.930 s | 26.091 s |

Subtotals sit inside coarse time, not in addition to it. Instrumentation and
ordinary timing variation explain some differences outside the changed stages.
Capturability was only 0.094 s in the six-game baseline: do not generalize the
middle-snapshot capture share to actual self-play. All builds extract the same
3,883 unique positions, 147,530 groups, 26,197,674 spans, and exact-coordinate rows;
policy/value/cache counters are identical.

Broader fixed seeds 806–817, 1,013 moves / 12 natural completions:
**89.431 → 72.573 s**, 11.33 → 13.96 moves/s. All 12 SGFs match byte-for-byte.
Across both sets, **18 identical games / 1,507 moves**, 133.719 → 108.083 s:
about **24% higher observed self-play throughput** for this workload.

## Actual training

Full ordinary and synchronized training phases were run after each change.
Ordinary phase times were 5.814 / 5.620 / 5.596 / 5.442 s respectively, with the
same 1,039 samples and board mix. To avoid treating one pass as a controlled
speedup, five additional alternating baseline/final runs used the saved binaries:

| Median ordinary metric | Before | After |
|---|---:|---:|
| Phase wall time | 5.701 s | 5.417 s |
| Positions/s | 182.24 | 191.82 |
| Exposed loader wait | 0.911 s | 0.641 s |
| Compute + transfer | 4.725 s | 4.715 s |

Wall ranges: 5.694–5.761 s before, 5.409–5.451 s after. About **5.3% observed
training throughput improvement** at six workers, not the much larger sparse
position kernel gain. Compute remains the dominant training limit.
CUDA losses vary even within builds: baseline 12.9447–12.9623, final
12.9566–12.9642. Identical inputs do not imply identical CUDA optimizer
trajectories; no learning/strength improvement is claimed.

Synchronized diagnostic worker coarse time fell from 9.747 to 7.523 s, while
exact extraction remained about 0.73–0.74 s. These are summed worker wall times,
not additive training phase time; use ordinary phases for throughput.

## Correctness and memory

- **All 1,507 training packets** match byte-for-byte between baseline/final
  binaries, covering coarse/exact features, legality, targets and outcomes.
- All nine dense fixed-position packets also match byte-for-byte. Policy/value
  values, queried logits, cells and masses match at all four measurement stages.
- Shared tests compare cached and uncached spans/capture-cover masks against
  exhaustive point geometry, including strict touching and board edges. Dense
  mixed-color capture tests still compare against exhaustive legal placements.
- Tests verify that move-only/empty-group use leaves the index uninitialized,
  subsequent group analyses reuse it, profiling is equivalent, borrowed groups
  preserve original order, and move validation is unchanged after cache creation.
- Existing frozen independent sort-and-merge area oracle matches float64 bits on
  randomized fragmented/bridging/duplicate-row spans, eight radii, and base/doubled
  density. Existing analytic area, capture/suicide, eye-mask and pass checks pass.
- Validation: **58 training Rust tests**, shared workspace tests (58 plus three
  doctests), **138 Python tests**, Clippy with warnings denied in both workspaces,
  Ruff checks/format and Rust formatting pass.

The index is transient prepared-position geometry, not additional neural context
cache payload. Linux six-worker summed lifetime RSS peaks were **26.25 → 26.68
MiB** in synchronized diagnostics; this is not a simultaneous tree peak.
Torch live/reserved peaks remained **48.76 / 82 MiB**, sampled training driver
memory **222 MiB**. Native sampled driver memory stayed **204 MiB**. Reference
native lifetime RSS was 525.2 → 514.3 MiB; broader seeds 517.7 → 516.1 MiB.
No memory improvement is claimed; allocator/process-lifetime peaks vary, and
100-ms-plus driver sampling can miss peaks.

Trained-actor measurements, broader tactical coverage, persistent-loader phases,
cache/concurrency tuning and a small-board learning trial remain pending.

## Artifacts and reproduction

Git-ignored `runs/20261104-group-geometry/` retains baseline/intermediate/final
feature, inference and self-play binaries; original sources and both repository
diffs; four full `pipeline-*` directories with hashes, raw logs, SGFs and metrics;
`repeat_training.py` / JSON; `extra_selfplay.py` / raw runs; packet hashes, native
parity summaries and test logs. Run experiments sequentially from the repo root.

Each pipeline stage used its corresponding saved release binaries and:

```sh
.venv/bin/python -m continugo_train.pipeline_profile \
  --positions runs/20261104-representative-positions \
  --model runs/20261104-000500-cell-policy-bench/selective \
  --checkpoint runs/20261104-000500-cell-policy-bench/selective.pt \
  --data runs/20261104-representative-cpu/selfplay \
         runs/20261104-feature-transport/extra-selfplay \
  --device cuda --iterations 20 --workers 6 --steps 600 \
  --positions-per-game 0 --warmup-steps 20 --output runs/NEW-OUTPUT
```

Build both path-dependent crates before measuring (`cargo build --locked --release
--bins` in this repository). The [profiling guide](../current/profiling.md) documents
stage overlap, first-index attribution and memory limitations.
