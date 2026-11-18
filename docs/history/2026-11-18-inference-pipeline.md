# Default bounded inference pipeline — 2026-11-18

Implemented and benchmarked in `scratch/20261119-pipeline/`, then adopted as the
only cross-game batching scheduler. No opt-in flag or legacy scheduler remains.
Direct `evaluate_batch` is still available for inference/parity tooling.

## Implementation

- One scoped CPU preparation thread feeds the single model-owning execution
  thread through a bounded one-batch channel. ONNX sessions stay on one thread.
- Coarse preparation overlaps previous-batch encoding/execution. Exact policy
  extraction and proposal/scorer work remain in the execution stage, using the
  same bounded feature pool as preparation.
- Actual batch size is capped by `min(configured_batch, ceil(game_slots/2))`,
  reserving outstanding requests for both stages. Default game slots / public
  batch cap remain 32/32; execution cap is consequently 16, not a forced size.
- Cache entries are immutable shared contexts; short locks cover lookups/inserts,
  never ONNX calls or feature extraction. Cached values can reply in preparation.
  Execution rechecks cache state, safely discarding redundant hints or falling
  back to extraction if a previously observed entry was evicted.
- Duplicate snapshots share preparation/encoding within each batch, including
  cache-disabled runs. Client errors/disconnects drain, and receiver lifetimes
  release a blocked producer on execution panic before scoped joins.
- Stage elapsed times now overlap and cannot be added to reconstruct wall time.
  Queue time covers collection and the prepared-to-execution queue. Coarse
  profiling counts actual work, including speculative extraction.

## Matched CUDA pilot

Same frozen cg10 round-22 model, GTX 1080, six-core i5-8400, six feature workers,
one ONNX host thread, 64 cached contexts, 2 ms collection wait, 32 active games /
batch cap 32. Each case: 64 size-9 games, seed 221000, maximum 32 moves, 32/128
search caps, 25% expensive searches; profiling on, training-cache creation off.
This is a reduced-search-budget early-game pilot, not full-campaign throughput
or playing-strength evidence.

| Scheduler | Moves/s | Wall seconds | GPU mean | CPU cores | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|
| Synchronous, first | 9.962 | 205.6 | 27.3% | 2.95 | 635.7 |
| Pipeline, prototype first | 12.790 | 160.1 | 36.1% | 3.85 | 610.7 |
| Pipeline, prototype repeat | 12.768 | 160.4 | 36.2% | 3.85 | 610.4 |
| Synchronous, repeat | 9.962 | 205.6 | 27.0% | 2.95 | 644.7 |
| Pipeline, final implementation | 12.729 | 160.9 | 35.9% | 3.83 | 621.2 |

Final implementation includes additional prepared-queue timing, speculative-work
profile accounting, and panic-unwind shutdown hardening. Its throughput gain
relative to the mean synchronous baseline is **27.8%**, a **21.7% wall-time
reduction**. The two original prototype repeats gained 28.3% on average.

All cases produced **64/64 byte-identical SGFs**, 2,048 played moves, 27,671
policy evaluations and 4,456,363 exact coordinates. Encoded positions differ by
about 0.01% due to cache timing. Numerical equality of future trajectories is not
generally guaranteed across batch sizes; native output parity uses tolerance.

`benchmark.py`, `validate-final.py`, binary fingerprints, commands, GPU samples,
`time.txt`, per-stage counters, SGFs and archived baseline/prototype binaries are
in the scratch directory. The detached worktree is experimentation-only; its
Cargo path overrides are not part of the adopted change.

## Validation / campaign status

- `cargo test --locked`: passed, benchmark/environment-dependent tests skipped.
- Full `test_loop`: 20 passed, including real subprocess interruption, cleanup,
  checkpoint/SGF recovery, export, second iteration and idempotent resume.
- Full `test_inference` + `test_schema_compatibility`: 21 passed, including CPU
  and CUDA worker parity/lifecycle, batch-one trajectory parity, default worker
  execution, output-failure shutdown, and schema-4 complete-cell worker coverage.
- Native worker parity/lifecycle also passed against the real cg10/schema-6
  model on CUDA, including bounded duplicate-query preparation with cache off.
- Profiling/loop default agreement test passed; formatting/diff checks passed.

Cg10 remains paused after round 22. Its frozen binary/runtime, loop configuration,
checkpoint and data were not changed or resumed; adopting source defaults does
not silently upgrade a frozen campaign.

## Follow-up: gather prepared inputs without a pool round trip

Further coarse-stage investigation (`scratch/20261118-coarse-investigation/`)
found that the execution thread submitted already-prepared inputs to the shared
Rayon pool merely to move owned buffers. This waited behind preparation work.
With explicit user approval, all-ready batches now gather on the execution
thread; genuine fallback extraction still uses the shared pool. There is no
new option and no change to feature values, model exports or cache semantics.

A matched diagnostic pilot with the workload above measured 159.630 -> 144.768 s
(12.830 -> 14.147 moves/s), **10.3% more throughput**, with all 64 SGFs byte-identical.
The all-ready gather counter fell from 22.746 s to .011 s; producer preparation
was essentially unchanged (122.850 -> 123.559 s). This is one matched pair, not
a full-campaign speed guarantee. The earlier ~149 s coarse counter combined
producer and execution-side work; it was not one 149 s preparation stage.

Validation after adoption: Rust suite passed; CPU/CUDA native worker parity,
legacy-schema worker coverage, concurrent self-play/output-error handling and
the real two-iteration interruption/recovery test all passed. Cg10 is still
paused with its frozen binaries unchanged. Group-level area caching and cheaper
area integration are separate, unimplemented changes.
