# ContinuGo AI — Open work

This is a living document.
Remove tasks when implemented.
Prioritize playing strength per wall-clock hour, not profiling for its own sake.

## cg13: return on expensive large-board self-play

- [ ] **Priority 1 — training/data reuse:** Check whether training is too light
  relative to expensive generation. Measure generated vs optimized positions,
  replay exposures/reuse and eviction before use **per board size**, including
  policy-enabled positions and the availability-capped equal-position mixture.
  Compare increased training budgets/reuse on fixed generated data, monitoring
  held-out losses, overfitting and separately measured playing strength per
  wall-clock hour; do not assume more epochs automatically help.
- [ ] **Priority 2 — self-play batching:** Investigate why larger inference
  batches do not improve large-board self-play throughput despite spare VRAM.
  Measure actual batch occupancy, queue waits, active games/tail effects, CPU
  rules/feature preparation, GPU utilization and inference/transfer time. Sweep
  concurrent games and inference batch limits separately on fixed models/search
  settings and board sizes; distinguish capacity limits from poor batch filling
  or CPU-bound work. Keep only measured end-to-end gains and check trajectories.

 ┌───────┬───────┬────────────────┬────────────┬────────────────┐
 │ Board │ Games │ Avg moves/game │ Moves/iter │ Self-play time │
 ├───────┼───────┼────────────────┼────────────┼────────────────┤
 │  9×9  │ 150   │  65            │  9,744     │  34 min        │
 ├───────┼───────┼────────────────┼────────────┼────────────────┤
 │ 11×11 │ 123   │  86            │ 10,521     │  48 min        │
 ├───────┼───────┼────────────────┼────────────┼────────────────┤
 │ 13×13 │ 104   │ 133            │ 13,878     │  85 min        │
 ├───────┼───────┼────────────────┼────────────┼────────────────┤
 │ 15×15 │  90   │ 193            │ 17,392     │ 127 min        │
 ├───────┼───────┼────────────────┼────────────┼────────────────┤
 │ 17×17 │  79   │ 279            │ 22,047     │ 196 min        │
 ├───────┼───────┼────────────────┼────────────┼────────────────┤
 │ 19×19 │  71   │ 350*           │ 24,850*    │ 275 min*       │
 └───────┴───────┴────────────────┴────────────┴────────────────┘

## Later, only for a measured bottleneck

- [ ] Trial a certified no-capture summary fast path in shared rules, reusing
  prepared liberty witnesses to avoid discarded child construction/repeated scans.
  Preserve splitting's use of actual child survivors. Unresolved
  candidates must use the authoritative validator. Obtain confirmation before
  modifying the sibling checkout; differential-check captures, suicide, joins,
  both colours and board sizes. Compare fixed seeded native searches, including
  value-only leaves, for throughput, peak memory and moves/visits/results; recheck
  trained schema-6 proposals rather than relying on old footprint timings.
- [ ] Only if unresolved candidates remain costly after that trial, profile
  `PreparedPosition::analyze_move` and test a general shared-rules summary-only
  path, preserving the child-survivor information needed by splitting.
- [ ] Parallelize/batch independent arena/rating games if evaluation dominates
  runtime; preserve paired seeds, recovery and promotion semantics.
- [ ] Optimize cached training loading/decoding/collation and transfers if data
  waiting remains significant.
- [ ] Investigate GPU-resident contexts if transfer costs justify them.
- [ ] Measure standalone rollout cost/strength before proposing its addition to
  the supervised loop/rating protocol.

Time-box performance experiments; keep them only for measured end-to-end benefit
within bounded memory. Keep model/search settings and source games fixed; check
decoded moves/visits/results when deterministic and report changed trajectories.
Use the [pipeline diagnostics](current/profiling.md), not isolated kernel timings.
Recheck settings on the trial workload/trained actor when available; this does not
block the first learning trial.

## Later integration work

- [ ] Integrate browser ONNX inference and verify parity/operator support.
- [ ] Move reusable components into the main ContinuGo repository with confirmation.
