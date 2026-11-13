# ContinuGo AI — Open work

This is a living document. Remove tasks when implemented.
Prioritize playing strength per wall-clock hour, not profiling for its own sake.
Use [KataGo's self-play training](https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md)
as a reference, adapting it to ContinuGo.

## Later, only for a measured bottleneck

- [ ] Trial a certified no-capture summary fast path in shared rules, reusing
  prepared liberty witnesses to avoid discarded child construction/repeated scans.
  Preserve splitting's use of actual child survivors. Unresolved
  candidates must use the authoritative validator. Obtain confirmation before
  modifying the sibling checkout; differential-check captures, suicide, joins,
  both colours and board sizes. Compare fixed seeded native searches, including
  value-only leaves, for throughput, peak memory and moves/visits/results; recheck
  trained schema-5 proposals rather than relying on old footprint timings.
- [ ] Only if unresolved candidates remain costly after that trial, profile
  `PreparedPosition::analyze_move` and test a general shared-rules summary-only
  path, preserving the child-survivor information needed by splitting.
- [ ] Tune the existing search-context LRU's capacity/reuse against memory cost.
- [ ] Parallelize/batch independent arena/rating games if evaluation dominates
  runtime; preserve paired seeds, recovery and promotion semantics.
- [ ] Profile cached training loading/decoding/collation and transfers if data
  waiting is significant. Investigate cross-phase live-worker reuse only if
  uncached extraction is actually needed and startup is a measured bottleneck;
  do not require another worker sweep before learning.
- [ ] Tune existing scorer batching/chunk limits and selected-cell budgets only
  if costly; retain tactical-coverage and strength checks.
- [ ] Investigate CPU/GPU pipelining and GPU-resident contexts if transfer costs
  justify them.
- [ ] Measure standalone rollout cost/strength before proposing its addition to
  the supervised loop/rating protocol.

## Optional cleanup

- [ ] Remove the test-only old local-footprint scalar and obsolete
  cache/projection/shortcut benchmarks, updating their diagnostic docs/scripts
  and the splitting comparison that still uses the old scalar as a reference.
  Keep comparison results and eye-making rationale in history. Preserve schema-4
  support and the separate coarse spatial liberty-center/footprint planes.

Time-box performance experiments; keep them only for measured end-to-end benefit
within bounded memory. Keep model/search settings and source games fixed; check
decoded moves/visits/results when deterministic and report changed trajectories.
Use the [pipeline diagnostics](current/profiling.md), not isolated kernel timings.
Recheck settings on the trial workload/trained actor when available; this does not
block the first learning trial.

## Later integration work

- [ ] Integrate browser ONNX inference and verify parity/operator support.
- [ ] Move reusable components into the main ContinuGo repository with confirmation.
