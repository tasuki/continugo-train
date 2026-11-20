# ContinuGo AI — Open work

This is a living document.
Remove tasks when implemented.
Prioritize playing strength per wall-clock hour, not profiling for its own sake.

## Feature planes — next fresh run

**Keep schema 7 open: do not increment its number until the next fresh run starts.**
Fold all pre-run input changes into schema 7.

- [ ] Trial friendly/enemy **Benson pass-alive planes**; measure integrated cost
  and benefit before including them.

## Move penalties

- [ ] Investigate a penalty (1 point?) for meaningless moves in own or opponent territory.

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
