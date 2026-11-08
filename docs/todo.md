# ContinuGo AI — Open work

This is a living document. Remove tasks when implemented.
Prioritize playing strength per wall-clock hour, not profiling for its own sake.
Use [KataGo's self-play training](https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md)
as a reference, adapting it to ContinuGo.

## First: avoid repeated fine-feature work

- [ ] Remove the obsolete local liberty-footprint scalar implementation and its
  test-only cache/projection/shortcut benchmarks now that binary region splitting
  replaces it. Keep the comparison results and eye-making rationale in history;
  preserve schema-4 support. This cleanup concerns the replaced fine scalar, not
  the separate coarse spatial liberty-center/footprint planes.
- [ ] Trial a certified no-capture summary fast path in shared rules, reusing
  prepared liberty witnesses and avoiding child construction/repeated scans on
  proven ordinary moves. Unresolved candidates must use the authoritative
  validator. Obtain confirmation before modifying the sibling checkout;
  differential-check captures, suicide, joins, both colours and board sizes.
  Compare any rules fast path in fixed seeded native searches for end-to-end
  throughput, peak memory and moves/visits/results, with concurrency, batching,
  model and source positions fixed. Include value-only leaves; retain only
  measured benefits and recheck trained schema-5 proposals.

The [friendly-region split replacement](history/2026-11-08-friendly-region-splitting.md)
is implemented; the old local-footprint field, projection optimization and
inherited scalar-cache proposal are obsolete. The prior
[fine-feature investigation](history/2026-11-08-fine-shortcuts.md) still motivates
avoiding discarded child construction in shared rules, with confirmation.

## Later, only if a bottleneck appears

- [ ] Tune the existing search cache: test whether retaining more already-computed
  board features/network encodings avoids enough repeated work to justify the
  extra memory. Keep concurrency and search settings fixed for this comparison.
- [ ] Run independent evaluation games in parallel/batches when evaluation is a
  substantial part of trial runtime. This speeds evaluation, not training steps.
- [ ] Investigate keeping feature workers alive between training phases if startup
  or waiting for features becomes a substantial cost. Use longer training runs
  to check this; recheck the six-worker setting on the trial corpus rather than
  making another worker sweep a prerequisite for learning.
- [ ] If unresolved candidates still dominate after the certified fast path,
  profile `PreparedPosition::analyze_move` internals and trial a general shared-rules
  summary-only path to avoid discarded children/repeated scans/allocations on
  those candidates too; differential-check captures and suicide.
- [ ] Tune scorer batching and cell budgets for speed if their measured cost
  becomes significant; retain tactical-coverage and strength checks.
- [ ] Investigate CPU/GPU pipelining and GPU-resident contexts if the remaining
  compute/transfer split justifies them.
- [ ] Measure rollout cost/strength before adding it to the supervised loop/rating protocol.
- [ ] Tune search/training budgets and manually scale network capacity; consider
  learned group embeddings only with evidence of better strength per compute.

Time-box performance experiments; keep them only for measured end-to-end benefit
within bounded memory. Keep model/search settings and source games fixed; check
decoded moves/visits/results when deterministic and report changed trajectories.
Use the [pipeline diagnostics](current/profiling.md), not isolated kernel timings.
Recheck settings on the trial workload/trained actor when available; this does not
block the first learning trial.

## Later integration work

- [ ] Integrate browser ONNX inference and verify parity/operator support.
- [ ] Move reusable components into the main ContinuGo repository with confirmation.
