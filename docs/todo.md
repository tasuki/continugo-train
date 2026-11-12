# ContinuGo AI — Open work

This is a living document. Remove tasks when implemented.
Prioritize playing strength per wall-clock hour, not profiling for its own sake.
Use [KataGo's self-play training](https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md)
as a reference, adapting it to ContinuGo.

## Priority: improve training generalization

The [network-size comparison](history/2026-11-11-network-size-comparison.md)
showed overfitting across all four architectures and heads. Keep 4×32 as the
baseline; prioritize these checks over larger networks or more passes on the
same games. Follow-up experiments need explicit budgets before launch.

- [ ] Audit generalization measurements: evaluate larger fixed train/validation
  probes in inference mode; compare split distributions (source rounds, game
  lengths, outcomes and scores); verify targets, masks and loss denominators.
  Test whether BatchNorm-statistics recalibration explains validation spikes.
- [ ] Test data scarcity with a fixed-recipe learning curve over progressively
  more unique, diverse games, keeping a common game-disjoint holdout. Distinguish
  new-data gains from gains due merely to extra position exposures.
- [ ] Test fewer passes per replay window with fresh self-play data rather than
  repeatedly fitting a static corpus; choose budgets from held-out curves,
  without assuming the observed 2–3-epoch optimum transfers to other corpora.
- [ ] If needed after the audits, test stronger weight decay, lower/decaying
  learning rates and auxiliary-loss reweighting separately. Compare per-head
  held-out progress and inference-mode gaps, not training loss alone.

## Later, only for a measured bottleneck

None of these performance experiments blocks the generalization work above.

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
