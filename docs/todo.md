# ContinuGo AI — Open work

This is a living document. Remove tasks when implemented.
Prioritize playing strength per wall-clock hour, not profiling for its own sake.
Use [KataGo's self-play training](https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md)
as a reference, adapting it to ContinuGo.

## Cheaper move proposals — implementation and alternatives

Goal: cut fine-feature extraction cost by at least 50%, rather than only optimize
individual checks. Geometry-generated shortlists are the schema-5 inference
default. Do not change any campaigns. Other approaches below remain alternatives,
not implementation tasks.
Halving fine work alone would save roughly 39% of current self-play wall time
(fine work is about 78%).

- [ ] **Validate geometry proposal coverage and strength:** see the
  [Extend / Join / Capture / Random implementation](current/geometry-proposals.md).
  Direct `geometry-v2-frontier` inference and shared-viewer visualizations
  (dots/translucent stones) are implemented; Restrict/Breathe scoring is removed.
  Schema-5 inference uses these proposals by default; training and existing
  campaigns are unchanged. Next: teacher-search
  coverage/equivalent outcomes, rare splitting fixtures, illegal-candidate refill,
  and equal-wall-clock strength checks. A candidate-only training objective is
  separate work requiring persisted domains and matching target normalization.
  - Preserve 1–3 real outward/connected-hugging extensions per friendly group,
    two lens-side plus two axial join extremes, capture representatives and a few
    random slots. No cell-corner padding, huge prescored pool or diversity ranking.
    Measure coverage and protected-set overflow, not just final rows.
  - The same fixed-position diagnostics now use 139–164 exact rows, with
    generation around 0.5–0.6 ms and whole CPU queries around 4.7–5.7 ms.
    These are not campaign self-play or playing-strength results.
- [ ] **Coarse cells plus deterministic coordinate selection:** omit the learned
  fine head; generate several alternatives per cell, preferring connections and
  captures without forcing them. Train cell policy from summed MCTS visits.
- [ ] **Cell-and-goal policy:** network outputs priorities for goals such as
  connect/capture/escape/ordinary placement per cell. Goal-specific generators
  propose coordinates; validate only those proposals and try alternatives if a
  goal is infeasible. Priorities allocate search attention, not feasibility
  probabilities. Investigate MCTS-visit credit assignment for overlapping goals;
  retain ordinary/exploratory moves and pass.
- [ ] Related variants: choose target groups or pairs of groups; score cheap
  spatial priority maps and take separated peaks; choose geometric move templates.
- [ ] **Cheap learned coordinate proposals:** retain learned coarse context,
  borders and precise within-cell coordinate information, but omit expensive fine
  inputs. Validate proposals on admission to the search tree and reject/resample
  illegal ones, not merely when executing the final move.

For any reduced candidate domain, explicitly redesign proposal probabilities,
normalization and training targets; do not silently treat sampled coordinates as
complete cells. Capture detection and splitting are not free: their generators
must avoid exhaustive move analysis, and shortlisted splitting queries still need
actual capture survivors. Compare shortlist coverage/equivalent tactical outcomes
against existing searches before retraining, then test end-to-end throughput and
playing strength. Candidate-count reductions are not guaranteed wall-time gains.
Keep these investigations separate from the frozen running campaign.

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
