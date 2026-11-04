# ContinuGo AI — Open work

This is a living document. Remove tasks when implemented.
Prioritize playing strength per wall-clock hour, not profiling for its own sake.
Use [KataGo's self-play training](https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md)
as a reference, adapting it to ContinuGo.

## 1. Prepare and run a small-board learning trial

This is the next strength milestone; do not block it on speculative optimization.

- [ ] Implement explicit bootstrap-only eye avoidance using shared
  `SmallEyeAnalysis`, preserving the capture exemption. Keep game legality and
  neural legality masks unchanged; check tactical eye fills and pass behavior.
- [ ] Generate genuine bootstrap search visits and natural outcomes with that
  guidance; record its configuration in SGFs. Do not invent policy/outcome labels.
- [ ] Configure size-9-only self-play and training sampling consistently.
- [ ] Set reference/actor match budgets, simulations, caps, and actor-match
  thresholds for the trial; report uncertainty and truncation/censoring bias.
- [ ] Check whether coarse-cell move selection misses captures or saves, including
  a comparison against selecting coordinates directly. Treat this as a playing-
  strength question; do not shrink move budgets just to improve timing results.
- [ ] Run a short inspected trial from random weights with fresh optimizer state
  and fresh bootstrap data, before committing to a long campaign.
- [ ] Evaluate without eye guidance: eye preservation, connections, captures/saves,
  sensible passing, natural completion, and matches against the reference bot.
  Keep guidance disabled for normal neural self-play/evaluation. Expand tactical
  test positions based on observed failures and compare move-selection choices.

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
- [ ] Profile `PreparedPosition::analyze_move` internals if exact move analysis
  becomes a significant end-to-end cost: legality, tactical summaries, discarded
  child construction, repeated scans, and allocations. Add a shared-rules
  summary-only path only if justified; differential-check against the validator,
  including captures and suicide.
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
