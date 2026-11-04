# ContinuGo AI — Open work

This is a living document. When something is implemented, remove it from here.

Prioritize playing strength per wall-clock hour. Complete the prioritized pipeline
fixes and ratings cleanup before the next training trial; defer speculative tuning.
Use [KataGo's self-play training](https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md)
as a reference, adapting it to ContinuGo.

## 1. Optimize the measured CPU bottleneck

- [ ] Split the remaining shared `analyze_group` timing into exact liberty spans
  and one-move capturability without copying shared-rule decisions. Audit its
  repeated blocker scans/allocations and duplicate group construction during
  preparation versus raster extraction. Keep populated early/middle/late 9/13/19
  positions and value-only cache misses in subsequent measurements.
- [ ] Split the training feature-server round trip into extraction, serialization,
  pipe transport, decoding/tensor copies, and exposed loader wait. Repeat longer,
  warmed training phases with a fixed genuine-policy corpus and controlled worker
  counts; account for Rust child RSS and native GPU memory, not only parent RSS
  and live PyTorch allocation.
- [ ] Remeasure actual self-play and training after each change, alongside the fixed
  positions. Keep model/search settings and source games fixed for comparisons;
  check decoded moves/visits/results when deterministic and report changed
  trajectories rather than claiming a controlled speedup. Expand seeds and
  tactical coverage; repeat with a trained actor when one becomes available.
- [ ] Use those results to tune coarse-context caching, feature-worker allocation,
  and cross-game concurrency/batching within bounded memory. Change one factor
  at a time and retain settings only for measured end-to-end benefit.
- [ ] Audit cell-size/budget choices for missed captures/saves, including a
  coordinate-first control, then compare playing strength in the trial. Treat
  tactical coverage as a correctness/strength question; do not shrink budgets
  merely to accelerate the currently small exact-feature/scorer stages.

## 2. Organize ratings and evidence

- [ ] Order `matches.csv` newest batch first, games stable within each batch.
  Keep rating reconstruction independent of display order; check refresh/recovery.
- [ ] Give rating SGFs readable, collision-safe dated names including players,
  board size, game number, and a hash. Preserve full identities, deduplication,
  and explicit ledger-to-file lookup.
- [ ] Update CLI/docs and support migration without losing ledger/evidence history.

## 3. Prepare and run a small-board learning trial

- [ ] Implement explicit bootstrap-only eye avoidance using shared
  `SmallEyeAnalysis`, preserving the capture exemption. Keep game legality and
  neural legality masks unchanged; check tactical eye fills and pass behavior.
- [ ] Generate genuine bootstrap search visits and natural outcomes with that
  guidance; record its configuration in SGFs. Do not invent policy/outcome labels.
- [ ] Configure size-9-only self-play and training sampling consistently.
- [ ] Set reference/actor match budgets, simulations, caps, and actor-match
  thresholds for the trial; report uncertainty and truncation/censoring bias.
- [ ] Run a short inspected trial from random weights with fresh optimizer state
  and fresh bootstrap data, before committing to a long campaign.
- [ ] Evaluate without eye guidance: eye preservation, connections, captures/saves,
  sensible passing, natural completion, and matches against the reference bot.
  Keep guidance disabled for normal neural self-play/evaluation.

## Later, if measurements justify it

- [ ] Profile `PreparedPosition::analyze_move` internals if exact move analysis
  becomes a significant end-to-end cost: legality, tactical summaries, discarded
  child construction, repeated scans, and allocations. Add a shared-rules
  summary-only path only if justified; differential-check against the validator,
  including captures and suicide.
- [ ] Tune scorer batching and cell budgets for speed if their measured cost
  becomes significant; retain tactical-coverage and strength checks.
- [ ] Investigate CPU/GPU pipelining and GPU-resident contexts if the remaining
  compute/transfer split justifies them. Loader/transport measurement is above.
- [ ] Batch independent evaluation games.
- [ ] Measure rollout cost/strength before adding it to the supervised loop/rating protocol.
- [ ] Tune search/training budgets and manually scale network capacity; consider
  learned group embeddings only with evidence of better strength per compute.
- [ ] Integrate browser ONNX inference and verify parity/operator support.
- [ ] Move reusable components into the main ContinuGo repository with confirmation.
