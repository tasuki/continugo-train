# ContinuGo AI — Open work

This is a living document. When something is implemented, remove it from here.

Prioritize playing strength per wall-clock hour. Complete optimization and ratings
cleanup before the next training trial. Use [KataGo's self-play training](https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md) as a reference, adapting it to ContinuGo.

## 1. Optimize the pipeline

- [ ] Profile `PreparedPosition::analyze_move`: legality, tactical summaries,
  child-position construction, repeated scans, and allocations.
- [ ] Add a shared-rules summary-only analysis path that avoids discarded child
  positions and repeated work. Verify against the existing validator, including
  captures and suicide.
- [ ] Compare cell-size/budget choices by tactical coverage and end-to-end speed,
  then playing strength in the trial. Include coordinate-first sampling as a control.
- [ ] Benchmark feature extraction, loader wait/transport, encoder, scorer, memory,
  and overall throughput. Tune concurrency/batching/cache settings from measurements.

## 2. Organize ratings and evidence

- [ ] Order `networks.csv` newest registration first with deterministic ties and
  each network's pool rows together; keep the reference clearly identifiable.
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

- [ ] Investigate CPU/GPU pipelining, GPU-resident contexts, and loader/transport overhead.
- [ ] Batch independent evaluation games.
- [ ] Measure rollout cost/strength before adding it to the supervised loop/rating protocol.
- [ ] Tune search/training budgets and manually scale network capacity; consider
  learned group embeddings only with evidence of better strength per compute.
- [ ] Integrate browser ONNX inference and verify parity/operator support.
- [ ] Move reusable components into the main ContinuGo repository with confirmation.
