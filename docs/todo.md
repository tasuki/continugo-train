# ContinuGo AI — Open work

Prioritize playing strength per unit of computation. Use [KataGo's self-play training](https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md) as a reference, adapting it to ContinuGo.

## High priority

### Fresh size-9 bootstrap with temporary eye-avoidance guidance

- [ ] Start from random weights, fresh optimizer/run state, and fresh size-9
  bootstrap data, without reusing the abandoned campaign's checkpoints or replay.
  Begin with a short inspected trial.
- [ ] Add an explicit bootstrap-only eye-avoidance option using the shared
  `SmallEyeAnalysis` predicate. Preserve the capture exemption; do not introduce
  a blanket ban on all eye-filling moves.
- [ ] Define how eye avoidance guides bootstrap search/proposals. Generate genuine
  search visit targets and natural outcomes, not fabricated policies from random
  moves or outcome labels for truncated games.
- [ ] Keep the heuristic separate from game legality and neural legality masks.
  Record its configuration in game/run provenance; test captures, necessary
  tactical eye fills, legal-mask invariance, and filtered/unfiltered behavior.
- [ ] Check terminal/pass behavior when heuristic-acceptable placements run out;
  do not equate heuristic rejection with the absence of legal placements.
- [ ] Evaluate the resulting network/search with guidance disabled on held-out
  eye/tactical positions and reference matches. Disable guidance for normal neural
  self-play/evaluation; configure and record any staged phase-out explicitly.

### Short-game board-size curriculum

- [ ] Start learning on size 9 for shorter games and faster outcome feedback, then
  introduce size 13 and later size 19. Test mixes such as
  100/0/0 → 70/30/0 → 40/40/20; choose transitions from observed play, not loss alone.
- [ ] Apply the curriculum consistently to self-play allocation and replay/training
  sampling. Record the configured mix and actual samples; make curriculum overrides
  of the normal 40/40/20 board mix explicit.
- [ ] Periodically evaluate frozen checkpoints on sizes 9 and 13, including before
  introducing size-13 training. Track eye preservation, connections, captures/saves,
  sensible passing, natural completion, and per-size playing strength.
- [ ] Measure transfer of local skills separately from larger-board strategy and
  value calibration. Introduce larger boards once basic habits emerge rather than
  waiting for size-9 mastery; retain smaller-board data to check forgetting.
- [ ] Compare strength gains per wall-clock hour against mixed-size training,
  measuring completed games/hour and per-query cost separately.

### Calibrate evaluation before scaling

- [ ] Validate cold-start search with fresh-network opening/pass diagnostics,
  natural completion rates, and reference matches; require meaningful play rather
  than just longer games or lower loss.
- [ ] Calibrate acceptance/reference match counts, simulation budgets, move caps,
  and acceptance thresholds. Measure strength against the fixed-100 reference bot
  and earlier checkpoints, not just validation loss or gate acceptance.
- [ ] Quantify rating uncertainty and per-size outcomes with enough completed
  games. Measure truncation/censoring bias and reference-match cost.
- [ ] After the bounded trials demonstrate useful learning, launch a resumable
  long campaign with periodic health/strength checks. Keep network-size changes
  manual and accepted-model publication explicitly user-approved and hash-pinned.

## Performance and scaling

### Reduce exact-feature work

- [ ] Profile shared `PreparedPosition::analyze_move` internally, separating legality,
  tactical summaries, child-position construction, scans, and temporary allocations.
- [ ] Investigate an authoritative summary-only API that avoids discarded child
  positions and repeated work. Compare results against the existing shared
  validator, including captures and suicide; do not duplicate rule semantics.
- [ ] Investigate CPU/GPU overlap, GPU-resident encoded contexts, and training
  extraction/transport overhead.
- [ ] Benchmark concurrency, batch sizes, cache settings, and feature threads across
  CPU/CUDA. Measure coarse extraction, exact rows, encoder, scorer, loader wait,
  memory, and end-to-end self-play/training throughput separately.

### Coarse-to-fine policy with overlapping candidate windows

- [ ] Add a cheap learned coarse-location selector, then compute exact features
  for every integer coordinate in selected neighborhoods and refine with the
  exact scorer. Compare against sampling coordinates before feature extraction.
- [ ] Use overlapping windows wider than coarse-grid spacing to tolerate selection
  errors near boundaries. Union/deduplicate coordinates before rule evaluation
  and scoring.
- [ ] Compare window sizes and selection budgets by exact coordinates evaluated,
  end-to-end time, coverage of strong MCTS moves, narrow tactical opportunities,
  and playing strength.
- [ ] Preserve independent whole-board exploration and a separate pass action.
  Validate every considered coordinate with authoritative exact rules.
- [ ] Redesign policy normalization/training explicitly for selective scoring.
  Compare a candidate-based objective with sampling-bias handling against a
  hierarchical location/coordinate policy with defined proposal probabilities
  and illegal-move handling. Prevent overlapping windows from double-counting
  action probability; do not silently replace full-domain normalization with a
  candidate softmax. Version model/objective compatibility as needed.

### Other scaling work

- [ ] Benchmark batching independent evaluation games.
- [ ] Measure cost/strength before exposing standalone rollout evaluation in the
  supervised loop/rating protocol.
- [ ] Integrate exported ONNX inference in the browser and verify runtime/operator
  support against native outputs.
- [ ] Scale network capacity and training/search budgets only when justified by
  measured strength per unit of computation; compare learned group embeddings
  if measurements justify the added complexity.
- [ ] Move stable reusable components into the main ContinuGo repository with
  confirmation.

## Ratings organization

- [ ] Write `networks.csv` with newly registered networks first, using an explicit
  registration timestamp and deterministic tie-breaker rather than hash order.
  Keep each network's protocol-pool rows together; do not reorder by rating or by
  refresh time. Keep the fixed reference clearly identifiable.
- [ ] Write `matches.csv` newest batch first, with stable game ordering within a
  batch. Separate display order from ledger reconstruction so newest-first storage
  cannot restore stale rating snapshots. Test refresh, restart, and crash recovery.
- [ ] Replace hash-only rating-game names with readable, collision-safe paths such
  as `ratings/games/YYYY-MM-DD/HHMMSS-batch/0001-size9-BLACK-vs-WHITE-HASH.sgf`.
  Include concise player names and retain full hashes/identities in the ledger;
  preserve deduplication and evidence integrity. Make ledger-to-SGF lookup explicit.
- [ ] Update ratings CLI output, storage documentation, and tests for ordering and
  naming. Define migration for existing ledgers/evidence without losing history.

## Implementation safeguards

- [ ] Preserve training/optimizer progress across inconclusive gates; reset only
  for an explicit fresh campaign. Keep experimental actor advancement independent
  of accepted-model publication.
- [ ] Test checkpoint/replay compatibility and recovery across these changes.
  Preserve self-contained SGFs, exact replay data, holdout history, hashes,
  deduplication, and recovery without `runs/`; keep generated artifacts in this repo.
