# ContinuGo AI — Open work

Use [KataGo's self-play training](https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md) as the primary reference, adapting it to ContinuGo. Prioritize playing strength per unit of computation.

This list contains outstanding work only. See [training](training.md), [inference](inference.md), [batching](batching.md), [loop](loop.md), [evaluation](evaluation.md), and [model storage](model-store.md) for implemented capabilities and operating instructions.

## High priority

### 1. Persistent, baseline-anchored static Glicko ratings

- [ ] Keep a durable, long-running table of **all accepted networks**, with readable names, acceptance dates, checkpoint identities, match counts, ratings, and **± rating deviation (RD)**. Preserve it independently of disposable `runs/`.
- [ ] Anchor the existing **random-but-not-eye-filling bot at rating 100**. Freeze/version its reference policy. Calibrate networks through actual matches against it, connecting later network comparisons to that baseline; untested networks remain **unrated** in reports, not assigned a claimed strength.
- [ ] Use **static-skill Glicko** for frozen bots: **no time decay, no inactivity inflation, and no volatility/process-noise inflation**, including on active updates. Ratings and uncertainty change only with new match evidence; the reference stays fixed at 100.
- [ ] Rate neural players at a fixed, recorded **simulation budget per move**, with paired colors. Keep separate rating pools for different budgets or board-size mixes and other strength-affecting evaluation settings. Equal simulations do not imply equal wall-clock compute across architectures. Default board mix remains 40%/40%/20% for 9/13/19.
- [ ] Retain the underlying match evidence, including rejected candidates rather than only successful promotions. Count genuine draws as half-points; report truncations explicitly and exclude unknown outcomes from rating updates. Keep evaluation games out of training replay.
- [ ] Make evidence ingestion restart-safe and idempotent so resumed or reimported matches cannot update ratings twice. Keep rating history auditable from authoritative SGFs; do not introduce game-record JSON sidecars.
- [ ] Wire the shared `../continugo/backend/rating` static-Glicko component and native adapter into the rating ledger; do not duplicate the math in Python. Add end-to-end tests for baseline calibration, persistent history, independent rating pools, and idempotent restart/reimport. The math and stateless adapter are implemented; see [evaluation](evaluation.md#ratings-shared-foundation-integration-pending).

### 2. Calibrate evaluation before scaling

- [ ] Calibrate acceptance-match count, simulation budget, and threshold; measure actual strength gains against the reference bot and historical checkpoints, not just validation loss or gate acceptance.
- [ ] Check rating uncertainty and per-size outcomes with sufficient completed games. A small finite-match gate is a regression guard, not statistical proof of improvement; ratings remain separate from the promotion decision.

## Performance and scaling

- [ ] Improve neural self-play throughput: reduce unnecessary full-domain feature/scoring work and investigate CPU/GPU overlap without compromising independent exploration, exact tactical distinctions, or full-policy normalization. Feature extraction remains the dominant measured cost.
- [ ] Benchmark inference concurrency, batch sizes, cache settings, and feature threads across CPU/CUDA rather than assuming batching is faster. Measure feature extraction, encoder inference, dense policy scoring, and end-to-end self-play separately.
- [ ] Consider batching independent evaluation games after measuring its benefit.
- [ ] Scale network capacity and training/self-play budgets only when justified by measured strength per unit of computation. Keep architecture configurable; consider learned group embeddings or more selective move evaluation only when measurements justify the added complexity.
- [ ] Move stable reusable components into the main ContinuGo repository with confirmation.

## Constraints to preserve

- Shared rules and rule-derived features in `../continugo` are authoritative; do not duplicate their semantics.
- Default to configurable **40% / 40% / 20%** coverage for 9/13/19 across generation, replay, actual training/validation samples, and evaluation.
- Use real search-policy targets and natural game outcomes. Never invent labels for truncated games or turn them into draws. Keep the random bot's narrow eye-avoidance filter separate from neural/unguided MCTS.
- Rejected or inconclusive candidates retain training and optimizer progress; self-play continues with the accepted model.
- Keep SGFs self-contained and generated artifacts inside this repository. Durable recovery must survive removal of `runs/`; ONNX alone is not a training checkpoint. Preserve exact replay data, holdout history, hashes, and store-owned deduplication when changing artifact layouts.
