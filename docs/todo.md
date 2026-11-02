# ContinuGo AI — Open work

Use [KataGo's self-play training](https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md) as the primary reference, adapting it to ContinuGo. Prioritize playing strength per unit of computation.

This list contains outstanding work only. See [training](training.md), [inference](inference.md), [batching](batching.md), [loop](loop.md), [evaluation](evaluation.md), and [model storage](model-store.md) for implemented capabilities and operating instructions.

## High priority

### Launch a user-approved long learning campaign

The [v5 loop](loop.md) implements the agreed [training and promotion policy](training-policy.md):
training and experimental actors advance independently, while every accepted
publication needs explicit hash-pinned user approval. Initial checkpoints and
favorable matches no longer grant acceptance. No promotion is authorized until
convincingly much stronger than the reference bot.

- [ ] Launch a resumable week-long campaign from the fresh `cold1` lineage, with periodic health/strength checks and a possible second-week extension. Keep network-size changes manual; do not substitute prolonged fitting of the bootstrap corpus for fresh self-play. Manual acceptance is a separate operation, not a prerequisite for continued learning.

### Calibrate evaluation before scaling

The [first bounded overnight run](overnight-2026-11-01.md) completed but exposed
opening-pass collapse. [Cold-start search tests and a fresh-network diagnostic](cold-start.md)
are underway; confirm meaningful play before scaling. Evaluation settings remain
provisional. Network-size changes stay manual.

- [ ] Validate the cold-start search changes with fresh-network opening/pass diagnostics, natural completion rates and reference matches. Exact terminal pass checks are implemented; optional rollout evaluation is experimental, not the default. Longer games or lower loss alone do not establish improvement.
- [ ] Calibrate acceptance/reference match counts, simulation budget, move caps, and acceptance threshold; measure actual strength gains against the fixed-100 reference bot and historical checkpoints, not just validation loss or gate acceptance. The CSV rating pipeline is implemented; the default budgets are not strength-tuned.
- [ ] Check rating uncertainty and per-size outcomes with sufficient completed games. Measure truncation/censoring bias and reference-match cost before scaling. A small finite-match gate is a regression guard, not statistical proof of improvement; ratings remain separate from the promotion decision.

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
- Failed or inconclusive candidates retain training and optimizer progress. Experimental actors may advance independently of acceptance; optional actor gating must not rewind training. Accepted-network changes always require explicit user approval. Never silently migrate old automatic-promotion loops or reset historical accepted metadata.
- Keep SGFs self-contained and generated artifacts inside this repository. Durable recovery must survive removal of `runs/`; ONNX alone is not a training checkpoint. Preserve exact replay data, holdout history, hashes, and store-owned deduplication when changing artifact layouts.
