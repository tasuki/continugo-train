# Network size comparison — 2026-11-11–12

## Question and conclusion

Does increasing depth or width improve held-out learning on the existing self-play
corpus? **This screen gives no convincing reason to enlarge the network.** All
four architectures overfit: larger models mostly improve training fit, not
held-out performance. This is a finding about this corpus and training recipe,
not a universal capacity ceiling or a playing-strength result.

## Completed experiment

Artifacts: `scratch/20261111-230713-network-size-cached/`.
All four models completed **20 exhaustive epochs**, finishing
**2026-11-12T06:51:42 UTC**, about **7 h 43 min** after launch.

- **Data:** cg5 rounds 80–100 and completed cg6 round 1; discarded cg6 round 2
  excluded. 1,150 source records deduplicate to **1,143 games / 112,501 positions**.
  Inherited game-disjoint splits: train **907 games / 90,256 positions**;
  validation **236 games / 22,245 positions**. All games are size 9.
- **Models:** fresh 4×32, 8×32, 4×64 and 8×64; one shared seed, **20261111**.
  No inherited weights, optimizer momentum or BatchNorm state.
- **Training:** batch **32**, constant SGD **lr=0.01**, momentum 0.9,
  weight decay 1e-4, gradient clipping 5; CUDA FP32, four Torch threads,
  six loader workers; required existing feature caches (`--feature-cache require`).
- **Held fixed:** features, head widths, leaky outcome activation, score loss
  weight 1, ownership loss weight 0.25, and size-9 reference shuffling.
  Full training and validation passes each epoch; no position/step caps.
- **Diagnostics:** fixed inference-mode probes (up to 16 positions from each of
  16 games per split), including outcome calibration bins. These small probes
  supplement, not replace, exhaustive validation.

Source/frozen SGF hashes, cache indexes and the inherited corpus fingerprint were
verified. Corpus fingerprint:
`736ee97e4d64f8b8625088592cccba9c6c265b020b5f97151b55a57e920f8f76`.
Protected source loop/checkpoint hashes were unchanged on completion.

## Results

Lower losses are better. Best total and best policy values may come from
**different checkpoints**; there is no single epoch optimizing every head.

| Blocks × channels | Best validation total | Epoch | Best validation policy CE | Training run time¹ |
|---|---:|---:|---:|---:|
| 4×32 | **9.62464** | 3 | 8.17935 | 1.75 h |
| 8×32 | 9.66020 | 3 | 8.18208 | 1.96 h |
| 4×64 | 9.67123 | 2 | 8.16821 | 1.81 h |
| 8×64 | 9.64144 | 3 | **8.16056** | 2.18 h |

¹ Per-model elapsed time through final epoch metrics, including startup,
validation and probes; not inference latency. Runs were sequential.

Final-epoch train / full-validation errors:

| Architecture | Policy CE | Outcome MSE | Score MSE | Ownership MSE |
|---|---:|---:|---:|---:|
| 4×32 | 8.071 / 8.179 | 0.169 / 1.213 | 0.165 / 0.561 | 0.807 / 0.853 |
| 8×32 | 8.047 / 8.187 | 0.145 / 1.169 | 0.156 / 0.535 | 0.768 / 0.855 |
| 4×64 | 8.009 / 8.182 | 0.122 / 1.204 | 0.142 / 0.535 | 0.782 / 0.845 |
| 8×64 | 7.990 / 8.187 | 0.108 / 1.276 | 0.133 / 0.534 | 0.719 / 0.862 |

Training metrics are accumulated in training mode while weights change;
validation uses the end-of-epoch model in inference mode. Exact gaps therefore
need care, but inference-mode probes also show outcome generalization gaps.

## Interpretation

- **All heads show overfitting, not just outcome/score.** Policy validation
  largely plateaus while training policy continues improving. Small late
  held-out gains do not overturn that pattern. Ownership validation similarly
  plateaus or worsens. Outcome and score show the strongest deterioration.
- **More epochs on this fixed corpus are not the promising direction.** Best
  combined validation errors occur at epochs 2–3, well before training ends.
- **Extra capacity mainly improves memorization.** Wider models achieve small
  best-policy gains, but no convincing overall held-out advantage. With one
  seed and checkpoint selection on this validation set, small differences are
  exploratory, not reliable architecture rankings.
- **Limited independent data is the leading hypothesis, not a demonstrated
  cause.** 112k correlated positions from 1,143 nearby self-play games provide
  much less diversity than 112k independent examples. Optimization/BatchNorm,
  target quality, loss balance and train/validation distribution differences
  could also contribute. Similar epoch-specific spikes across architectures
  warrant checking shared ordering and evaluation effects.
- Caching and batching made the experiment practical. Larger-network training
  costs rose much less than nominal trunk compute ratios; this does **not**
  establish their inference cost or strength per unit move time.

## Suggested follow-ups — not launched

1. Audit generalization measurements: larger fixed inference-mode training
   probes, split distributions, targets/masks/loss denominators, and whether
   BatchNorm-statistics recalibration changes validation behavior.
2. Test more **unique, diverse games**, ideally with a fixed-recipe data-volume
   learning curve. Verify board-symmetry augmentation rather than assuming it.
3. Favor fresh replay data over many passes on the same window. Test stronger
   regularization, learning-rate changes or auxiliary-loss reweighting separately
   if needed; this experiment did not establish which remedy works.

Retain 4×32 as the economical baseline for further experiments. No architecture
promotion follows from these losses. No playing matches, inference benchmarks,
self-play continuation, shared ratings changes or model publication were run or
queued; any such tests need separate budgets and approval.

## Reproducibility and prior attempt

The completed experiment contains `launch.json`, runtime/configuration hashes,
`corpus-manifest.json`, split ledger, all epoch checkpoints, `metrics.jsonl`,
`run.log` and final `status.json`. Its `metrics.png` compares the usual five loss
graphs side by side for all four architectures; additional plots compare full
validation against training exposures and per-model wall time.

The initial batch-2/five-epoch attempt in
`scratch/20261111-211434-network-size-comparison/` was stopped before any full
epoch completed. The completed batch-32/20-epoch run is a revised optimization
budget, not an equivalent replay of that plan. Cache-build artifacts remain in
`scratch/20261111-214102-feature-sidecars/`; the batch-size pilot is summarized
separately in [the batch-size screen](2026-11-11-training-batch-size.md).
