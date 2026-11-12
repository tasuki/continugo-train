# D4 augmentation versus the frozen 4×32 baseline

## Outcome and stop

Stopped at the user's request on 2026-11-12 after **17 completed epochs of the
planned 20**. Training and the automatic report updater are stopped; nothing is
queued. Epoch 17 finished and its atomic checkpoint was saved before termination.
Any subsequent unsaved work is discarded. No self-play, promotion or rating jobs
were launched, and the original baseline files remain byte-identical to the copy.

**D4 substantially reduced overfitting and improved held-out value, score and
ownership losses. It did not improve policy loss, nor beat the baseline's best
early joint validation loss.** This supports keeping augmentation enabled for the
next campaign, not claiming that every head or playing strength has improved.

## Controlled recipe

Experiment: `scratch/20261112-113200-d4-augmentation/`.

The completed `scratch/20261111-230713-network-size-cached/b4c32/` run was copied
into `baseline-b4c32/`, including checkpoints and logs. The augmented `b4c32/` run
started from **fresh initialization**, not from the baseline or cg6 weights.
The original frozen runtime was copied, with only `augmentation.py`, `data.py`
and `train.py` changed. The original Rust binary and frozen SGFs were copied;
cache sidecars retained links to the original caches. The corpus fingerprint and
inherited game-disjoint split were asserted; `launch.json` records runtime hashes.

Both runs used:

- Four residual blocks × 32 channels; seed 20261111; FP32 CUDA.
- Batch 32; SGD LR .01, momentum .9, weight decay .0001; gradient clipping 5.
- The same 907 train / 236 validation games: **90,256 / 22,245 positions per epoch**.
- Exhaustive positions, 9-only board-size weights, six loader workers and four
  Torch threads; score weight 1 and ownership weight .25.
- The same fixed inference-mode train/validation probes after every epoch.

The only training-recipe change was uniform random selection of one of eight
square symmetries per training position per epoch, using an independent seeded
RNG. Coarse inputs and ownership targets are rotated/reflected together; action
coordinates and directional border features transform consistently. Candidate
rows are not reordered: policy indices, visits, legality and pass retain their
meaning. Outcome and score are unchanged. Validation and probes are unaugmented.
There are no extra position exposures. This transforms decoded tensors rather
than re-extracting transformed SGFs; raster discretization/antialiasing need not
be bit-identical to re-extraction.

## Matched-epoch results

Epoch 17 corresponds to **1,534,352 training position exposures** per run.
Lower losses are better. The training measurements below are online training-mode
metrics; they are not fixed inference-mode probes.

| Metric | Baseline train | Baseline validation | D4 train | D4 validation | Validation Δ D4 − baseline |
|---|---:|---:|---:|---:|---:|
| Joint loss | 8.663163 | 10.157603 | 9.758823 | 9.895657 | -0.261947 |
| Policy CE | 8.080769 | 8.201794 | 8.423216 | 8.487290 | +0.285497 |
| Value MSE | 0.204884 | 1.195566 | 0.728547 | 0.797547 | -0.398019 |
| Score MSE | 0.175965 | 0.574378 | 0.408859 | 0.429750 | -0.144628 |
| Ownership MSE | 0.809315 | 0.851714 | 0.797960 | 0.825315 | -0.026399 |

The joint train/validation gap shrank from **1.494441 to 0.136834**. The original
run's value and score training losses kept falling while validation worsened;
D4 largely prevented that divergence. A smaller gap alone is not evidence of a
better model, but these heads also have lower absolute validation losses.
Policy is the exception: it remains worse with D4 at the matched budget.

Best joint validation checkpoints:

- Baseline: **9.624640 at epoch 3** (over the full original 20 epochs).
- D4: **9.844191 at epoch 15** (over the completed 17 epochs).

Thus D4's best joint validation loss is **0.219551 higher**, despite its better
late-epoch behavior. The baseline's epoch-20 loss was 10.138777; comparing only
late epochs would conceal its better early checkpoint.

### Fixed inference-mode probes at epoch 17

These small probes corroborate the per-head trade-off, but should not replace
the exhaustive holdout. They select up to 16 positions from each of 16 games per
split using the original fixed sampling procedure.

| Validation probe metric | Baseline | D4 |
|---|---:|---:|
| Joint loss | 10.548731 | 10.871864 |
| Policy CE | 9.224407 | 9.679017 |
| Value MSE | 0.893151 | 0.782306 |
| Score MSE | 0.220092 | 0.153703 |
| Ownership MSE | 0.798530 | 0.770373 |

Both probes run in inference mode. Value, score and ownership again improve;
policy and joint loss do not. This is one seed on one frozen size-9 corpus, with
checkpoint selection using the same validation split—not a playing-strength test.
D4 reached epoch 17 after approximately 109 minutes including validation, probes
and plotting; the comparison is matched by exposures, not wall time.

## Decision for the next campaign

Keep **D4 as the default for new training and warm-start jobs**. Old resumes
preserve their original saved augmentation recipe; validation stays unaugmented.
The user agreed to the proposed cg7 recipe: warm-start cg6 weights, reset SGD
momentum, retain the latest 1,000 unique eligible games with inherited game-level
split assignments, add fresh self-play, and review after one exhaustive epoch.
**No cg7 run was launched by this experiment.**

Overfitting is not by itself a reason to throw away cg6's learned weights. This
experiment did not compare warm-starting cg6 against fresh initialization; that
remains a separate question. Track policy and absolute held-out losses, not just
the size of the train/validation gap, during the next review.

## Checks and saved artifacts

33 targeted augmentation, training, auxiliary and cache tests passed. Symmetry
tests cover all eight distinct transforms and inverses, directional fields,
schema-4/5 widths, empty candidate sets, pass/index preservation, deterministic
seeds, epoch variation, unchanged validation and unchanged exposure counts.
The broader test-discovery invocation timed out; it was not a full-suite pass.

Within the experiment directory:

- `metrics.png` / `comparison.png`: two columns (baseline and D4), five metric
  rows, training versus validation in each cell, shared scales across columns.
- `baseline-metrics.json` / `metrics.jsonl`: complete raw comparison data and probes.
- `b4c32/epoch-0017.pt` / `latest.pt`: last completed checkpoint.
- `b4c32/epoch-0015.pt` / `best.pt`: best joint-validation D4 checkpoint.
- `baseline-b4c32/`: untouched copy of the original baseline.
- `status.json`, `run.log`, `launch.json`, `corpus-manifest.json`: stop state,
  console history, provenance and frozen split metadata.
