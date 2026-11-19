# PyTorch policy/value network

**Schema 6 inputs: 14 coarse planes and 9 exact fields (fresh-training default).** Group size
and total per-group liberty area use capped log scaling, saturating at 100.
Exact legality and resulting group size are removed; the separate legality mask
is unchanged. Both spatial liberty representations and exact splitting remain.
Schema-4 and schema-5 inference keep their own input contracts and scaling;
schema-6 training starts fresh. See the [schema-6 feature contract](features.md#schema-6-contract).

## Architecture

`PolicyValueNet` defaults to 64 channels and eight BatchNorm/ReLU residual blocks.
Blocks **3 and 6** additionally use learned global pooling (`global-pooling-v1`).
A parallel 3×3 convolution → BatchNorm → ReLU produces **16 global channels**;
concatenated spatial mean/max are linearly projected to 64 channels and broadcast
across the local branch, after its first BatchNorm and before ReLU. Later spatial
layers can therefore condition local decisions on whole-board context, rather
than leaving global information only to the output heads.

`--gpool-every 3` and `--gpool-channels 16` configure the interval and width;
`--gpool-every 0` disables trunk pooling for ablations. Block numbering is one-based.
This follows KataGo's pool/project/broadcast mechanism, not its exact architecture:
we omit the board-size-scaled mean, since raster dimensions also depend on feature
density. No board padding/mask is needed; training batches share raster dimensions.
One set of weights handles multiple spatial resolutions; batches group matching
resolutions. A shared value/score branch projects the trunk to 16 spatial channels
with leaky-ReLU, concatenates spatial mean/mean-square/max, and maps 48 pooled
features to 32 hidden units. Separate scalar outputs predict side-to-move outcome,
score mean and score standard deviation. Ownership projects the shared spatial
features to one channel. Pass independently uses global trunk pooling.

The shared dense layer uses linear → activation; outcome and score mean use
linear → tanh. Its checkpoint setting
`value_activation` is `relu` or `leaky_relu` (fixed negative slope **0.1**).
Fresh models default to `leaky_relu`. The `shared-gaussian-v1` architecture requires fresh training; earlier independent
value/score checkpoints are not supported.
A parameter-free activation change is supported by an explicit training
warm-start, preserving all weights and optionally SGD momentum; strict resume
cannot change it. It does not change the policy head, feature schema or parameter
shapes. Activation choice is recorded in checkpoint and ONNX metadata.

The network also predicts **final score and ownership** using authoritative
completed-game targets. Score predicts a mean and uncertainty; ownership has the coarse raster
resolution and is training-only. See [auxiliary supervision](auxiliary.md) for
labels, losses, search utility and diagnostics.

The coarse raster has four intervals per stone diameter by default: 38/54/78
samples per side for 9/13/19. **Each sample owns the integer coordinates nearest
it**, giving non-overlapping cells; edge cells are half-width. For an axis
coordinate `x`, its cell index is `(x * (resolution - 1) + 499) // 999`.
This partitions coordinates 0..999 exactly, including board margins; rules decide
which placements are legal. Cell spacing is roughly a quarter stone diameter.

A cheap learned head scores every cell. Inference selects **32 top cells plus 8
pseudorandom remaining cells**, configurable with training/export architecture
options `--top-cells` and `--exploration-cells` (both positive). Exploration uses
the game's search RNG seed and position, independent of batching/cache order.
Thus different games can explore different cells even at the same opening.

Default schema-6 inference uses the [geometry proposal policy](geometry-proposals.md)
to choose coordinates inside selected cells; only those coordinates receive
exact move analysis. It reuses the same encoder/scorer without changing the
training contract. Legacy schema-4 models and `infer --complete-cells` analyze
every integer coordinate inside selected cells. No overlapping windows or full-board exact scan. Within each cell,
the coordinate scorer bilinearly samples 16-channel projected coarse context,
adds a shared linear projection of nine exact features, applies ReLU, and reads
out a scalar. Schema-6 column 4 is binary `splits_friendly_region`: using actual
child survivors, does placing the new stone split one friendly-bounded
available-center region into at least two nonempty components? Exact integer
scanline connectivity retains narrow splitting moves; there is no per-candidate
area integration or raster dilation. This is a learned geometric input, not an
explicit eye reward or proof of life. Exact features are never interpolated. Pass is always
available. Legacy schema-4 models retain ten exact features.

With default widths and schema-6 inputs, the network has **627,687 trainable
parameters**: trunk 622,656 (including 22,592 for global context); policy branches including pass 2,307; shared value
spatial/dense branch 2,608; outcome/score mean/score stdev outputs 33 each;
ownership 17.
Region indices are shared within a position; exact captured-survivor indices use
a bounded cache. See the [feature definition](features.md#binary-splitting-predicate)
and [replacement costs](../history/2026-11-08-friendly-region-splitting.md).
The synthetic benchmark below preallocates exact inputs and does not measure
extraction cost.

## Probabilities and training objective

The coarse softmax covers **all cells and pass**. Each nonempty cell has a local
softmax over all its exact legal coordinates for training. Training uses hierarchical weights
`P(cell) * P(coordinate | cell)`; pass uses its global coarse probability.

Native inference preserves this **global pass/placement split**. Let `S` be the
selected cells containing at least one retained legal coordinate, and `p = P(pass)`.
Native local probabilities are normalized over each cell's shortlisted legal
coordinates, not the complete training domain:

- Pass retains probability `p`.
- Selected cells divide the entire placement mass `1-p`, proportional to their
  coarse weights: `P(move) = (1-p) * P(cell)/sum_S P(cell) * P(move | cell)`.
- Illegal/unselected coordinates remain unavailable. If `S` is empty, pass has
  probability one; this is not proof that the whole board has no legal placement.

Do **not** normalize the selected hierarchical weights together with pass: dropping
placement cells that way artificially boosts pass (for example, 10% can become
74%). Computation uses stable log probabilities for both strata. This remains a
selective policy, not an exhaustive full-board legal softmax. Cell selection
changes placement locations, not the learned pass/placement balance.

Training uses genuine sparse MCTS visit counts and natural game outcomes:

- Coarse cross-entropy uses visits summed by cell, with pass separate; its
  denominator always covers the entire coarse grid and pass.
- Local cross-entropy uses every legal coordinate in each target-bearing cell,
  weighted by that cell's fraction of total visits. Rust extracts **complete cells
  containing positive placement targets**, not just visited coordinates. Cells
  with zero target mass need no exact features, so their omission introduces no
  sampled-denominator bias or importance correction.
- Value MSE has equal configured weight to total coarse-plus-local policy CE.
  Randomized-budget SGFs mask cheap positions out of policy CE only, normalizing
  over enabled positions per batch (zero when none). Value uses all sampled
  positions; Gaussian score NLL and area-weighted ownership MSE do too, each with
  weight 0.5 by default. See [loop budgets](loop.md#randomized-policy-search-budget).

`joint_loss` adds the auxiliary terms to `hierarchical_policy_value_loss`, which
implements this factorized proposal objective.
It does **not** apply another candidate-only softmax. The objective is not the
inference-time CE: inference conditions placement on selected nonempty cells and
shortlisted coordinates while preserving the global coarse pass probability. Coarse proposals can put mass on empty
cells during learning; no claim of an exhaustively legal-normalized distribution
is made. Illegal positive targets are rejected. Ragged batches pad exact rows with
false legality and remap pass to the final column; pass-only targets need no rows.

Compared with a single candidate-union softmax, this hierarchy provides direct
whole-board supervision for selection without scoring a million exact moves or
estimating their partition function. Coverage and playing strength require validation.

## Format and benchmarks

Checkpoint and ONNX metadata require policy version **`cells-v1`** and value
version **`shared-gaussian-v1`**, plus trunk version **`global-pooling-v1`**.
The trunk change requires fresh training and new exports. The old dense
policy is not supported: start fresh training and export new graphs. Existing
validated SGF visit/outcome records remain usable. New training uses feature
schema 6 with fourteen coarse planes: antialiased occupancy, connected group
size, spatial liberty centers/footprints, per-group total liberty area and borders;
no capturability flags. Schema-4 score/ownership inference and viewing explicitly preserve all
twelve original coarse inputs and ten exact fields. Schema-5 inference/viewing uses
its own twelve coarse planes and eleven exact fields, restored from Git commit
`71f8daf`. Mixed-schema arenas extract inputs independently for each model;
schema-5 checkpoints/tensors cannot initialize schema-6 training. Metadata names
and widths must match the selected numbered schema.
See [features](features.md).
The pass-preserving normalization correction changes native execution only, not
weights, features, graph outputs, or the training objective. Current exports additionally require `targets: final-area-v1`; old policy/value-only
checkpoints/graphs are incompatible. Changed native fingerprints require a new
loop rather than replacing a running loop's frozen binaries. The new loop uses
the same shared model store and global rating table.

```sh
mise run model-bench -- --device cuda
mise run model-bench -- --device cpu --top-cells 16 --exploration-cells 4
```

The synthetic benchmark reports coarse encoding/proposals, selected-cell scoring,
combined inference, coarse backward/SGD, and hierarchical policy/value training.
Exact inputs are preallocated; it excludes Rust extraction, transport, proposal
selection, and game search. `--actions` supplies an approximate coordinate budget,
rounded to complete cells. Timings are not end-to-end playing strength; retain
measurements in [benchmarks](../benchmarks.md).

Training/checkpoint contracts are in [training](training.md); native execution and
search sampling are in [inference](inference.md). Execution is FP32 eager (PyTorch
2.7.1 / CUDA 11.8 for the local Pascal GPU), without mixed precision or compilation.
