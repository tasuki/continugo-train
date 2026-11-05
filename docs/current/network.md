# PyTorch policy/value network

## Architecture

`PolicyValueNet` defaults to 32 channels and four BatchNorm/ReLU residual blocks.
One set of weights handles multiple spatial resolutions; batches group matching
resolutions. The value head globally pools the trunk and predicts side-to-move
outcome in [-1,1]. Pass also uses global pooling.

The value head is linear → activation → linear → tanh. Its checkpoint setting
`value_activation` is `relu` or `leaky_relu` (fixed negative slope **0.1**).
Historical checkpoints without this field retain ReLU; they are never silently
reinterpreted. New jobs can explicitly choose `--value-activation leaky_relu`.
A parameter-free activation change is supported by an explicit training
warm-start, preserving all weights and optionally SGD momentum; strict resume
cannot change it. It does not change the policy head, feature schema or parameter
shapes. Activation choice is recorded in checkpoint and ONNX metadata.

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

Only selected cells receive exact move analysis: every integer coordinate inside
them, once. No overlapping windows or full-board exact scan. Within each cell,
the coordinate scorer bilinearly samples 16-channel projected coarse context,
adds a shared linear projection of exact features, applies ReLU, and reads out a
scalar. Exact features are never interpolated. Pass is always available.

## Probabilities and training objective

The coarse softmax covers **all cells and pass**. Each nonempty cell has a local
softmax over all its exact legal coordinates. Training uses hierarchical weights
`P(cell) * P(coordinate | cell)`; pass uses its global coarse probability.

Native inference preserves this **global pass/placement split**. Let `S` be the
selected cells containing at least one legal coordinate, and `p = P(pass)`:

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
- Value MSE has equal weight to total coarse-plus-local policy CE.

`hierarchical_policy_value_loss` implements this factorized proposal objective.
It does **not** apply another candidate-only softmax. The objective is not the
inference-time CE: inference conditions the placement stratum on selected nonempty
cells while preserving the global coarse pass probability. Coarse proposals can put mass on empty
cells during learning; no claim of an exhaustively legal-normalized distribution
is made. Illegal positive targets are rejected. Ragged batches pad exact rows with
false legality and remap pass to the final column; pass-only targets need no rows.

Compared with a single candidate-union softmax, this hierarchy provides direct
whole-board supervision for selection without scoring a million exact moves or
estimating their partition function. Coverage/strength remain experimental.

## Format and benchmarks

Checkpoint and ONNX metadata require policy version **`cells-v1`**. The old dense
policy is not supported: start fresh training and export new graphs. Existing
validated SGF visit/outcome records remain usable. Feature schema 4 is unchanged.
The pass-preserving normalization correction changes native execution only, not
weights, features, graph outputs, or the training objective. Existing `cells-v1`
checkpoints/graphs remain compatible; changed native fingerprints require a new
loop/protocol pool rather than replacing a running loop's frozen binaries.

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
