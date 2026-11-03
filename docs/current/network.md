# PyTorch policy/value network

## Architecture

`PolicyValueNet` defaults to 32 channels and four BatchNorm/ReLU residual blocks.
Input/feature widths and head sizes are configurable. One set of weights handles
multiple spatial resolutions; batches group matching resolutions without padding.

The value head globally pools the trunk and predicts side-to-move outcome in
[-1,1]. The pass logit also uses global pooling. The coordinate scorer projects
coarse context to 16 channels before bilinear sampling, adds a shared linear
projection of exact move features, then applies ReLU and a scalar linear readout.
There is no full-resolution convolution. Exact features are not interpolated, so
neighboring coordinates can have different tactical scores.

## Output and loss

Inputs, labels, and transport are defined in [features](features.md). Outputs are
logits `[batch, candidates+1]` (pass last) and values `[batch]`. Illegal coordinates
get `-inf`; pass stays finite without legal placements. Coordinate subsets use the
same kernel, but a subset softmax is conditional unless every legal move is present.
Chunking bounds inference intermediates; autograd retains all chunks for backward.

`policy_value_loss` combines sparse MCTS-visit cross-entropy normalized over legal
logits plus pass with outcome MSE, using equal weights. Zero-count target padding
is allowed; positive counts on illegal moves are rejected. Outcomes come from
naturally completed games, flipped to the pre-move player's perspective. Joint
training uses the full domain, not a visited-action softmax.

Training/checkpoint/resume contracts are in [training](training.md); native ONNX
export/runtime and sampled search priors are in [inference](inference.md).

## Synthetic benchmark command

```sh
mise run model-bench -- --device cuda
mise run model-bench -- --device cpu --batch-sizes 1
mise run model-bench -- --channels 64 --blocks 6 --policy-hidden 32
# Small-domain wiring check, not million-coordinate throughput:
mise run model-bench -- --device cpu --actions 128 --iterations 1 --warmup 0 --batch-sizes 1
```

The command reports coarse inference, dense policy evaluation with precomputed
context, combined inference, coarse backward/SGD, and full policy/value training.
Coarse-only training uses a synthetic squared-output objective; the full step uses
sparse visit targets and outcome MSE. Inputs/targets are synthetic and preallocated
on-device. Timings include dispatch and loss/optimizer work, exclude Rust feature
extraction/transfers, and synchronize CUDA. Peak GPU memory is live allocation,
not reserved/context/other-process usage. These measurements are not end-to-end
self-play throughput or playing strength; retain results in
[benchmarks.md](../benchmarks.md).

Setup pins PyTorch 2.7.1 with CUDA 11.8 wheels for the local Pascal GPU. CPU-only
installation can set `TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu`.
Execution is FP32 eager; mixed precision and compilation are not enabled.
