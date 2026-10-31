# Initial policy/value network

`continugo_train.model.PolicyValueNet` defaults to 32 channels, 4 ordinary
BatchNorm/ReLU residual blocks, and configurable input/feature widths. A single
set of weights handles different spatial resolutions; batch similar resolutions
together, without padding. `resolution_for_diameter` consumes the **actual stone
diameter from Rust**, with four coarse intervals per diameter: 38/54/78 grid
samples for diameters 110/76/52 (9/13/19 equivalents).

The value head globally pools the trunk and predicts the side-to-move expected
outcome in [-1,1]. The pass logit also uses global pooling. The coordinate scorer
projects coarse features to 16 channels before bilinear sampling, adds a shared
linear projection of exact move features, then applies ReLU and a scalar linear
readout. No full-resolution convolution is used. Exact features are never
interpolated, so neighboring coordinates can retain different tactical scores.

## Input contract

- Coarse tensor: `[batch, input_planes, height, width]`, side-to-move relative.
- Exact features: `[batch, candidates, move_features]`.
- Coordinate IDs: shared `[candidates]` int64 tensor, `x + 1000*y`, excluding pass.
- Legality: `[batch, candidates]` bool, from the authoritative Rust validator.
- Output: raw logits `[batch, candidates+1]` (pass last), and values `[batch]`.

The initial 16 input planes and 12 exact feature columns are **configurable
placeholder widths, not an implemented feature schema**. Rust extraction of
stone/group/liberty/history/border features and exact per-candidate features is
next. Python neither guesses legality nor implements any rules. The random
bootstrap's eye filter must NOT enter neural policy legality or MCTS.

Full evaluation supplies `torch.arange(1_000_000)`; output columns then match the
shared Rust action IDs, including pass at 1,000,000. Candidate subsets use the
same scoring kernel, but softmax over a subset is only a **conditional** policy.
Illegal coordinates get `-inf`; pass remains finite, even with no legal placement.
Inference uses chunks of 65,536 coordinates by default. Chunking bounds temporary
inference activations; autograd still retains all chunks during training.

`policy_value_loss` combines sparse MCTS visit-count cross-entropy (normalized over
all supplied legal logits plus pass) and value MSE, with equal initial weights.
Zero-count target padding is allowed; positive counts on illegal moves are rejected.
Outcome labels must come from completed games and be flipped to the pre-move
player's perspective. This stage does **not** load SGF, train a real self-play model,
export ONNX, or replace the Rust baseline evaluator yet.

Checkpoints can store `dataclasses.asdict(model.config)` and `model.state_dict()`
in a single `torch.save` file, restoring with `NetworkConfig(**saved["config"])` and
`load_state_dict`; the tests cover configuration and BatchNorm-state round trips.

## Synthetic benchmarks

```sh
mise run model-bench -- --device cuda
mise run model-bench -- --device cpu --batch-sizes 1
mise run model-bench -- --channels 64 --blocks 6 --policy-hidden 32
# Quick small-domain wiring check, not a million-coordinate performance estimate:
mise run model-bench -- --device cpu --actions 128 --iterations 1 --warmup 0 --batch-sizes 1
```

Reports coarse inference, dense policy evaluation with precomputed coarse context,
combined inference, coarse backward/SGD, and full policy/value backward/SGD. The
coarse-only training stage uses a synthetic squared-output objective; the full
step uses sparse visit targets and outcome MSE. All inputs/targets are synthetic
and preallocated on-device. Timings include Python dispatch and (for training)
loss validation and optimizer work, but exclude feature extraction and transfers.
CUDA is synchronized; reported GPU peak memory is total PyTorch live allocation,
including inputs and optimizer state, not reserved memory or other GPU processes.
These numbers are **not** end-to-end self-play throughput or playing strength.

The initial stack pins PyTorch 2.7.1 because CUDA 11.8 wheels support the local
GTX 1080 (Pascal). `mise run setup` uses that wheel index by default; CPU-only
installation can use `TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu`.
FP32 eager execution is the baseline; mixed precision and compilation are not
assumed to benefit this older GPU.
