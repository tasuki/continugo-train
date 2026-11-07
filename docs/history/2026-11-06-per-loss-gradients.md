# Per-loss gradients, cg4 round 18 — 2026-11-06

User requested measurement before changing loss weights. Read-only diagnostic in
`scratch/20261106-213310-loss-gradients/`: checkpoint copy, metadata, source, pinned
sample references, per-batch JSON, summary, descent projections and
**`gradients.png`**. The active network, optimizer, runtime and campaign settings
were not modified.

## Method and checks

- Frozen round-18 epoch-4 checkpoint, family cg4, **154,000 optimized rows** /
  53,629 generated rows, SHA-256
  `4e7760c092fcfceb0961a547b644adc9254b5ddc4635a85c6d08be84be93fa5c`.
- **256 minibatches / 512 training positions**, batch two, production sampling
  rules and final-epoch seed/position planning. 146 positions have policy
  supervision: 128 batches contain at least one, 128 contain none. All critics
  use all positions. First four batches are bitwise checked against the actual
  six-worker production loader (all tensors), despite serial diagnostic feature
  extraction limiting CPU contention.
- CPU FP32 model operations, one Torch thread, lower OS scheduling priority.
  Train-mode BatchNorm, restoring checkpoint buffers before each batch. Each
  batch has one shared forward graph; separate autograd gradients isolate
  policy, value, score and ownership at identical activations.
- Actual loss weights **1 / 1 / 1 / 0.25**, including policy masking and its
  enabled-position normalization. Unused gradients are zero. Weighted component
  gradients sum to the direct total-loss gradient (checked on two batches).
- No optimizer is instantiated or stepped. Parameter `.grad` fields remain None;
  all copied model parameters and restored buffers equal the starting checkpoint
  afterward. Checkpoint file hashes are unchanged. This measures a current
  fixed-checkpoint gradient distribution, not historical online optimizer steps.

## Results

Mean **weighted shared-trunk gradient L2 norms**:

| Loss | All 256 batches | 128 policy-active batches |
|---|---:|---:|
| Policy | 0.8569 | **1.7137** |
| Outcome/value | 0.5318 | **0.5827** |
| Score | 0.0561 | **0.0595** |
| Ownership | 0.0460 | **0.0448** |

On policy-active batches, policy is roughly **2.9× value** and **28.8× score** at
the shared trunk. On the other half of batches, policy contributes exactly zero.
Across **all parameters**, value is larger on average (1.357 versus policy 0.954),
because its own head has substantial gradients. Do not claim policy dominates
every parameter or every update.

Own-head mean gradient norms on policy-active batches:

- Policy heads: 0.7410.
- Value head: **1.2389**.
- Score head: **0.3469**.
- Ownership head (after 0.25 weight): 0.1135.

There is no dead-gradient explanation for either critic head here. Score's direct
contribution to shared-trunk learning is comparatively small despite weight 1;
scalar loss coefficients do not equalize gradient influence.

### Direction, not just magnitude

Shared-trunk cosine similarities, policy-active batches:

| Pair | Mean cosine | Fraction negative |
|---|---:|---:|
| Policy / value | **+0.010** | 52.3% |
| Policy / score | **-0.009** | 54.7% |
| Value / score | **+0.544** | 15.6% |

Policy is mostly near-orthogonal, not consistently opposed to both critics.
Value and score gradients tend to align, supporting keeping both rather than
sacrificing score. Negative minibatch cosines alone do not prove harmful
interference or poor generalization.

Global gradient clipping at 5 activates in **6/256 batches (2.34%)**, all
policy-active. Average clip scale is 0.9963 overall, 0.9925 on active batches.
Clipping is not the main explanation for weak critic learning in this snapshot.

## Counterfactual reweighting — no training performed

Recombine the same measured gradients with policy weights 1, 0.25 or 0, keeping
value/score at 1 and ownership at 0.25. On policy-active batches:

| Quantity | Policy 1 | Policy 0.25 |
|---|---:|---:|
| Combined trunk-gradient alignment with value | 0.359 | **0.773** |
| Combined trunk-gradient alignment with score | 0.209 | **0.456** |
| Value-opposing trunk direction | 4.7% of batches | 0% |
| Score-opposing trunk direction | 18.8% | 14.8% |

However, improved cosine **does not mean stronger critic descent**: removing an
orthogonal vector improves angle simply by reducing the total vector's length.
The mean first-order descent proxy `clip_scale * dot(total_gradient, critic_gradient)`
across all parameters is essentially unchanged for value, **2.6807→2.6812**, and
slightly lower for score, **0.2664→0.2641**. Shared-trunk-only proxies are also
slightly lower on average (value 0.5861→0.5661, score 0.0594→0.0562).
These are instantaneous per-batch proxies; they exclude optimizer momentum and
weight decay, curvature, later representation changes and held-out outcomes.

## Interpretation

There is real evidence that policy controls much of the shared representation's
update direction on enabled batches. Reducing policy weight would rebalance that
movement while retaining critic supervision. But the measurement **does not
establish that policy is simply starving/blocking the critics**, nor that weight
0.25 will improve learning or reduce bad passes. Healthy head gradients, rare
clipping and almost unchanged first-order critic descent are important caveats.

A fixed-data matched-checkpoint loss-weight A/B remains justified. Measure held-out
midgame calibration and the same pass/search probes after actual training; do not
infer strength from gradient angles alone or discard the prior evidence of weak
midgame targets/representation. No loss weight or active campaign was changed.
