# 2026-11-05 value-head diagnostic: inactive ReLU layer

Both campaign segments completed: 30 original rounds plus 30 pass-balanced rounds.
The final corrected checkpoint is
`runs/20261104-215404-pass-balanced/loop/iteration-0030/train-attempt-0001/epoch-0004.pt`
(SHA-256 `06a812a79cbb2f0a674ad865af0634557e5a5b51f4c8a6bd3116fe925af181c4`).

## Concrete finding

The value head is `Linear(32,32) -> ReLU -> Linear(32,1) -> Tanh`.
At the final checkpoint, **all 32 hidden preactivations were nonpositive on every
one of 664 sampled positions**. Thus ReLU emits an all-zero vector, and the value
prediction is just `tanh(output_bias)` = **0.1424806565**, independent of the board
on this sample. Prediction standard deviation was exactly zero.

All 664 positions were also tested in deterministically shuffled batch-two
training mode: zero active hidden units, maximum preactivation -0.00218, and the
same constant prediction. A training-mode, value-loss-only backward probe had
zero active units too. Gradient norms for the first value layer (weights and bias), final value
weights and first trunk convolution were **zero**. Only the final value bias
received a loss gradient (norm 1.680222). Momentum/weight decay can still move
other parameters; this finding concerns the value loss's learning signal.
No model was optimized or overwritten by the diagnostic. The subsequent
[user-approved leaky-ReLU continuation](2026-11-05-leaky-value.md) unblocked value
gradients without resetting learned weights or SGD state; its checked first epoch
also improved held-out late-game prediction.

## Sample and method

Read-only Rust feature extraction from the final checkpoint's corpus manifest:
64 deterministically sampled training games and all 102 held-out games of at
least 16 plies. Four positions per game: opening, midpoint, two plies remaining,
and one ply remaining. Exact features and outcomes were supplied by the frozen
Rust feature server with file-hash checks. No approximate SGF replay or invented
labels were used. Games were kept in their original train/validation splits.

Held-out MSE at the final checkpoint:

| Position | Samples | Network MSE | Always-zero MSE |
|---|---:|---:|---:|
| Opening | 102 | 1.0818 | 1.0000 |
| Midpoint | 102 | 1.0259 | 1.0000 |
| Two plies remaining | 102 | 0.9700 | 1.0000 |
| One ply remaining | 102 | 1.0706 | 1.0000 |

The lower error in the penultimate row reflects target imbalance versus a
constant positive prediction, not board discrimination. Stage-balanced sampling
is diagnostic and differs from ordinary randomly sampled validation rows.

The same positions were evaluated with twelve saved checkpoints. Already after
the first bootstrap epoch, predictions varied only from 0.3645 to 0.3773; most
nonopening positions had no active hidden units. Original rounds 20 and 30 had
prediction ranges only about 0.00005 and 0.00016 wide. Corrected rounds 10, 20 and
30 were exactly constant on all 664 positions. The collapse/near-collapse
predates the normalization correction; this does not establish exactly which
optimization choice originally caused it.

## Interpretation and next experiment

This is a concrete explanation for flat value loss, not merely speculation about
unpredictable early outcomes. The critic has mostly become a bias estimator, not
a position evaluator. Policy learning can still proceed through its separate
head/trunk gradients; terminal search scoring still works, but ordinary learned
leaf estimates cannot distinguish boards on this sample.

A controlled next experiment should unblock the value head with a non-dying
activation (e.g. SiLU or leaky ReLU), possibly reinitializing only that head and
its optimizer state if necessary, while preserving the policy/trunk. First verify
nonzero gradients and learnability on held-out late positions before another long
joint campaign. A score feature may still help, but adding information alone does
not address the demonstrated zero-gradient pathway. No such model change was
implemented by this investigation.

Reproduction and full raw results:
`runs/20261104-215404-pass-balanced/value-diagnostics/{probe.py,probe.log,report.json,positions.json,inputs.npz}`.
The probe updates BatchNorm buffers only in its disposable in-memory training-mode
copy; saved checkpoints, optimizers and SGFs remain unchanged.
