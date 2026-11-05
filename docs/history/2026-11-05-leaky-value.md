# 2026-11-05 leaky-ReLU value continuation

Following the [inactive-value-head diagnostic](2026-11-05-value-head-collapse.md),
the user approved switching to leaky ReLU and continuing existing training.
The previous campaign finished all 60 global rounds; no live job was replaced.

## Implementation and compatibility

`NetworkConfig.value_activation` is now explicit: `relu` or `leaky_relu`, with
leaky negative slope **0.1**. Historical configs without the field still mean
ReLU. The value head alone changes activation; the trunk and policy keep their
existing activations. Parameter names, shapes, feature schema and policy objective
are unchanged.

An explicit trainer `--warm-start --value-activation leaky_relu` copies all weights
and BatchNorm buffers. `--warm-start-optimizer keep` preserves every SGD momentum
buffer. The activation change and parent hash are recorded in initialization
provenance. Strict resume rejects an activation change; subsequent jobs inherit
it normally. Loop checks compare canonical configs so historical omitted ReLU
and newly explicit ReLU remain compatible. Old checkpoints/ONNX graphs are not
silently reinterpreted or overwritten.

All **150 Python tests passed**, including negative-preactivation gradient
regression, exact pre-update parameter/momentum retention, provenance, strict
resume guards, historical-config loop recovery, and native CPU/CUDA ONNX parity.
Ruff checks passed. No native executable change was required; this continuation
uses byte-identical frozen pass-balanced binaries.

## Checked warm-start

Run: `runs/20261105-092434-leaky-value/`.
Source: final global-round-60 durable checkpoint, SHA-256
`06a812a79cbb2f0a674ad865af0634557e5a5b51f4c8a6bd3116fe925af181c4`.

The recovery phase started at **09:38:24 UTC** and finished at **09:39:18 UTC**:

- One ordinary joint policy/value epoch on the existing 512-game replay.
- All weights, BatchNorm, SGD momentum, ancestral holdouts and lineage counters
  retained; no value-head reinitialization or optimizer reset.
- Warm-start RNG reseeded normally (seed 2,103,000); local job epoch/step counters
  restarted, cumulative optimized rows advanced from 490,000 to **492,000**.
- The eligible-data counter stayed **249,987**: no data was regenerated/invented.
- Before training, value-loss gradients through the first value layer, output
  weights and trunk became nonzero with leaky ReLU. Disposable in-memory probes
  did not change saved weights or BatchNorm statistics.
- Training value MSE: **1.053110** across the epoch's updating model; held-out
  validation value MSE: **0.968804**, 408 sampled rows. These are not strength
  measurements or a direct same-row comparison against the previous job.

On the **same** stage-balanced diagnostic positions used for the collapse report,
using the unchanged 102 held-out games:

| Position | Old ReLU MSE | After one leaky epoch | Always-zero MSE |
|---|---:|---:|---:|
| Opening | 1.0818 | 1.1087 | 1.0000 |
| Midpoint | 1.0259 | 0.9957 | 1.0000 |
| Two plies remaining | 0.9700 | 0.8553 | 1.0000 |
| One ply remaining | 1.0706 | 0.8444 | 1.0000 |

These held-out assignments were checked against the new checkpoint's ancestral
split ledger. Across all 664 diagnostic positions, prediction standard deviation
changed from exactly zero to **0.1022**, with predictions ranging -0.1304 to
+0.1817. Value gradients remained nonzero after the recovery epoch. Opening error
has not improved, and playing strength is not yet established, but the value head
is no longer merely an output-bias predictor on this sample.

Raw probes: `before-relu.json`, `before-leaky.json`, `after-recovery.json`, and
`recovery-comparison.json` under the new run. The last comparison reuses
`value-diagnostics/inputs.npz` and `positions.json` from the previous run.

## Bounded continuation

After the checked epoch, the launcher starts **ten further rounds**, corresponding
to global rounds **61–70**, four training epochs and 50 fresh size-9 games each.
Existing search/training/replay settings are retained: 32 simulations, 1,024-move
self-play cap, 1,000 batches/epoch, six feature workers, 12 active neural games,
512-game / 16-round replay, latest actor, optimizer keep. Natural outcomes and
search visits remain the only labels. First fresh self-play seed: **2,103,000**.

An initial 20-game reference batch precedes new self-play; another is scheduled
at local round ten. Reference cap remains 2,048 and no actor gate is enabled.
The frozen native evaluation protocol is unchanged, so evidence uses the same
rating pool; the new network has its own checkpoint identity and activation
metadata. Durable store remains `models/20261104-151420-size9-training/`.
No accepted publication or score feature is introduced.

```sh
tail -f runs/20261105-092434-leaky-value/run.log
mise run loop-status -- --output runs/20261105-092434-leaky-value/loop
# Resume only with no live launcher; increase the local target explicitly if desired:
.venv/bin/python -u runs/20261105-092434-leaky-value/run.py --iterations 10
```

The launcher checks parent/binary hashes and uses inherited locks/process-group
cleanup. Interrupt its PID with SIGINT; never remove a live lock. Review value
prediction spread, held-out error by stage, natural completion and reference
strength before another large extension. Actual reference/round results should
be appended as they arrive.
