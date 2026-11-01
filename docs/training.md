# Offline policy/value training

This is a real SGF → Rust features → PyTorch training job, without ONNX. It learns
from saved baseline MCTS games; it does **not** generate neural-guided self-play
or establish playing strength yet. The first goal is a usable initial checkpoint
and verified data/optimization plumbing, not prolonged fitting of weak data.

## First run

After `mise run setup`, generate a small corpus with policy visits and scored RE:

```sh
mise run selfplay -- --engine mcts --games 24 --sizes 9,13,19 --seed 100 --simulations 32 --output runs/bootstrap
mise run train -- --data runs/bootstrap --output runs/train-001 --device cuda --epochs 3 --steps-per-epoch 20 --positions-per-game 8 --validation-positions-per-game 4
```

Use a new output directory. `--device auto` (default) chooses CUDA when available,
otherwise CPU; `--device cpu` forces CPU. Inspect options with
`mise run train -- --help`. Architecture options include `--channels`, `--blocks`, `--policy-hidden`,
`--value-hidden`, and `--chunk-size`. Defaults are the existing 32-channel/4-block
network, FP32 eager mode, SGD lr 0.01 / momentum 0.9 / weight decay 1e-4, gradient
norm clipping at 5, batch size 2, and two feature workers. This fits the local GTX
1080 comfortably; mixed precision is not assumed to help Pascal.

The example deliberately limits the run. With omitted limits (all default to
zero), each epoch uses every eligible position once, and validation evaluates all
held-out positions. Nonzero `--positions-per-game` chooses a fresh reproducible
subset each training epoch. `--validation-positions-per-game` chooses a fixed
seeded subset per held-out game. `--steps-per-epoch` caps optimizer batches;
otherwise an epoch exhausts the data. Final partial batches are retained.

**Random SGFs have no MCTS policy labels.** They are not silently assigned played-
move targets. This initial joint policy/value trainer uses only positions with
both real visits and a completed-game outcome. It reports skipped unlabeled,
no-policy and duplicate games. Invalid moves, policies or scores fail rather than
being silently skipped. At least two eligible unique games in one resolution
bucket are needed to obtain a held-out set. Train board sizes are 9..19.

## Data loading and validation

- Rust validates **every** move/policy in each SGF and checks scored RE against
  authoritative final scoring. Value targets are side-to-move win/loss/draw,
  not score margins. Missing RE means no value label, even after two passes.
- A canonical identity hashes board size plus decoded plays. Copies with changed
  names/comments/seeds cannot land in both train and validation. Duplicate games
  use one deterministic representative (lowest raw-file hash, then path).
- A seeded split holds out **whole unique games**, stratified by input resolution
  when a bucket has at least two games. Singleton buckets stay in training. Common
  openings may still occur in different games; positions are not globally deduped.
- Each background thread owns a persistent Rust process, which retains immutable
  pre-move snapshots of one validated game. A game is replayed/scored once per
  worker visit, not once per sampled position. Workers never touch CUDA.
- Games and within-game positions are shuffled; workers are consumed in stable
  round-robin order. A bounded position shuffle buffer (default 8) mixes their
  output. Positions are batched by actual resolution without spatial padding.
- Every sample includes **all million coordinates plus pass** for exact policy
  normalization, not a sampled/conditional action softmax. Sparse visit targets
  are padded with zero counts; no dense target distribution is allocated.

SGFs stay the primary game records. The Rust `features --inspect` command emits
metadata only. `features --serve` accepts local JSON request lines containing a
path, ply, density and expected file hash; replies are status-u8 / length-u64
framed CGFEAT01 tensors or UTF-8 errors. These are ephemeral IPC commands/tensors,
**not JSONL game records or SGF sidecars**. The tensor schema did not change.

Corpora are immutable during a job. Workers check the inspected raw-file SHA-256
when loading a game, then serve that snapshot. Do not point a run at files being
rewritten by self-play. The implementation stores per-game snapshots, so unusually
long games consume more worker memory. Dense feature packets are never cached as
an entire corpus: one packet is about 45 MB. Reduce `--shuffle-buffer`,
`--batch-size`, or `--workers` for less host memory. Worker queues are bounded;
early epoch termination closes readers and reaps Rust processes.

## Checkpoints and resume

An epoch produces `epoch-0001.pt`, etc., plus atomic `latest.pt` and `best.pt`.
Best means the lowest held-out **policy CE + value MSE**, not playing strength.
Checkpoints contain weights and BatchNorm buffers, SGD momentum/state, optimizer
step count, completed epochs, RNG states, architecture/options, feature names,
existing compatibility marker/density, and an embedded corpus/split manifest.
No separate game-data manifest file is written.

```sh
mise run train -- --data runs/bootstrap --resume runs/train-001/latest.pt --epochs 5
# Continue with fuller training passes while preserving the same held-out set:
mise run train -- --data runs/bootstrap --resume runs/train-001/latest.pt --epochs 6 --steps-per-epoch 0 --positions-per-game 0
```

`--epochs` is a **total target**, not an additional count. Without it, a resume
runs one more epoch. Resume in the checkpoint's existing run directory; newer
epoch files prevent accidentally rewinding/overwriting progress. Training settings
are restored. Batch size, workers, shuffle buffer, training position/step caps,
learning rate, gradient clipping and policy chunk size may be explicitly changed.
Feature semantics/density, architecture weights, split seed/fraction and validation
sample selection must match. Changed eligible SGF contents/targets are rejected;
relocating an otherwise identical corpus is allowed.

Checkpoints are at **completed epoch boundaries**, after validation. Ctrl-C does
not save a partial epoch. Resume the latest completed one; if none completed,
start a fresh run directory. A `.train.lock` prevents concurrent writers; after a
hard crash, remove it only after confirming no trainer is alive. Writes use
same-directory temporary files plus atomic replacement. Existing epoch snapshots
are not overwritten. GPU training is not promised bitwise deterministic
(`grid_sample` backward uses atomics); unchanged single-thread CPU continuation
is tested against an uninterrupted run, including optimizer state. Changing
workers/device/batch settings can also change numerical or sampling order.

Load a trained model for later inference/export:

```python
import torch
from continugo_train.model import NetworkConfig, PolicyValueNet

saved = torch.load("runs/train-001/best.pt", map_location="cpu", weights_only=True)
model = PolicyValueNet(NetworkConfig(**saved["config"]))
model.load_state_dict(saved["model"])
model.eval()
```

Resume refuses incompatible feature definitions. No new feature-schema revision
was introduced just for this training command. Extending training to a changing
neural self-play corpus is future loop work, not the fixed-corpus resume contract.

## Measurements

`training.log` and stdout report policy cross-entropy, value MSE, sample/batch
counts, end-to-end positions/s, loader wait, compute-plus-transfer time, CUDA peak
live allocation and parent peak RSS. Metrics are weighted by **positions**, not
batches, so leftovers do not bias validation. CUDA timings synchronize. Phase
elapsed time includes loader startup/shuffle fill and shutdown of prefetched work;
wait and compute subtotals need not sum to it.

CUDA memory excludes reserved allocator/context/desktop usage. Parent peak RSS is
the process lifetime high-water mark including Python threads, **not** Rust child
RSS. More workers/buffering use more memory; they do not eliminate the current
full-domain Rust feature extraction and binary transport bottleneck. See
[benchmarks.md](benchmarks.md) for the actual first run and its limitations.
