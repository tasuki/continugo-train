# Representative pipeline profiling

Use real positions and end-to-end work before optimizing a synthetic kernel.
The synthetic [model benchmark](network.md#format-and-benchmarks) excludes rules,
features, transport, and search; it cannot identify pipeline bottlenecks.

## Run

Provide a current checkpoint and its matching exported ONNX bundle. The profiler
checks embedded checkpoint hashes and the feature contract. Use trusted
checkpoints only (as with training/export).

```sh
# Completed dense source games; no invented training targets.
mise run selfplay -- --engine random --games 3 --sizes 9,13,19 \
  --size-weights uniform --seed 0 --output runs/profile-positions

mise run pipeline-profile -- --positions runs/profile-positions \
  --model runs/model-001 --checkpoint runs/train-001/best.pt --device cpu
# Repeat sequentially with --device cuda, not alongside another benchmark.
```

Omitting `--output` reserves a dated `runs/*-pipeline-profile` directory. Explicit
output must not exist. The command builds release binaries before measurement;
compilation and Python runtime-location work are excluded from native timings.
Density is currently fixed at four samples per diameter.

The profiler:

1. Validates source SGFs with Rust. For each size 9/13/19 it deterministically
   chooses the longest completed game (at least ten moves), then inspects early
   (10% of non-final-pass turns), middle (50%), and late (before the final two
   passes) pre-move positions. These are phase **proxies**; inspect the SGFs for
   actual stones, connections, captures, and tactical diversity.
2. Runs policy and value-only native queries on all nine positions. One warmup,
   then `--iterations 3` measured queries by default. Context caching is disabled
   so repeated queries do not hide coarse/rules extraction and encoding.
3. Generates six genuine neural MCTS games, two per size, seeds 800–805, eight
   simulations, cap 512. Default production cache settings apply here; games run
   sequentially. Records completion, truncation, throughput, and inference stages.
4. Validates/deduplicates the generated games for training and creates a game-level
   holdout. `--data PATH...` can supply a fixed real-policy corpus instead. Missing
   labels are excluded, never manufactured; insufficient completed data fails
   explicitly after retaining self-play diagnostics.
5. Measures warm persistent feature-server round trips on actual eligible policy
   positions (first/middle/last eligible ply per size). These cells come from
   genuine visits, unlike dense random-bot games without policy targets.
6. Runs real FP32 SGD twice from the supplied weights and fresh optimizer state:
   an ordinary baseline and synchronized stage diagnostics. Each gets one untimed
   warmup step, then weights, BatchNorm buffers, RNG and optimizer are reset.
   Default caps: 12 steps, batch two, at most 12 sampled positions per game, two
   feature workers. Short caps can skew the size mix; inspect `samples_by_size`.
   No checkpoints are saved or promoted. This is a cost diagnostic, not a trial.

## Output and interpretation

`summary.json` retains configuration, revision/dirty status, source hashes, model
hashes, corpus manifest, per-position results, self-play totals, and training
metrics. Raw native stdout/stderr and generated SGFs remain beside it. Intermediate
results are saved before subsequent stages, so a corpus error does not lose the
native measurements. SGFs remain the only primary game records.

- Native `stats`: coarse/rules features, move preparation/selection/exact features,
  encoder, scorer, counts, and cache payloads. `feature_seconds` is the sum of
  coarse/move times, **not an additional disjoint stage**. Scorer timing includes
  host output handling/transfer; coarse includes shared-rule analysis. Overall
  query seconds include normalization, dense result construction and other work.
- Training baseline: wall time, loader wait, compute plus transfer, positions/s,
  memory, losses, and actual board mix. Stage diagnostics separately synchronize
  transfer, encoding, policy scoring/normalization, and loss/backward/optimizer.
  Synchronization can perturb overlap; use the ordinary baseline for throughput.
  Small CUDA backward differences are possible even with the same initialization.
- Feature-server round trip includes extraction, serialization, pipe wait,
  decoding and tensor copies. It is **not pure transport time**. Loader wait is
  exposed consumer stall, not total worker CPU time; prefetch overlaps compute.
- Linux `/usr/bin/time` records native process-lifetime peak RSS, including model
  load, validated SGF replay and warmup. Elsewhere it is null. Training parent RSS
  excludes Rust children and is cumulative for the profiler process. CUDA training
  memory is peak live PyTorch allocation, not reservation/context. Native GPU peak
  allocation is not measured by this command. Cache bytes are payload, not RSS.

Do not add overlapping stage totals or compare process startup with steady-state
query times. Repeat with multiple seeds, representative trained weights, larger
training windows, and controlled thread/cache settings before claiming a general
speedup. Random-weight early passing is not playing strength. Retain dated results
in [history](../history/), not in this operating guide.

## Lower-level diagnostics

`mise run infer -- ... --profile-json` adds `seconds`, `iterations`, and structured
`stats` to its usual JSON output; ordinary output remains unchanged. Warmup/model
load/replay and diagnostic output summarization are excluded from query timing.
Warmup retains cache contents: use `--cache-positions 0` to measure uncached work.

Self-play's `--stats-json NEW_FILE` writes machine-readable game totals and native
inference stages without changing SGF annotations. Existing files are not replaced.
For random-engine candidate/eye timings, use the separate existing `--profile`
option. See [native inference](inference.md) and [training metrics](training.md#measurements).
