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
   positions (first/middle/last eligible ply per size). Opt-in server telemetry
   separates replay/target-cell selection, coarse/exact extraction and serialization;
   the client records request writes, response reads, decoding/copies and an IPC
   residual. These cells come from genuine visits, unlike dense random-bot games
   without policy targets.
6. Runs real FP32 SGD twice from the supplied weights and fresh optimizer state:
   an ordinary baseline and synchronized stage diagnostics. Each gets one untimed
   warmup step by default, then weights, BatchNorm buffers, RNG and optimizer reset.
   Default caps: 12 steps, batch two, at most 12 sampled positions per game, six
   feature workers. Short caps can skew the size mix; inspect `samples_by_size`.
   Use `--positions-per-game 0 --steps 120 --warmup-steps 20` for a longer
   fixed-corpus window. These are caps, not promises: corpus exhaustion can end
   a phase early. Each phase opens its own feature workers; this is not a fully
   persistent warmed-loader benchmark. No checkpoints are saved or promoted.
   This is a cost diagnostic, not a trial.

## Output and interpretation

`summary.json` retains configuration, revision/dirty status, source hashes, model
hashes, native binary hashes, corpus manifest, per-position results, self-play
and training metrics. `shared_rules` also records the sibling ContinuGo revision,
dirty status and hashes of all Rust rule sources; local training-repo provenance
alone does not identify changes in a path dependency. Raw native stdout/stderr and generated SGFs remain beside it. Intermediate
results are saved before subsequent stages, so a corpus error does not lose the
native measurements. SGFs remain the only primary game records.

- Native `stats`: coarse/rules features, move preparation/selection/exact features,
  encoder, scorer, counts, and cache payloads. `feature_seconds` is the sum of
  coarse/move times, **not an additional disjoint stage**. Scorer timing includes
  host output handling/transfer; coarse includes shared-rule analysis. Overall
  query seconds include normalization, dense result construction and other work.
  `stats.coarse_stages` splits preparation (including shared group/witness work),
  `liberty_spans`, `capturability`, area integration, and raster
  allocation/painting/borders. Span/capturability clocks live inside the shared
  rule implementation, not a duplicate capture algorithm. The span stage includes
  lazy blocker-row index construction on the first nonempty group; later groups
  reuse it. Raster extraction borrows prepared groups, so the former second
  `group_construction` stage and combined `spans_and_capturability` field are gone.
  These subtotals sit **inside** `coarse_seconds`. They exclude flattening and
  instrumentation/bookkeeping overhead. For concurrent extraction, subtotals
  sum worker wall times and may exceed elapsed batch time; never add them
  to the batch total. Counts include only extractions on cache misses, after
  deduplicating identical batch positions, and exclude diagnostic warmup.
- Training baseline: wall time, loader wait, compute plus transfer, positions/s,
  memory, losses, and actual board mix. Stage diagnostics separately synchronize
  transfer, encoding, policy scoring/normalization, and loss/backward/optimizer.
  Synchronization can perturb overlap; use the ordinary baseline for throughput.
  Small CUDA backward differences are possible even with the same initialization.
- `training_transport[*].feature_profile` records warm round-trip stages.
  `server_coarse_seconds`, `server_exact_seconds`, `server_serialization_seconds`
  and `server_replay_and_selection_seconds` are disjoint server work.
  `response_read_seconds` **includes waiting for that work**, not just transport.
  `ipc_residual_seconds` subtracts server work from request-write + response-read
  time: it includes framing, dispatch, allocation cleanup and scheduling, not pure
  pipe bandwidth. It is a signed residual; do not clamp measurement noise to zero.
  `server_output_write_seconds` includes pipe backpressure and **overlaps** client
  reads; never add it to the residual or round trip. `decode_seconds` contains the
  `decode_arrays_seconds` (header/NumPy copies/conversions),
  `decode_validation_seconds`, and `decode_tensors_seconds` (Torch wrappers/legal
  copy) subtotals. Request encoding/path resolution and timing-trailer reading are
  recorded separately. Telemetry itself adds overhead; ordinary loads have no
  per-stage clocks/trailer.
- Synchronized training diagnostics include the same `feature_profile` aggregated
  across workers. These are **summed worker wall times**, including prefetch and
  any samples extracted before early shutdown, not disjoint training wall time.
  Loader wait is exposed consumer stall, not total worker CPU time; prefetch
  overlaps compute. Ordinary baseline throughput has no feature telemetry.
- Linux `/usr/bin/time` records native process-lifetime peak RSS, including model
  load, validated SGF replay and warmup. Elsewhere it is null. Training parent RSS
  is cumulative and excludes Rust children. Diagnostic `child_peak_rss_mib` reads
  each Rust worker's Linux `VmHWM`; `sum_child_peak_rss_mib` sums those lifetime
  peaks, **not a simultaneous process-tree peak**. Unsupported readings are null.
  CUDA training reports both peak live PyTorch allocation (`cuda_peak_mib`) and
  reservation (`cuda_reserved_peak_mib`), neither of which includes all driver memory.
- On Linux CUDA runs with `nvidia-smi`, native commands and synchronized training
  also sample driver-reported compute-process memory for the measured process
  tree, excluding unrelated GPU users. `sampled_gpu_peak_mib` includes contexts,
  native ONNX allocations, and reservations outside live Torch tensors. This is a
  **sampled**, not exact, peak; the polling gap is at least 100 ms plus query cost.
  `gpu_memory_samples` counts successful polls; missing support/errors yield null
  if no poll succeeds. Monitoring can perturb timings. Native samples include
  startup/replay/warmup; training samples include sampler teardown in phase time.
  Cache bytes remain payload, not RSS.

Do not add overlapping stage totals or compare process startup with steady-state
query times. Repeat with multiple seeds, representative trained weights, larger
training windows, and controlled thread/cache settings before claiming a general
speedup. Random-weight early passing is not playing strength. Retain dated results
in [history](../history/), not in this operating guide.

For a controlled worker sweep, reuse `--data` with unchanged games, weights,
thread/batch settings and sample caps, changing only `--workers`. Keep caps above
corpus exhaustion when checking full-corpus counts. Worker counts change the
loader's round-robin sample order and hence SGD trajectories, even with the same
seed; compare throughput and actual board counts, not loss as a strength result.
Worker scaling need not be monotonic because of shard imbalance/backpressure.
The [feature-transport diagnostic](../history/2026-11-04-feature-transport.md)
found six workers fastest among the tested counts on the current machine. Six is
now the user-selected default for fresh training, new loops and profiling;
`--workers` can override it. Recheck allocation on other machines/workloads.

## Lower-level diagnostics

`mise run infer -- ... --profile-json` adds `seconds`, `iterations`, and structured
`stats` to its usual JSON output; ordinary output remains unchanged. Warmup/model
load/replay and diagnostic output summarization are excluded from query timing.
Warmup retains cache contents: use `--cache-positions 0` to measure uncached work.

`--profile-json` also enables per-group coarse clocks; ordinary queries avoid
those clocks. Self-play's `--stats-json NEW_FILE` enables the same clocks and
writes machine-readable game totals and native inference stages without changing
SGF annotations. Existing files are not replaced.
For random-engine candidate/eye timings, use the separate existing `--profile`
option. See [native inference](inference.md) and [training metrics](training.md#measurements).
