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
  --size-weights uniform --seed 0 --output scratch/profile-positions

mise run pipeline-profile -- --positions scratch/profile-positions \
  --model scratch/model-001 --checkpoint runs/train-001/best.pt --device cpu
# Repeat sequentially with --device cuda, not alongside another benchmark.
```

Omitting `--output` reserves a dated `scratch/*-pipeline-profile` directory. Explicit
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
   simulations, cap 512. Default production cache settings apply here. Concurrency
   defaults to twelve games / batch limit twelve / collection wait two ms, capped
   by the six games requested. Records completion, truncation, throughput and
   inference stages. Override with `--concurrent-games`, `--inference-batch-size`
   and `--inference-batch-wait-ms`. For historical sequential comparisons, pass
   `--concurrent-games 1 --inference-batch-size 8 --inference-batch-wait-ms 2`.
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
  instrumentation/bookkeeping overhead. `coarse_stages.raster` further splits
  `painting` into allocation, group shapes, occupancy, liberty union, group
  reduction, liberty centers, liberty footprints and borders (schema 5; legacy
  painting is not fully subdivided). These are nested subtotals, not additional
  coarse work. For concurrent extraction, subtotals
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

## Benson versus eye heuristics

The opt-in Rust benchmark lives in `src/action/analysis_bench.rs` in **this repo**;
it reads the shared analysis/rules crates without changing `../continugo/`.
No network, GPU, training run, or model is required. Supply completed SGFs with
at least ten moves. Each contributes early/middle/late snapshots using the same
phase proxies described above. Replay and geometric-index initialization are
outside the repeated timers. The availability list is rebuilt from each snapshot;
its order is not reconstructed from the original self-play trajectory.

```sh
# Reuse dense random games, or generate sources with the selfplay command above.
export CONTINUGO_BENCH_SGFS="runs/20261104-representative-positions/game-000001-size9-seed0.sgf:runs/20261104-representative-positions/game-000002-size13-seed1.sgf:runs/20261104-representative-positions/game-000003-size19-seed2.sgf"
CONTINUGO_BENCH_OUTPUT="scratch/$(date -u +%Y%m%d-%H%M%S)-benson/results.json" \
  CONTINUGO_BENCH_FULL_SCAN=1 mise run analysis-bench
```

The output path must be new. Defaults: five samples, minimum 50 ms per measured
batch; override `CONTINUGO_BENCH_SAMPLES` and `CONTINUGO_BENCH_MIN_MS`. Calibration
batches and warmups are discarded. Each timed operation constructs and drops a
fresh analysis, so repeated positions cannot hide work behind retained caches.
Run sequentially without competing benchmarks. This uses the release **libtest
system allocator**, not the production executable's mimalloc.

The JSON preserves source/game hashes, revisions and dirty status, exact plies,
counts, individual sample times/iteration counts, and median/min/max per operation:

- `benson_both_colors`: complete Benson extraction for **both colors**, including
  groups, geometry, allocations, pruning, and destruction. No raster painting.
- `eye_trace`: only the cached eye queries actually needed for a random decision,
  for the side to move. The real selector is first profiled outside timing; its
  proposal prefix is replayed to recover legal, noncapturing eye queries, and
  query/rejection counts are asserted equal. The timed trace uses a fresh
  `SmallEyeAnalysis` and the existing availability index. No rule validation is
  included. Seeds are `0..samples`; each sample fixes that seed during its timing
  batches. Zero means that seed needed no eye queries, not a measured zero-cost
  function. This median describes both query-order variation and timing noise.
- `random_move`: the actual unprofiled production selector, with fresh RNG/cache
  per decision and the same seeds. Includes preparation, legality, capture checks,
  and eyes; **eye time is inside this total**, not an additional stage. Excludes
  index construction/synchronization, committing the chosen move, and passing.
- `eye_full_scan` (only with `CONTINUGO_BENCH_FULL_SCAN=1`): both colors over every
  geometrically available coordinate, including suicidal moves, sharing one
  fresh pocket cache per scan. Reports query and eye-filling counts. Query-vector
  construction is outside timing. This intentionally exhaustive diagnostic is
  **not** the normal random-selector workload and can take minutes in openings.

Benson certifies pass-alive chains; the eye heuristic rejects candidate fills.
They are not interchangeable computations. In particular, a fast whole-board
Benson result versus an exhaustive eye scan does not establish a self-play
speedup: normal play often asks only one eye question. Neither policy nor rules
are changed by the benchmark. Keep measured results in dated history notes.

## Historical footprint audits

The parent/child and shortcut diagnostics below retain the **replaced local
footprint contract**, not current schema-5 column 6. They explicitly extract the
test-only old reference. Use the region-splitting comparison for the current
replacement; coarse inputs are unchanged.

### Whole-game parent/child feature audit

This opt-in release diagnostic requires a naturally completed, scored SGF with
current-format genuine policy targets. It extracts the historical footprint
contract at density four at every pre-move ply, using complete positive-visit target cells. It does
not run a network/search, change SGFs, or install a production feature cache.

```sh
OUT="scratch/$(date -u +%Y%m%d-%H%M%S)-parent-child-features"
mkdir "$OUT"
CONTINUGO_CACHE_SGF=GAME.sgf CONTINUGO_CACHE_OUTPUT="$OUT/results.json" \
  cargo test --locked --release --lib \
  features::cache_bench::profile_parent_child_features -- --ignored --exact
```

The output file must be new. `CONTINUGO_CACHE_REPEATS` defaults to three ordinary
whole-game repeats. Replay, cell selection and audit work are outside extraction
timers. Coarse extraction and the separately prepared fine extractor match
native inference's preparation lifetime, but use serial release-libtest/system
allocation, not production mimalloc or concurrent feature workers.

JSON retains source/revision hashes, raw per-ply timings, fine/footprint stage
decomposition, group/span preservation, conservative blocker-row reuse, physical
raster equality after colour-channel swaps, and requested-coordinate overlap.
All instrumented coarse/exact/legality results are checked bitwise against full
historical-reference extraction. Fine `footprint` stages subdivide `local_footprint`;
`parent_reusable_footprint` and `ancestor_reusable_footprint` are overlapping
subsets of it, never extra work. Group-spans and scalar invalidation assertions
cover every audited edge. Finished children are excluded.

A metadata-only 65,536-slot coordinate cache simulates inherited local-footprint
values along the played line, invalidating around every added/removed stone.
The extractor still computes every value and checks cache hits against it.
Reported reusable computation seconds exclude cache maintenance and are **gross
opportunity estimates**, not a measured optimization. A played line's target
cells are not a native search-tree proposal trace. In particular, value-only
leaves never extract fine rows. Measure bounded memory and controlled end-to-end
search throughput before retaining a cache. See the
[initial whole-game investigation](../history/2026-11-08-parent-child-features.md).

### Fine-feature shortcut experiment (historical footprint)

Before adding an inherited cache, compare cheaper within-position extraction:

```sh
OUT="scratch/$(date -u +%Y%m%d-%H%M%S)-fine-shortcuts"
mkdir "$OUT"
CONTINUGO_FINE_SGF=GAME.sgf CONTINUGO_FINE_OUTPUT="$OUT/results.json" \
  cargo test --locked --release --lib \
  features::fine_bench::profile_fine_shortcuts -- --ignored --exact
```

Same scored/current-format source requirements and target-cell selection as the
parent/child audit. `CONTINUGO_FINE_REPEATS` defaults to three. Four methods rotate
order per position/repeat: the old reference fine extraction, a test-only
no-capture summary certificate with authoritative fallback, exact integer
projection lookup, and both. Each timer includes fresh fine preparation and any
extra tables/witness preparation. Matrix/legality comparisons are outside timers;
an untimed ordinary extraction at each position warms code/allocator. No extractor
cache survives between methods. Output includes all samples/counts and provenance.

This measures complete serial fine extraction over a real game, not a kernel or
native search throughput. The certificate is diagnostic only: a production rules
fast path belongs in shared rules, not a parallel validator in this crate.
For paired component clocks (baseline versus both shortcuts), use the same
`CONTINUGO_FINE_*` variables and a new output file, replacing the test name with
`features::fine_bench::profile_fine_shortcut_components`. It records preparation,
borders, shared rules/counts and nested footprint stages. These instrumented
seconds differ from ordinary totals; footprint subtotals must not be added to
the parent footprint stage. All outputs are checked against the old reference.

Projection lookup changes arithmetic implementation, not geometry or schema.
Neither prototype is installed in production by this test. See the
[fine-feature results](../history/2026-11-08-fine-shortcuts.md) and measure fixed
native searches before retaining either optimization.

## Friendly-region splitting versus former footprint

```sh
OUT="scratch/$(date -u +%Y%m%d-%H%M%S)-region-splitting"
mkdir "$OUT"
CONTINUGO_SPLIT_SGF=GAME.sgf CONTINUGO_SPLIT_OUTPUT="$OUT/results.json" \
  cargo test --locked --release --lib \
  features::splitting::tests::benchmark_region_splitting -- --ignored --exact
```

Use a completed scored current-format SGF. At every pre-move ply, complete
positive-visit target cells feed two fresh extractors: test-only former local
footprint and current binary splitting. `CONTINUGO_SPLIT_REPEATS` defaults to three;
method order rotates per position/repeat. Timers include all fine extraction,
full authoritative move analysis and lazy index/capture-cache construction.
Legality and the ten unchanged fine columns must match bitwise. Splitting is
checked binary; positive counts and index payloads are reported per position.
The output file must be new. JSON retains every sample and source provenance.
Set `CONTINUGO_SPLIT_NO_FEATURE_BASELINE=1` to compare against the ten shared move
fields alone instead of the old footprint. That ablation uses schema-4 move
fields (no coarse extraction), preserving identical shared facts, and measures
the incremental cost of adding the splitting bit, indices and caches.

This is serial release-libtest/system allocation, not production mimalloc, a
network/MCTS run, or a trained-schema-5 proposal trace. Both methods use the same
full rules path, not the historical no-capture shortcut. Do not interpret this
as a measured search-speed or strength improvement. See the
[replacement results](../history/2026-11-08-friendly-region-splitting.md).

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
