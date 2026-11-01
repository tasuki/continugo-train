# Cross-game neural inference batching

Native neural self-play can run bounded independent games against a single frozen
ONNX model. CPU and CUDA use the same batching interface. The default remains one
game at a time; benchmark concurrency before enabling it for larger runs.

```sh
mise run neural-selfplay -- --model runs/onnx-initial --games 12 --sizes 9,13,19 --seed 200 --simulations 8 --max-moves 64 --concurrent-games 6 --inference-batch-size 6 --inference-batch-wait-ms 2 --output runs/batched-001
# Concurrent control: identical scheduling interface, but no neural batching.
mise run neural-selfplay -- --model runs/onnx-initial --games 12 --sizes 9,13,19 --seed 200 --simulations 8 --max-moves 64 --concurrent-games 6 --inference-batch-size 1 --output runs/concurrent-001
```

Use fresh directories. Add `--device cpu` for CPU inference. Baseline random and
unguided MCTS engines do not use this scheduler.

## Scheduling and bounds

- `--concurrent-games`: 1..64, default 1. Fixed game workers claim stable game
  indices as slots become available. Active workers are capped by `--games`.
- `--inference-batch-size`: 1..32, default 8; further capped by active game slots.
- `--inference-batch-wait-ms`: 0..1000, default 2. The worker flushes partial batches
  at this deadline; zero drains only immediately available requests. Runs with
  only one game slot never wait to fill a batch. This is a **collection** bound,
  not an end-to-end latency promise: queued requests can also wait for prior
  feature/inference work.

Each tree runs its original synchronous search and has one outstanding query at a
time. A shared model-owning inference worker collects requests through a bounded
channel. There is no intra-tree concurrency, virtual loss, modified search budget,
heuristic evaluator or altered proposal law. One shared feature pool (default four
threads) serves every game; there is not a feature pool or ONNX session per game.

Encoder cache misses are grouped by actual coarse resolution, without padding.
Identical complete Game snapshots within a collected batch share one encoder
execution, even with caching disabled. Cached value requests return as soon as the
worker dequeues them, without waiting for batch collection. Uncached values return
after encoding, **before** the batch's dense policy work. Value-only leaves still
never extract or score a million coordinates.

Full policies are grouped separately by resolution. Exact features are extracted
and scored in coordinate chunks, not retained as a batch of million-row matrices.
Chunks contain at most 65,536 coordinates per position and 262,144 rows across the
batch, also respecting the model's configured chunk size. The shared CPU pool
parallelizes both across positions and within coordinate chunks. Extraction and
ONNX execution currently alternate: CPU/GPU pipelining and GPU-resident contexts
are not implemented.

Each policy output still contains all million logits and its legality mask (about
5 MB per position), with at most 32 positions in an inference batch. Queued replies,
active game/search state and completion buffers are additionally bounded by game
concurrency. This is not a bound on total RSS or CUDA allocation. The existing
32-MiB spatial context cache budget and per-session CUDA arena limits are unchanged.
Full-domain normalization, independent exploration and sampled search priors remain
as documented in [inference.md](inference.md).

## Reproducibility, records and failure handling

Game index determines size, seed and filename independently of scheduling. Each
game owns its RNG. Concurrent output is emitted in **completion order**, including
stdout previews; file numbering is stable, not completion-based. SGFs remain the
only game records, with unchanged policy/outcome semantics and model identity.
Existing files are never overwritten.

Batch-one concurrent execution is checked against sequential SGFs byte-for-byte.
Larger batches are tested against single-position native outputs for every logit,
value and legality row on nonempty sizes 9/13/19, on CPU and CUDA. Floating-point
changes with batch shape can still change tree choices: batched runs are not
promised byte-identical or bitwise reproducible trajectories.

An inference error is returned to every affected request, never converted into an
unguided move. Game/output errors stop new game assignments; in-flight games are
drained and threads joined. Already saved SGFs remain. A missing worker or dropped
reply becomes an error instead of an indefinitely blocked client. There is no
persistent replay-loop restart state in this low-level scheduler; the separate
[loop supervisor](loop.md) provides per-game/epoch/export recovery.

## Measurements

The final stderr summary reports **whole-run moves/s**; per-game generation times
are concurrent latencies and must not be summed as wall time. `InferenceStats`
adds encoder/scorer batch counts and maximum actual batch sizes, plus summed
`queue_seconds` (including collection), `request_seconds` and `requests`.
The latter two give mean end-to-end request latency. Queue/request totals sum
across clients and can exceed elapsed wall time; queue/request counters are zero
for sequential execution without the worker. Scorer batch counts count
coordinate-chunk calls; `scorer_positions` counts completed full policies.

Compare sequential execution, concurrent batch one, and larger batches with the
same sizes, seeds and search budget. Report completed moves and trajectories as
well as timing, since numeric changes can alter game length. Full-domain features
remain CPU work; batching is not assumed to improve end-to-end throughput. See
[benchmarks.md](benchmarks.md) for local results.
