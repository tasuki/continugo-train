# Native inference and search

Rust MCTS uses CPU/CUDA ONNX inference with shared authoritative rules. Python
launchers select runtime/library paths; search runs in-process in Rust. Browser
execution is not integrated. Use [training](training.md) to produce a checkpoint.

## Run

```sh
mise run setup-gpu
mise run export -- --checkpoint runs/train-001/best.pt --output runs/model-001
mise run neural-selfplay -- --model runs/model-001 --games 3 --sizes 9,13,19 --seed 200 --simulations 8 --max-moves 64 --output runs/neural-001
```

`mise run setup` installs inference dependencies; an existing environment can use
`.venv/bin/python -m pip install -e '.[inference]'`. Use fresh output directories.
Omitting `--output` reserves UTC-dated `runs/YYYYMMDD-HHMMSS-export/model`,
`-selfplay`, or `-arena` paths with collision counters. Explicit paths are literal;
`--output -` previews neural SGFs on stdout. Bare Rust baseline self-play defaults
to stdout and needs no inference runtime.

CUDA is the default; `--device cpu` needs no `setup-gpu`. The latter installs ONNX
Runtime GPU 1.20.1 / CUDA 11.8 / cuDNN 8.9.6 in `.venv/ort-gpu/`, reusing PyTorch's
CUDA 11 libraries without replacing PyTorch/cuDNN or CPU ONNX Runtime 1.22.1.
The Rust binding targets API 1.20, compatible with CPU 1.22.1. Official newer GPU
wheels target newer CUDA; see the [compatibility table](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).
Installation targets Linux; other platforms require a compatible supplied runtime.

An explicit CUDA request fails if provider setup fails; there is no whole-session
CPU or unguided fallback. Shape/control operators may run on CPU. Execution uses
FP32, TF32 disabled, heuristic convolution selection, and a 2-GiB CUDA arena limit
per session—not total process/device memory. CPU thread spinning is disabled.
Rules, feature extraction, normalization, and sampling remain on CPU.

For standalone binaries, set `ORT_DYLIB_PATH` or `--ort-library` to a compatible
runtime and supply its dependency paths:

```sh
export ORT_DYLIB_PATH="$(.venv/bin/python -m continugo_train.export --runtime-path)"
cargo run --locked --release -- --engine neural --device cpu --model runs/model-001 --sizes 9 --simulations 8
```

## Model bundle and verification

Export atomically publishes two opset-17 graphs as one new directory:

- `encoder.onnx`: coarse inputs → projected spatial context, pass logit, and value.
- `policy.onnx`: context + exact candidate features + shared int64 action IDs →
  raw placement logits. Batch, spatial, and candidate axes are dynamic.

Both graphs embed matching feature definitions, density, architecture, checkpoint
hash, and epoch. There is no metadata sidecar. Loading rejects incompatible
metadata or mixed graph pairs. Chunking avoids rerunning the encoder; default
chunks are 65,536 coordinates. See [features](features.md) for input semantics.

Export verifies eager/ONNX parity across default resolutions, batch sizes, and
candidate counts. Native tests compare full-domain logits, legality, values,
neighboring coordinates, chunks, and value-only queries on CPU/CUDA. The
checkpoint—not ONNX—is the trainable/recoverable artifact.

## Diagnostics

```sh
mise run infer -- --model runs/model-001 --sgf FILE --ply 1
mise run infer -- --model runs/model-001 --sgf FILE --ply 1 --value-only
mise run infer -- --device cpu --model runs/model-001 --sgf FILE --ply 1 --warmup 1 --iterations 3
```

Stdout is a numeric JSON report, not a game record. `--actions` selects printed
logits, not evaluated coordinates; ordinary diagnostics still compute the full
policy. Illegal scores print as null. `--value-only` skips exact rows/dense scoring.
Set `CONTINUGO_ORT_PROFILE` to an existing directory for encoder/scorer runtime
profiles, including actual operator placement. Timers exclude warmup/model load.

## Feature extraction and context caching

- `--feature-threads 6`: bounded Rayon pool sharing one immutable prepared position;
  disjoint rows match serial extraction bit-for-bit. Set one for a serial control.
  This is separate from ONNX `--inference-threads` / diagnostic `--threads`, whose
  default is one. Native self-play and `infer` use mimalloc. The standalone feature
  exporter remains serial per process; training uses multiple loader processes.
- `--cache-positions 64`: host-side LRU of encoded spatial context, pass logit, and
  value. A value query populates context reusable for subsequent value/policy work.
  Keys use exact immutable Game equality, including record/turn/pass history,
  within one frozen model/backend/density. Reconfiguration clears it; terminal
  queries are rejected before lookup. There is no approximate or cross-model reuse.
- Spatial payload is capped at 32 MiB; game keys/scalars/allocator overhead are
  separate and bounded by entry count. Oversized contexts are not cached; zero
  positions disables caching. Raw coarse rasters, dense exact rows, and million-
  action logits are not cached. Search retains only small candidate reservoirs.

Warmup retains cache contents, so repeated queries may report zero encoder calls.
Use `--cache-positions 0` for uncached timings. Encoded context stays host-side;
CPU feature extraction and GPU execution alternate without pipelining.

## Search semantics

Neural and unguided `--engine mcts` share PUCT, progressive widening, side-to-move
backup signs, terminal scoring, and visit-count move sampling. Terminal outcomes
override estimates. Default nonterminal leaves use neural values or neutral
unguided values. Full policies are deferred until node expansion; their dense
vectors are discarded after candidate sampling. Neither search applies the direct
random reference's eye filter or a handcrafted positional value bonus.

### Exact terminal-pass checks

Both searches inspect legal terminal pass continuations before leaf estimates:

- A winning second pass is chosen regardless of prior/value bias.
- A pass losing immediately, or allowing the opponent an immediate winning pass,
  is avoided when an alternative exists, without claiming that alternative wins.
- Being ahead does not prove a first pass wins: the opponent need not pass back.
- Opening draws remain legal; incomplete placement sampling never proves all
  moves lost. Only terminal hypothetical games are scored, not unfinished territory.

### Optional standalone rollout evaluator

Self-play/arena accept `--rollouts N` (default zero), replacing new-leaf estimates
with N legal playouts; a neural model then supplies policy only.
`--rollout-max-moves 512` caps each playout without forced scoring.
`--rollout-pass-probability 0.03125` controls passes; other moves use bounded
full-domain legal proposals. Exhausting attempts passes but does not prove there
are no legal placements. Immediate winning passes are taken.

There is no eye filter. Only naturally terminal playouts supply values; capped
ones stay unknown. With none complete, selection falls back to neutral lack of
evidence. Completed-only estimates can be censored. Root comments record settings
and `rollout_stats`; hypothetical games are not extra training rows. The supervised
loop/rating protocol does not expose rollouts. `arena --unguided --reference-random`
compares unguided search against the direct reference without ONNX.

### Huge-action-space adaptation

Expanded nodes score **all million coordinates plus pass**, apply exact legality,
and compute a stable full-policy normalizer—not a visited-action softmax. Finite
candidate search then approximates full-domain PUCT using stratified importance
sampling ([Sampled MuZero](https://arxiv.org/abs/2104.06303)):

1. Let `P` be placement mass and `r(a) = p(a)/P` its conditional policy.
2. Draw a bounded reservoir from `q(a) = epsilon/L + (1-epsilon)*r(a)`, where `L`
   is the legal-coordinate count. `--independent-exploration` defaults to 0.25,
   configurable in `(0,1]`.
3. Sample with replacement; merge duplicates but retain multiplicity. Candidate
   weight is `multiplicity * r(a)/q(a)`. Deduplicating without counts changes the law.
4. Admit candidates in first-sampled order under progressive widening. Their priors
   share `P` proportional to those weights; pass retains its full-policy mass.
   If weights underflow to zero, admitted placements share mass uniformly.
5. Every newly admitted coordinate gets a real leaf evaluation, even with tiny prior.

A tiny candidate softmax would over-weight pass; weighting policy-sampled moves by
policy again would double-count it. The correction avoids both mistakes but remains
a finite-sample approximation, not exhaustive search.

## Cross-game batching

Standalone neural self-play defaults to one active game. For concurrency:

```sh
mise run neural-selfplay -- --model runs/model-001 --games 12 --concurrent-games 6 --inference-batch-size 6 --inference-batch-wait-ms 2 --output runs/batched-001
```

- `--concurrent-games`: 1..64, default 1; capped by games requested.
- `--inference-batch-size`: 1..32, default 8; capped by active game slots.
- `--inference-batch-wait-ms`: 0..1000, default 2. Zero drains immediately available
  requests; a single active game never waits to collect a batch. The bound covers
  collection, not queueing/execution latency. Batch-one concurrency is a control
  for overlap without neural batching. [Loop](loop.md) defaults may differ.

Each synchronous tree has one outstanding query. One model-owning worker drains a
bounded channel; one feature pool serves all games. There is no intra-tree
parallelism or virtual loss. Baseline random/unguided engines bypass this scheduler.

Cache misses batch by actual resolution without padding. Identical Game snapshots
share an encoder execution even with cache disabled. Cached values return when
dequeued; uncached values return after encoding, before dense policy work. Policies
batch separately, streaming at most 65,536 coordinates/position and 262,144 aggregate
rows per chunk, also respecting model chunk size. The pool parallelizes positions
and rows. Each reply still contains a million logits and legality (~5 MB/position).
Queued replies/search state are bounded by concurrency, not the cache payload cap;
these bounds are not total RSS/VRAM limits.

Game index fixes seed, size, and filename regardless of scheduling; output arrives
in completion order. Existing files are never overwritten. Batch-one concurrency
is checked against sequential SGFs; larger batches have native output parity tests,
but floating-point differences can change trajectories. An inference error fails
requests rather than falling back. Game/output errors stop new assignments, drain
in-flight games, and join workers. Disconnects become errors, not blocked clients.
Per-game restart recovery belongs to [loop](loop.md#interruption-semantics).

## Records and measurements

Neural SGFs carry player/model identities, seed/search settings, and pre-move root
visits; replay/label rules are in [features](features.md#sgf-replay-and-labels).

Stderr reports whole-run moves/s, full-policy/value calls, coarse/exact feature,
encoder/scorer time, cache statistics, and encoder/scorer batch counts/maxima.
`scorer_batches` counts chunk calls; `scorer_positions` counts full policies.
Queue/request seconds sum across clients and can exceed wall time; request seconds
/ requests gives mean latency. These counters are zero without the worker.
Per-game concurrent latencies must not be summed as batch wall time.

Compare the same seeds, sizes, and budgets across sequential, concurrent batch-one,
and batched execution; report outcomes/trajectories as well as time. Model load is
outside game time. Memory includes transient full-domain outputs, not a million-
action vector per tree node. Historical results are in [benchmarks](../benchmarks.md).
