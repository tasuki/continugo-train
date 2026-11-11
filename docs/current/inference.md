# Native inference and search

Rust MCTS uses CPU/CUDA ONNX inference with shared authoritative rules. Python
launchers select runtime/library paths; search runs in-process in Rust. Browser
execution is not integrated. Use [training](training.md) to produce a checkpoint.

## Run

```sh
mise run setup-gpu
mise run export -- --checkpoint runs/train-001/best.pt --output scratch/model-001
mise run neural-selfplay -- --model scratch/model-001 --games 3 --sizes 9,13,19 --seed 200 --simulations 8 --max-moves 64 --output scratch/neural-001
```

`mise run setup` installs inference dependencies; an existing environment can use
`.venv/bin/python -m pip install -e '.[inference]'`. Use fresh output directories.
Omitting `--output` reserves UTC-dated `scratch/YYYYMMDD-HHMMSS-export/model`,
`-selfplay`, or `-arena` paths with collision counters. Explicit paths are literal;
`--output -` previews neural SGFs on stdout. Bare Rust baseline self-play defaults
to stdout and needs no inference runtime.

On-disk self-play automatically builds [training-feature sidecars](feature-cache.md)
for completed MCTS/neural games after generation. This post-generation step uses
the project's Python and Rust feature binary; `--no-feature-cache` opts out. Random,
truncated and stdout-only games are not cached. Arena/rating games are unaffected.

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
cargo run --locked --release -- --engine neural --device cpu --model scratch/model-001 --sizes 9 --simulations 8
```

## Model bundle and verification

Export atomically publishes two opset-17 graphs as one new directory:

- `encoder.onnx`: coarse inputs → projected spatial context, cell logits, pass
  logit, outcome value, and normalized expected final score. Ownership remains
  training-only; see [auxiliary supervision](auxiliary.md).
- `policy.onnx`: context + exact candidate features + per-position int64 IDs `[B,N]` →
  raw placement logits. Batch, spatial, and candidate axes are dynamic.

Both graphs embed matching feature definitions, density, architecture, checkpoint
hash, and epoch. Schema-5 models have twelve coarse planes: antialiased
occupancy, connected group size, spatial liberty centers/footprints and borders,
without scalar liberty area or capturability flags. Their eleven exact fields
include binary `splits_friendly_region`; the user-authorized revision of unused
schema 5 rejects the former footprint feature metadata. Existing schema-4 score/ownership models keep all twelve
original point-sampled inputs, selected explicitly from this metadata.
Re-exporting does not upgrade or truncate a model's inputs. There is no metadata sidecar. Loading rejects incompatible
metadata or mixed graph pairs; `cells-v1` is required and old dense bundles must
be retrained/re-exported. Chunking avoids rerunning the encoder; default
chunks are 65,536 coordinates. See [features](features.md) for input semantics.

Export verifies eager/ONNX parity across default resolutions, batch sizes, and
candidate counts. Native tests compare selected-cell probabilities, legality, values,
neighboring coordinates, chunks, and value-only queries on CPU/CUDA. The
checkpoint—not ONNX—is the trainable/recoverable artifact.

## Diagnostics

```sh
mise run infer -- --model scratch/model-001 --sgf FILE --ply 1
mise run infer -- --model scratch/model-001 --sgf FILE --ply 1 --value-only
mise run infer -- --device cpu --model scratch/model-001 --sgf FILE --ply 1 --warmup 1 --iterations 3
```

Stdout is a numeric JSON report, not a game record. `--actions` selects printed
log probabilities, not evaluated coordinates. Diagnostics use the same selective
policy as search; illegal **and unselected** scores print as null. `cells` lists
selected nonempty cells; `policy_mass` is the sum of the returned inference
probabilities (approximately one, including pass). Pass keeps its global coarse
probability whenever a selected legal placement exists; otherwise it is one. `--value-only` skips all
cell selection and exact scoring.
Set `CONTINUGO_ORT_PROFILE` to an existing directory for encoder/scorer runtime
profiles, including actual operator placement. Timers exclude warmup/model load.
`--profile-json` adds structured stage counters and total query seconds to stdout;
ordinary diagnostic JSON remains unchanged. Self-play can write run totals with
`--stats-json NEW_FILE`. For representative positions plus real training/self-play,
use the [pipeline profiler](profiling.md).

For a visual per-move dashboard with embedded images, all input planes and heads,
use the [self-contained neural game microscope](nn-viewer.md): `mise run nn-view`
for existing SGFs, or `--nn-html --nn-checkpoint FILE` on self-play/arena. This
opt-in checkpoint replay includes ownership, which production exports omit.

## Feature extraction and context caching

- `--feature-threads 6`: bounded Rayon pool sharing one immutable prepared position;
  disjoint rows match serial extraction bit-for-bit. Set one for a serial control.
  This is separate from ONNX `--inference-threads` / diagnostic `--threads`, whose
  default is one. Native self-play and `infer` use mimalloc. The standalone feature
  exporter remains serial per process; training uses multiple loader processes.
- `--cache-positions 64`: host-side LRU of encoded spatial context, cell logits,
  pass logit, and value. A value query populates context reusable for subsequent value/policy work.
  Keys use exact immutable Game equality, including record/turn/pass history,
  within one frozen model/backend/density. Reconfiguration clears it; terminal
  queries are rejected before lookup. There is no approximate or cross-model reuse.
- Spatial plus cell-logit payload is capped at 32 MiB; game keys/scalars/allocator overhead are
  separate and bounded by entry count. Oversized contexts are not cached; zero
  positions disables caching. Raw coarse rasters, dense exact rows, and million-
  action logits are not cached. Search retains only small candidate reservoirs.

Warmup retains cache contents, so repeated queries may report zero encoder calls.
Use `--cache-positions 0` for uncached timings. Encoded context stays host-side;
CPU feature extraction and GPU execution alternate without pipelining.

## Search semantics

Search defaults to `--score-weight 0.5`, using
`(1-weight)*outcome + weight*normalized_score`: equal bounded contributions.
Weight 1 is score-only and 0 outcome-only. This replaces the old relative-weight
formula; native fingerprints change. Terminal, neural and completed-rollout
values share the objective. Old policy/value-only ONNX graphs are rejected.

Neural and unguided `--engine mcts` share PUCT, progressive widening, side-to-move
backup signs, terminal scoring, and visit-count move sampling. Terminal outcomes
override estimates. Default nonterminal leaves use neural values or neutral
unguided values. Selective policies are deferred until node expansion; their dense
vectors are discarded after candidate sampling. By default neither search applies
the direct random reference's eye filter or a handcrafted positional value bonus.

For fresh bootstrap data only, `--engine mcts --bootstrap-avoid-eyes` rejects
noncapturing own-small-eye fills from placement proposals throughout the tree,
using shared `SmallEyeAnalysis`. Captures, connections between distinct groups,
larger pockets and pass remain available. The flag is recorded as
`search.bootstrap_avoid_eyes: true` in SGF root comments; when disabled the field
is omitted. It changes proposal guidance, not rules, neural legality masks,
visit accounting or outcome labels. Neural search and evaluation reject this
guidance; combining it with rollout evaluation is also rejected.

### Exact terminal-pass checks

Both searches treat pass as an ordinary PUCT action:

- A terminal second pass has exact utility available even before selection, but
  no visits are invented. Winning does **not** force it or suppress alternatives;
  a higher-utility continuation may be preferred.
- Losing passes likewise compete normally. A placement's existence does not
  make it better than passing; utility can favor avoiding a larger loss.
- A first pass is evaluated through search, not by assuming an opponent pass.
  Being ahead does not prove a first pass wins: the opponent need not pass back.
- Opening draws remain legal; incomplete placement sampling never proves all
  moves lost. Only terminal hypothetical games are scored, not unfinished territory.

### Optional standalone rollout evaluator

Self-play/arena accept `--rollouts N` (default zero), replacing new-leaf estimates
with N legal playouts; a neural model then supplies policy only to search.
Post-move SGF display annotations still query its outcome/score heads independently.
`--rollout-max-moves 512` caps each playout without forced scoring.
Completed rollouts use the same outcome/score blend as search.
`--rollout-pass-probability 0.03125` controls passes; other moves use bounded
full-domain legal proposals. Exhausting attempts passes but does not prove there
are no legal placements. Winning passes are not forced.

There is no eye filter. Only naturally terminal playouts supply values; capped
ones stay unknown. With none complete, selection falls back to neutral lack of
evidence. Completed-only estimates can be censored. Root comments record settings
and `rollout_stats`; hypothetical games are not extra training rows. The supervised
loop/rating protocol does not expose rollouts. `arena --unguided --reference-random`
compares unguided search against the direct reference without ONNX.

### Huge-action-space adaptation

Expanded nodes select **32 top coarse cells plus 8 exploratory cells** by default,
then compute exact legality/features for every coordinate inside them. Cells do
not overlap. Exploration is seeded from each tree's RNG, so opening coverage can
vary across games without depending on queue/batch order. Both budgets are pinned
in model configuration (`--top-cells`, `--exploration-cells` at training/export).

Pass retains its global coarse probability. Selected nonempty cells share the
entire remaining placement mass, proportional to coarse cell weights and exact
within-cell legal probabilities. Conditioning only the placement stratum avoids
boosting pass merely because cells were pruned. All-unselected/all-illegal
placements mean pass only, not proof that the board has no legal moves. See
[network](network.md) for the formula and unchanged training objective.

Finite candidate search then approximates PUCT **within that selected domain**
using stratified importance sampling ([Sampled MuZero](https://arxiv.org/abs/2104.06303)):

1. Let `P` be placement mass and `r(a) = p(a)/P` its conditional policy.
2. Draw a bounded reservoir from `q(a) = epsilon/L + (1-epsilon)*r(a)`, where `L`
   is the selected legal-coordinate count. `--independent-exploration` defaults to 0.25,
   configurable in `(0,1]`.
3. Sample with replacement; merge duplicates but retain multiplicity. Candidate
   weight is `multiplicity * r(a)/q(a)`. Deduplicating without counts changes the law.
4. Admit candidates in first-sampled order under progressive widening. Their priors
   share `P` proportional to those weights; pass retains its global coarse mass.
   If weights underflow to zero, admitted placements share mass uniformly.
5. Every newly admitted coordinate gets a real leaf evaluation, even with tiny prior.

A tiny candidate softmax would over-weight pass; weighting policy-sampled moves by
policy again would double-count it. The correction avoids both mistakes but remains
a finite-sample approximation, not exhaustive search. This correction concerns
coordinate sampling inside selected cells; it does not recover omitted cells.

### Randomized policy-search caps

The training loop enables randomized 32/256 budgets by default. Standalone
self-play retains fixed-budget behavior unless explicitly enabled:

```sh
mise run neural-selfplay -- --model scratch/model-001 --simulations 32 \
  --policy-simulations 256 --policy-search-probability 0.25 --output scratch/mixed-budget
```

The expensive budget is the **total**, not 256 additional simulations. A separate
seeded RNG chooses each cap before search. Cheap moves still have at least 32
simulations, preserve their actual visit counts in SGF, and train only the
value/score/ownership heads. Only expensive moves supply policy supervision.
Zero probability disables randomization and keeps ordinary all-policy training.
The same options work for bootstrap MCTS, but not direct random play; evaluation
rejects randomized caps. See [loop budgets](loop.md#randomized-policy-search-budget)
and [SGF masks](features.md#sgf-replay-and-labels).

## Cross-game batching

Neural self-play and new loops default to twelve active games, a twelve-request
batch limit and a two-ms collection wait. Actual concurrency is capped by the
number of games requested; request at least twelve games to use all slots:

```sh
mise run neural-selfplay -- --model scratch/model-001 --games 18 --concurrent-games 12 --inference-batch-size 12 --inference-batch-wait-ms 2 --output scratch/batched-001
```

See the [concurrency measurements](../history/2026-11-04-concurrency.md) for the
fixed workload, parity checks and memory costs. These user-selected defaults are
hardware/model-specific, not a universal optimum; override them when appropriate.
Training feature workers are separate and unchanged. Random/unguided self-play
remains sequential. Existing loops restore their frozen configuration.

- `--concurrent-games`: 1..64, neural default 32; capped by games requested.
- `--inference-batch-size`: 1..32, default 32; capped by active game slots.
- `--inference-batch-wait-ms`: 0..1000, default 2. Zero drains immediately available
  requests; a single active game never waits to collect a batch. The bound covers
  collection, not queueing/execution latency. Batch-one concurrency is a control
  for overlap without neural batching. Use `--concurrent-games 1` for a sequential
  control. [Loop](loop.md) resumes may retain older saved settings.

Each synchronous tree has one outstanding query. One model-owning worker drains a
bounded channel; one feature pool serves all games. There is no intra-tree
parallelism or virtual loss. Baseline random/unguided engines bypass this scheduler.

Cache misses batch by actual resolution without padding. Identical Game snapshots
share an encoder execution even with cache disabled. Cached values return when
dequeued; uncached values return after encoding, before exact policy work. Policies
batch separately with per-position selected cells. Chunks stream at most 65,536
coordinates/position and 262,144 aggregate rows, also respecting model chunk size.
Ragged coordinate lists are padded only for scorer transport; padded rows receive
no rules analysis and never enter a cell's probability normalizer. The pool parallelizes positions
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

Stderr reports whole-run moves/s, selective-policy/value calls, coarse/exact feature,
encoder/scorer time, cache statistics, and encoder/scorer batch counts/maxima.
`exact_coordinates` counts real extracted rows (not padding), and `selected_cells`
counts proposed cells. `scorer_batches` counts chunk calls; `scorer_positions`
counts selective policies.
Queue/request seconds sum across clients and can exceed wall time; request seconds
/ requests gives mean latency. These counters are zero without the worker.
Per-game concurrent latencies must not be summed as batch wall time.

Compare the same seeds, sizes, and budgets across sequential, concurrent batch-one,
and batched execution; report outcomes/trajectories as well as time. Model load is
outside game time. Memory includes transient full-domain outputs, not a million-
action vector per tree node. Historical results are in [benchmarks](../benchmarks.md).
