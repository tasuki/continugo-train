# Constants, defaults and budget formulas

A pipeline-ordered inventory of the current implementation's numerical choices,
including formulas that act like constants even though they have no named constant
in code. **Shared code defaults are the normal tuning surface.** When changing
intended behavior, update the defaults/formulas at the linked source definitions
and update this document, rather than adding campaign-specific overrides.
Campaign overrides are for deliberately exceptional experiments, not the usual
way to tune the system.

Propagation is a separate implementation concern: currently, loop resumes restore
saved settings, campaign forks mostly inherit their parent, and architecture and
optimizer settings come from checkpoints. A shared-default change therefore needs
an explicit adoption step for an existing/forked campaign where inheritance would
otherwise preserve the old value; editing a recipe alone is not a replacement for
changing the shared default. Frozen runtimes do not silently pick up source edits.
See [campaigns](campaigns.md) and [loop](loop.md).

The tables distinguish **choices** (hyperparameters and heuristics), **contracts**
(feature/label/format semantics), and **limits** (resource or validity bounds).
A choice is not a claim of optimality. Contract changes need compatibility handling,
not just a different CLI default. Source links are authoritative. This inventory
covers production pipeline settings and diagnostic command defaults, not test
fixtures, historical experiments, UI styling or dependency-internal defaults.

- [Boards and self-play](#1-boards-and-self-play-generation)
- [Training/validation game windows](#2-choosing-training-and-validation-games)
- [Position sampling and optimization](#3-position-sampling-and-optimization)
- [Representation and model](#4-representation-and-model-contracts)
- [Inference proposals and resource bounds](#5-native-inference-proposals-and-resource-bounds)
- [Evaluation and ratings](#6-evaluation-champions-and-ratings)
- [Formats, seeds and operational defaults](#7-formats-reproducibility-and-operational-defaults)

## 1. Boards and self-play generation

Sources: [`LoopConfig`](../../python/continugo_train/loop.py),
[native self-play CLI](../../src/main.rs),
[Python allocation](../../python/continugo_train/sampling.py),
[native allocation](../../src/sampling.rs).

| Setting | Current value / rule | Kind |
|---|---|---|
| Supported training board equivalents | Integers 9–19 inclusive | Limit |
| Default sizes | `9,13,19` | Choice |
| `size_weights` / `DEFAULT_SIZE_WEIGHTS` | `9:1,13:0,19:0`: 100% / 0% / 0%, size 9 only; former `9:2,13:2,19:1` mix remains commented out in source | Choice |
| Weight bounds | Integer 0–1000; zero excludes a size; unspecified sizes have weight 1 | Limit / rule |
| Weighted cycle | Smooth weighted round-robin after GCD reduction; default `[9]`; maximum 4096 entries | Rule / limit |
| `uniform` | Literal supplied size cycle; disables position-quota downsampling | Choice |
| Games per loop iteration | 100; standalone self-play defaults to 1 game | Choice |
| Engine | Standalone `mcts`; loop explicitly uses `neural` | Choice |
| Game move cap | 512 plies | Choice |
| Initial seed | 0 | Choice |
| Loop iteration target | 1 for a new loop unless supplied; total target, not additional rounds | Choice |

For loop round `r` (one-based) and game index `i` (zero-based), let
`o = (r - 1) * games_per_iteration + i`. The size is `cycle[o % len(cycle)]`
and the game seed is `seed + o`. The cycle continues across round boundaries.
Seeds must fit u64; native game counts and simulation counts fit positive u32.

A move cap is a safety cap, **not a scoring rule**: truncated games do not acquire
outcome/score/ownership labels. SGFs are deduplicated by decoded board and plays,
not filename, comments, seed or player name.

### Search effort and action selection

Sources: [search](../../src/search.rs), [self-play](../../src/selfplay.rs),
[optional rollouts](../../src/search/rollout.rs). See [inference](inference.md).

| Setting | Loop default | Standalone self-play / `SearchConfig` default |
|---|---:|---:|
| Cheap / fixed simulations | 100 | 32 |
| Expensive `policy_simulations` | 400 | 256 (CLI; inactive by default) |
| `policy_search_probability` | 0.25 | 0 (fixed-budget, all-policy) |
| PUCT coefficient `cpuct` | 1.5 | 1.5 |
| Progressive-widening coefficient | 2.0 | 2.0 |
| Widening exponent | 0.5 | 0.5 |
| `proposal_attempts` | 128 | 128 |
| Independent placement exploration fraction | 0.25 | 0.25 |
| Score fraction `score_weight` | 0.5 | 0.5 |
| Bootstrap eye avoidance | Not enabled by the loop | Off; unguided MCTS opt-in only |
| Optional rollout count | Not exposed by the loop | 0 (disabled) |
| Rollout move cap / pass probability | — | 512 / `1/32` |

These are choices, with the following formulas and constraints:

- Each move independently uses the expensive budget with probability `p`.
  Expected simulations are `(1-p)*cheap + p*expensive`: **175/move** for the loop
  defaults. This is not a wall-time guarantee. Only expensive moves train policy;
  cheap moves still train all three critics. With `p=0`, all moves train policy;
  with `p=1`, all use the expensive budget. Enabled randomization requires
  `cheap >= 32` and `expensive >= cheap`; `p` is in `[0,1]`.
- Allowed placement children at node visit count `N`:
  `ceil(widening * N**widening_exponent)`, default `ceil(2*sqrt(N))`.
  Pass is a separate child, outside this placement quota. The exponent is strictly
  between 0 and 1; at most one placement child is added per simulation.
- Neural PUCT selection: `Q + cpuct * sqrt(N) * prior / (1 + child_visits)`.
  Placement proposals use `q = epsilon*uniform + (1-epsilon)*policy`, with
  `epsilon=0.25`, and importance weights `policy/q`. Their stratum retains the
  full placement probability; pass retains its own mass. `epsilon` must be in `(0,1]`.
- Search utility: `(1-score_weight)*outcome + score_weight*normalized_score`.
  The weight is a fraction in `[0,1]`, **not** an additive coefficient. Observed
  terminal jigo is a loss for both sides: outcome `-1`, score `0`; backup retains
  the shared penalty `1-score_weight` rather than turning it into a win by negation.
- Self-play samples moves proportional to `visits**(1/T)`, using the same
  temperature schedule for cheap and expensive searches. Defaults match KataGo's
  published `selfplay8mainb18.cfg`: `chosen_move_temperature_early=0.75`,
  `chosen_move_temperature=0.15`, `chosen_move_temperature_halflife=19`.
  For square board width `B` and `m` individual moves already played,
  `T = late + (early-late)*2**(-m*19/(halflife*B))`.
  All three settings are configurable through matching kebab-case CLI flags in
  both the generator and training loop. Temperatures must be finite and nonnegative;
  half-life must be finite and positive. `T=0` selects maximum visits; setting both
  temperatures to 1 restores legacy sampling. Older resumed loops retain `T=1`.
  The schedule changes played moves, not policy-training visit targets.
  Evaluation still takes maximum visits, randomly breaking ties. Winning passes
  are not forced.
- Rollouts average naturally completed evidence only. Unfinished playouts are
  unknown, not draws; no completed evidence yields no value observation.

## 2. Choosing training and validation games

Sources: [window selection and `LoopConfig`](../../python/continugo_train/loop.py),
[split assignment](../../python/continugo_train/data.py). These formulas are
**policy choices**, not consequences of the rules or network architecture.
See [window semantics](loop.md#training-and-validation-windows).

### Growing windows (the default)

| Symbol / setting | Default / meaning |
|---|---|
| `f = training_window_fraction` | `1/2`; valid range `[0,1]`; zero selects legacy mode |
| `v = validation_fraction` | `0.02`; strictly between 0 and 1 |
| `D` | Cumulative eligible data positions in the **parent checkpoint's `progress.data_positions`** |
| Training target | `T_target = max(1, ceil(D*f))` positions |
| Selected training positions | Whole newest games until the target is reached, or the eligible accessible positive-weight pool is exhausted |
| Validation target | `V_target = max(1, ceil(T*v/(1-v)))`, where `T` is actually selected training positions |
| Selected validation positions | Whole newest games from the separate pinned validation pool until its target is reached or the pool is exhausted |

At the defaults, request half of historical eligible data positions for training
and one validation position per 49 **actually selected** training positions.
Both targets can overshoot by the final selected game's position count.
For example, `D=900000` targets 450000 training positions; with only 430000
available, retain all of them and target 8776 validation positions.
There is no fixed game-count cap in this mode.

Important qualifications:

- `D` counts historical eligible generated/imported positions, including validation,
  not optimizer exposures or game counts. Repeated epochs do not increase it.
  Generated data can increase it even if not selected for training; inaccessible
  history still contributes to the target but cannot be selected. Unknown legacy
  position totals are rejected in growing mode rather than replaced by game counts.
  An explicit initial `--data-positions 900000` can supply the agreed cg10 estimate
  when training its fresh checkpoint; the loop then adds new generation normally.
- All local rounds through the current round and explicitly supplied bootstrap
  games are candidates. Selection is newest-round-first within each size,
  weighted round-robin across sizes, with canonical duplicates removed.
- Splits are pinned **before** growing-window selection, including identities not
  selected this round. A held-out identity never becomes training data after a
  window/seed change or fork. The chosen corpus and budgets are frozen per round.
- Both splits must be nonempty; a minimum target of one does not invent missing
  games. `window_games` and `window_iterations` do **not** cap growing windows.

### Assigning permanent holdouts

Split assignment is stratified by raster resolution. For a bucket of `n` unique
games, the desired validation count is:

```text
H = max(1, min(n-1, round(n*v)))    if n > 1
H = 0                             if n = 1
new_validation = max(0, min(unseen_games, H - already_held_out))
```

`round` is Python's ties-to-even rounding, unlike the windows' `ceil`.
Game-ID-sorted buckets are shuffled with `Random(seed)`; unseen identities fill
only the missing holdouts. Existing assignments are never undone, so the final
fraction need not equal `v` exactly (or have precisely `H` holdouts).

### Legacy fixed windows

With `training_window_fraction=0`, `window_iterations=4` admits the current and
previous three generated rounds, plus explicit bootstrap games. `window_games=256`
limits the **combined** selected corpus before its permanent split is applied.
The combined limit must be at least 2. This is a different policy, not an additional
cap on the one-half formula.

## 3. Position sampling and optimization

Sources: [training `Options` and CLI](../../python/continugo_train/train.py),
[data loader](../../python/continugo_train/data.py),
[position quotas](../../python/continugo_train/sampling.py),
[augmentation](../../python/continugo_train/augmentation.py),
[losses](../../python/continugo_train/losses.py).

### How much of the selected corpus is used

| Setting | Standalone fresh training | Loop |
|---|---|---|
| Epochs per job | 3; resume without `--epochs` adds one to completed epochs | 1 |
| `steps_per_epoch` | 0 = exhaust stream | 0 = exhaust stream |
| `positions_per_game` | 0 = all eligible pre-move positions | 0 = all eligible pre-move positions |
| `positions_per_epoch` | 0 = entire eligible stream | 65,536 globally sampled training positions; 0 disables |
| `validation_positions_per_game` | 0 = all eligible positions | 0 = all eligible positions |
| `batch_size` | 64 | 64 |
| Feature workers | 6; 0 = serial | 6 |
| Shuffle buffer | 8 samples; 0 = off | 8 |
| Feature-cache mode | `require` | `require` |
| D4 augmentation | `d4`: uniform draw among 8 symmetries during training only | Inherited from parent |
| Training device / Torch threads | `auto` / `min(4, cpu_count or 1)` | `cuda` / 4 |

The loop defaults to one epoch of up to 65,536 training positions, uniformly
sampled without replacement from its selected window before feature loading.
The seeded sample refreshes each epoch/iteration. Smaller pools are exhausted.
At batch 64 this is 1,024 optimizer steps when the pool is large enough.
Validation remains exhaustive and is not subject to this budget. Standalone
initial training remains uncapped; use `--positions-per-epoch 0` explicitly when
starting a fresh job from a previously capped checkpoint. Explicit per-game and
step caps can further restrict the sampled stream.
Policy sample count can be smaller than total samples because of the cheap-move
policy mask.

Training shuffles games/plies each epoch. Capped validation plies are a stable
seeded random subset, sorted in ply order; validation is not D4-augmented.
Batches contain one spatial resolution and keep final partial batches. Weighted
plans already shuffle within sizes and disable the global shuffle buffer.

For positive-weight sizes with available position counts `a_s`, weights `w_s`,
`W = sum(w_s)` and `k` available sizes, the weighted position budget is:

```text
B = min(sum(a_s), max(k, floor(min_s(a_s * W / w_s))))
```

Distribute `B` references by the weighted cycle, skipping exhausted sizes, without
replacement. This **downsamples** surplus positions rather than upweighting their
loss or oversampling scarce sizes. Absent sizes renormalize; tiny pools may not
achieve exact proportions. The quotas apply independently to train and validation,
after per-game caps. `uniform` bypasses this quota reduction.

### Optimizer and objective

| Choice | Fresh-training default / rule |
|---|---|
| Optimizer | FP32 SGD with momentum; no learning-rate schedule |
| Learning rate | `0.01` |
| Momentum | `0.9` |
| Weight decay | `1e-4` |
| Gradient-norm clip | `5.0`; 0 disables clipping |
| Policy CE / outcome MSE coefficients | 1 / 1 |
| Final-score MSE coefficient | `score_loss_weight=0.5` |
| Ownership MSE coefficient | `ownership_loss_weight=0.5` |
| Ownership pixel weights | Control-volume area / 1,000,000; sum to 1, not uniform pixel counts |
| Outcome targets | Win `+1`; loss **or jigo** `-1` (`jigo-loss-v1`) |
| Final-score target | Side-to-move `(black_area-white_area)/1,000,000`, clipped to `[-1,1]` |
| Ownership targets | Side-to-move signed area fraction per control volume, in `[-1,1]` |
| Policy target | Actual root visits divided by their sum; cheap positions have policy weight 0 |

The default objective is `policy_CE + outcome_MSE + 0.5*score_MSE + 0.5*ownership_MSE`.
Policy CE averages only policy-enabled samples; critics average all samples.
These are **training-loss multipliers**, distinct from search `score_weight=0.5`.
Per the user's instruction, cg9 remains unchanged at `score_loss_weight=1.0` and
`ownership_loss_weight=0.25`. For identical errors, its score-loss contribution is
twice the new default and its ownership-loss contribution is half the new default.
Coefficients do not rescale labels or specify fractions of the total gradient;
loss magnitudes and shared-trunk gradients also matter.
Training uses full coarse normalization and complete target-bearing cells, not a
candidate-only softmax. Score/ownership targets come from the final completed game.

Loop warm-start defaults to **`keep`** optimizer state; standalone warm-start
defaults to **`reset`**. The loop inherits architecture, density, optimizer/loss
settings, augmentation and clipping; `lr=None` means inherit, not zero learning
rate. Local epoch/step counters restart; cumulative lineage progress persists.
The loop exports the latest completed epoch, not the loss-selected `best.pt`.

## 4. Representation and model contracts

Sources: [features](../../src/features.rs), [spatial raster](../../src/features/spatial.rs),
[cells](../../src/cells.rs), [targets](../../src/targets.rs),
[model and `NetworkConfig`](../../python/continugo_train/model.py).
See [features](features.md), [network](network.md) and [auxiliary targets](auxiliary.md).

### Coordinates and inputs

| Contract / setting | Value / formula |
|---|---|
| Coordinate range | `COORD_RANGE=1000`; integer coordinates 0–999 |
| Action IDs | `x + 1000*y`; `COORDINATE_ACTIONS=1,000,000`; pass ID `1,000,000` |
| Board area | `BOARD_AREA=1,000,000` coordinate-area units |
| Current / supported legacy feature schema | `SCHEMA_VERSION=5` / `POINT_SAMPLED_SCHEMA=4` |
| Coarse planes / exact columns | Schema 5: 12 / 11; schema 4: 12 / 10; names/order are part of the contract |
| Policy / target version | `cells-v1` / `final-area-v1` |
| `samples_per_diameter` | 4 by default; positive integer |
| Raster side length `R` | `ceil(999*density/stone_diameter)+1`; maximum 2048 |
| Default 9/13/19 diameters and resolutions | Diameters 110/76/52 from shared rules; `R=38/54/78` |
| One-axis cell owner for coordinate `x` | Integer `(x*(R-1)+499)//999`; cell ID `cx + R*cy` |
| Count transform | `ln(1+count)/ln(1+1,000,000)`, without clipping |
| Border transform | Distance from permitted stone-center boundary / diameter, clipped to `[0,4]`, then divided by 4 |
| Group bridge radius | `0.40 * stone_radius` (representation only, not game geometry) |
| Group/footprint antialias samples per pixel axis | `min(8, max(1, 1024//R))`; normally 8×8 |
| Physical occupancy / liberty-center coverage | Analytic disc coverage / exact integer-center cell coverage; not the 8×8 approximation |
| Friendly-region splitting | Binary; at least 2 remaining nonempty components; exact 8-neighbour integer connectivity |
| Legacy scalar liberty-area integration | Midpoint slices with `ceil(32/radius)` slices/unit, minimum 1; normalized by `pi*radius²` |
| Schema-6 scalar liberty-area precision | Eight times the legacy midpoint spacing; clipped last slice at its own midpoint/height; analytic empty/single-span shortcuts preserved; no per-group cache |

Rounded raster sample locations have endpoints 0 and 999; control-volume outer
edges are 0 and 1000. Features are side-to-move relative, float32; raster geometry
uses float64. Legality is u8 on the wire / bool in Torch; action IDs are u32 on
the wire / int64 in Torch. The complete feature-name lists, including legacy
inputs, are in [features.md](features.md), not duplicated here.

### Architecture choices for a new model

| `NetworkConfig` field | Default |
|---|---:|
| `channels` / residual `blocks` | 32 / 4 |
| `policy_hidden` | 16 |
| `value_hidden` / `score_hidden` | 32 / 32 |
| `ownership_hidden` | 16 |
| Exact scorer `chunk_size` | 65,536 coordinates |
| `top_cells` / `exploration_cells` | 32 / 8 |
| `value_activation` | `leaky_relu`, slope 0.1; `relu` is supported |

The trunk and each residual convolution use 3×3 kernels, padding 1, no convolution
bias before BatchNorm. Spatial policy/ownership projections use 1×1 kernels.
Score and ownership hidden activations also use leaky-ReLU slope 0.1; all three
critics end in tanh. Fine policy uses ReLU and bilinear sampling with
`align_corners=True`, mapping coordinates by `2*x/999-1`. These architecture
choices are checkpoint-pinned, not replaced when defaults change.

## 5. Native inference proposals and resource bounds

Sources: [geometry proposals](../../src/geometry_proposals.rs),
[inference](../../src/inference.rs), [batch scoring](../../src/inference/batch.rs),
[batch worker](../../src/inference/batching.rs),
[parallel games](../../src/selfplay/parallel.rs),
[splitting cache](../../src/features/splitting.rs).

### Candidate-domain choices

| Choice / contract | Current value |
|---|---|
| Schema-5 default proposal contract | `geometry-v2-frontier` |
| Legacy schema-4 / reference mode | `complete-cells-v1`; `infer --complete-cells` opts in |
| Geometry `DEFAULT_BUDGET` | 16 per selected cell, **soft**, not a hard candidate cap |
| `infer --geometry-budget` range | 1–128 |
| Extend quota | At most 3 points per originating friendly group per selected cell; includes reserved connected hugging |
| Join proposals | Up to 4 lens extremes per overlap region; 4 cardinal extremes for multiway regions |
| Capture proposals | Up to 4 cardinal extremes per connected capture patch per selected cell |
| Random reservation | `slots = min(free_coordinates, clamp(budget//4, 2, 4))`; up to `4*slots` sampling attempts |
| Source bits | Extend 1, Join 2, Hug 4, Capture 16, Random (`COVERAGE`) 32; Hug also carries Extend |

Top-cell selection uses the model's 32 top + 8 distinct exploratory cells by
default, capped by grid availability. Geometry operates only in their union;
multiple groups/regions can overflow the soft budget. Only the shortlist gets
exact move features. Full coarse pass mass is preserved; placement mass is
redistributed over selected nonempty cells, with a local legal softmax. With no
selected legal placement, pass probability is 1. See [proposal details](geometry-proposals.md).

### Parallelism and memory limits

| Setting / bound | Current value |
|---|---|
| Concurrent neural games | 32 default, valid 1–64; active slots capped by game count |
| Non-neural concurrency | 1 |
| Inference batch size | 32 default, hard maximum 32; also capped by active game concurrency; homogeneous resolution |
| Collection wait | 2 ms default; valid 0–1000 ms; forced to zero with one active game; partial batches still run |
| Worker request-queue capacity | 1–64; supplied from active game concurrency |
| ONNX intra-op / inter-op threads | 1 default / fixed 1; thread spinning off |
| Exact feature pool | 6 threads default; 1 = serial |
| Parallel exact extraction chunk | 4096 coordinates |
| Native scorer width for batch size `b` | `min(model.chunk_size, 65,536, floor(262,144/b))` |
| Encoded-context LRU | 64 positions default; 0 disables; 32 MiB accounted payload cap |
| Captured-survivor splitting-index cache | 16 states and 4 MiB accounted index/key payload per extractor |
| Loader sample queues | One queued sample per worker; active workers `min(workers, plans)` |
| CUDA device / arena limit | Device 0 / 2 GiB ONNX allocator limit |
| CUDA convolution options | TF32 off; heuristic algorithm search; max-workspace option off |

Context-cache accounting includes encoded tensors and optional proposal geometry,
not Game keys or allocator overhead. Splitting-cache accounting excludes the
parent index and thread-local scratch. Oversized entries are computed without
retention. The ONNX arena limit is not a whole-process/GPU memory cap. Native
commands and the loop default to CUDA; requesting CUDA never silently falls back
to CPU.

## 6. Evaluation, champions and ratings

Sources: [shared champion policy](../../python/continugo_train/rate_saved.py),
[evaluation config](../../python/continugo_train/evaluation.py),
[paired-color planning](../../python/continugo_train/gate.py),
[arena CLI](../../src/bin/arena.rs). See [evaluation](evaluation.md).

Training and experimental actors do not depend on these results. Champion changes
are automatic evaluation decisions; **accepted publication still requires explicit
user approval**, with no numerical auto-publication threshold.

| Choice | Current value / rule |
|---|---|
| Loop model-saving cadence | Every round; 0 disables; independent of rating cadence |
| Shared evaluator eligibility / challenge cadence | Round multiples of 5 / 5 |
| Initial evaluation champion | Random bot, `random-no-eye-fill-v1` |
| Challenge screen | 10 games; at least 7 **wins** to continue |
| Challenge extension | 30 more games; at least 28 wins across all 40 to replace champion |
| Random calibration | 10 games after promotion only, not for rejected challengers; initial random challenge is reused, not duplicated |
| History-opponent policy | Up to 2 previously evaluated neural models nearest in rating, excluding the candidate and challenged opponent; 10 games each, after promotion only |
| Neural rating simulations | 128; standalone native arena defaults to 32 |
| Shared evaluator neural / random move cap | 1024 / 2048 |
| Generic `EvaluationConfig` match / random defaults | 20 / 10 games; move caps 512 / 2048 |
| Native arena defaults | 20 games, move cap 512, seed 0, start index 0 |
| Shared evaluator initial seed / family | 97,000,000 / `cg5` |

Draws/truncations are not wins for champion thresholds. Match scores count a draw
as 0.5 and exclude unresolved games; conservative score bounds count unresolved
points as 0 at the lower bound and 1 at the upper bound.

For global match ordinal `o`, size is `cycle[(o//2) % len(cycle)]`, seed is
`base_seed + o//2`, and the candidate is Black iff `o` is even. Thus each pair
shares size/seed and swaps colors. Evaluation game counts must be even; allocating
a batch of `n` games advances the shared seed cursor by `n//2`.

### Rating scales and optional matchmaking

Sources: [rating state](../../python/continugo_train/ratings.py),
[joint fit](../../python/continugo_train/joint_ratings.py),
[adaptive scheduler](../../python/continugo_train/adaptive_ratings.py),
[round-robin](../../python/continugo_train/round_robin.py).

The shared artifact store rebuilds ratings with the joint Bradley–Terry fit
(`joint-bt-v2`). Older/incremental rating jobs use static Glicko; the initialization
and adaptive-information rules below also cover those paths.

| Choice / formula | Value |
|---|---|
| Random anchor | Rating 100, deviation 0 |
| Initial neural deviation | 350; incremental state starts from parent rating when available, otherwise 100 |
| Elo/Glicko log-odds scale | `Q = ln(10)/400` |
| Joint Bradley–Terry smoothing | Half a win + half a loss for each observed opponent pair |
| Disconnected joint-fit component prior | Mean rating `N(100, 350²)`; no relative-strength shrinkage |
| Joint-fit solver limits | 100 Newton steps; gradient max-abs `<1e-11`; Armijo factor `1e-4`; backtracking ×0.5; fail below `1e-12` |
| Adaptive targets / opponent pool | Latest 3 networks / all registered neural networks by default |
| Adaptive completion evidence | Anchored, RD ≤75, ≥20 completed color pairs, ≥2 opponents with ≥4 pairs each |
| Adaptive batch / hard game cap | 4 / 360 attempted additional games |
| Adaptive seed | 9,300,000 |
| Adaptive diversity discount | ×0.01 for already-covered opponents while target diversity is unmet |
| Adaptive completion discount | `(completed_games+2)/(attempted_games+2)` |
| Round-robin defaults | Latest 4 networks, 20 games per matchup, seed 9,100,000 |

Adaptive expected information per game is `(Q*g)²*p*(1-p)`, where
`g=1/sqrt(1+3*(Q*opponent_RD/pi)²)` and
`p=logistic(Q*g*(rating-opponent_rating))`. For target variance `v`, batch size `b`
and information `I`, predicted reduction is `v-v/(1+b*v*I)`. Match scoring sums
reductions normalized by `target_RD²`, with the discounts above. These are
scheduling heuristics, not extra outcome evidence.

## 7. Formats, reproducibility and operational defaults

### Durable formats and caches

Sources: [wire encoder](../../src/feature_io.rs),
[wire decoder](../../python/continugo_train/features.py),
[feature caches](../../python/continugo_train/feature_cache.py),
[cache-generation CLI](../../src/feature_cache.rs),
[export](../../python/continugo_train/export.py),
[runtime](../../python/continugo_train/runtime.py).

| Contract / setting | Value |
|---|---|
| Loop control version | 6; not a game-record format |
| Outcome semantics tag | `jigo-loss-v1` |
| Feature packet magic / header | `CGFEAT02`; little-endian `<8s11If2If`, 68 bytes |
| Feature-cache magic / version | `CGCACHE1` / format 1 |
| Cache prefix | Little-endian `<8sQ32s`, 48 bytes |
| Cache compression | Independent zlib packets, level 3 |
| Maximum cache index / uncompressed packet | 32 MiB / 1,000,000,000 bytes |
| Cache build synchronization / extractor-hash LRU | 64 striped locks / 16 entries |
| Cache modes | `require` (default, including low-level loaders), `auto`, `off`, `prefer` |
| Cache CLI density / build workers | 4 / 4; generation writes caches by default unless opted out |
| ONNX export | Opset 17; metadata key `continugo`; encoder + policy graphs |
| Export verification threads / seed | CPU, intra/inter-op 1 / 1; seed 0 |
| Export parity relative tolerance | `1e-4` |
| Export parity absolute tolerances | Spatial tensors `5e-4`; logits `1e-4`; critics/probabilities `1e-5` |
| Export verification `(batch, resolution, coordinates)` | `(1,38,7)`, `(1,54,73)`, `(2,78,19)` |
| Isolated CUDA runtime pins | ONNX Runtime GPU `1.20.1`; cuDNN `8.9.6.50`; CUDA 11.8 wheel channel |

Caches additionally pin feature names, target version, density, extractor SHA-256
and SGF SHA-256. Matching a schema number alone is not enough. SHA-256 identities
are 64 lowercase hexadecimal characters. `artifacts/` is the default model store;
training/loop outputs default under `runs/`, experiments under `scratch/` (see
[artifact paths](artifacts.md)). Native standalone self-play defaults to stdout
(`--output -`). Saved model snapshots default to every round, including later
backfills with `save-models`; `--every N` explicitly selects a sparser cadence.

### Seed derivations (implementation choices)

Sources: [loader](../../python/continugo_train/data.py),
[self-play](../../src/selfplay.rs),
[cell selection](../../src/inference/proposals.rs),
[geometry](../../src/geometry_proposals.rs).

| Purpose | Seed / mixing rule |
|---|---|
| Loop training job, round `r` | `loop_seed + r - 1` |
| Training position plans, epoch `e` | `seed + e*1,000,003`; validation uses `seed` |
| Validation capped plies for one game | `seed XOR int(game_id[:16],16)` |
| Validation size-quota subset | `seed + size*100,003` |
| Training weighted-cycle starting index | `(seed+e) % cycle_length`; validation starts at 0 |
| Shuffle-buffer RNG | `seed + e*1,000,033` |
| Separate D4 RNG | `seed + e*1,000,037 + 17,041` |
| Separate policy-budget RNG | `game_seed XOR 0x706f6c6963796361` |
| Per-cell geometry random RNG | `query_seed XOR wrapping_u64(cell*0x9e3779b97f4a7c15)` |
| Exploratory-cell ordering | First 8 SHA-256 bytes of diameter/query seed/relative stones, then SplitMix-style hashing of seed XOR cell |

The cell mixer uses multipliers `0xbf58476d1ce4e5b9`, `0x94d049bb133111eb` and
right shifts 30, 27, 31. Separate RNGs keep augmentation and expensive-search
sampling independent of loader ordering/search consumption. These choices support
repeatability but do not promise bitwise GPU reproducibility.

### Diagnostics and cleanup

Sources: [synthetic benchmark](../../python/continugo_train/benchmark.py),
[pipeline profiler](../../python/continugo_train/pipeline_profile.py),
[infer CLI](../../src/bin/infer.rs),
[diagnose](../../python/continugo_train/diagnose.py),
[memory sampler](../../python/continugo_train/memory.py),
[feature client](../../python/continugo_train/features.py),
[supervisor](../../python/continugo_train/loop.py).

| Tool / purpose | Defaults / limits |
|---|---|
| Synthetic model benchmark | Diameters 110/76/52; batches 1/4; density 4; 2 warmups, 5 measured iterations; `auto` device; threads `min(6, cpu_count or 1)` |
| Real pipeline profiler | Sizes 9/13/19; CPU; 3 iterations; 1 warmup training step, 12 measured steps; 12 positions/game; 8 search simulations; 512 moves; seed 800 |
| Profiler parallelism | 32 games, inference batch 32, wait 2 ms; 6 feature threads, 6 loader workers; training batch 64; Torch threads 1 |
| Native `infer` | Ply 0; seed 0; 0 warmups, 1 measured iteration; default probes `0,999,1000,999999,500500,500501,1000000` |
| Offline `diagnose` | CPU; at most 32 games; threads 1 |
| Neural HTML viewer | Threads 1; rated rendering contract `rated-own-actor-v2` |
| Driver GPU-memory sampler | Poll every 0.1 s; `nvidia-smi` timeout 2 s; sampled high-water mark, not exact peak |
| Explicit timing-chart watcher | Poll every 30 s; refresh on state/journal changes; exit with supplied supervisor PID |
| Loader queue stop polling | 0.1 s |
| Feature subprocess cleanup | Wait 10 s on normal close; 2 s after unexpected stdout EOF; then kill if still running |
| Supervisor interrupted-child escalation | SIGINT: wait 20 s; SIGTERM: wait 5 s; then SIGKILL and wait |

Benchmark/profiler settings are diagnostic workloads, not training recommendations
or implicit overrides of a campaign recipe. Historical campaign-specific numbers
belong in [history](../history/), not in these default tables.
