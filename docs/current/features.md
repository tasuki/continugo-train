# Real neural features — schema version 5

**On 2026-11-08 the user explicitly reopened unused schema 5 and replaced the
local-liberty-footprint scalar with binary `splits_friendly_region`.** There are
still twelve coarse planes and eleven exact fields, with the new bit before the
four borders. Coarse inputs and schema 4 are unchanged. No training is started.
Checkpoint/ONNX feature names must match the revised contract; earlier schema-5
footprint checkpoints/exports are rejected. Discard all earlier derived schema-5
tensor packets: their header has no feature-name fingerprint, so matching version
and width alone do not establish compatibility. Validated source SGFs remain usable.

Extraction lives in `src/features.rs`; rule facts come from `continugo-rules`.
`Position::liberty_spans` returns the exact integer liberty-center intervals used
for UI-style geometry. Schema 5 uses `PreparedPosition::liberty_spans` to share
lazy blocker rows between groups, without running an opposing one-placement capture
search, then rasterizes per-colour unions of centers and footprints. Coarse extraction
borrows the prepared groups rather than rebuilding them. Legacy schema-4 extraction
still uses `PreparedPosition::analyze_group`, including its exact capture witness
and shared blocker-row cache, to preserve old model inputs. This changes no rules,
move validation, witnesses or capture decisions.
`PreparedPosition::analyze_move` uses the existing prepared validator for
connections, captures and resulting group size. Binary region splitting uses the
actual returned child survivors, with the candidate temporarily absent, and an
exact integer scanline connectivity index in `src/features/splitting.rs`.
The replaced scalar remains only as a test/benchmark reference in
`src/features/footprint.rs`; continuous reference/legacy integration lives in
`src/liberty_area.rs`. No game rules live in Python.

## Coarse input `[12, H, W]`

All features are relative to the **pre-move player**. Grid endpoints correspond
to action coordinates 0 and 999. `H = W = ceil(999 * density / diameter) + 1`,
using the actual Rust stone diameter; density defaults to four intervals per
stone diameter (38/54/78 samples for 9/13/19 equivalents). Extraction caps grids
at 2048 samples per side. Raster sample locations
are rounded to the nearest integer coordinate. Occupancy is **area-antialiased**:
each pixel contains the fraction of its control volume covered by physical stone
discs, not their larger move-exclusion discs. Pixel boundaries are midway between
rounded samples, with outer boundaries 0 and 1000 (the same rectangles used for
ownership targets). Rust computes circle/rectangle intersections analytically;
this is not viewer interpolation, blur, or a supersampling approximation. Fully
covered pixels are 1, empty pixels 0, and boundary pixels fractional. Legal stone
discs have disjoint interiors, so coverage adds when several discs touch one pixel.

Schema 5 removes both capturability flags and the per-group liberty-area scalar.
It includes **both** spatial liberty representations. Grid resolution, cell
partition and labels are unchanged. Coarse extraction computes neither capture
witnesses nor per-group scalar areas: coarse-profile `capturability.calls` and
`liberty_area.calls` are zero. Exact candidate extraction additionally computes
the binary friendly-region split described below, without footprint integration.

| Channels | Contents |
|---|---|
| 0, 1 | Friendly / enemy physical stone-disc area coverage, float in [0,1] |
| 2, 3 | Friendly / enemy log group size on connected group shapes |
| 4, 5 | Friendly / enemy liberty-center union coverage |
| 6, 7 | Friendly / enemy liberty-footprint union coverage |
| 8–11 | Left / right / top / bottom border distances |

**Connected group size:** each group's member discs are joined by thin capsules
along exactly the same-colour adjacency edges supplied by the rules. Bridge radius
is 0.30 times stone radius. The shape carries `ln(1 + size) / ln(1 + 1_000_000)`;
where shapes overlap, the largest group feature wins before pixel averaging.
These bridges are an input representation, not additional stones or changed
connections. Physical occupancy channels never include bridges.

**Liberty centers:** Rust's exact integer row spans are unioned per colour, so
shared liberties are counted once. Each integer center occupies a 1×1 unit cell
about its coordinate. Circle-independent rectangle intersections give exact area
coverage of those cells in each coarse pixel. An isolated center remains a tiny
positive signal rather than being missed by point sampling.

**Liberty footprints:** sweep the physical stone radius over those same unioned
spans, forming round-ended capsules. The plane is union coverage in [0,1], never
a sum over alternatives or groups. It shows possible occupied regions, not group
area in stone-disc units. Nearby alternatives overlap; one isolated center still
produces a stone-sized footprint. Per-group liberty identity is not encoded in
either spatial liberty map.

Group shapes and footprint unions use 8×8 midpoint antialias samples per pixel at
normal training resolutions. To bound intermediate memory at unusual densities,
samples per axis are `min(8, max(1, 1024 // resolution))`. Zero-area control volumes
are zero. Occupancy remains analytic and center coverage remains exact. Rasterization
lives in `src/features/raster.rs` and `src/features/spatial.rs`. Empty colours skip
spatial canvases entirely; footprint rasterization maintains disjoint row unions
and accumulates sample counts directly into coarse pixels, without a second
supersampled canvas. These optimizations preserve the original raster bits.

All schema-5 coarse values are in [0,1]. Border distances are measured from the
permitted stone-center boundary in stone diameters, clipped to [0,4], then divided
by four; outside that boundary they are zero. Liberty centers are physically
unblocked, in-bounds integer points adjacent to a group, not a fully validated
move-legality mask. There are no move-history, per-group area, or capturability
planes. All coarse planes and exact columns are **float32**. Coarse raster
geometry uses float64; region splitting uses exact integer intervals and
connectivity. Legality is separately u8 on the wire / `torch.bool` in Python; action
IDs are u32 on the wire / `torch.int64` in Python.

## Exact coordinate input `[N, 11]` and legality `[N]`

Coordinate IDs are `x + 1000*y`, excluding pass. Rows preserve caller order.
Default schema-5 inference uses the [geometry proposal policy](geometry-proposals.md):
it retains exact per-group spans during coarse extraction and analyzes only
proposed coordinates inside chosen cells. `infer --complete-cells` and legacy
schema-4 inference evaluate complete chosen cells. Training still evaluates
complete target-bearing cells (see [network](network.md)). Retention does not change any feature bits. The explicit diagnostic
`--full-domain` mode uses IDs 0 through 999,999 as a reference only. Pass is always
the separate final policy-output column, including when no placements are supplied.

| Column | Contents |
|---|---|
| 0 | Authoritative placement legality, 0 or 1 |
| 1 | Number of friendly groups connected by the placement |
| 2 | Total stones in those connected friendly groups |
| 3 | Number of enemy groups captured |
| 4 | Number of enemy stones captured |
| 5 | Resulting merged friendly group size, including the new stone |
| 6 | `splits_friendly_region`: binary split into at least two nonempty friendly-bounded regions |
| 7–10 | Left / right / top / bottom border distances, as above |

Columns 1–5 use the logarithmic count transform. Column 6 is exactly **0 or 1**.
The authoritative legality/capture/connection columns remain unchanged.

### Binary splitting predicate

For an authoritative legal placement at `p`:

1. Start with the actual child position, including enemy captures and any removed
   friendly stones, then temporarily remove just the newly placed stone.
2. Available centers are all integer coordinates inside the board's permitted
   center rectangle, not overlapping any surviving stone: squared distance must
   be at least `diameter²`. Connect available centers by **eight-neighbour** unit
   steps. No grid subsampling or interpolation is used.
3. Find the available-center component containing `p`. It is **friendly-bounded**
   iff its blocked unit neighbours touch at least one friendly stone and no
   enemy stone. Board edges are neutral permitted boundaries. A blocked neighbour
   overlapping both colours counts as touching both, so does not qualify.
4. Insert the candidate's strict diameter-radius exclusion disc into that one
   component. Return **1** iff at least two nonempty components remain; otherwise
   return **0**. Each surviving component contains at least one exact integer
   center, including single-coordinate pockets and passages.

Thus this tests a split, not simply whether two pockets already exist. Filling
an eye, open-board play without a split, and splitting enemy/mixed-boundary space
return zero. Capturing an enemy intruder can turn the region friendly: the test
uses the actual surviving stones, not fixed pre-move blockers. It is a geometric
input, **not a proof of two true eyes, independent playable stones, one owning
chain, unconditional life, or a legality rule**. Friendly boundary stones may
belong to different groups; tactical and group inputs remain separate.

### Intended signal: rare moves that divide space into two eyes

**The user's motivation for this feature is helping the network learn to make
two eyes, not merely rewarding generic open space.** A placement can divide a
local region into two separate eye pockets. If each pocket admits only one
stone center and their physical footprints are disjoint, their union is about
**two stone-discs**. Other placement directions may leave only one such pocket
and less footprint. The useful dividing placement can occupy **only one exact
coordinate, or very few coordinates**, while nearby alternatives lose the second
pocket. A tiny change in candidate position can therefore remove an entire disc's
worth of signal; the feature need not be smooth across neighboring actions.

This was the motivation for the original footprint proxy and is preserved here
as design rationale. Area alone could also be large in one connected region and
was colour-blind; the replacement tests a friendly-region split directly, rather
than integrating area. The binary predicate above is still not a life-and-death
solver or a replacement for authoritative legality/captures.

**Optimization must preserve the rare action's contrast with its alternatives.**
Use targeted candidate-neighborhood tests: does the one/few-coordinate dividing
move retain a positive split bit versus nearby one-pocket moves? For the old
area proxy, this was the roughly two-disc versus one-disc contrast; small
average/p95 area error over random positions was not enough. Preserving zero versus nonzero, or retaining one isolated center,
is also not enough: both alternatives may be nonzero, yet losing the second
pocket destroys precisely the intended distinction. Interpolating between moves,
sharing one scalar across a cell, or sparsely probing directions can smooth away
that narrow peak even when their aggregate error looks small. Evaluate this
failure mode explicitly before adopting any approximation or removing the input.

### Implementation and regression coverage

The extractor builds exact per-row exclusion unions, available runs, component
labels and boundary colours once per position, lazily on the first legal query.
Most candidates immediately reject a non-friendly source component. Qualifying
queries subtract the candidate disc from that component's runs and count its
remaining components with scanline union-find. Thread-local scratch is reused;
there is no footprint raster, dilation, area integration or million-point flood
fill per candidate. The source region may extend beyond the candidate's local
annulus: this is intentionally a region property, not the replaced local scalar.

Capturing moves share indices keyed by exact surviving old stones, capped at 16
states and 4 MiB accounted index/key payload per extractor; excess entries are
evicted and oversized entries computed without retention. The parent index and
thread-local scratch are additional geometry-bounded memory, not included in
that capture-cache payload budget. Parallel extraction shares immutable indices;
ordinary move validation and search selection are unchanged.

Tests compare scanline results against an independent point flood-fill oracle,
including both colours, mixed boundaries, edge-bounded space, actual captures,
zero/one-center remnants and diagonal connectivity. A synthetic corridor has
exactly one dividing coordinate. A real size-3 fixture has only five neighboring
dividing x coordinates, with immediate adjacent moves returning zero. An enemy
intruder captured by the dividing move tests the difference from fixed blockers.
Serial/parallel feature extraction remains bitwise identical.

The old `local_liberty_footprint_stones_cap3` and earlier draft footprint names
are no longer accepted in checkpoint/export metadata. Their implementation and
coverage tests remain test-only historical references. See the
[replacement benchmark](../history/2026-11-08-friendly-region-splitting.md),
[old footprint coverage](../history/2026-11-08-footprint-coverage.md) and
[old cap-three costs](../history/2026-11-08-cap3-footprint.md).

For illegal moves, columns 0–6 are zero and only border geometry remains. Legality also has a separate boolean
mask for the network. It includes suicide and captures, not just geometric
availability. It **never applies the direct random bot's eye-avoidance filter**.
Finished games cannot be policy inputs. The shared rules do not enforce ko.

## Legacy one-move capturability (schema 4 only)

These two coarse planes are absent from new schema-5 models. They remain available
only to existing twelve-plane networks, not as zero-filled placeholder inputs.
Exact per-candidate captures in the detailed policy are unchanged. Schema 4
retains its original ten exact columns, with borders at 6–9; the new splitting
bit at column 6 is schema-5-only, moving schema-5 borders to 7–10.

Each group's flag is 1 iff its **opposite color** has a legal placement that
removes the whole group on the unchanged position. It is painted on all of that
group's stone pixels, with zero elsewhere. For enemy groups this is an available
capture by the current player; for friendly groups it is a counterfactual threat
if the opponent could play on this unchanged board. The owner may defend first:
this flag is not a forced-capture claim or a multi-turn capture-depth estimate.

An attacking stone must block every original liberty center. For each candidate
y row, intersect the strict diameter-radius exclusion constraints of all liberty
span endpoints. Endpoints suffice because squared distance along a horizontal
span is convex. Subtract existing stones' geometric exclusion intervals, then
validate the remaining placements with the shared prepared rules, including
simultaneous captures and suicide. Stop only on a verified whole-group capture.
Wide-spread liberties reject early; there is no sampling budget or heuristic
false result. All geometry is exact integer arithmetic, independent of the
numerical area measurement and coarse raster resolution.

Queries use the current position-level rules, which do not implement ko. Tests
compare the filter against exhaustive point geometry and the capture result
against exhaustive legal placements in dense mixed-color setups. Corner captures
with many liberties, protected separate pockets, both colors, and turn changes
are covered. Normal self-play/search behavior is unchanged.

## Liberty-footprint geometry and legacy scalar area

Schema 5 represents the following shape spatially, unioned per colour and
antialiased as described above. The scalar integration below is used only by
legacy schema-4 coarse extraction and optional accuracy/preview diagnostics.
Current coarse inputs use spatial maps; the exact splitting bit does not measure
area at all.

We measure the region drawn by `Coverage.elm`: each exact integer liberty-center
row span is swept by a disk of the **physical stone radius** (a horizontal capsule,
matching the SVG round-ended stroke). Integrate the **union** of those capsules,
not their summed overlapping areas. This deliberately includes the same tiny
between-center smoothing as the UI, not fractional centers invented by the rules.
Valid centers are at least one radius inside the board, so these capsules already
fit within its boundaries.

An empty region has area 0. One isolated touching liberty center has area exactly
**1 stone-disc**, not zero. A single span of length L has normalized area
`1 + 2*L/(π*r)` analytically. Multiple overlapping spans use continuous horizontal
union lengths integrated with midpoint slices of height `1/ceil(max(1,32/r))`:
at most one board unit and at most r/32. Slices align with integer capsule
activation boundaries. This is a **numerical area estimate**, not a certified
worst-case error bound; exact liberty centers and move legality are never sampled.
Tests compare analytic disk/capsule unions and doubled integration density, plus
pointwise checks of the exact center spans against the shared rule geometry.

Legacy coarse area is computed once per existing group. Current schema 5 does
not integrate child-group areas. Per-move legality, connections, captures and
resulting group sizes remain exact; binary splitting is exact for its stated
integer-center connectivity and boundary-colour definition.

Area is **not** a moves-to-capture counter. Equal-area regions can have different
shapes or separated pockets; attack legality and defensive extensions/connections
change the problem further. Blocking liberty centers uses an exclusion radius of
one stone **diameter**, whereas the UI sweep uses one **radius**. Keep exact
capture features separate and leave multi-turn capture difficulty to learned
evaluation/MCTS, not a handcrafted capture-depth estimate.

## SGF replay and labels

`src/replay.rs` uses the shared SGF codec, `SZ`, and authoritative move replay.
`ply` is the number of moves already played: ply 0 means the empty pre-first-move
position. The loader validates the *entire* game, including moves after the chosen
position, before trusting the outcome. Unsupported setup, komi or rule overrides
are rejected rather than silently ignored.

Generated move comments put post-move evaluation first, then the alternatives:

```json
{"win_pct":63.4,"score":5.2,"policy":[["aMaM",3],["dSdS",1],["",0]]}
```

`win_pct` and `score` describe the position **after the recorded move**, from the
**mover's perspective**, using that mover's frozen neural model. Both are numeric,
rounded to one decimal. `win_pct = 50*(1 + outcome)` is expected points percentage
(draws count half), not a separate estimate of strict win probability. Score is
expected final signed margin in the same stone-area units as SGF `RE`, converting
normalized area by `1_000_000 / (pi * stone_radius²)`. Positive favors the mover.
These are raw neural head estimates, not the score-weighted search utility.
Neural annotations use a value-only query (reusing cached context when available),
not another search or extra simulations in the policy visit counts.

All generated moves have a comment. Non-neural/random turns have `null` estimates
unless the move ends the game, in which case all engines use exact final outcome
and score (win/draw/loss percentages 100/50/0). Random turns still omit `policy`;
optional `profile` follows the alternatives. Predictions are display annotations
only: they neither guide move selection nor supply/override training labels.
A final prediction on a truncated game does not supply a result.

- Policy targets come only from the chosen move node's JSON `policy` visit counts.
  Each pair is `[move, visits]`,
  using the shared SGF's four-character coordinates; `""` is pass, and color is
  implicit in the move node. Writers order entries by descending visits, then
  ascending internal action ID for ties, without changing search/move selection.
  Replay accepts **only** this pair format: old numeric-action objects are rejected,
  with no migration or compatibility reader. Generate fresh training SGFs.
  Moves must be unique and legal, with nonnegative integer counts and positive
  total visit mass. Zero-count entries are allowed. Moves become numeric action
  IDs internally; the feature transport and loss are unchanged. These are MCTS
  root visits, not raw network priors. Unannotated/random games do **not** acquire
  invented policy labels.
- Randomized-budget self-play adds boolean `train_policy` to each move comment:
  `true` for expensive searches, `false` for cheap ones. Both retain genuine,
  fully validated `policy` visits; `false` masks only policy CE, not final labels.
  Missing `train_policy` preserves legacy policy supervision. An explicit flag
  without real visits, or a non-boolean flag, is rejected. The SGF root also
  records `policy_search: {"simulations":256,"probability":0.25}` when enabled;
  `search.simulations` remains the cheap budget.
  `features --inspect` exposes `policy_training_plies` separately from
  `eligible_plies` (all naturally labeled real-policy positions). The loader
  carries this mask from hash-pinned inspection into training batches; CGFEAT02
  tensors and raw `PositionFeatures.policy_targets()` remain unchanged.
- A value label requires natural two-pass completion and a scored `RE` matching
  authoritative final scoring. It is +1/-1 for a win/loss from the chosen pre-move
  player's perspective, or 0 for a draw. Margin is not a value target. Games without
  `RE` have no value label; a scored `RE` on an unfinished game is rejected.
- Resignation/time-result labels and branching/setup SGFs are not supported here.

## Rust → Python bridge

```sh
mise run build-features
mise run selfplay -- --engine mcts --games 1 --sizes 9 --seed 41 --simulations 8 --output scratch/mcts
```

```python
from continugo_train.features import load_position
from continugo_train.model import PolicyValueNet

sample = load_position("scratch/mcts/game-000001-size9-seed41.sgf", ply=0)
model = PolicyValueNet()
logits, value = model(*sample.network_inputs())
columns, visits = sample.policy_targets()  # Output columns, with pass last.
# A trainer must require sample.outcome is not None before using an outcome loss.
```

The Python loader invokes the release Rust binary and decodes its binary stdout.
`CONTINUGO_FEATURE_BIN` can override the executable path. `actions=[...]` requests a
coordinate subset for diagnostics; positive targets omitted from it cause
`policy_targets()` to fail, not silently renormalize. An arbitrary subset is not
valid hierarchical training input: each represented cell must be complete for
its local softmax to be correct. Model outputs are hierarchical log weights;
do not apply a candidate-only softmax in training. A tensor packet can also be
cached for inspection:

```sh
mkdir -p runs
mise run features -- --sgf FILE --ply 0 --actions 200200,500500 --output scratch/position.cfeat
```

New extraction/training defaults to schema 5 (twelve planes: antialiased occupancy,
connected group size, liberty centers, liberty footprints and borders; no scalar
liberty-area or capturability planes). Existing schema-4 score/ownership models retain their
original twelve point-sampled planes, including the capture flags. Native inference
reads the schema from ONNX metadata, and the viewer reads it from the checkpoint.
Explicit offline diagnostics can request `--feature-schema 4`, or the corresponding
`feature_schema=4` Python argument. Packet decoding requires the requested schema;
it never silently accepts old features for new training. New training requires
fresh schema-5 checkpoints; old weights/sidecars are not converted or truncated.

The one-shot `load_position` convenience helper replays the SGF per request.
The offline trainer instead uses `FeatureClient` / `features --serve`: each worker
keeps one fully validated game's pre-move snapshots and samples positions without
replaying/scoring each request. `features --inspect` exposes validated metadata
and eligible plies without implementing a Python SGF parser. See
[training.md](training.md) for the bounded loader, game-level holdout and checkpoints.
The CGFEAT02 packet includes final auxiliary targets. Server/default one-shot extraction
returns complete cells containing positive placement targets, with full-domain
flag false; a pass-only/no-policy position has zero exact rows. Explicit
`--actions` remains a diagnostic subset; `--full-domain` is a slow reference only.
Packet size scales with target-cell coverage, not a mandatory million-row matrix.
SGFs remain the sole primary game records. Binary packets/cache files are derived, disposable tensors, not JSON
sidecars or another game-record format.

### Optional server diagnostics

Ordinary `features --serve` responses remain `status u8 + length u64 + payload`,
where status 0 carries CGFEAT02 and status 1 carries a UTF-8 error. A request may
set `"profile": true`: **successful** responses then append `length u64 + JSON`
with server stage times and Linux process-lifetime peak RSS. Errors never append
telemetry, so the next request remains aligned. Profiling does not change the
CGFEAT02 payload; these are disposable transport diagnostics, not SGF annotations.

```python
from continugo_train.features import FeatureClient

with FeatureClient(profile=True) as client:
    sample = client.load("GAME.sgf", ply=0)
    print(client.last_profile)
```

The Python client adds its own request/read and decode/copy clocks. No profile
flag is sent by ordinary clients. Rebuild both clients and servers for CGFEAT02.
See [profiling](profiling.md) for stage overlap, IPC residuals and memory limits.

Requests may specify `feature_schema: 4` to preserve an existing model's original
twelve inputs; omitted/default `feature_schema: 5` produces the new twelve inputs.
The returned header contains that schema. Python clients explicitly select the
expected schema when decoding, so a server/client mismatch is rejected.

The [NN viewer](nn-viewer.md) additionally uses optional `cells: [id, ...]`
(explicit complete-cell extraction; `[]` gives coarse-only tensors) and
`view: true`. A successful view request appends the same u64-length JSON trailer
with a `view` field containing Rust-replayed stone geometry and the move played;
`FeatureClient.last_view` exposes it. Optional `view_geometry: true` additionally
includes exact group membership, rule-adjacent member pairs, normalized group size,
liberty area and integer liberty-center spans. These are rendering facts for input
design previews, not extra tensor channels or changed rules. Python exposes the
option as `FeatureClient.load(..., view_geometry=True)`; it implies view metadata.
Profile and view can be combined in one trailer. Errors never append a trailer. Omitting these fields preserves ordinary
training selection and transport, and the tensor packet/schema is unchanged.

### CGFEAT02 packet layout

Little endian, no padding. Header: Python struct `<8s11If2If` (68 bytes):

1. Magic `CGFEAT02` (8 bytes).
2. Eleven u32 fields: schema version, coarse-plane count, move-feature width,
   board size, stone diameter, resolution, ply, player (0 Black / 1 White),
   coordinate count N, target count K, final-label-valid flag.
3. One f32: side-to-move outcome (zero when unlabeled).
4. Two u32 fields: samples per diameter, full-domain flag.
5. One f32: side-to-move normalized final score (zero when unlabeled).

Payload: coarse f32 `[12,H,W]`; exact f32 `[N,11]` (schema 4: `[N,10]`);
legal u8 `[N]`; coordinate
IDs u32 `[N]`; target action IDs u32 `[K]`; target visit counts u32 `[K]`;
final ownership f32 `[H,W]` (zero-filled when unlabeled). Score and ownership
labels share the outcome validity flag. See [auxiliary supervision](auxiliary.md).
Target IDs are canonical Rust IDs, including pass at 1,000,000; they are **not**
subset output columns. Python checks schema, dimensions, length, finiteness,
legality encoding and action domains before creating writable tensors.
Schemas 4 and 5 both have twelve coarse planes, but their channel meanings
differ. Schema 5 has eleven exact fields; schema 4 retains ten. Schema 5 uses
spatial centers and footprints instead of coarse scalar liberty area and capture
flags, and adds binary friendly-region splitting to exact inputs. Schema 4 remains supported for existing
score/ownership inference and visualization. Header schema and model feature names,
not width alone, identify the semantics. Packets/checkpoints
without `targets: final-area-v1` remain incompatible.
Rebuild
the extractor and discard incompatible derived packets/checkpoints; source SGFs
remain usable. This unused schema-5 revision was explicitly user-authorized;
checkpoint/ONNX feature names must match, and all older schema-5 tensor packets
must be discarded.
