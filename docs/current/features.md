# Real neural features — schema version 4

Extraction lives in `src/features.rs`; rule facts come from `continugo-rules`.
`Position::liberty_spans` returns the exact integer liberty-center intervals used
for UI-style geometry. `PreparedPosition::analyze_group` reuses those spans and
finds an exact legal one-placement capture witness when one exists.
`PreparedPosition::analyze_move` uses the existing prepared validator for
connections, captures and resulting group size. Area integration
lives in `src/liberty_area.rs`; normal move validation never pays for it. No game
rules live in Python.

## Coarse input `[12, H, W]`

All features are relative to the **pre-move player**. Grid endpoints correspond
to action coordinates 0 and 999. `H = W = ceil(999 * density / diameter) + 1`,
using the actual Rust stone diameter; density defaults to four intervals per
stone diameter (38/54/78 samples for 9/13/19 equivalents). Extraction caps grids
at 2048 samples per side. Raster sample locations
are rounded to the nearest integer coordinate. A stone paints its physical disk
(strict distance less than its radius), not its larger move-exclusion disk.

| Channels | Contents |
|---|---|
| 0, 1 | Friendly / enemy stone disks, 0 or 1 |
| 2, 3 | Friendly / enemy group size, painted on member stones |
| 4, 5 | Friendly / enemy group liberty area in stone-disc units, painted on member stones |
| 6, 7 | Friendly / enemy group is capturable in one opposing placement, 0.0 or 1.0 |
| 8–11 | Left / right / top / bottom border distances |

Group-size and other count features use `ln(1 + count) / ln(1 + 1_000_000)`.
Area is directly divided by `π * stone_radius²`, with **no cap, clipping or log
transform**. There are no move-history planes, liberty-coordinate count fields,
or exactly-one-integer-liberty flags.
Border distances are measured from the permitted stone-center boundary in stone
diameters, clipped to [0,4], then divided by four. Outside that boundary they are
zero. Channels other than borders are zero on empty raster samples.

Liberty centers are physically unblocked, in-bounds integer points adjacent to at
least one group member, exactly as in the shared rules. They are not fully
validated moves or counts of cached witnesses; see the area definition below.

All coarse planes and per-coordinate feature columns are **float32**, including
binary occupancy/capture flags. Area integration uses float64 internally before
conversion. The separate legality mask is u8 on the wire / `torch.bool` in Python;
action IDs are u32 on the wire / `torch.int64` in Python.

## Exact coordinate input `[N, 10]` and legality `[N]`

Coordinate IDs are `x + 1000*y`, excluding pass. Rows preserve caller order.
Selective inference evaluates complete chosen coarse cells; training evaluates
complete target-bearing cells (see [network](network.md)). The explicit diagnostic
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
| 6–9 | Left / right / top / bottom border distances, as above |

Columns 1–5 use the logarithmic count transform. For illegal moves, columns 0–5
are zero and only border geometry remains. Legality also has a separate boolean
mask for the network. It includes suicide and captures, not just geometric
availability. It **never applies the direct random bot's eye-avoidance filter**.
Finished games cannot be policy inputs. The shared rules do not enforce ko.

## Exact one-move capturability

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

## UI-style liberty area and numerical accuracy

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

Area is computed once per existing group, not per hypothetical candidate.
Actual child positions get fresh group areas when their features are extracted.
Per-move legality, connections, captures and resulting group sizes remain exact.

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

- Policy targets come only from the chosen move node's JSON `policy` visit counts.
  IDs must be unique and legal, with positive total visit mass. Zero-count entries
  are allowed. Unannotated/random games do **not** acquire invented policy labels.
- A value label requires natural two-pass completion and a scored `RE` matching
  authoritative final scoring. It is +1/-1 for a win/loss from the chosen pre-move
  player's perspective, or 0 for a draw. Margin is not a value target. Games without
  `RE` have no value label; a scored `RE` on an unfinished game is rejected.
- Resignation/time-result labels and branching/setup SGFs are not supported here.

## Rust → Python bridge

```sh
mise run build-features
mise run selfplay -- --engine mcts --games 1 --sizes 9 --seed 41 --simulations 8 --output runs/mcts
```

```python
from continugo_train.features import load_position
from continugo_train.model import PolicyValueNet

sample = load_position("runs/mcts/game-000001-size9-seed41.sgf", ply=0)
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
mise run features -- --sgf FILE --ply 0 --actions 200200,500500 --output runs/position.cfeat
```

The one-shot `load_position` convenience helper replays the SGF per request.
The offline trainer instead uses `FeatureClient` / `features --serve`: each worker
keeps one fully validated game's pre-move snapshots and samples positions without
replaying/scoring each request. `features --inspect` exposes validated metadata
and eligible plies without implementing a Python SGF parser. See
[training.md](training.md) for the bounded loader, game-level holdout and checkpoints.
The tensor packet layout below is unchanged. Server/default one-shot extraction
returns complete cells containing positive placement targets, with full-domain
flag false; a pass-only/no-policy position has zero exact rows. Explicit
`--actions` remains a diagnostic subset; `--full-domain` is a slow reference only.
Packet size scales with target-cell coverage, not a mandatory million-row matrix.
SGFs remain the sole primary game records. Binary packets/cache files are derived, disposable tensors, not JSON
sidecars or another game-record format.

### Optional server diagnostics

Ordinary `features --serve` responses remain `status u8 + length u64 + payload`,
where status 0 carries CGFEAT01 and status 1 carries a UTF-8 error. A request may
set `"profile": true`: **successful** responses then append `length u64 + JSON`
with server stage times and Linux process-lifetime peak RSS. Errors never append
telemetry, so the next request remains aligned. The tensor payload/schema is
unchanged; these are disposable transport diagnostics, not SGF annotations.

```python
from continugo_train.features import FeatureClient

with FeatureClient(profile=True) as client:
    sample = client.load("GAME.sgf", ply=0)
    print(client.last_profile)
```

The Python client adds its own request/read and decode/copy clocks. No profile
flag is sent by ordinary clients, which remain compatible with existing servers.
See [profiling](profiling.md) for stage overlap, IPC residuals and memory limits.

### CGFEAT01 packet layout

Little endian, no padding. Header: Python struct `<8s11If2I` (64 bytes):

1. Magic `CGFEAT01` (8 bytes).
2. Eleven u32 fields: schema version, coarse-plane count, move-feature width,
   board size, stone diameter, resolution, ply, player (0 Black / 1 White),
   coordinate count N, target count K, outcome-valid flag.
3. One f32: side-to-move outcome (zero when unlabeled).
4. Two u32 fields: samples per diameter, full-domain flag.

Payload: coarse f32 `[12,H,W]`; exact f32 `[N,10]`; legal u8 `[N]`; coordinate
IDs u32 `[N]`; target action IDs u32 `[K]`; target visit counts u32 `[K]`.
Target IDs are canonical Rust IDs, including pass at 1,000,000; they are **not**
subset output columns. Python checks schema, dimensions, length, finiteness,
legality encoding and action domains before creating writable tensors.
Schema 4 has 12 planes / 10 move fields. Schemas 1–3 are incompatible: rebuild
the extractor and discard incompatible derived packets/checkpoints; source SGFs
remain usable. Feature changes require a schema-version bump and compatible
checkpoint/ONNX feature metadata.
