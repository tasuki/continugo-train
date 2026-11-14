# Geometry move proposals

The geometry proposal policy replaces exhaustive coordinate enumeration inside
selected coarse cells with a small set of geometric alternatives. The coarse NN,
exact shortlisted features, fine NN scorer and MCTS are unchanged.

The policy is the **schema-5 inference default**, with soft budget 16, including
neural self-play, arena and `infer`. `infer --geometry-budget N` overrides the
budget; `infer --complete-cells` selects complete-cell reference analysis.
Legacy schema-4 models retain complete-cell inference. `proposal-view` visualizes
the same geometry proposals. The candidate contract is **`geometry-v2-frontier`**. Implementation lives in
`src/geometry_proposals.rs`; inference integration is in `src/inference/`.

## Placement domain and shared geometry

The coarse NN selects its top cells plus seeded exploratory cells. Their union is
the allowed placement domain: no generator proposes coordinates outside it. Cell
borders are selection boundaries, not physical connection or overlap boundaries.

`src/features.rs` retains exact liberty spans per group before merging them into
coarse colour rasters. The generator uses these spans, not raster pixels. The
owned geometry is stored alongside encoded context and included in the bounded
LRU's memory accounting. Value-only queries do not generate candidates.

Group reasoning uses complete geometry, including stones and liberties outside
selected cells. In particular, a capture proposal must block **all** of its
target's liberties, not just those inside the candidate's cell.

## Proposal sources

### Extend, including connected hugging

For each relevant friendly group in each selected cell, Extend proposes at most
three coordinates:

1. A **connected hugging** point, when geometrically available, is reserved even
   if it lies inside maximum extension reach. The generator intersects the
   group's free connected centers with nearby enemy-contact bands. Contact means
   that one integer-neighbour step closer would overlap an enemy stone. Among
   contact points, it chooses the one that extends farthest.
2. Remaining slots contain the farthest points of separate outward-frontier
   pieces. Distance is measured to the nearest member of the group, not its
   centroid.

The true frontier is where stepping farther would disconnect or leave the board.
Lazy per-group union-of-adjacency-disc row intervals and integer eight-neighbour
erosion distinguish this frontier from overlap holes and artificial cell edges.
Only rows needed by selected cells are calculated, with reuse across cells.

Farthest-point selection examines interval endpoints and nearest-site bisector
breakpoints rather than enumerating coordinates. A cell with no outward frontier
or connected contact contributes no Extend points; it is not filled with corner
samples.

`extension_groups` records the groups for which each point was generated. The
three-point quota is per originating group, with deduplication across sources.
**Hug is a subset of Extend, not a separate move generator:** “Connected hugging”
and “Extend + Hug” select the same candidates.

### Join — lens extremes

A sweep of friendly liberty-span endpoints finds exact overlap regions for each
connected group set. For two groups, the locally supporting member stones define
the group-to-group axis. Join proposes:

- The **two side extremes perpendicular to that axis**.
- The other two extremes along the axis.

Coinciding coordinates are deduplicated. There are no additional center or
compass samples. For more than two groups, Join uses four cardinal extremes of
the common region; regions connecting only a pair remain available. Tiny and
one-coordinate overlaps are retained.

Extremes are taken within the selected placement domain. Global lens tips in
unselected cells remain unavailable. Source tags identify generated join
extremes, not random coordinates that happen to join the same groups.

### Capture

Separated liberties can rule out a one-move capture. Otherwise, the generator
intersects strict exclusion-disc constraints for every liberty-span endpoint,
then subtracts placement blockers and board bounds. It does not run exhaustive
`analyze_group` capture searches or enumerate the resulting coordinates.

Up to four cardinal extremes are proposed per connected capture patch in a
selected cell. Capture regions are cached across cells, and coordinates are
deduplicated with the other sources. Actual legality and capture outcomes are
checked afterward by the authoritative rules.

### Random

Each usable cell reserves 2–4 seeded, geometrically free random coordinates:
`clamp(budget / 4, 2, 4)`, limited by available coordinates, with bounded duplicate
retries. These supply quiet and unconnected exploratory alternatives.

There is no separate Restrict or Breathe generator/scorer. Hugging belongs to
Extend; neither blocked-opponent-liberty counts nor hypothetical local-breathing
counts are computed to select candidates.

## Combining and scoring proposals

Duplicate coordinates are merged, retaining source and target tags. The direct
pool receives exact analysis without preliminary heuristic scoring or diversity
ranking. `preliminary_candidates` equals the final pre-legality coordinate count.

The budget is a **soft target and overflow-reporting threshold**, not a hard cap
or fill requirement. Cells may contribute fewer points; multiple groups, join
regions or capture patches can exceed it. Selected geometric opportunities and
random reservations are not silently discarded to meet the target.

Only the shortlist receives authoritative legality, capture, connection and
splitting features. Illegal proposals are rejected. Splitting uses actual child
survivors after captures. The existing fine NN ranks the legal alternatives.

Policy normalization preserves global coarse pass mass. Selected nonempty cells
share placement mass according to their coarse weights, with a local softmax over
each cell's shortlisted legal coordinates. This is a restricted-domain policy,
not complete-cell or exhaustive full-board inference.

Source bits are `1` Extend, `2` Join, `4` Hug, `16` Capture and `32` Random.
Sources overlap; every Hug proposal also has the Extend bit.

## Visualization and diagnostics

`src/bin/proposal-view.rs` is the CLI entrypoint. Report generation lives in
`src/proposal_view/mod.rs`, with its page template in `src/proposal_view/page.html`.
It uses the shared ContinuGo embedded viewer, with a host-side proposal overlay. It supports source filters, coarse-cell shading, native hover
previews, and a Dots / mover-coloured Stones toggle. Stone-layer opacity starts
at 0.2 and is adjustable; overlapping alternatives do not accumulate opacity.
The viewer bundle is copied into the output directory, not modified. See
[inference usage](inference.md#geometry-proposals).

`infer --profile-json` reports disjoint generation-stage timings and overlapping
source counts. Tests cover integer interval operations, actual versus artificial
frontiers, connected hugging inside maximum reach, per-group quotas, rotated
lens side tips, multiway/single-coordinate joins, captures, deterministic seeds
and profiling parity. Native tests cover normalization/pass mass, legality,
batching/cache/parallel parity and mode switching. Browser tests check replay
against Rust, filters, opacity, resize alignment and hover passthrough.

Example measurements use saved size-9 positions with cg8/r0005, CPU ONNX, one
feature thread, and 30 uncached queries after three warmups:

| | Ply 12 | Ply 35 | Ply 60 |
|---|---:|---:|---:|
| Coordinates before legality | 164 | 144 | 139 |
| Legal coordinates | 164 | 142 | 113 |
| Generation, ms/query | 0.55 | 0.50 | 0.58 |
| Whole policy query, ms | 4.67 | 5.38 | 5.65 |

Previews: `scratch/20261114-140252-geometry-proposals/index.html`.
Commands/raw timings:
`scratch/20261114-140252-geometry-proposals/previews/timings.json`.
These are policy-query measurements, not full search/self-play throughput or
playing-strength results. The [historical stage breakdown](../history/2026-11-14-geometry-proposal-stages.md)
describes the earlier prescored generator, not this implementation.

## Limitations and training

- Geometric eligibility does not prove suicide legality. A reserved hugging point
  can be removed by final validation; adaptive illegal-candidate refill is not
  implemented.
- Neither empty proposals nor an empty selected domain prove that the board has
  no legal placements. Coarse-cell omissions and within-cell omissions are
  separate coverage limitations.
- These sources do not guarantee rare splitting moves or tactical/playing-strength
  equivalence to complete-cell searches. Teacher-search coverage, equivalent
  tactical outcomes and equal-wall-clock strength remain unverified.
- Training retains the complete-cell `cells-v1` objective, distinct from native
  inference's restricted-domain normalization. A candidate-only training objective
  would require persisted/reconstructed domains and seeds, matching local target
  normalization, and versioned metadata. A shortlist must not be treated as a
  complete cell; teacher moves must not be injected into generator-coverage tests.
