# Schema-6 coarse feature cost breakdown

Channel numbers below record the measured pre-reorder layout. The user subsequently
moved group liberty area to channels 4–5, liberty centers to 6–7, and footprints to
8–9; costs and geometry are unchanged. The [current contract](../current/features.md)
and regenerated scratch previews use the new ordering.

Brief CPU release-build profile on Intel Core i5-8400 @ 2.80 GHz, density 4.
Nine positions from the same three SGFs as the schema-5/6 speed smoke test:
size 9 plies 10/53/100, size 13 and 19 plies 3/16/31. The latter games only
contain 32 moves, so this does not cover dense large-board endgames.

Three warmups, 15 measured extractions per position, alternating instrumented and
ordinary extraction. Replay/scoring, exact candidates, serialization, neural
execution and search are excluded. Coarse tensors are checked identical between
instrumented and ordinary extraction on every repetition. Numbers below are the
mean of per-position medians, not a realistic self-play position distribution.

## Results (milliseconds per coarse extraction)

| Plane family / work | Size 9 | Size 13 | Size 19 | Share, all nine positions |
|---|---:|---:|---:|---:|
| Occupancy, 0–1 | 0.055 | 0.031 | 0.032 | 0.8% |
| Group size, 2–3 | 0.833 | 1.046 | 2.129 | 26.4% |
| Liberty centers, 4–5 | 0.041 | 0.083 | 0.091 | 1.4% |
| Liberty footprints, 6–7 | 0.357 | 0.746 | 0.671 | 11.7% |
| Per-group liberty area, 8–9 | 1.796 | 2.306 | 3.525 | 50.3% |
| Borders, 10–13 | 0.019 | 0.038 | 0.079 | 0.9% |
| Shared preparation, spans, colour unions | 0.416 | 0.190 | 0.140 | 4.9% |
| Output allocation | 0.002 | 0.012 | 0.061 | 0.5% |
| Other / unattributed | 0.075 | 0.137 | 0.256 | 3.1% |
| **Total instrumented** | **3.595** | **4.589** | **6.985** | **100%** |
| Ordinary extraction | 3.576 | 4.591 | 6.989 | |

Instrumentation overhead in aggregate: approximately 0.08%, below the noise
of this brief run. Friendly/enemy channels are grouped: liberty computations
and union preparation are shared, so presenting independent complete costs for
every individual channel would misleadingly double-count work. The group-size
and area rows include their own shape painting/canvas allocation and reduction;
centers and footprints exclude their explicitly listed shared geometry costs.
Residual includes grid setup, sorting, output copies, cleanup and unclocked work;
medians of separate stages also need not sum exactly to a median wall clock.

## Where the new area-plane cost goes

Average over all nine positions:

- Area integration: **1.226 ms**, 24.2% of total coarse time.
- Area shape painting/canvas allocation: **0.903 ms**, 17.9%.
- Area supersampled-canvas reduction: **0.413 ms**, 8.2%.
- Combined: **2.542 ms**, 50.3%.

Thus the increase is not all liberty-area numerical integration. `Grid::group`
currently traverses the identical group geometry separately for size and area;
it also reduces separate full 8×8-supersampled canvases. Sharing geometric
coverage tests between the two scalar channels is a promising exact optimization.
Sparse/touched-pixel reductions may also help, particularly on sparse 19-sized
boards. Neither optimization was implemented as part of this measurement.

## Reproduction

`tests/feature_costs.rs` is an ignored release profiling harness. Supply a JSON
array of `{ "sgf": "path", "plies": [10, 53, 100] }` cases:

```sh
CONTINUGO_FEATURE_COST_CASES=scratch/20261117-111959-schema6-speedcheck/cost-cases.json \
CONTINUGO_FEATURE_COST_OUTPUT=scratch/20261117-111959-schema6-speedcheck/plane-costs.json \
cargo test --locked --release --test feature_costs -- \
  --ignored --exact coarse_plane_cost_breakdown --nocapture --test-threads=1
```

Local case definitions and raw per-repetition profiles are under
`scratch/20261117-111959-schema6-speedcheck/`. Raw output includes SGF SHA256, ply, stone/group
counts, resolution, ordinary/instrumented elapsed seconds and nested stage costs.
These `RasterProfile` stages also appear in native inference's existing
`stats.coarse_stages.raster` profile and merge normally across batches. See the
follow-up below for the updated shared-stage names. Instrumentation does not
alter feature schemas, tensors or rules.

## Follow-up: shared size/area shape coverage

Disc and bridge membership predicates and group adjacency traversal are now
computed once for both scalar channels. Each channel still has its own canvas,
independent maximum across overlapping groups and unchanged reduction order.
Tests compare against a frozen pre-change single-channel rasterizer at five
grid densities, including overlapping shapes with different size/area winners.

A paired release-server comparison with 3 warmups and 15 alternating repetitions
per position/mode checked all **270 pairs of complete tensor packets identical**.
The same SGFs and plies were used, in coarse-only and training-cell modes:

| Mode / size | Before | Shared coverage | Time reduction |
|---|---:|---:|---:|
| Coarse only, 9 | 3.481 ms | 3.274 ms | 5.9% |
| Coarse only, 13 | 4.542 ms | 4.470 ms | 1.6% |
| Coarse only, 19 | 6.409 ms | 6.341 ms | 1.1% |
| Coarse only, equal sample average | 4.811 ms | 4.695 ms | 2.4% |
| Training cells, equal sample average | 6.809 ms | 6.737 ms | 1.1% |

Small gains are noisy; these are smoke timings, not a guaranteed throughput gain.
Raw pairs: `scratch/20261117-111959-schema6-speedcheck/shared-comparison.json`; reproduction:
`.venv/bin/python scratch/20261117-111959-schema6-speedcheck/bench-shared.py`.

### Why sharing helps less than the previous profile suggested

The previous shape-painting stages included the lazy allocation/zeroing of the
full supersampled canvases. They were not just geometric coverage tests. That
allocation is now timed separately as `group_canvas_allocation`, while the
actual shared geometry/painting is `group_shared_shapes` (replacing the previous
`group_shapes` / `group_liberty_shapes` fields).

An updated instrumented run, mean of nine position medians:

| Stage | Time | Share of coarse extraction |
|---|---:|---:|
| Four group canvases: allocation/zeroing | 1.396 ms | 29.0% |
| Shared group size/area coverage and painting | 0.181 ms | 3.8% |
| Group-size canvas reduction | 0.404 ms | 8.4% |
| Group-area canvas reduction | 0.408 ms | 8.5% |
| Per-group scalar area integration | 1.221 ms | 25.4% |
| Colour-union liberty footprints | 0.586 ms | 12.2% |
| All other work | 0.611 ms | 12.7% |
| Total coarse extraction | 4.806 ms | 100% |

At size 19, group-canvas allocation/zeroing alone was 2.478 ms (36.4%);
their two reduction stages were another 1.430 ms (21.0%). Each canvas is
`(8*78)^2` floats; four canvases occupy about 6.2 MB, cleared anew even with
only three stones. This explains why group-size inputs were expensive despite
trivial scalar size calculation. Coverage predicates were not the main cost.
Buffer reuse and/or sparse touched-pixel clearing/reduction are the next targets;
they are not implemented in this change.

### Why area was roughly 4.3 times the footprint-plane cost

The earlier 2.542/0.591 ratio combined **per-group numerical area integration,
shape rasterization, canvas allocation and reduction**, whereas the footprint
row measured only its rasterization after shared span/union preparation.
Furthermore, area integration runs independently per group at vertical spacing
at most `min(1, radius/32)` board units. Overlaps between different groups do not
eliminate this work, because their scalar areas must remain distinct.

Footprints first union spans per colour, then rasterize the two unions at the
coarse grid's 8 subrows per pixel (approximately diameter/32 board units apart
at normal density 4). The footprint implementation directly accumulates covered
sample counts into coarse pixels, avoiding a full float supersampled canvas
and its later reduction. Thus a spatial footprint map can be cheaper than a
single scalar per group. The current 25.4% area-integration cost alone is about
2.1 times the 12.2% footprint rasterization cost; after sharing, geometry costs
should no longer be assigned exclusively to either group-size or area planes.

Updated raw stages: `scratch/20261117-111959-schema6-speedcheck/plane-costs-shared.json`.

## Current schema 6 versus schema 5, after reorder and wider bridges

Reran the release-server comparison after shared group coverage, channel reorder,
and bridge radius increase from 0.30 to 0.40 stone radii. Same nine positions,
15 alternating repetitions per position and mode, warmed feature servers. Times
are means of per-position medians; extraction only, not neural/search execution.

| Size | Coarse v5 | Coarse current v6 | Ratio | Training cells v5 | Training cells current v6 | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| 9 | 1.726 ms | 3.320 ms | 1.92× | 6.011 ms | 7.840 ms | 1.30× |
| 13 | 2.215 ms | 4.512 ms | 2.04× | 3.028 ms | 5.441 ms | 1.80× |
| 19 | 2.956 ms | 6.470 ms | 2.19× | 3.764 ms | 7.317 ms | 1.94× |

Worst individual coarse position: 2.62×. Performance remains in the same range
as the initial comparison: sharing predicates did not remove the dominant
supersampled-canvas allocation/zeroing and reduction costs. Larger-board samples
are still early-game positions, not a dense-endgame performance guarantee.

Raw results: `scratch/20261117-111959-schema6-speedcheck/results-current.json`.
Original binaries/results were preserved; reproduction:

```sh
cargo build --locked --release --bin features
cp target/release/features scratch/20261117-111959-schema6-speedcheck/features-v6-current
.venv/bin/python scratch/20261117-111959-schema6-speedcheck/bench.py \
  --schema6-bin scratch/20261117-111959-schema6-speedcheck/features-v6-current \
  --output scratch/20261117-111959-schema6-speedcheck/results-current.json --repeats 15
```

## Current per-plane-family breakdown, including wider bridges

Reran the instrumented harness on the final reordered layout and 0.40-radius
bridges: same nine positions, 3 warmups and 15 alternating profiled/plain
extractions. Mean of per-position medians, milliseconds:

| Planes / work | Time | Share |
|---|---:|---:|
| 0–1 occupancy | 0.040 | 0.8% |
| 2–3 group size, own reduction | 0.407 | 8.4% |
| 4–5 group liberty area, integration + own reduction | 1.634 | 33.6% |
| 6–7 liberty centers | 0.071 | 1.5% |
| 8–9 liberty footprints | 0.590 | 12.1% |
| 10–13 borders | 0.045 | 0.9% |
| Shared size/area canvas allocation and zeroing | 1.395 | 28.7% |
| Shared size/area coverage and painting | 0.203 | 4.2% |
| Shared preparation, liberty spans and colour unions | 0.251 | 5.2% |
| Output allocation and other/unattributed | 0.221 | 4.5% |
| **Total coarse extraction** | **4.858** | **100%** |

Rows for size and area do not include their shared allocation/painting rows;
those must be counted only once. Area integration alone was 1.225 ms, with
0.409 ms spent reducing the area canvases. Ordinary extraction averaged 4.840 ms.
At size 19, allocation/zeroing was 2.467 ms (36.1%), while both group reductions
together were 1.438 ms (21.0%); shared shape painting was only 0.113 ms (1.6%).

Raw repetitions: `scratch/20261117-111959-schema6-speedcheck/plane-costs-current.json`.
