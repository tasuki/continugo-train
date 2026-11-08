# Local-only liberty footprint: implementation, preview and measurements

This supersedes the whole-resulting-group field and cost investigation earlier
on 2026-11-07. The intended input is the **new stone's own local geometry**, not
its merged group's distant liberties. No proposal/reranker architecture change
or candidate subsampling was needed.

## Definition and implementation

Exact column 6 is now named `local_liberty_footprint_stones`, followed by the
four borders. Renaming the feature rejects earlier whole-group draft checkpoints
in metadata validation. Schema 5 remains unreleased; regenerate derived packets
and start fresh training. Existing schema-4 inputs remain unchanged.

For candidate `p`, use in-bounds integer centers `q` with
`distance²(p,q) < 2*diameter²`, excluding overlap with `p` or any existing stone
(`distance² < diameter²`). Stones of both colours remain **fixed obstacles**:
no capture removal, group joins, child positions or distant group liberties are
used for this feature. Authoritative legality, connection/capture counts and
resulting group size are still separate, unchanged inputs.

The shared rules library adds `PreparedPosition::visit_local_liberty_spans` and
its collected diagnostic counterpart. These reuse the lazy obstacle-row union
already used for coarse liberties; candidate queries only visit local rows and
nearby blocker intervals. Exhaustive tests compare the spans with `Board`'s
integer geometry. No game-rule or move-validation behavior changes.

`src/features/footprint.rs` projects every exact local center span onto a
**128×128 bitset at 16 samples per physical stone radius**. Nearest-bin projection
retains even an isolated center; bitwise circular dilation measures the footprint
**union**, never a sum. Popcount is divided by the raster disc's 797 pixels, so
one isolated center is exactly one stone-disc and two disjoint raster footprints
are exactly two. The scalar is raw, not clipped or log-scaled.

This is an **approximate raster area**, not the old continuous-area quadrature.
The support fits inside the fixed raster because the footprint reaches less than
`(sqrt(2)+0.5)*diameter` from the candidate. Existing stone centers farther than
`(sqrt(2)+1)*diameter` cannot affect the local center set. A 1024-slot, direct-mapped
cache memoizes full center rasters, bounded at about 2 MiB per extractor. Hash
collisions only replace entries; complete bitmap equality is required for a hit.
Workers share the cache safely and serial/parallel outputs remain bit-identical.

## CPU comparison

Artifacts and reproducible benchmark wrapper:
`scratch/20261107-235226-local-footprint/`.

Same size-9 game and 32 complete diagnostic cells / **23,328 candidate
coordinates per position** as the original audit. Medians of three paired serial
Rust-server extractions, alternating schema order. The ten shared exact columns
were checked bit-for-bit against schema 4, and current tensors were checked
against the updated preview hashes. Replay, transport, neural execution and
search are excluded. These are isolated extraction measurements, not a search
throughput or strength result.

| Ply | Legal candidates | Existing 10 exact inputs | Added local footprint | All 11 exact inputs | Previous added whole-group cost |
|---|---:|---:|---:|---:|---:|
| 0 | 23,328 | 3.00 ms | 165.52 ms | 168.53 ms | 7,769 ms |
| 42 | 14,321 | 140.73 ms | 92.95 ms | 233.68 ms | 1,646 ms |
| 84 | 3,433 | 13.09 ms | 10.72 ms | 23.82 ms | 882 ms |
| 126 | 11,992 | 64.24 ms | 96.93 ms | 161.17 ms | 946 ms |

The corrected/rasterized input costs about **10–82 times less** than the original
whole-group implementation, or about 3.1–8.1 microseconds extra per legal
candidate in these snapshots. This comparison includes a definition correction
and a numerical-method change, not an identical-feature algorithm benchmark.
All twelve coarse planes took 0.027 / 2.35 / 1.41 / 1.36 ms respectively.

The feature is **not free**: it still dominates the exceptionally cheap empty
position, adds about 66–82% to exact extraction at plies 42/84, and about 151% at
ply 126. Parallelism can reduce elapsed time but does not remove CPU work.

## Numerical accuracy

`features::footprint::tests::profile_local_footprints` compares against continuous
capsule-union integration of the **same local, fixed-obstacle spans**, not the
old whole-group definition. Every eighth diagnostic coordinate was audited:
6,630 legal candidates across the four snapshots.

| Ply | Mean absolute error | 95th percentile | Maximum absolute error |
|---|---:|---:|---:|
| 0 | 0.205 discs | 0.272 discs | 0.315 discs |
| 42 | 0.028 discs | 0.106 discs | 0.258 discs |
| 84 | 0.001 discs | 0.010 discs | 0.039 discs |
| 126 | 0.061 discs | 0.105 discs | 0.149 discs |

Maximum error divided by `max(1, reference area)` was **4.4%**. These are sampled
error measurements, not certified bounds. Raster quantization tends to increase
coverage; it is intentionally disclosed rather than called exact integration.
Tests also cover radii from sizes 3/9/13/19/500, isolated/disjoint/overlapping
footprints, collision-safe bounded memoization, capture-independent obstacles,
and a connected group's remote growth leaving the local scalar unchanged.

## Visualization and validation

Updated the same stable preview:
`scratch/20261107-144658-schema5-input-preview/schema5-preview.nn.html`.

All **70 frames / 1,635,525 original diagnostic coordinates** were re-extracted,
including 936,000 legal candidates. New local maps and candidate values replace
the whole-group scalar immediately before borders. Coarse variants and all
original schema-4 policy/critic/context outputs are preserved. The report clearly
states that those old predictions did not consume the new input. No checkpoint,
source game or original rated sidecar was changed. A pre-local backup, regeneration
script, tensor hashes and timings are in that directory; `local-footprint-1920.png`
and `local-footprint-browser-check.json` record the browser check.

Checks passed: 105 Rust training-crate tests (nine explicit diagnostics/benchmarks
ignored); 48 shared-rule tests/doctests; 60 directly affected Python tests; all
16 native inference tests, including Torch/ONNX agreement and legacy policy
execution; clippy and Python lint/format checks. Chrome verified loaded images,
eleven maps, fifteen candidate-table columns, ordering and navigation without
JavaScript exceptions. No continuing training campaign was started.
