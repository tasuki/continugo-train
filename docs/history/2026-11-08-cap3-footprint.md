# Local footprint capped at three: cheap saturation and updated costs

At the user's request, the input is now exactly **`min(previous raster area, 3)`**.
Values below three keep the same precision. No candidate filtering, policy
normalization, capture/legality rule or training-target change was made.

## Fast paths

`src/features/footprint.rs` first tests four outer cardinal centers, followed by
four diagonals when needed (especially at board corners). All probes remain
strictly inside the local adjacency annulus. Shared rules provide a cached,
single-point fixed-obstacle query, `PreparedPosition::is_local_liberty_center`.
No hypothetical captures, group joins or child positions are used.

Three free probes certify saturation only if their **projected raster discs are
pairwise disjoint**. Compatibility is precomputed per stone radius, with strict
center separation greater than the raster disc diameter. This is a sufficient
lower bound on the full raster union, not a guess based on neighbor count. The
shortcut returns three before span extraction, center projection or raster work.
Failed probes do not imply empty space and never discard a narrow pocket.

Fallback raster dilation counts newly covered pixels and stops as soon as the
union reaches `3 * 797` pixels. A result below the cap still uses the complete
local geometry. The existing bounded memoization stores capped results.

The feature name is now `local_liberty_footprint_stones_cap3`, identifying the
changed semantics in checkpoint/ONNX contracts. It remains column 6, before
borders. Schema 4 is unchanged. Earlier uncapped draft-schema-5 weights are not
silently reused; regenerate derived tensors and start fresh schema-5 training.
The draft schema number remains five until the first new training run.

## Existing four-position benchmark, repeated only for the new code

Artifacts: `scratch/20261108-093419-cap3-footprint/`.

Same source game, same 32 cells / **23,328 coordinates per position**, median of
three serial paired extractions alternating schemas 4 and 5. Replay, transport,
network execution and search are excluded. The before column reuses the previous
recorded uncapped-local measurements; no new old-code benchmark or training
benchmark was run.

Before rebuilding, uncapped tensors were retained for these four cases. Every
new matrix was checked bit-for-bit against those matrices with only column 6
clipped to three. Actions, legality, the other ten exact columns and coarse
rasters were unchanged. The paired schema-4 baseline additionally verifies all
shared exact features in each repeat.

| Ply | Legal coordinates | Coarse planes | Existing 10 exact inputs | Added footprint before | Added cap-3 footprint | All exact inputs now |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 23,328 | 0.027 ms | 3.21 ms | 165.52 ms | **1.10 ms** | 4.31 ms |
| 42 | 14,321 | 2.32 ms | 144.82 ms | 92.95 ms | **86.67 ms** | 231.48 ms |
| 84 | 3,433 | 1.41 ms | 13.40 ms | 10.72 ms | **8.73 ms** | 22.13 ms |
| 126 | 11,992 | 1.41 ms | 64.81 ms | 96.93 ms | **93.15 ms** | 157.96 ms |

The opening footprint overhead falls by about **151 times**. Populated-position
improvements are modest: about 7%, 19% and 4% for the added field. Do not describe
this as a general 151-times speedup. Only 1,772/14,321 legal coordinates are
saturated at ply 42, none at ply 84, and 8,645/11,992 at ply 126. Even saturated
positions cannot always be certified by this small probe pattern, so constrained
geometry can still require the raster. Ordinary move analysis remains a major
cost. These are inference-sized feature workloads, not training-step timings.

## Validation and previews

111 Rust training-crate tests passed (nine optional audits/benchmarks ignored),
48 shared-rule tests/doctests passed, and 76 Python/native tests passed. Clippy and
Python lint/format checks passed. New tests verify:

- capped dilation equals the full raster clipped at three on varied bitmaps;
- three almost-touching discs below three are not rounded up;
- valid probe offsets and disjointness certificates across radii 1–500;
- empty-board center/corner fast paths across board sizes;
- point queries match exact span geometry, including boundaries;
- isolated/off-grid/corner coverage tests still pass below the cap.

The stable NN preview was re-extracted for all 70 frames and now displays [0,3]
with the cap-three name. Original schema-4 predictions are unchanged. The visual
coverage gallery now distinguishes full verification footprints from the capped
production input; it checks the latter against `min(full raster,3)`.

- `scratch/20261107-144658-schema5-input-preview/schema5-preview.nn.html`
- `scratch/20261108-073503-footprint-coverage/coverage.html`

Pre-cap backups, fresh tensor hashes, screenshots and browser checks are retained
alongside those reports. No continuing training campaign was started.
