# Post-move footprint: cost investigation, not a completed performance fix

The initial scalar input is not practical at current complete-cell inference
budgets. Adding a useful input does not justify ignoring its extraction cost.

Artifacts: `scratch/20261107-232101-footprint-optimization/`.

## Profile

`src/features/footprint.rs` adds an explicitly ignored, test-only diagnostic. It
separately clocks validation/child construction, group discovery, exact liberty
spans and numerical union area, with replay excluded. Four cases use the same
record/cells as the original comparison. Every eighth coordinate is evaluated
for the stage profile; a separate keys-only pass covers all coordinates.

- Empty position: area integration ~0.95 s versus spans ~0.010 s per 2,916 legal
  sampled candidates. Integration is overwhelmingly dominant.
- Ply 42: area ~0.148 s, spans ~0.064 s, validation ~0.020 s.
- Ply 84: area ~0.056 s, spans ~0.052 s. Accelerating integration alone would
  still leave a substantial child-geometry cost.
- Ply 126: area ~0.086 s, spans ~0.033 s.

## Experiments

- Spatial bounds/BVH pruning: insufficient benefit; slower on some positions.
- SIMD plus certified one/two/four-interval union fast paths: helped the empty
  case, but setup/fallback costs hurt other cases. Not retained.
- Adaptive Gauss–Kronrod integration of the actual capsule union: frequently
  slower or exhausted its refinement budget. Relaxing tolerance produced an
  error of ~0.00462 stone-discs at ply 126/action 718837, outside the existing
  reference-check tolerance. Not retained. The width function has horizontal
  edge events from the discrete source rows; treating it as a smooth integrand
  is unsafe without handling those events.
- Translation-normalized area memoization: repeated shapes exist, but full-domain
  snapshots had 9,855 / 4,954 / 524 / 11,102 distinct keys. It does not solve the
  worst cases, and does not remove child-span extraction. Not deployed.
- Reusing circular square-root extensions for binary-exact midpoint grids:
  modest but safe. **Retained.** The original midpoint samples, continuous union,
  interval ordering and outputs are unchanged. Other radii fall back to the
  original computation. Legacy schema-4 extraction keeps the original path.

## Retained optimization: isolated repeat

Medians of three paired serial extractions, same 23,328 coordinates / 32 cells
per case. Timings exclude replay, transport, neural inference and search.

| Ply | Previous exact extraction | New exact extraction | Reduction |
|---|---:|---:|---:|
| 0 | 7,772 ms | 6,714 ms | 14% |
| 42 | 1,797 ms | 1,658 ms | 8% |
| 84 | 896 ms | 821 ms | 8% |
| 126 | 1,010 ms | 923 ms | 9% |

This is **not a practical solution**. Current inputs and preview are retained;
no reranker, reduced candidate domain, proxy feature, missing-value convention,
or policy-normalization change has been implemented.

The next design decision is between a substantially more incremental geometry
algorithm and a bounded expensive reranking stage. A reranker should receive
spatially diverse/center/exploration proposals, not just the cheap scorer's top
coordinates, or it could exclude the eye center before footprint analysis. It
would require explicit proposal/reranker training and policy semantics; silently
zero-filling expensive inputs on unexamined candidates would not be equivalent.

## Checks

Rust release tests pass (101 passed; eight explicit diagnostics/benchmarks
ignored), and clippy passes. Randomized fragmented/bridging capsule cases compare
the optimized integrator bit-for-bit against the original; sampled real child
positions do likewise. Existing connection/capture/isolated-liberty and
serial/parallel tests continue to pass.
