# Visual coverage tests: local footprint centers must not disappear

The earlier tests covered bitset unions, isolated raster discs and area error,
but did not directly visualize the whole exact-center → bin → footprint path.
Added four permanent regression tests in
`src/features/footprint/coverage_tests.rs` and a manual renderer in
`tests/visualize_footprint_coverage.py`. Production projection/dilation were
factored into small shared functions without changing their calculations.

## What is tested

- **708,518 radius/axis-offset phases**, covering every physical radius 1–500 and
  every integer offset in the local adjacency bound. Binning is monotone and
  leaves the full 16-pixel disc margin on both axes. An independent integer
  distance bound also checks nearest-bin positioning and half-cell ties.
  Projection is separable, so this is not a random phase sample.
- **9,216 single-center positions**: every interior bit position with the required
  margin, including x=63/64 and all buffer edges, must retain its center and the
  full 797-pixel disc, matching an independent pixel-by-pixel stamp at every
  position. A center may not disappear, shift incorrectly, or have its footprint cropped.
- **Four board reflections** of a real, single-center corner pocket: exact
  extraction returns precisely that point, and the production scalar is 1.
- **Nine visual fixtures** exercise actual production `project_span` and `dilate`.
  Every individual integer center in every span must retain a projected bit and
  footprint coverage. The entire footprint must equal an independent oracle
  that stamps every circle pixel individually (no production spread/shift/row
  table reuse). Nonempty spans may not become zero area.
- The four real fixed-obstacle fixtures additionally compare every center with
  exhaustive independent `Board` predicates and check actual production `area`,
  cold and cached. Synthetic fixtures explicitly test only projection/dilation,
  not rule extraction.

The real off-grid eye has center `(500,500)`, candidate `(566,588)`, radius 55,
and fixed stones at `(434,412)`, `(412,566)`, `(588,434)`. Its relative offset
`(-66,-88)` is between raster samples. The point has no other exact centers in
its eight neighboring integer coordinates, yet its full raster disc survives.
There are outer liberties too; the highlighted eye is one component, not the
only liberty of the new stone.

## What is not guaranteed

Several nearby exact centers may map to one occupied bin. In the deliberate
resolution-limit fixture, centers `(610,444)` and `(610,555)` have a one-unit gap
between their radius-55 physical discs. Quantization closes that gap into one
shared pixel; both centers and both footprints remain present, with scalar
`1.9987453` discs. The raster is not a topology/eye-count guarantee, nor does this
say that coarse policy selection examines every possible candidate move.

## Visual evidence and regeneration

Self-contained interactive gallery (click any panel to enlarge):

`scratch/20261108-073503-footprint-coverage/coverage.html`

Contact sheet:

`scratch/20261108-073503-footprint-coverage/coverage-contact-sheet.png`

Red = exact centers (markers enlarged for visibility), green = continuous
physical-radius capsule union, orange = exported production bitset. Gray discs
are fixed obstacles; blue is the candidate; pale raster cells are center bins.
The renderer reads Rust test output only; it performs no rule extraction or
footprint reconstruction. Insets show the real one-point eye and the tiny-gap
resolution limit.

```sh
mkdir -p scratch/YOUR-TIMESTAMP-footprint-coverage
CONTINUGO_FOOTPRINT_COVERAGE_REPORT=scratch/YOUR-TIMESTAMP-footprint-coverage/coverage.json \
  cargo test --locked --release --lib features::footprint::coverage_tests -- --nocapture
.venv/bin/python tests/visualize_footprint_coverage.py \
  scratch/YOUR-TIMESTAMP-footprint-coverage/coverage.json \
  scratch/YOUR-TIMESTAMP-footprint-coverage/coverage.html
```

Tests normally run without writing files; JSON export is explicit. The renderer
refuses to overwrite an existing report unless passed `--force`. Gallery JSON,
source renderer copy, test log, browser-check JSON and screenshot are retained
in the artifact directory. Chrome checked all nine cards / 31 SVG panels,
click-to-zoom, and absence of JavaScript exceptions. Validation also passed all
109 Rust tests (nine explicit audits/benchmarks ignored), 19 feature/viewer
Python tests, clippy, and Python lint/format checks.
