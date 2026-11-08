# Fine features: avoid repeated work before adding parent/child caches

**Superseded input:** the user subsequently replaced the footprint scalar with
[binary friendly-region splitting](2026-11-08-friendly-region-splitting.md).
These measurements and test-only prototypes retain the old footprint contract;
the projection optimization is no longer a production priority.

Follow-up to the [parent/child audit](2026-11-08-parent-child-features.md), at the
user's request: 6–13% gross cache opportunities are not compelling enough to make
caching the first priority. **There are simpler, measured fine-feature wins.**

## Paired whole-game feature experiment

Same validated 224-ply, 9-equivalent source game and genuine complete target
cells as the preceding audit: 1,576,098 coordinate rows, 1,191,243 legal.
Schema 5 meanings, candidate coordinates, legality and all eleven fields remain
unchanged. No parent/child cache is used.

Artifacts: `scratch/20261108-111223-fine-shortcuts/`, particularly `results.json`,
`summary.json`, `summarize.py` and `benchmark.log`. Source SGF SHA256:
`3474b745f712d09b8105305328fcb768fc9ea51bccbe2127d0d67ae0acd43405`.

Three repeats of each method, rotating order across positions/repeats, serial
release libtest/system allocator. A fresh fine extractor is constructed inside
each timer, including experiment-specific preparation. Full matrix comparisons
against ordinary extraction are outside the timers. An untimed ordinary extraction
at each position warms code/allocator; no extractor memoization survives between
methods. Each reported sample is the summed extraction time over the whole game.

| Method | Median seconds | Less fine-feature time | Speed ratio |
|---|---:|---:|---:|
| Current production extractor | 10.852 | — | 1.00× |
| Certified no-capture summary fast path | 8.738 | **19.5%** | 1.24× |
| Exact footprint projection lookup | 10.009 | **7.8%** | 1.08× |
| Both | 7.887 | **27.3%** | **1.38×** |

The three combined totals are 7.887, 7.888 and 7.882 s; baseline totals are
10.852, 10.862 and 10.850 s. Compare methods *within this paired run*, not against
the previous audit's somewhat slower baseline. These are **measured extraction
reductions**, unlike the earlier cache's sums of potentially avoidable clocks.

Still not a search-throughput result: no ONNX, IPC, batching, MCTS or training is
inside these timers. This is one played line's target cells, not a trained
schema-5 search-tree proposal trace. Inference's value-only leaves do no fine
work. Production adoption still needs controlled end-to-end measurements.

## Before/after component table

At the user's request, a separate paired, instrumented run breaks down both
paths. Artifacts: `scratch/20261108-120939-fine-feature-table/`, including raw
`results.json`, `summary.json`, `summarize.py` and `benchmark.log`. Same source,
coordinates and three rotated repeats; every matrix/legality output again matches
ordinary extraction bitwise. Values are medians of whole-game stage sums.

| Feature/work | Before (s) | Both shortcuts (s) |
|---|---:|---:|
| Local footprint scalar, column 6 | **5.81** | **4.71** |
| ↳ Exact center spans + bitmap projection | 3.14 | 2.13 |
| ↳ Bitmap dilation / union pixel counting | 2.07 | 2.08 |
| ↳ Probes, memoization and footprint bookkeeping | 0.60 | 0.51 |
| Legality + connection/capture/resulting-size fields, columns 0–5 | **5.41** | **3.24** |
| Four border fields, columns 7–10 | 0.05 | 0.05 |
| Preparation, including certificates/lookup table | 0.04 | 0.12 |
| Allocation, other loop work and instrumentation | 0.12 | 0.13 |
| **Total with component clocks** | **11.43** | **8.25** |

Indented footprint rows subdivide column 6; do not add them to its total.
Legality and tactical columns come from shared analysis; there are no independent
per-column times to assign honestly. The "other" row is a signed residual,
including output allocation and timer/bookkeeping costs. Component clocks perturb
runtime, so these totals differ from the earlier ordinary **10.85 → 7.89 s**.
The footprint remains approximately **57%** of the instrumented current path;
shared rules/counts are **39%**. Dilation is essentially unchanged, and spans plus
projection still cost about two seconds even after removing divisions.
"Current" here remains the test-only prototype, not a deployed optimization.

## 1. Stop constructing children when no captures or suicide are possible

`PreparedPosition::analyze_move` currently calls the full prepared validator,
constructing a child position. It then rediscovers connected groups, recounts
enemy stones, checks which enemy groups disappeared, and collects witnesses even
when the requested liberty count is zero. Fine extraction throws the child away.
This happens separately for every coordinate, including neighboring coordinates
in the same cell.

The test-only fast path is deliberately a **sufficient certificate**, not an
approximate replacement for the validator:

1. Check ordinary board bounds/stone overlap. Geometrically invalid placements
   are definitively illegal.
2. Retain two actual liberty-center witnesses per existing group. If their
   separation is at least two stone diameters, no one strict diameter-radius
   exclusion disc can cover both; that group cannot die from one placement.
3. For the remaining groups, require at least one witness outside the candidate's
   exclusion disc. Failure goes to the authoritative validator; it does not claim
   that a capture occurs.
4. Count adjacent friendly groups/stones. If any join, their surviving liberty
   certifies the new group's survival. Otherwise, try four exact cardinal local
   liberty centers; one genuinely free center certifies the new stone's survival.
   If no certificate succeeds, fall back to `analyze_move`.
5. With all old groups and the new group certified alive, captured counts are
   exactly zero, and resulting group size is exactly one plus joined stones.
   Write those facts directly, without constructing a child.

Results: **940,323 candidates**, or **78.9% of legal rows**, take the certified
legal path. 362,906 are rejected by overlap/bounds; only 272,869 rows require the
full validator (including the remaining legal rows and non-geometric illegality).
On average just 3.46 groups per position require candidate-specific witness
checks; the other groups have the separated-witness proof.

The prototype derives witnesses from exact group spans, and this preparation is
included in its times. Shared rules already retain witnesses, so a production
implementation should use those rather than create a second preparation scheme.
The certificate and summary API belong **in the shared rules crate**, not as a
parallel production validator in training code. The sibling repository was not
changed by this experiment.

This only bypasses uncomplicated candidates. A more general summary-only rules
API could also avoid child construction/rescans on difficult candidates, but its
benefit and capture/suicide correctness are not established by this experiment.
That is a larger follow-on, not a promised additional speedup.

## 2. Replace repeated projection divisions with a tiny exact table

The local footprint projects each emitted integer span's y and two x endpoints
onto its fixed 128×128 bitmap. Each projection repeats a runtime integer
Euclidean division. The radius is fixed for the entire position, and all deltas
are bounded by the local adjacency disc.

The experiment tabulates the **exact existing** `bin(delta, radius)` result over
that bounded delta range, once per extractor. Span projection becomes three byte
lookups. The 9-equivalent board needs only 311 bytes. This is arithmetic lookup,
not coordinate subsampling, a coarser raster, or a new approximation. Footprint
probes, center spans, bitmap unions, dilation, cap, memoization and outputs are
otherwise unchanged. Table construction/allocation is included in the timings.

This is the lowest-risk implementation candidate. The complete mapping is
checked against the original division for every supported radius 1–500 and every
supported delta. The full-game experiment checks all output bits as well.

## Next priority

1. With confirmation before modifying the sibling checkout, add the certified
   summary path to shared rules, keeping the full validator for unresolved cases.
2. Trial the exact projection lookup in normal extraction. Do not pursue a
   parent/child cache first: these two within-position changes already remove
   more work, with no branch-lineage or inherited-cache memory machinery.
3. Differential-test captures, suicide, joins, both colours and board sizes, then
   benchmark identical seeded native searches with fixed model, budgets,
   concurrency/batching, and source positions. Record moves/visits/results, memory
   and throughput; retain optimizations only for measured end-to-end benefit.
4. If fine rules still dominate, investigate a general allocation-free summary
   API rather than assuming the remaining work is intrinsically necessary.

## Validation

All four methods produced bitwise-identical exact matrices and identical legality
bytes at every source-game ply, in every repeat. Focused fixtures include pass,
capture (explicitly requiring fallback), group-join candidates and all projection
phases. Additional seeded dense positions check both colours on board sizes
9/13/19 against ordinary extraction. 116 Rust tests passed, eleven optional
benchmarks/tests ignored; clippy passed with warnings denied. No production
feature semantics or search behavior changed.
