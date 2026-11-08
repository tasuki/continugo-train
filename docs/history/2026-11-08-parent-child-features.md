# Parent/child feature reuse: whole-game cost and invalidation audit

## Conclusion

**Follow-up:** the [fine-feature experiment](2026-11-08-fine-shortcuts.md) found
simpler within-position wins and supersedes the implementation priority below.
This report's cache opportunity measurements remain unchanged.

**Start with bounded reuse of the local-footprint scalar, not incremental coarse
rasters or whole exact rows.** Coarse extraction is only 3.5% of the measured
feature workload. Fine work is approximately half authoritative move analysis
and half the local-footprint column. A metadata-only, 768-KiB coordinate cache
inherited along the played line found reusable footprint computations accounting
for **12.9% of instrumented feature time**, before cache maintenance costs.
Keeping only the immediate parent's rows found about **6.0%**.

These are opportunities, **not measured speedups**. No production cache, neural
search throughput comparison, training run, rule change or feature-schema change
was introduced. A small prototype followed by controlled end-to-end search is
still needed before keeping an optimization.

## Workload and provenance

Artifacts: `scratch/20261108-103436-parent-child-features/`.
Use `results-inherited.json`, `summary.json`, `summarize.py` and
`inherited-benchmark.log` for the final audit. Earlier `results.json` and
`results-detailed.json` retain the preceding decompositions; do not mix their
clocks with the final run. An initial rated-game candidate was rejected by the
current replay parser because its comments used the obsolete policy format; no
SGF was migrated or edited.

The source is an existing completed 9-equivalent neural self-play game:

```
runs/20261106-142353-scratch-score/loop/iteration-0019/selfplay/attempt-0001/game-000022-size9-seed10100921.sgf
SHA256: 3474b745f712d09b8105305328fcb768fc9ea51bccbe2127d0d67ae0acd43405
```

Rust validates the entire SGF and its final score, `B+43.3`. There are 224 moves,
201 placements and 23 passes, with up to 53 stones present. The 223 audited
nonterminal parent/child edges include 22 passes, 57 moves removing existing
stones (149 removed in total, maximum 35), and 21 group-merge transitions. The
terminal edge is excluded because finished games are not policy inputs.

At **every pre-move ply**, extract finalized schema 5, density four, resolution
38, and the complete cells containing that ply's positive-visit placement
targets. This gives 1,576,098 coordinate rows, 1,191,243 legal, mean 9.65 cells
per position, maximum 16,767 rows. No candidate subsampling or invented targets.
The recorded actor used schema 4; the game/visits remain valid source data, but
this is **not** a trained-schema-5 proposal trace. These are target-bearing cells,
not all cells selected by native search.

The release libtest diagnostic is serial and uses the **system allocator**, not
production mimalloc. Replay, target-cell selection, transport, networks, scoring
and MCTS are outside extraction timers. Three ordinary whole-game repeats use
fresh extractors and caches at every position. As native inference does, fine
extraction prepares a fresh rules view rather than retaining the coarse view.
A separate instrumented pass decomposes the work. Audit/cache-simulation work is
outside stage timers. Machine/source/lockfile provenance, shared-rules revision
and source hashes are in the JSON; the shared checkout was clean and unchanged.

## Where the time goes

Median ordinary totals over all 224 positions:

| Work | Seconds | Fraction of feature time |
|---|---:|---:|
| All twelve coarse planes and preparation | 0.414 | 3.5% |
| All eleven exact fields, legality and separate preparation | 11.404 | 96.5% |
| Combined | 11.819 | 100% |

Ordinary ranges: coarse 0.407–0.416 s; fine 11.382–11.436 s. These totals
are **one extraction per played position**, not the time spent generating the
original game or searching all its branches.

The separately instrumented pass takes 0.395 s coarse and 11.903 s fine,
including fine preparation. Clock/bookkeeping overhead and measurement variation
mean its totals differ from the ordinary baseline. Do not add nested subtotals.

| Fine stage | Instrumented seconds |
|---|---:|
| Local-only footprint, column 6 | 6.206 |
| Authoritative analysis of legal candidates | 5.034 |
| Authoritative analysis of illegal candidates | 0.269 |
| Constructing prepared groups/witnesses again | 0.036 |
| Four border columns | 0.052 |
| Writing legality and five tactical count fields | 0.062 |
| Dropping analyzed child positions | 0.052 |
| Output allocation | 0.002 |

`analyze_move` supplies the legality/connection/capture/resulting-size fields
together; this diagnostic does not invent separate costs for those columns.
Its legal-candidate stage includes child construction and rule scans, even though
the caller only retains a summary. Internal rule optimization remains a separate
possible task, not evidence that caching those summaries is safe.

The footprint stage subdivides into:

| Footprint work | Seconds |
|---|---:|
| Fixed-obstacle spans and projection into the center bitmap | 3.523 |
| Bitmap dilation/popcount, including cap-three early exit | 2.076 |
| Saturation probes | 0.164 |
| Existing center-bitmap cache lookup/hash | 0.115 |
| Existing cache insertion/allocation | 0.199 |

373,377 legal candidates return immediately from successful saturation probes;
208,890 fallback center sets are empty; 53,638 hit the existing bitmap cache.
**Merely retaining that existing cache across positions still pays span extraction
and bitmap construction.** A coordinate-keyed scalar with sound invalidation can
avoid both major footprint stages.

Coarse decomposition, in milliseconds over the whole game:

| Coarse stage | ms |
|---|---:|
| Prepared groups and witnesses | 36.5 |
| Exact group liberty spans, including first lazy blocker index | 78.0 |
| Connected group shapes/bridges | 80.7 |
| Physical occupancy | 19.2 |
| Per-colour span union | 8.2 |
| Group-canvas reduction | 39.3 |
| Liberty-center coverage | 10.6 |
| Liberty-footprint coverage | 115.0 |
| Border planes | 4.3 |
| Output raster allocation | 1.4 |

Capturability searches and continuous liberty-area integration have **zero
calls** in schema 5. Remaining coarse time includes geometry/grid setup, copies,
destruction and bookkeeping. Raster subtotals live inside `painting`, which lives
inside coarse total.

For scale, median per-position coarse/fine times are 0.024/1.46 ms at ply 0,
1.81/128.69 ms at ply 22, 1.96/67.74 ms at ply 112, and 1.69/9.41 ms at ply 222.
Their exact row counts differ; these are examples, not a density-controlled phase
comparison.

## What survives an edge?

Use absolute colours for retained geometry; every move, including pass, swaps
friendly/enemy channel pairs. Compare group membership using sorted stone
colour/coordinates, **not vector indices**: captures compact the stone vector.

Across the 223 edges:

- **90.8%** of child groups have exactly the same members as a parent group.
  Only **73.5%** have both the same members and identical exact liberty spans.
  Membership alone is not a valid liberty-cache key.
- A conservative distance test proves **66.8%** of child groups clean without
  checking span equality. Every such group's full spans matched the child.
- **76.6%** of in-bounds blocker rows lie outside all changed stones' exclusion
  row bands. This is a guaranteed-clean bound, not equality measured after
  rebuilding the private blocker index. For placement edges alone it is 74.0%;
  for removal edges 64.3%.
- After correcting for side-to-move channel swaps, unchanged float32 values in
  the two-colour occupancy/group-size/center/footprint planes are respectively
  **98.8% / 95.6% / 98.6% / 95.9%**. These counts include zeros. They do not imply
  that the remaining raster work is proportional to changed pixels.
- Only **13.5%** of child target-cell coordinates were requested in the immediate
  parent. Requiring both rows legal and a provably unchanged local neighborhood
  leaves **117,228** reusable footprint values: **7.5%** of child requests.
  Their actual footprint computation cost is 0.736 s in this instrumented pass,
  about 6.0% of all feature time before any cache-maintenance costs.

The second simulation retains legal-coordinate footprint scalars from **all
surviving ancestors on this one played line**, rather than discarding everything
not requested by the immediately preceding position. It uses 65,536 direct-mapped
slots with exact coordinate tags; collisions evict, never cause false hits. Before
each position, invalidate every entry within the conservative influence radius
of **any added or removed stone**, then add newly extracted legal values.

The simulation holds 786,432 bytes of slots, reaches 27,091 occupied entries,
and finds **300,237** legal reusable values (25.2% of legal rows, 19.0% of all
requested rows). Every retained scalar is checked bit-for-bit against full child
extraction. Their original footprint computations consume **1.586 s**, or
**12.9%** of instrumented feature time. This is a gross opportunity estimate:
extraction still computes all values, and invalidation/lookup/branch-management
costs are not deducted. Cache hits also change the existing bitmap-cache access
stream, so the sum of skipped clocks is not a predicted net runtime.

The one-edge snapshot's coarse + complete exact matrices + action/legal arrays +
stable group keys/spans average 422 KiB, maximum 897 KiB, excluding containers,
prepared rules, blocker rows and allocator overhead. Keeping everything at every
search node is not bounded merely because one node looks small. The 768-KiB
simulation is bounded **for one line**, not a memory design for a branching tree.

## Implementation difficulty and recommended experiment

1. **Borders / occupancy: easy, little payoff.** Border fields depend only on
   board size and coordinate; coarse borders can be shared by board/density.
   Occupancy updates are geometrically local. But removing float32 contributions
   by subtraction need not preserve original bits; repaint dirty pixels/tiles
   from contributors in the existing stable order. These stages are tiny here.
2. **Blocker rows and unchanged-group spans: moderate.** A changed stone affects
   rows within `diameter - 1` of its y coordinate. Rebuild just those row unions
   from child contributors (or retain multiplicity); deleting an interval from a
   merged union cannot recover what another blocker still covers. Unchanged
   member sets still need blocker-dependency invalidation. Group joins, removals,
   member-index remapping and liberty witnesses need explicit treatment. The
   shared `PreparedPosition` owns borrowed, position-local lazy data; it is not
   currently an owned incrementally reusable feature state.
3. **Coarse group/liberty rasters: moderate to hard, low ceiling here.** Extending
   or merging a group changes its size feature along the **entire** connected
   shape, potentially far from the move. Removing a max/union contributor needs
   the remaining contributors. Liberty footprints are unions, not additive group
   textures: recompute affected union rows/sample regions, expanding changed
   center rows by the physical radius. Reusing already-averaged coarse pixels or
   subtracting a group's coarse map is not generally correct. Even eliminating
   *all* coarse extraction saves only 3.5% of this feature workload.
4. **Local-footprint scalar: best first prototype, moderate integration.** It is
   colour-blind and intentionally ignores each candidate's hypothetical captures.
   Its center annulus reaches `< sqrt(2)*diameter`; existing blockers influence
   it only inside `(sqrt(2)+1)*diameter`. Conservatively invalidate a candidate
   within the upward-rounded radius of every actual added/removed stone (266
   coordinates on this board). Pass changes no geometry. Reuse only the scalar;
   child legality must still be checked, and illegal rows must retain zero in
   column 6. Bound retained bytes across active games/branches, not just per node.
5. **Whole exact rows: hard.** Side to move flips, suicide/capture legality is
   colour-dependent, and distant group merges can change tactical counts near
   unchanged stones. Swapping friendly/enemy columns or applying the scalar's
   local radius to the whole row is unsound. Keeping both colours would require
   counterfactual rows not normally requested and dependency tracking for groups
   and liberties. Do not begin here.

Native integration also needs a parent/feature-state hint: search calls
`Evaluator` with only a `Game`, and batching sends only game/policy/seed. The
model context cache contains encoded **exact-state** outputs, not parent lineage
or these scalar maps. A single mutable last-position cache cannot safely handle
interleaved games and sibling branches. Consider immutable bounded states or
copy-on-write tiles with a global byte budget and ordinary-extraction fallback.
Incremental raw features do not make neural encodings incrementally reusable.

Next: time-box the scalar-only prototype, preserve full extraction as an oracle,
then run identical seeded native searches with fixed model, concurrency, batching,
cache budget and source positions. Include value-only leaves (which have **no fine
work**) and report visits/moves/results, cache coverage, invalidation cost, peak
memory and end-to-end throughput. This single game's target-cell overlap and
played-line survival are not search-tree coverage estimates; recheck a trained
schema-5 actor and larger boards before claiming a general gain.

## Validation

The audit compares every coarse float, exact float and legality byte against
ordinary extraction at all 224 plies, and checks all conservatively reusable
group spans and scalar cache hits against full child results. Captures, group
merges and pass transitions occur in the real game; focused tests additionally
exercise these dependencies and collision/budget handling. All feature meanings,
source games and search settings are unchanged.

114 Rust tests passed, ten opt-in tests/benchmarks ignored; 15 focused Python
unittests passed. Clippy passed with warnings denied. The diagnostic itself also
passed separately in release mode. See the retained test/build logs.
