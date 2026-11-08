# Replace the local footprint with binary friendly-region splitting

The user explicitly reopened **unused schema 5** and requested this replacement.
Column 6 is now `splits_friendly_region`, a float32 bit; the other ten exact
columns, twelve coarse planes, action selection and authoritative rules are
unchanged. No training was started. Existing schema-4 actors remain supported.

## Meaning

For a legal move, use its actual child survivors, with just the new stone
removed. Available integer stone centers are connected by eight-neighbour unit
steps, with strict diameter exclusion around every surviving stone. Find the
component containing the candidate. It qualifies only if its blocked neighbours
touch at least one friendly stone and no enemy; board edges are neutral.

Return one if inserting the candidate's exclusion disc divides **that single
component** into at least two nonempty components. Otherwise return zero. This
preserves single-coordinate pockets/passages and responds to boundary colour.
Capturing an enemy intruder can make the source region qualify. Removing friendly
stones is handled from the actual child too, not inferred from enemy capture
counts. There is no sampled raster, spatial interpolation or area threshold.

This is deliberately a geometric input, **not** certified two eyes, unconditional
life, independent future stone placements or a single owning chain. Boundaries
can involve multiple friendly groups. Existing distinct pockets alone do not
count: the candidate must split one component. See the complete
[contract and retained motivation](../current/features.md#binary-splitting-predicate).

## Fast path and memory

`src/features/splitting.rs` builds a position-level exact scanline component index
lazily. Ordinary moves share it; most queries reject a mixed/enemy region
immediately. Qualifying queries subtract the candidate disc only from its source
region's runs and count components with scanline union-find. Reused thread-local
scratch avoids allocations after warmup. The old per-candidate span projection,
footprint dilation/popcount, and 2-MiB bitmap cache are absent from production.

Different captured-survivor sets share exact-keyed immutable indices, capped at
16 states and 4 MiB accounted index/key payload per extractor. Oversized entries
are computed without retention. Parent-index and scratch storage are additional,
bounded by the fixed integer geometry. Cache locks are released before queries;
parallel calls use their own scratch. The shared rules checkout was not modified.

## Brief paired comparison

Artifacts: `scratch/20261108-144932-region-splitting/`. Use **`results-final.json`**,
`summary.json`, `summarize.py` and `benchmark-final.log`; initial results/logs
retain the preceding run. The first `/usr/bin/time` log includes compilation and
its RSS must not be read as feature-process memory. The final warmed invocation
has no compilation and reports about 29.6 MiB lifetime peak RSS, including cargo,
replay and both methods, not separate per-method peaks.

Same source game as the earlier feature investigations:

```
runs/20261106-142353-scratch-score/loop/iteration-0019/selfplay/attempt-0001/game-000022-size9-seed10100921.sgf
SHA256: 3474b745f712d09b8105305328fcb768fc9ea51bccbe2127d0d67ae0acd43405
```

All **224 pre-move positions**, **1,576,098 coordinate rows**, using complete
positive-visit target cells. Three repeats, rotating method order per position.
Serial release libtest/system allocator; both methods include fresh fine/rules
preparation, allocations and full authoritative `analyze_move` for every query.
The old scalar is retained as a **test-only** reference. Neither method uses the
earlier experimental no-capture summary shortcut or exact projection lookup.
Compilation, replay, target-cell selection, assertions, coarse extraction,
networks, transport and MCTS are outside extraction timers.

| Full fine extraction over the entire game | Median | Mean per position |
|---|---:|---:|
| Former local-footprint input | **10.949 s** | 48.88 ms |
| Binary friendly-region split | **5.379 s** | 24.01 ms |

**50.9% less total fine-feature time, 2.04× extraction throughput.** Samples:
old 10.951 / 10.949 / 10.936 s; new 5.383 / 5.373 / 5.379 s. All legality bytes
and all ten unchanged fine columns match bitwise in every repeat. This is a
semantic replacement, so column 6 is intentionally different, not a parity claim.
Do not compare methods across earlier runs with different warmup/clocks. This is
not a measured native-search speedup or evidence of increased playing strength.

9,218 candidates are positive across 30 positions. Maximum accounted retained
parent/capture index payload is **509,644 bytes (~498 KiB)**, with at most five
capture states retained in this workload. This excludes transient build memory,
container overhead and thread-local scratch. These are observed payloads, not
universal bounds or a per-method RSS comparison. Some positions are not faster;
remaining rule analysis and capture/index work still vary substantially.

## Is the splitting bit still the bottleneck?

No. A follow-up paired ablation (`no-feature-ablation.json` and
`no-feature-ablation.log` in the same artifact directory) compares the current
extractor with the ten shared exact fields only. This baseline uses schema-4
**move fields**, not its coarse extraction: legality, tactical counts and borders
are bitwise identical, and no coarse work is timed. Both methods still use full
shared move analysis. Same game/coordinates, three rotated repeats.

| Whole-game fine extraction | Median seconds |
|---|---:|
| Shared rules, counts, borders and preparation only | 5.135 |
| Same work plus binary splitting | 5.458 |
| **Incremental splitting cost, including indices/caches** | **0.323 (~5.9%)** |

Paired increments are 0.327 / 0.324 / 0.323 s. These are ablation differences,
not per-function self clocks; the extra output column/allocation effects are
included. The fresh paired total differs slightly from the preceding run and
must not be subtracted from that run's timings. The shared move-analysis path,
not splitting, now accounts for nearly all fine-feature time. Keep schema 5;
obsolete local-scalar code/benchmarks are a todo cleanup, not production work.

## Contract and tests

Rust/Python feature names, packet binary-value validation, viewer [0,1] rendering,
training/export metadata, and operating docs use the replacement. Old footprint
schema-5 checkpoints/exports fail name matching. **Discard all old schema-5 tensor
packets**, even those whose old scalar happens to be zero or one: unchanged
packet version/width cannot distinguish their meaning. Source SGFs remain valid.

Independent point-flood-fill differential tests cover dense mixed positions on
sizes 3/5/9. Fixtures check a single-coordinate synthetic split, a five-coordinate
real dividing window, diagonal connectivity, mixed boundaries, opposite colours,
ordinary eye filling, and an enemy intruder actually captured by the dividing
move. Existing serial/parallel tests preserve exact feature bits. Historical
footprint audits now explicitly call the old reference rather than pretending it
is the current schema-5 contract.

120 Rust tests passed, thirteen optional tests/benchmarks ignored. 56 focused
Python features/network/viewer/inference/profiling tests, including stale-metadata
rejection, passed; clippy and formatting/lint checks passed. No old models, SGFs, or unrelated
working files were rewritten.
