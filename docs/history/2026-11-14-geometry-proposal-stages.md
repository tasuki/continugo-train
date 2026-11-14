# Geometry proposal stage breakdown — 2026-11-14

**Historical v1 measurements.** The current [v2 direct generator](../current/geometry-proposals.md)
removes Restrict/Breathe scoring and diversity ranking. These numbers and the
captured script refer to the earlier implementation, not current performance.

These are **whole policy-query** timings, not self-play throughput, search time or
playing-strength results. No campaigns or shared-rule sources were changed.

## Workload / method

- Intel i5-8400; optimized branch binary, CPU ONNX, one feature thread.
- Frozen `artifacts/models/cg8/r0005`; saved size-9 game
  `game-000004-size9-seed88100003.sgf` from cg8 iteration 1; plies 12/35/60.
- Same top/exploratory cell selection and seed 0; geometry soft budget 16/cell.
- Three runs of 20 uncached queries each, after three warmups per process.
  Alternate complete/geometry order and reverse position order between rounds.
- Excludes model loading, SGF replay, rendering and search. Includes coarse
  extraction (and geometry retention), encoder, proposal generation, exact
  shortlisted features, fine NN scoring and policy-output bookkeeping.
- `infer --profile-json` now also collects opt-in disjoint `proposal_stages`.
  No per-point clock reads. Profiled/unprofiled outputs match in unit and native
  ONNX tests. Generator work order was split into separate tagging/restriction/
  breathing passes for attribution; candidate choices/features are unchanged.
- Raw commands/results and rerunnable script:
  `scratch/20261114-140252-geometry-proposals/profiling/v1/{measure.py,results.json}`.

## Whole query, milliseconds per position (mean of 60 queries)

| Stage | Ply 12 | Ply 35 | Ply 60 |
|---|---:|---:|---:|
| Coarse features + retained geometry | 1.27 | 1.50 | 1.61 |
| Coarse NN encoder | 2.12 | 2.22 | 2.14 |
| Geometry proposal generation, including heuristic scoring/selection | 10.30 | 10.98 | 7.58 |
| Exact move features + preparation/packing | 0.34 | 1.31 | 2.31 |
| Fine NN scoring + result handling | 0.28 | 0.29 | 0.28 |
| Other: cell selection, allocations, normalization, bookkeeping | 0.56 | 0.74 | 0.57 |
| **Geometry-mode whole query** | **14.86** | **17.03** | **14.50** |
| **Complete-cell whole query** | **24.78** | **30.40** | **79.89** |

Complete-cell fine NN scoring alone costs 10.78–11.34 ms on CPU; exact extraction
costs 8.89/14.27/64.22 ms. This is not the campaign's GPU/multithreaded workload.
The three geometry run means were 14.79–14.93 / 16.45–17.98 / 14.28–14.73 ms.

## Inside proposal generation, milliseconds

These execution-stage costs are disjoint. Candidate source labels overlap; they
are not exclusive buckets to which full fine-feature/NN costs can be assigned.
Source construction includes its own insertion/deduplication. Restrict/Breathe
scores are evaluated over the shared preliminary pool, not just points bearing
those source labels.

| Stage | Ply 12 | Ply 35 | Ply 60 |
|---|---:|---:|---:|
| Shared setup, free regions, cell/group clipping | 0.207 | 0.376 | 0.312 |
| Extend point construction | 0.203 | 0.106 | 0.196 |
| Join point construction | 0.071 | 0.033 | 0.057 |
| Restrict/contact point construction | 0.102 | 0.124 | 0.089 |
| Capture region + representatives | 0.004 | 0.049 | 0.004 |
| Coverage + Breathe seed construction | 0.163 | 0.155 | 0.150 |
| Preliminary diversity/pruning | 1.413 | 1.228 | 1.156 |
| Joined-group tagging of preliminary points | 0.031 | 0.015 | 0.026 |
| Restrict: count blocked opponent liberties | 3.290 | 4.699 | 1.824 |
| Breathe: count new stone's local liberties | 4.105 | 3.514 | 3.159 |
| Final reservations/diversity/pruning | 0.669 | 0.638 | 0.568 |
| Finish + unassigned overhead | 0.043 | 0.046 | 0.040 |
| **Total proposal generation** | **10.302** | **10.982** | **7.580** |

| Pool size | Ply 12 | Ply 35 | Ply 60 |
|---|---:|---:|---:|
| Preliminary points receiving heuristic scores | 1,415 | 1,246 | 1,268 |
| Shortlisted points receiving full exact analysis | 544 | 517 | 512 |
| Legal proposals afterward | 544 | 508 | 432 |

Previously nonempty selected cells and global pass mass match the complete-cell
runs; that does **not** establish tactical coverage or strength equivalence.

## Interpretation

Actual point/region construction is cheap: roughly 0.54/0.47/0.50 ms combined for
Extend, Join, Restrict, Capture and coverage/Breathe seeds. The expensive part is
**exact heuristic scoring of the preliminary pool**: Restrict + Breathe consumes
7.40/8.21/4.98 ms, or 66–75% of generation. Shared diversity selection costs another
1.72–2.08 ms. Capture is not the bottleneck in these positions.

The next targeted experiment should reduce/cheapen Restrict/Breathe scoring and
then diversity selection, checking unchanged outputs for implementation-only
optimizations, or reporting coverage/trajectory changes if heuristics/pool size
change. Do not infer gains from simply removing those heuristics, and do not
change any campaign based on this diagnostic.
