# Benson connectivity optimization investigation — 2026-11-20

## Production adoption follow-up

With explicit user approval, the exact prototype was promoted to
`../continugo/backend/analysis/src/benson.rs`. Neural inputs, feature schemas,
Benson's public API and certification semantics are unchanged. Existing running
campaigns/binaries were not replaced or restarted.

The shared workspace's release tests and training crate's library tests pass
(100 training tests passed, eight opt-in tests ignored). Production additionally
matches the original baseline on all 3,184 replay snapshots, for both colours.
The randomized integer-point graph oracle and a many-separated-fragments stress
case now live in `benson/graph_tests.rs`. Validation logs are in
`scratch/20261120-benson/adoption/`.

For populated size-19 positions, **1–3 ms for both colours** is a provisional
planning estimate, not a measured real-game average. Sparse real size-19
snapshots measured 0.15–0.3 ms; dense synthetic size-19 snapshots measured
2.1–2.7 ms. The timings below remain the original contended investigation results.

## Original investigation conclusion

Benson need not cost the previously recorded 4–8 ms for both colours. An **exact,
scratch-only prototype** measured **0.15–1.40 ms for both colours** on the sampled
real-game positions, versus 1.00–6.46 ms for the current implementation on those
same nonempty snapshots. Dense synthetic positions took up to **2.65 ms**, versus
7.62 ms. There is no neural-feature/schema change, production adoption, or
self-play throughput/strength claim.

At investigation time the shared analysis crate was unchanged. Prototype, source patch,
standalone harness, copied regression fixtures, raw samples and source/machine
provenance are under `scratch/20261120-benson/`. `candidate.patch` contains only
the proposed shared Benson algorithm changes; the equivalent changes were later
promoted as described above.

**Timing caveat:** an existing cg10 self-play run was left running. Builds and
measurements used `nice -n 19`, one Cargo build job, and serial benchmark calls.
These are provisional alternating-order comparisons under contention, not
clean-machine latency guarantees. Another real-position repeat ranged up to
1.89 ms for the prototype; do not interpret the final 1.40 ms maximum as a bound.
No existing campaign, binary, model, cache, or configuration was changed.

## What changed in the prototype

Profiling points to connected-region extraction, not Benson's final pruning.
For example, on the size-9 ten-stone snapshot the original region computation
used about 2.2 ms per colour out of about 2.3 ms total. Its run graph repeatedly
compares nearby rows even after most runs are already connected.

The prototype contracts that graph before searching its remaining edges:

1. Connect same-row runs and adjacent-row runs using the **original exact stone
   adjacency**, not pixel connectivity or an approximate flood fill.
2. Stop searching for more connections if all runs already belong to one
   component. This is especially effective in openings.
3. Otherwise group runs into forest fragments. Reject fragment pairs whose
   bounding boxes are too far apart; test the remaining cross-fragment run pairs
   until an exact adjacency is found. Already-connected fragment pairs need no
   further tests.
4. Precompute integer disk widths instead of repeating integer square roots for
   every stone/row pair during disk-span construction.

This is exact graph contraction: initial edges are a subset of the original
edges; every remaining pair of distinct components is either separated by a safe
bounding-box lower bound or checked using the original integer-radius cutoff.
No opponent removal, vital-region definition, pruning rule, or output semantics
changes. No retained position cache is involved.

An intermediate row-skipping prototype reduced the middle size-9 example only
to about 2.4 ms; contracting and testing fragments reduced it to under 0.9 ms.
The fragment-pair loop is quadratic in the number of initial fragments, so a
production promotion should retain a many-fragment stress test and consider a
fallback if unusually fragmented positions expose a regression. This report
does not establish a worst-case runtime bound or a globally optimal algorithm.

## Matched measurements

All columns are **milliseconds per complete fresh analysis of both colours**,
including groups, disk spans, allocation, exact regions, vitality, pruning and
destruction. No feature-plane painting is included. Each result is the median
of seven samples of at least 40 ms; baseline/prototype execution order alternates.
The standalone release binary uses the system allocator, not production mimalloc.
Replay and file parsing are outside timers.

Selected real-game snapshots from `real-final.csv`:

| Source / size / ply | Stones | Current | Prototype | Speedup |
|---|---:|---:|---:|---:|
| Older game / 9 / 10 | 10 | 5.053 | 0.192 | 26.4x |
| Older game / 9 / 52 | 38 | 4.636 | 0.802 | 5.8x |
| Older game / 9 / 105 | 51 | 2.778 | 0.729 | 3.8x |
| Recent cg10 / 9 / 48 | 46 | 6.033 | 1.399 | 4.3x |
| Recent cg10 / 9 / 96 | 44 | 0.998 | 0.536 | 1.9x |

The final real-position suite samples six games: four size-9 games plus two
short size-13/19 games. Across its 18 nonempty snapshots the speedup is
1.9–27.0x. Size-13/19 real sources end after only 32 moves: they **do not represent
dense large-board endgames**. Empty positions immediately return in both versions
and are excluded from these ranges.

To exercise dense large boards without inventing a gameplay claim, six
explicitly synthetic SGFs construct scattered grids and two-colour grids with
two eyes per colour, for sizes 9/13/19. Placements are validated by `Position::play`,
but colours are not constrained to alternate. Selected results from
`synthetic.csv`:

| Synthetic position | Stones | Current | Prototype | Speedup |
|---|---:|---:|---:|---:|
| Scattered, size 13 | 125 | 6.761 | 1.951 | 3.5x |
| Scattered, size 19 | 260 | 7.616 | 2.651 | 2.9x |
| Two-colour eyes, size 13 | 165 | 3.987 | 1.243 | 3.2x |
| Two-colour eyes, size 19 | 357 | 4.303 | 2.096 | 2.1x |

## Correctness checks

- All nine existing Benson regressions pass for both baseline and prototype,
  including edge cutoffs, invaded eyes, two-centre eyes and cascading pruning.
- An additional 512 deterministic randomized interval graphs across nine board
  sizes (1 through 500) match an all-pairs integer-point adjacency oracle for
  both implementations. Including the original regressions: **20 tests pass**.
- Both implementations return exactly the same complete chain-index vectors
  for both colours at every replay snapshot across **34 input records / 3,184
  snapshots**: 28 real-game records and six synthetic records. This count
  includes empty states, passes and final snapshots, not 3,184 unique boards.

These checks support exactness but are not an exhaustive proof over every board.

## Artifacts and reproduction

From the repository root:

```sh
nice -n 19 cargo test --offline --release -j 1 \
  --manifest-path scratch/20261120-benson/Cargo.toml
nice -n 19 cargo build --offline --release -j 1 \
  --manifest-path scratch/20261120-benson/Cargo.toml
python scratch/20261120-benson/run_followup.py
```

`run_followup.py` records source hashes/revisions and machine/process information
in `environment.json`, runs the six real-game comparisons, and compares every
snapshot in 25 recent cg10 games. The three recent benchmark games are also in
that 25-game correctness set; the snapshot count above does not count that
repeat twice. Run only when the machine is idle for a cleaner follow-up.

Synthetic inputs can be regenerated with `GENERATE=scratch/20261120-benson/synthetic`
on the standalone executable. Passing those six SGF paths to it produces the
synthetic comparison. `CHECK_ONLY=1` skips timers; `PROFILE=1` prints cumulative
stage clocks from the separately instrumented copies. `make_profiles.py`
regenerates those copies. Logs retain all seven per-position sample times.

## Implication for a future input plane

The earlier "roughly doubles coarse extraction" concern is not a fixed cost:
there is a credible exact implementation that adds hundreds of microseconds to
around a millisecond on many sampled size-9 states, rather than 4–8 ms. Some
states and dense larger boards remain more expensive. Painting and integration
still need measurement, and useful playing-strength benefit remains untested.
Promoting the algorithm, doing an idle-machine comparison, and measuring the
actual new coarse stage should precede any decision to add neural inputs.
