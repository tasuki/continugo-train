# Useless move penalty

Implemented the agreed ownership-based useless move penalty in neural search, shared by
self-play, rated games and the human-facing move worker. Defaults to a maximum
0.5 board-point-equivalent cost, configurable with `--useless-move-penalty-points`;
zero disables it. No running/frozen campaigns or saved model bundles changed.

## Rule

- Ownership is side-to-move-relative, sampled at the placement's nearest raster cell.
- Cost ramps linearly from zero at absolute ownership 0.95 to the maximum at 1.
- Friendly placements are exempt if their four-connected >0.95 friendly region
  overlaps an enemy stone's physical footprint. Positive-area disc/pixel overlap,
  not just centre containment, is sufficient.
- Opponent-owned placements have no exemption, including captures. Pass is free.
- Fragmentation and small enemy-owned holes can miss cleanup; conversely an
  invader can exempt an entire large region. These are intentional heuristic
  limitations, not a proof of meaningful play or a certified life detector.

The region mask is built once per root. Expanded root candidates cache their
cost; PUCT subtracts it from Q only at the root. No costs enter backups, game
scores, value/score/ownership labels or SGF results. Policy targets still reflect
search visits. Point conversion uses a root-local finite difference of the
existing expected curved score utility over one board point, multiplied by
score weight. Thus a pure outcome objective gives no point penalty.

## Inference/export

The former ONNX encoder omitted ownership. New exports include its small 1x1
head in the existing encoder pass, preserving the two-file bundle and requiring
no extra inference call or trunk execution. This adds ownership work to encoded
leaf contexts too; only root mask/penalty construction is root-only. The bounded
host cache accounts for and shares ownership rasters. Existing exports without
ownership automatically skip the preference and retain their old behavior, even
with a nonzero configured maximum. Only new ownership-capable exports apply it.
No old bundles need changing; optional re-export of existing weights enables the
preference with no retraining/schema migration.

Ratings protocol v3 records the maximum cost. Versions 1/2 and old CSV headers
remain readable without inventing a historical penalty setting.

## Measurements

Scratch: `scratch/20261127-173504-useless-move-penalty-benchmark/`.

Release Rust microbenchmark, 2,000 masks/case, synthetic mixed-colour positions
and fully owned or fragmented ownership maps:

| Board | Raster | Stones | Solid friendly map | Fragmented map |
|---|---:|---:|---:|---:|
| 9 | 54² | 20 | 49.5 µs | 41.5 µs |
| 13 | 78² | 65 | 102.4 µs | 83.3 µs |
| 19 | 114² | 81 | 249.9 µs | 196.0 µs |

Reproduce: `cargo test --release --lib search::useless_move::tests::benchmark_root_masks -- --ignored --nocapture`.
These times exclude the two score-utility evaluations used for point conversion.
Candidate cost lookup happens once per expanded root edge, not each simulation.

Paired CPU ONNX encoder benchmark with actual cg12 initial weights, synthetic
inputs, one inference thread, three warmups and 15 alternating measurements.
The baseline removes the ownership output from the identical exported graph;
ONNX Runtime prunes its unused head. Medians:

| Batch | Raster | Without ownership | With ownership | Difference |
|---:|---:|---:|---:|---:|
| 1 | 54² | 34.47 ms | 34.39 ms | −0.24% |
| 1 | 114² | 152.17 ms | 152.51 ms | +0.22% |
| 8 | 54² | 284.28 ms | 284.06 ms | −0.08% |
| 8 | 114² | 1280.20 ms | 1281.43 ms | +0.10% |

The differences are tiny and partly noise, not evidence of a speedup. New export
maximum eager/ONNX parity error was 2.48e-5 across all checked outputs. GPU
ownership-head overhead and whole-selfplay throughput were not measured. No
playing-strength or shorter-game claim is established by these timing tests.

## Checks

Added tests for the linear ramp, pass exemption, cleanup regions, physical-disc
overlap, malformed ownership, root-only/cached costs, untouched backups, actual
root selection preference, settings forwarding, and old/new rating protocols.
Native integration checks exercise ownership parity across batch/cache paths and
penalty-enabled multi-game arena shutdown. Export/native parity, neural self-play,
legacy feature exports and a real CPU loop/restart also passed.
