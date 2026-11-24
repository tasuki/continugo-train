# Komi

New self-play defaults to mean **7**, Gaussian standard deviation **2**. Once per
actual game, sample `N(komi, komi_stddev²)`, round to the nearest half point, then
clamp to **[0,14]** (clipping, not rejection sampling). A separate seeded RNG stream
makes the sample independent of search, move selection and batching.

`--komi 7 --komi-stddev 0` disables variation. The training loop exposes the same
`komi` and `komi_stddev` configuration fields. Arena games default to fixed
**7**, with `--komi VALUE` selecting another fixed value for both games of a
colour-swapped pair; no Gaussian variation applies there. Champion challenges
remain fixed at 7. Explicit cross-lineage comparisons can use 0.
Existing rated results are not rescored.

Komi awards White board points, where the whole board is `SZ²` points, not stone
area units. Shared Rust `Game::with_komi` stores a finite `f64`. Half-point rounding and
[0,14] clipping apply only to randomized training sampling, not storage, SGF,
or fixed-komi settings. Immutable/prepared
children retain it. Equality includes komi, so inference cache
entries and duplicate batch openings cannot cross komi values. Low-level
`Game::new`/`Default` retain zero for compatibility; new-game generators explicitly
choose the new default.

## SGF and scoring

Generated SGFs include root `KM`. The shared codec validates a single finite
floating-point value without range restrictions and rejects per-move overrides. **Missing KM means 0**,
not 7: legacy records must retain their original outcomes. Replay checks `RE`
against the komi-adjusted shared scorer. SGF results and move annotations express
net margins in board points. Draw tolerance is unchanged. Setup and other rule
overrides remain unsupported by training replay.

With `A = SZ²` and `d = (black_area - white_area) / 1_000_000`, schema-7 final
Black score targets and terminal estimates are `d - komi/A`, flipped for
White-to-move. The score mean head is **linear**, with no clipping or nonlinear
squashing; uncertainty is `softplus(raw) + 0.001`, positive with no upper cap.
Multiply predicted mean/uncertainty by `A` for board-point presentation. Search
retains its two-`sqrt(A)`-point utility scale, using the actual board size.
Ownership targets remain pure territory; add the signed komi scalar to their
integral before comparing with net score.

Legacy schema-4/5/6 packets retain their raw territory score targets and no komi
scalar. Native inference subtracts side-to-move-relative compensation from legacy
score means for search; uncertainty already uses board-area units. Their outcome/policy heads cannot condition on komi;
they are not upgraded or retrained implicitly.

## Schema 7 scalar contract

Schema 7 remains open; its number is unchanged. There are still **17 coarse
planes and 9 exact fields**, plus two named globals:

1. `coarse_width_div100`: actual raster width / 100 (unclipped).
2. `komi_self_div_board_area`: `-komi/SZ²` for Black, `+komi/SZ²` for White.

The scalar vector is projected into trunk context after normalization, not
broadcast as another spatial plane. D4 augmentation leaves komi unchanged.
CGFEAT02 schema-7 packets add one little-endian f32 immediately after the existing
68-byte header: the signed komi scalar. Legacy packet layouts are unchanged.
The ONNX encoder takes a separate `komi: [batch]` input; width remains derived
from raster shape. Both graphs declare the global names in feature metadata.

The score contract is `shared-gaussian-linear-v2` / `final-area-komi-v2`.
Discard bounded-score schema-7 tensors, caches, checkpoints and exports. Start
fresh; old score/global metadata is rejected, and cache provenance includes the
extractor hash. Source SGFs remain usable if their `RE` matches current board-point
presentation and their actual `KM` (or legacy zero).
