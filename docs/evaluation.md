# Candidate acceptance matches

The loop gates a trained/exported candidate against the **last accepted model**.
It does not use training or validation loss for promotion. This follows KataGo's
optional [gatekeeper](https://github.com/lightvector/KataGo/blob/master/cpp/command/gatekeeper.cpp)
pattern: head-to-head win points, with a configurable required proportion (default
50%, candidate wins ties). A finite match is a regression guard, not statistical
proof that the network improved.

## Budget and board mix

Loop controls:

- `--gate-games 20`: even number of games, paired by candidate color. Zero explicitly
  disables gating. Defaults give ten pairs, split **40% size 9 / 40% size 13 / 20%
  size 19**: 8/8/4 games. Ten games give 4/4/2.
- `--gate-simulations 32`: same MCTS simulation budget for each competitor per move.
- `--gate-max-moves 512`: a safety cap, **not a forced-scoring rule**.
- `--acceptance-rate 0.5`: required win-point proportion; configurable in 0.5..1.
- `--sizes` and `--size-weights 9:2,13:2,19:1` share the generation/training policy.
  Per-size counts/outcomes are always reported. No separate size-19 acceptance
  requirement overrides its intended 20% contribution.

Pairs share board size and RNG seed and swap candidate/incumbent colors. They start
from the empty authoritative board; randomized opening positions are not implemented.
The native `arena` binary selects maximum root visits with random tie breaks, rather
than training's temperature-one move sampling. Both use the existing PUCT,
progressive widening, independent exploration, exact legality and full-domain
policy normalization. Each network evaluates its own search; neither uses a
handcrafted value function or the random bot's eye filter.

Matching is currently sequential within one native process, with two frozen ONNX
evaluators, bounded feature pools and context caches. It does not yet batch across
independent evaluation games. Equal simulations are not necessarily equal wall time
when comparing different architectures or board sizes.

## Outcomes and decisions

- Natural win: **1** point; loss: **0**; natural draw: **½**.
- A draw is the shared scorer's actual `RE[0]`, not a small rounded score margin.
  It emits **WARNING DRAW**, with the SGF retained for inspection.
- A capped game has **no RE/outcome** and emits **WARNING TRUNCATED**. It is never
  converted to a draw, never force-scored and never used as a training label.

Let `P` be candidate points from completed games, `T` the number of truncated games,
and `N` the planned match count. The possible final match score is bounded by
`P/N .. (P+T)/N`:

- Accept only if the **lower** bound reaches the configured threshold.
- Reject if even the **upper** bound cannot reach it.
- Otherwise mark the test **inconclusive** and retain the incumbent.

Thus missing outcomes cannot manufacture acceptance or rejection. A test may still
be decided when it is mathematically guaranteed regardless of the unknown games,
similar to KataGo's mathematically-decided early-stop logic. This implementation
still plays the full planned match rather than stopping early.

The report includes wins/losses/draws/truncations, per-size breakdown, completed-only
score, planned-score bounds, threshold and decision. With no completed games, the
completed-only score is `None`, not 50%. Reports and warnings live in `loop.log`,
`loop.pt`, and native arena logs. All SGFs have standard `SZ`, `PB`/`PW`, and RE only
when naturally scored; model/search/seed metadata and actual visits are comments.

## Training is not rolled back

Accepted model and latest training checkpoint are separate pointers. Rejected or
inconclusive candidates keep their weights and optimizer progress for the next
training phase. Self-play remains on the accepted model until a later candidate
passes. Every candidate checkpoint/replay corpus is also backed up in the durable
[model store](model-store.md), not just the accepted ones. Gate SGFs are excluded
from the training replay selection.

## Manual matches

```sh
mise run arena -- --candidate PATH/TO/CANDIDATE --incumbent PATH/TO/ACCEPTED --games 20 --sizes 9,13,19 --size-weights 9:2,13:2,19:1 --simulations 32 --max-moves 512 --output runs/arena-001
```

Use a fresh output directory, or omit `--output` in this launcher for
`runs/YYYYMMDD-HHMMSS-arena` here automatically (UTC). Explicit paths are literal;
add `--device cpu` for CPU inference. Inputs can be
plain ONNX bundles or durable accepted bundles containing the two ONNX graphs.
`--game-offset` exists for the supervisor's missing-game recovery; even global
ordinals put the candidate on black. Each pair advances the seed once.
The standalone binary writes records and raw outcome totals; it does not move
models or update any accepted pointer by itself.

## Limits

Twenty games are cheap enough for initial wiring but noisy. The threshold, visit
budget, number of games, model quality and sample diversity need calibration before
scaling or trusting fine strength differences. There is no SPRT/confidence-bound
promotion test, resignation, opening suite or per-size minimum-strength constraint
in this first gate. The conservative bounds above account for **missing outcomes**,
not statistical uncertainty. Diagnostics/parity tests still cover all board sizes.
