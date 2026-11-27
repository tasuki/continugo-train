# Final score and ownership supervision

The network jointly trains policy, win/loss/draw outcome, final score mean and
uncertainty, and final ownership. Fresh schema-7 models use
**`shared-gaussian-linear-v2`**: linear score mean and positive, unbounded
softplus uncertainty. Bounded-score schema-7 checkpoints/exports are incompatible;
start fresh and regenerate packets/caches. Validated SGFs remain usable;
see [komi](komi.md).
No checkpoint migration or historical campaign rewrite is performed; do not
resume old loop control files with the new model/runtime.

## Score presentation compatibility

SGF `RE`, move annotations and viewers now express margins in board points:
raw territory fraction × `SZ²`, minus White's komi. Schema-7 predicted net scores
and uncertainty convert to board points by multiplying by `SZ²`. See
[komi normalization and compatibility](komi.md). Legacy schemas retain their
raw-territory heads; native inference subtracts compensation from their means. Old stone-area SGF `RE`
values fail authoritative replay validation; regenerate or migrate those records
before rebuilding caches. Existing historical annotations are not rewritten.

## Labels and resolution

Only naturally completed games with `RE` verified against shared-rule scoring
supply labels. Missing `RE` and truncated games have no outcome, score or ownership
label. No unfinished-board territory is used as an intermediate reward.

Schema-7 Black score is `(final_black_area - final_white_area) / 1_000_000
- komi / SZ²`, flipped for White-to-move. The physical board is a 1000-unit square;
komi is in board-point equivalents. Use the unrounded authoritative area, not
the formatted SGF margin. Raw territory is bounded to [-1,1] only to correct
floating-point polygon roundoff; **net scores and predictions are not clipped**.
Labels, terminal search utility and move annotations share board-area units.
Outcome retains the shared scorer's draw tolerance.

Ownership is **38×38 / 54×54 / 78×78** on 9/13/19 at default feature density.
Each output is the friendly-minus-enemy fraction of final area in a continuous
control volume about a coarse raster sample. Boundaries are midway between the
rounded sample coordinates; outer edges are 0 and 1000. Rust clips the final
shared-rule Voronoi polygons to these rectangles. Boundary cells can have soft
labels in [-1,1]; an empty final board has zero ownership everywhere. The
area-weighted ownership integral plus signed side-to-move komi/SZ² equals the
net score within floating-point precision. Zero-area control volumes at unusually high densities have zero label
and zero loss weight.

Final polygons are built lazily; a worker caches one final ownership raster for
its current game/density and reuses it across sampled plies. Metadata inspection
does not rasterize. `CGFEAT02` carries these derived labels; SGFs remain the game
records. `cells-v1` policy semantics and auxiliary labels are unchanged by the
schema-5 input update (antialiased occupancy, connected groups, spatial liberty
centers/footprints instead of scalar liberty area and coarse capture flags).
Schema-4 inference and schema-5 checkpoint/ONNX feature metadata both require
`targets: final-area-v1`; fresh schema 7 requires `final-area-komi-v2`.

## Heads and objective

- Shared spatial features: trunk → 1×1 projection to **16 channels** → leaky-ReLU.
  Concatenate spatial mean, mean-square and maximum; **48 → 32** dense units plus
  leaky-ReLU form the shared outcome/score representation. Configure these widths
  with `--value-channels` and `--value-hidden`.
- Outcome: shared representation → scalar/tanh.
- Score mean: shared representation → **linear scalar**, in net board-area units.
  Search accepts any finite score mean, including values outside [-1,1]; only
  the outcome head is range-checked. Invalid-output errors include actual values.
- Score standard deviation: shared representation → scalar → **softplus + 0.001**,
  with no upper cap, in the same units as the mean. It models conditional final-score
  variability, not epistemic uncertainty or a full multimodal score distribution.
- Ownership: shared spatial features → one channel/tanh. Predicts expected final
  signed ownership fractions. It is supervision only, never a legality mask,
  life/death rule, pass proof or handcrafted search reward.

Outcome and score signs are not forced to agree: being likely to win narrowly can
coexist with a negative expected margin because of occasional large losses.

```
score_NLL = mean(log(sigma) + 0.5*((final_margin - score_mean)/sigma)**2)
loss = policy_CE + outcome_MSE + 0.5*score_NLL + 0.5*ownership_MSE
```

`--score-loss-weight` and `--ownership-loss-weight` configure the last two factors
and are saved in training options. Both fresh-training defaults are **0.5**.
Per the user's instruction, cg9 keeps its inherited score coefficient **1.0** and
ownership coefficient **0.25**. For identical errors, cg9 gives score loss twice
and ownership loss half the influence of the new defaults. These ratios describe
the individual loss terms and their gradients, not the total loss/gradient.
The score coefficient now weights Gaussian NLL rather than MSE, so inherited
coefficients are not directly comparable. NLL omits its constant and may be negative.
Targets and predictions keep the same normalized scale. This is independent of
search's `score_weight=0.5` below; raw loss scales and shared-trunk gradient
interactions matter, so no coefficient is automatically better.

Ownership MSE is weighted by physical cell
area within each position and then averaged across positions. Report raw losses
separately; the logged total includes the configured factors. Saved metrics also
include score NLL/MSE/MAE/bias, mean predicted standard deviation, standardized
squared residuals, one-sigma coverage, score-versus-integrated-ownership discrepancy
and prediction spread. Training diagnostics use training games; validation is
inference-only, never optimizer input. Under a calibrated Gaussian, standardized
MSE is approximately 1 and one-sigma coverage approximately 68%; non-Gaussian
score distributions can depart from the latter. Best validation loss is not a
strength rating.

Production ONNX exports include score and ownership. Full policy and cheap
value-only requests share encoded contexts containing outcome, score mean/standard
deviation and the ownership raster. Only the root uses ownership to build the
[useless move penalty](inference.md#useless-move-penalty); there is no ownership
loss or geometry calculation inside search. Existing weights only need re-exporting.

## Search utility

```
scale = 2 / board_size
f(margin) = (2/pi) * atan(margin / scale)
score_utility = E[f(S)], S ~ Normal(score_mean, score_stdev**2)
utility = (1 - score_weight)*outcome + score_weight*score_utility
score_weight = 0.5  # default; 1 = curved score only, 0 = outcome only
```

Outcome is `P(win)-P(loss)`, with draws contributing zero, not raw win probability.
At the default, outcome and expected bounded score utility contribute equally. Displayed win
percentage is an affine transform of outcome, not a number directly averaged with
raw SGF points. Neural leaves, root initialization, completed rollouts and exact
terminal backups share this utility and side-to-move sign flips.

There is **no winning-pass shortcut**, in either search or rollouts. A terminal
pass has exact utility available to PUCT without fabricated visits; it competes
with placements. A first pass does not assume a cooperative reply. Higher utility
can justify continuing despite an available guaranteed win; passing can still be
best when continuation is worse. Truncated playouts remain unknown.

Weights must be finite and in [0,1]. The weight is still a linear blend, but the
score contribution is now nonlinear and uncertainty-aware. Retune old weights;
`0.5` is not equivalent to the old raw-margin `0.5` or KataGo's coefficients.
Search weights are separate from training loss weights. Changed native
fingerprints isolate protocols; do not resume old frozen campaigns.

The static curve is centered at zero, with diminishing returns for large margins.
Scale is **2 / board_size in normalized area**, equivalent to two square roots of
board area when measuring area in `board_size²` point-equivalents; it is unrelated
to feature-raster resolution. Numerical Gaussian integration uses paired Simpson
quadrature on [-8,8] standard deviations, 256 intervals, with adaptive refinement
for sigma > 1 to resolve sharp transitions under larger uncertainty. Exact terminal scores
have sigma zero and use the curve directly. Raw displayed score stays the expected
margin, not utility. A bounded score term can still trade win probability for margin.
No dynamic centering, score-distribution head, or sign-agreement penalty is implemented.

## Diagnostics before a large fresh run

```sh
# Stage-balanced rows from ancestral validation games only; never resplit training games:
.venv/bin/python -m continugo_train.diagnose --checkpoint NEW.pt --data REPLAY --output diagnostics.json
# Search controls: same SGF/ply/seed and simulations, change only score weight:
mise run infer -- --model NEW_MODEL --sgf GAME.sgf --ply 60 --simulations 32 --seed 7 --score-weight 0
mise run infer -- --model NEW_MODEL --sgf GAME.sgf --ply 60 --simulations 32 --seed 7 --score-weight 0.5
```

The first command reports outcome/score calibration, score MAE/RMSE/bias,
area-weighted ownership error, prediction spread and score/ownership agreement by
size and game stage. Source/checkpoint hashes and plies are retained in the
report. The search diagnostic reports visits, best action, pass/eye-fill visit
fractions and verified one-move capture exposure. Labels are applied **after**
search, never as guidance. The targeted old eye-fill cases complement, not replace,
held-out stage-balanced diagnostics.

Correctness tests and tiny fresh-model overfit/smoke runs do not establish playing
strength. A learning trial must still measure unassisted neural-opponent strength,
natural completion, game lengths, truncations, eye preservation, wall-clock cost
and memory. Hold search budget and visit sampling fixed while testing score
utility. No full campaign or accepted publication is started automatically.
