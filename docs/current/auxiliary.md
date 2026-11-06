# Final score and ownership supervision

The network jointly trains policy, win/loss/draw outcome, final score and final
ownership. This implementation requires **fresh training**: old policy/value-only
checkpoints, exports, feature packets and loop control files are incompatible.
No migration or historical campaign rewrite is performed. Source SGFs remain
usable after ordinary authoritative validation.

## Labels and resolution

Only naturally completed games with `RE` verified against shared-rule scoring
supply labels. Missing `RE` and truncated games have no outcome, score or ownership
label. No unfinished-board territory is used as an intermediate reward.

Score is `(final_black_area - final_white_area) / 1_000_000`, flipped for a
White-to-move training position. The physical board is always a 1000-unit square;
normalizing by `SZ²` would be incorrect. Use the unrounded authoritative area,
not the formatted SGF margin. Outcome retains the shared scorer's draw tolerance.

Ownership is **38×38 / 54×54 / 78×78** on 9/13/19 at default feature density.
Each output is the friendly-minus-enemy fraction of final area in a continuous
control volume about a coarse raster sample. Boundaries are midway between the
rounded sample coordinates; outer edges are 0 and 1000. Rust clips the final
shared-rule Voronoi polygons to these rectangles. Boundary cells can have soft
labels in [-1,1]; an empty final board has zero ownership everywhere. The
area-weighted ownership integral equals the normalized score within floating-point
precision. Zero-area control volumes at unusually high densities have zero label
and zero loss weight.

Final polygons are built lazily; a worker caches one final ownership raster for
its current game/density and reuses it across sampled plies. Metadata inspection
does not rasterize. `CGFEAT02` carries these derived labels; SGFs remain the game
records. Input schema 4 and `cells-v1` policy semantics are unchanged, while
checkpoint/ONNX feature metadata additionally requires `targets: final-area-v1`.

## Heads and objective

- Outcome: globally pooled trunk → existing hidden layer → tanh. Fresh models
  default to **leaky-ReLU, slope 0.1**, avoiding the old inactive-ReLU failure.
- Score: pooled trunk → 32 hidden units, leaky-ReLU → scalar/tanh. Predicts
  expected normalized final margin in [-1,1].
- Ownership: spatial trunk → 16 channels, leaky-ReLU → one channel/tanh. Predicts
  expected final signed ownership fractions. It is supervision only, never a
  legality mask, life/death rule, pass proof or handcrafted search reward.

```
loss = policy_CE + outcome_MSE + 0.25*score_MSE + 0.25*ownership_MSE
```

`--score-loss-weight` and `--ownership-loss-weight` configure the last two factors
and are saved in training options. Ownership MSE is weighted by physical cell
area within each position and then averaged across positions. Report raw losses
separately; the logged total includes the configured factors. Saved metrics also
include score MAE/bias, score-versus-integrated-ownership discrepancy and prediction
spread. Best validation loss is not a strength rating.

Production ONNX exports include the score scalar but **omit ownership**. Full
policy and cheap value-only native requests both return outcome and score;
context caching and cross-game batching retain both. Eager training diagnostics
can inspect ownership without adding a dense output to every MCTS leaf.

## Search utility

```
utility = (outcome + score_weight*normalized_score) / (1 + score_weight)
score_weight = 0.3  # default; 0 is the outcome-only diagnostic control
```

Outcome is `P(win)-P(loss)`, with draws contributing zero, not raw win probability.
The coefficients are approximately 76.9% outcome / 23.1% score. Leaf estimates,
root initialization and exact terminal backups use the same utility and normal
side-to-move sign flips. Pass proofs remain exact **outcome** facts, independent
of the magnitude of score-adjusted utility. Nonzero score weight with the optional
rollout evaluator is rejected; explicitly use `--score-weight 0` for rollouts.
Weights must be finite and in [0,1]. Search settings, including score weight, are
frozen in new v6 loops and recorded in SGFs and v2 evaluation protocols.
Use a fresh `--model-store` for the new campaign if the existing store contains
old-format ratings; no rating-ledger migration is provided.

A bounded score term can still trade winning chances for margin. KataGo's
standard factors (1.0 outcome, 0.1 static score, 0.3 dynamic score) use different,
nonlinear score transforms and are not directly equivalent to this simpler mix.
No dynamic centering or score-distribution head is implemented.

## Diagnostics before a large fresh run

```sh
# Stage-balanced rows from ancestral validation games only; never resplit training games:
.venv/bin/python -m continugo_train.diagnose --checkpoint NEW.pt --data REPLAY --output diagnostics.json
# Search controls: same SGF/ply/seed and simulations, change only score weight:
mise run infer -- --model NEW_MODEL --sgf GAME.sgf --ply 60 --simulations 32 --seed 7 --score-weight 0
mise run infer -- --model NEW_MODEL --sgf GAME.sgf --ply 60 --simulations 32 --seed 7 --score-weight 0.3
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
