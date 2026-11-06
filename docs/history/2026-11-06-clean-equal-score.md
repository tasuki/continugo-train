# Clean cg4 campaign and objective audit — 2026-11-06

User requested stopping the old campaign, clearing generated artifacts, and
starting genuinely fresh with sensible, visible settings. After discussing
score-only search, the user explicitly selected **score_weight=0.5**, meaning
**equal outcome and normalized-final-score contributions** in a corrected linear
blend. This is not a score-only campaign.

## Cleanup and retained history

Safely interrupted launcher 1097099 and verified its loop/native children exited.
Moved repository-root `runs/`, `models/`, `models_20261105/` and
`models_20261106/` to **recoverable system trash** using `gio trash`.
No permanent purge was performed. Source, Git history and all 20 pre-existing
`docs/history/*.md` documents remain. Their historical generated-artifact paths
are no longer live in this checkout; original artifacts are recoverable from
trash if needed. Do not resume old controls against a new store.

## Audit findings and fixes

1. Neural and unguided MCTS had a hard-coded terminal-winning-pass shortcut.
   Removed it. A terminal pass now has exact utility available to PUCT without
   inventing visits, and competes normally with placements. First passes never
   assume an opponent reply. A guaranteed small win does not exclude a
   higher-utility continuation; losing passes also remain eligible.
2. Rollouts independently forced immediate winning passes. Removed that too.
3. Completed rollouts backed up only outcome and rejected nonzero score weight.
   They now share the terminal/neural outcome-score utility, while capped
   continuations remain unknown and do not acquire fabricated draws or scores.
4. Old score weighting was `(outcome + w*score)/(1+w)`; even w=1 meant a 50/50
   blend. It is now **`(1-w)*outcome + w*score`**, default **w=0.5** everywhere
   (self-play, inference, arena and loop). w=1 is score-only; w=0 outcome-only.
   Native binary fingerprints change, so old frozen runtimes are not reused.
5. Bootstrap eye guidance can bias legal placement proposals for this geometric
   score objective. It is **disabled in this fresh campaign**. The versioned
   random-no-eye-fill reference is deliberately unchanged, not silently redefined.

All estimates remain side-to-move relative. Exact final-score normalization still
bounds floating-point polygon roundoff; neural head validation stays strict.
Final labels require authoritative natural completion. Self-play deliberately
samples visits for exploration; evaluation chooses maximum visits. W/D/L Glicko
ratings are diagnostics, not score-margin ratings or an actor-promotion rule.

Regression tests cover higher expected score beating an available smaller
terminal win, exact pass utility with zero invented visits, pure-score/half-blend/
outcome-only semantics, rollout margin backups, no forced rollout pass, signs,
visit conservation and truncation handling. Full Rust tests, formatting, Clippy
and **201 Python tests**/Ruff passed before launch.

## Fresh launch

`runs/20261106-142353-scratch-score/`, launched **14:38:46 UTC**,
PID **1147801** at startup; verify live `run.pid` identity before signaling.
Family **cg4**. No imported weights, BatchNorm, optimizer, SGFs, split ledger,
ratings or model store. Source code/patch snapshot, private read-only native
binaries, passing validation logs and exact `plan.json` / `plan.pt` are retained
inside the new run. Its README is the operating card.

- Size 9, default 32-channel/four-block network with outcome/score/ownership heads.
- 500 new unguided MCTS bootstrap games, four CPU shards, cap 1024,
  seeds 9,100,000–9,100,499. Five initial CUDA epochs, seed 910,001.
- Then 60 rounds: 50 fresh neural self-play games + four training epochs each;
  neural seeds start at 10,100,000. Latest actor, no automatic acceptance.
- Bootstrap/neural self-play: **32/256 simulations at approximately 75%/25%**;
  policy only on 256-simulation positions; other heads use both. Utility is
  **0.5*outcome + 0.5*normalized_score**. No forced winning pass or eye guidance.
- Training: SGD 0.01 / momentum 0.9 / decay 1e-4 / clip 5; batch two, six feature
  workers, four Torch threads; up to 1000 batches/epoch, eight train/four held-out
  positions per game. Outcome MSE weight 1, **score MSE weight 1** (raised from
  the old 0.25 auxiliary weight), ownership weight 0.25. These training weights
  are separate from the user's 0.5 search blend.
- Replay: 512 games / 16 rounds plus only this run's own fresh bootstrap fallback.
  Neural concurrency/batch limit 12, wait 2 ms, six feature threads, one ONNX
  thread, 64 cached contexts, CUDA FP32.
- Ratings: **fixed 32 visits**, 20 initial reference games; scheduled every ten
  rounds with 20 parent/20 reference games and unchanged adaptive precision
  limits. Reference cap 2048, other matches 1024. Evaluation never enters replay.
  Fresh `models/ratings/networks.csv` always leads with random-no-eye-fill-v1,
  fixed at 100 / RD 0, even before completed match evidence exists.

## Initial verification

All **500/500** bootstrap games finished naturally: one draw, zero truncations,
17,405 eligible rows, including 4,378 expensive policy-training positions.
Audited SGFs have the linear score fraction 0.5, no bootstrap eye guidance, and
actual 32/256 visit totals consistent with their per-move masks.
Verified the first CUDA checkpoint: initialization `fresh`, family `cg4`,
2,000 optimized rows / 17,405 generated rows, score-MSE training weight 1.
The new launcher is active. These are startup checks, not strength evidence.

Root `metrics.png` initially links to the live initial-training plot; later loop
updates replace it with the combined loss plot. `draws.png` appears after the
first fully recorded neural batch; `models/ratings.png` is a separate strength
plot. No data-less placeholders are produced. Use the new run's `run.py` for
resumption, and do not change Python code while it is active.
