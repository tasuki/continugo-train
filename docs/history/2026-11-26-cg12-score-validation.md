# cg12 round-33 score validation failure — 2026-11-26

## Failure and reproduction

cg12 completed round 32, then self-play round 33 failed with the generic
`invalid neural evaluation` error. The first attempt retained 125/150 games.
An unchanged resume adopted those games and retried the first contiguous gap,
seeds 12104913–12104914. Seed 12104914 completed; seed 12104913 failed again.

A diagnostic native build reproduced the failure on CUDA with the round-32
model (model ID `d4a8daec67ecd3bb1ddb848071d8c87afa2298fef030aebe48b0a134b769d41c`):

- logits length: 1,000,001; legal-mask length: 1,000,000 (both correct)
- outcome: -0.9999969601631165
- score mean: **-1.0068011283874512**
- pass logit: -0.5395242

All outputs were finite. `SampledPolicy::new` still enforced the obsolete
bounded-score-head range [-1,1]. Current linear score means are explicitly
unbounded; net-komi scores also need not lie in [-1,1]. Encoder validation and
`LeafEstimate::utility` already accepted finite unbounded means. This was a
stale policy-search check, not evidence of NaNs or training divergence.

## Fix and validation

Remove only the score-mean range restriction in `src/search.rs`. Preserve finite
score validation, bounded outcome validation, tensor lengths and pass-logit
checks. Include output values/dimensions in future validation errors.

A regression test fails before the fix and passes after it, covering the actual
failing score, positive/negative unbounded means, nonfinite means and invalid
outcomes. Full `cargo test --locked --release` and formatting pass. Strict Clippy
is blocked by existing warnings in replay, arena and feature tests (not changed
for this fix).

The corrected CUDA binary completes both original retry games:

- seed 12104913: 99 moves, naturally completed
- seed 12104914: 72 moves, naturally completed (same length as before)

Both feature sidecars were built. Logs, diagnostic/fixed binaries and reproduction
commands are under `scratch/20261126-cg12-invalid-evaluation/`.

The frozen campaign runtime, control files, historical games and weights were
not modified during debugging. Scratch reproduction games are diagnostic-only
and have not been adopted into campaign data or ratings.

## User-approved runtime update and resume

At 12:58 UTC the user explicitly requested deployment and resumption. Old
self-play/arena binaries, manifests, loop control, evaluator control and champion
registry were backed up under
`runs/20261124-164042-cg12/runtime-upgrades/20261126-125849-linear-score-validation/`.
The migration receipt includes old/new hashes and the source patch. Self-play
and arena received the fix; feature/rating binaries and frozen Python are unchanged.
Campaign manifest/launch and loop executable fingerprints were updated together.
The evaluator's audited `upgrade_runtime` operation updated its fingerprints and
protocol hashes, preserving champion/history and all evaluation settings.
Weights, optimizer, game records, configuration and iteration progress were not
changed. Resume launched with the original recipe and total target of 60 rounds
(supervisor PID 2817124 at startup).

## Cache recovery after interrupted generation

The fixed runtime completed all 150 round-33 self-play games, then training
stopped at 13:14:39 UTC because adopted games from the failed generation attempts
had no feature sidecars: native cache creation runs only after successful game
generation. Of the 150 SGFs, only 24 had caches. No new neural-validation failure
occurred.

On the user's request, the explicit frozen-runtime cache builder scanned all
round-33 self-play attempts using the unchanged frozen `bin/features`, density 4
and four workers. It built **126** missing caches and verified **24** existing
caches in 103.7 seconds, with no skipped games or SGF modifications. Build log:
`scratch/20261126-cg12-invalid-evaluation/rebuild-round33-caches.log`.
The campaign was resumed unchanged with the original recipe and 60-round target
(supervisor PID 2847380 at startup).
