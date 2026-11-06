# Terminal-score roundoff fix and cg3 continuation — 2026-11-06

## Failure and reproduction

The fresh cg3 campaign completed its five initial epochs (10,000 optimized rows),
then stopped during initial reference game 17, size 9, seed 8,100,008, candidate
Black. Sixteen completed reference SGFs remained uncommitted to the rating ledger.
The generic error was `invalid neural outcome/score estimate`.

Adding actual values to the error reproduced the fault exactly on CUDA with the
saved network: **value=1, score=1.0000000000000002**. Encoder validation was not
at fault: the terminal move annotation computed a final normalized score from
floating-point Voronoi polygons, whose sum exceeded the mathematical board area
by a few ulps. The strict neural-style bounds check then rejected that exact
terminal annotation.

## Fix

`src/targets.rs::normalized_score` clamps authoritative normalized final scores
to their mathematical [-1,1] bounds. Auxiliary score labels, terminal search
utility and terminal display annotations share it. Shared scoring/rules,
authoritative SGF RE, learned head outputs and neural validation are unchanged.
`LeafEstimate::utility` now includes invalid values in its error diagnostics.

Regression tests cover real polygon-accumulation overshoot in terminal games,
both signs at the normalization boundary, ordinary/draw scores, and continued
strict rejection of nonfinite or even one-ulp-out-of-range **neural** estimates.
Full Rust tests/Clippy/format and **201 Python tests**/Ruff pass.

The exact failing CUDA game now completes naturally: 124 moves, W+105.2.
Before/after logs and SGF are retained under
`runs/20261106-130621-fresh-policy256/error-fix/`. This is a correctness check,
not evidence of improved strength.

## Operational continuation

Launched **13:39:12 UTC**, PID **1097099** at startup:
`runs/20261106-133757-policy256-score-bound/`.

The new binary fingerprints require a new loop control, not replacing binaries
under a frozen old loop. The continuation starts from the exact saved
`initial-training/epoch-0005.pt` (SHA-256
`99c60651cf359a964c5e10d249954d40443768bd61c3d0b3b4ba8188f90be264`), preserving
all model/SGD/BatchNorm state, family `cg3`, 10,000 optimized rows, 20,823 generated
rows and inherited whole-game holdouts. It imports only the same 500 fresh
bootstrap games. No new scratch initialization or re-bootstrap occurred.

Size-9 scope, 60 rounds, 50 self-play games/four training epochs per round,
randomized 32/256 self-play, policy-only-at-256 masking and fixed **32-visit rating
games** remain unchanged. The same shared `models/` store continues. Old frozen
controls/binaries/logs and 16 partial reference games are retained unchanged;
those games were never rated and are diagnostic-only. The new initial reference
batch reruns under the corrected protocol, so no old uncommitted games are counted
twice. No network is automatically accepted.

The continuation's README/plan JSON, source patch/snapshot, private binaries and
validation logs document its exact runtime. Use its `run.py` to resume, never the
old launcher. Verify live process identity before signaling; do not remove live
locks or modify Python code during an active campaign.
