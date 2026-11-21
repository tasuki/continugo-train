# cg11 fresh start — lineage 20261122

Launched `runs/20261121-121524-cg11/` through the shared campaign launcher,
using its small `cg11.json` recipe. Supervisor PID at launch: 3478271.
This is a launch receipt, not evidence of playing strength.

Fresh MCTS bootstrap uses completed random rollouts. Initial training and neural
self-play use the frozen current defaults, without tuning overrides. Operational
target: 60 rounds; independent evaluation every fifth round, initially against
random. Durable store: `artifacts/20261122/`. No automatic publication.
No ancestral weights, optimizer, games, holdouts, caches or ratings are imported.

A first attempt was stopped during bootstrap, before training, after finding that
the shared launcher incorrectly assumed an evaluator already existed. That
attempt is retained under `scratch/20261121-121524-cg11-aborted/`, with no data
reused. The custom per-run wrapper was retired in favor of shared fresh-start
support and an initialization/resume regression test.

Release build and Rust tests passed. A real fresh-data/default-CUDA-training smoke
test passed under `scratch/20261121-cg11-fresh-smoke/`. The final focused Python
suite passed 27 tests; lint passed. The earlier full Python suite passed 353 of
355 tests, with existing legacy artifact-path and ratings-plot assertion failures
recorded explicitly in `preflight.json`. Do not describe the full suite as green.

Partial bootstrap or uncommitted initial training stops for inspection rather
than silently adopting or overwriting data. If abandoning cg11, stop and verify
all children have exited before archiving/removing the run and its isolated store
together. A clean restart must not import abandoned-attempt state.
