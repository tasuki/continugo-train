# cg6 exhaustive-position replay — 2026-11-11

User approved launching the previously prepared cg6 proposal and requested timing
of features, NN inference/training and independent ratings, with overhead checks.

Campaign: `runs/20261111-145926-cg6-full-window/`. Launcher PID at start **1360542**.
The first full epoch is the review boundary: no automatic new self-play or rating
jobs after it. The later continuation used `continue.py`, superseding
`run.py --continue-loop`.

Warm-start cg5 round 100, SHA-256
`feedb3adfc610af5fcea47f28205281199a0d03f9bf5462d169bc95b134fdffb`, keeping
weights, BatchNorm, SGD lr=0.01/momentum and cumulative counters. Rename family to
cg6; inherit all 4,738 ancestral game assignments. Newest 1,000 unique eligible
neural self-play games, rounds 80–100: **799 train / 201 validation games**,
**79,560 / 18,154 positions**. Initial production trainer independently verified
these exact counts, no duplicates/unlabeled games skipped. A naturally completed
draw is retained. Bootstrap MCTS, matches and diagnostics are not imported.

All three position/batch caps are zero. One exhaustive epoch, batch two, six
workers, four Torch threads, CUDA FP32, unchanged architecture, loss/search
semantics and size-9 reference shuffling. No model is automatically accepted.

Private Python/runtime and independent read-only native binaries are frozen by
`launch.json`. Project source/old campaigns/shared ratings policy are untouched.
The private loop subclass restores descending acquisition-round order of imported
historical games before the existing newest-first replay selector. The prepared
future loop uses 1,000 games / 32 rounds, 50 new games/round, one full epoch, and
32/256 search at 75/25. Models are saved every fifth round; ratings stay separate.

## Timing and overhead

`timings.jsonl` records subprocess wall time and phase metrics. Production adds
worker service clocks and detailed server/IPC/decode data on **one request in
128 per worker**, with no additional GPU synchronization. Exposed loader wait and
compute+transfer/loss/backward/optimizer timing already exist in the base trainer.
Native self-play already logs feature, encoder and scorer counters; no fine
per-group native profiling is enabled. Worker/service sums overlap and must not
be added together as wall shares.

Before training, warmed matched ABBA trials on disposable model/SGD copies used
identical data and initial states, 768 optimized positions per trial. Baseline
phase durations **14.874 / 15.053 seconds**, instrumented **15.153 / 14.926**:
measured median overhead **+0.50%**, within the run-to-run variation. Two
observations per arm are a pilot, not a statistical bound. Six detailed requests
were actually sampled in each instrumented trial. A >5% median-overhead guard
would disable optional telemetry; it remained enabled. This comparison measures
added cg6 telemetry, not timers already present in the baseline implementation.

A separate fully instrumented 14.926-second diagnostic trial recorded transfer
0.164s, encoder 1.005s, scorer 0.765s and loss/backward/optimizer 4.001s. These are
representative synchronized samples, **not** campaign-wide exclusive fractions.
Benchmark weights were discarded. Full raw results: `measurement-overhead.json`.

`timing_report.py` summarizes available completed phases; `measure_ratings.py`
can time an explicitly invoked standalone evaluator without changing its queue
or policy. No rating jobs have been launched by cg6. Four campaign-specific and
22 upstream loop/progress tests passed. An initial unittest invocation lacked
`PYTHONPATH=tests`; the corrected invocation passed all 22 tests.

The copied package roots are explicitly corrected to use the project's shared
artifact store. Its startup imports the trainer for root configuration, so
`python -m continugo_train.train` emits a runpy pre-import warning; the executed
main module receives its own telemetry wrapper, and the real trainer reports the
correct source, caps and data/split counts. No source changes were made during
active training.

## Continuation and stop

Later approved 10 rounds: 100 fresh games, one exhaustive epoch over the latest
1,000 games, saving and challenging the shared champion every round via
`continue.py`. Iteration 1 completed; its 4/10 challenge left cg5-r0080 champion.
User stopped the campaign on 2026-11-11: iteration 2's 18 SGF files and loop-state
entry were deleted before training; target reset to 1. Iteration 1 and ratings
are unchanged. Nothing is queued; further continuation needs approval.
