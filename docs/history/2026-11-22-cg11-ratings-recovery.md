# cg11 round-20 ratings recovery

The campaign stopped at 2026-11-22 07:39:29 CET (06:39:29 UTC), after
round 20 training, export and model saving succeeded. The shared evaluator
failed fitting the 280 SGFs with `joint ratings did not converge`.

Reproduced the failure from the authoritative SGFs. Near the optimum, the
objective's floating-point resolution defeated Armijo backtracking: a full
Newton correction of roughly 0.0000002 rating points was repeatedly reduced to
a step that did not change the parameters. The absolute gradient threshold
was never reached. This was not a model/training failure.

The original `python/continugo_train/joint_ratings.py` now stops when the full
Newton correction is below **one rating point**, as requested. It applies
that last correction before calculating uncertainty. It does not use a tiny
damped step/objective change as proof of convergence, change the statistical
model, or suppress genuine solver/data errors. Regression tests cover the
exact cg11 pair totals, an unbeaten chain, symmetry/order invariance, and
one-point fits against a 0.01-point reference across 25 randomized graphs.
Ten joint-rating tests, nine campaign tests, four lineage-store tests, lint and
format checks pass. Another 500 deterministic randomized graphs fitted
successfully. The broader 60-test ratings suite passed 59 tests; its existing
legacy `test_two_csvs_all_networks_fixed_reference_and_durable_evidence`
plot-file expectation still fails. Do not describe that suite as green.

Applied only this numerical source file to cg11's frozen runtime, explicitly
updating its hash in `runtime.json` and `launch.json`, with an audit receipt.
Original files/checkpoints/registry/CSV/chart are backed up under
`scratch/20261122-cg11-ratings-repair/backup/`; the reproducible repair script,
receipt and test logs are beside that directory. No weights, momentum,
training data, search settings, binaries, evaluation policy or champion
history were reset or changed by the repair.

All 20 saved checkpoints and ONNX exports matched the training-state hashes.
Cold replay validated every one of the 280 evaluation SGFs, rebuilt
`artifacts/20261122/networks.csv` and `ratings.png`, and agreed exactly with
cached replay. Off-cadence models correctly have no fabricated ratings.
The existing unfinished SGF remains counted as unfinished, not as a draw.

Resumed the existing recipe at 09:20 CET with total target 60, supervisor
PID 3988802. At 09:23 it was actively generating the missing round-20 neural
reports as part of resuming the interrupted evaluation. Further training
follows that synchronous evaluation; round 21 was not yet started at this
observation. This is a recovery receipt, not a claim that the target finished.
