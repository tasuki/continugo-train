# Restartable experimental neural self-play loop

`mise run loop` runs native self-play → bounded SGF replay → warm-start training →
ONNX export, with run-owned recovery. **Training does not run rating matches.**
It saves each tenth model for separate evaluation under `artifacts/ratings/`.
It never automatically promotes a network; accepted publication follows the
[approval procedure](model-store.md#approval-operation).

## Three independent roles (v6)

New loops require score/ownership checkpoints and record `--score-weight` (default
0.5, the linear score fraction). See [auxiliary supervision](auxiliary.md).

| Role | Behaviour | Location |
|---|---|---|
| Training | Latest completed candidate; weights/optimizer progress survive interruptions | Campaign `loop.pt` and checkpoint |
| Experimental actor | Latest exported model generating the next self-play batch | Campaign `loop.pt` and export |
| Accepted | User-approved publication, or historical accepted model; may be absent | Store `LATEST-ACCEPTED` |

Initialization snapshots an unaccepted actor inside the campaign. After each
candidate is trained, exported and verified, it becomes the next actor. There is
no actor gate or evaluation verdict. Approval changes neither training nor actors.
The accepted head is only read at startup and iteration boundaries.

## Start, resume, inspect

Requires a full training checkpoint and the CPU/CUDA setup in [inference](inference.md).
The supervisor requires POSIX process groups and `flock`.

```sh
mise run loop -- --checkpoint PATH/TO/checkpoint.pt --bootstrap-data PATH/TO/replay --iterations 2 --games-per-iteration 10 --train-epochs 1 --steps-per-epoch 20
# Use the dated path printed by the first command:
mise run loop -- --output runs/YYYYMMDD-HHMMSS-loop --resume
# Increase the TOTAL iteration target, not an additional count:
mise run loop -- --output runs/YYYYMMDD-HHMMSS-loop --resume --iterations 3
mise run loop-status -- --output runs/YYYYMMDD-HHMMSS-loop
```

New loops reserve `runs/YYYYMMDD-HHMMSS-loop` (UTC, collision-safe). Explicit
`--output` paths are literal; resume must name the existing output. Status is
read-only and reports roles, cumulative progress, metrics, phase, disk and live logs.

Use `mise run loop -- --help` for defaults. New loops default to twelve concurrent
games / batch limit twelve / collection wait two ms; active slots are capped by
`--games-per-iteration` (default ten). See [inference batching](inference.md#cross-game-batching).
`--device cpu` selects CPU. Play, training and export are separate supervised
processes; export verification uses CPU.

Resume restores frozen training/search configuration. Conflicting overrides fail;
the total iteration target may increase, never decrease. `--stop-after-iteration N`
pauses after that round without lowering the target. Resume without it to continue.
Self-play/feature executable hashes are checked on resume and around child stages.
Keep the same absolute path and compatible Python/rules/runtime versions. Do not
replace active binaries; changed generation executables require a new loop/fork.

**v1–v5 control files and old policy/value-only models are not resumed.** Preserve
historical campaigns with their frozen runtime; historical accepted bundles stay intact.

Existing v6 loops with embedded evaluation are detached on resume: obsolete rating
settings/jobs are removed from control state, with the original checkpoint preserved
as `before-standalone-ratings.pt`. An interrupted post-training evaluation does not
need to finish: its existing checkpoint/export is reused and the actor advances.
This does not delete old directories or reset the shared ratings ledger. The removed
`--gate-*`, `--reference-*`, `--adaptive-*`, `--evaluation-interval`, `--actor-policy`
and evaluation-enabling flags are no longer loop options.

## Milestone saving, separate ratings

`--save-model-every 10` saves independent checkpoint and ONNX copies in
`artifacts/models/<family>/rNNNN/`, alongside small portable `metadata.json`. Original campaign files stay
intact. No replay copy, hardlink, approval or rating match is created. Saving is
idempotent; resume repairs a missed save after a state-commit interruption without
retraining. `--save-model-every 0` disables it.

Training creates no `gate/`, `reference/`, `initial-reference/` or
`adaptive-ratings/` jobs and does not register models or update rating plots.
Evaluate saved models separately:

```sh
mise run rate-saved -- --family cg5
mise run rate-saved -- --resume
```

The standalone evaluator owns its protocol, 10+30-game champion challenges and
post-promotion ten-game random calibration (not for rejected challengers).
Seven wins in ten permit continuation; 28 wins
in forty **automatically replace the evaluation champion**, initially the random
bot. Rated neural moves use 128 simulations by default. Publication is still a
separate user decision. The shared evaluator state/log and champion lineage live under `artifacts/ratings/`;
SGFs live under `artifacts/sgfs/YYYY-MM-DD/`, with no duplicate game ledger. See
[evaluation](evaluation.md). Ratings never affect training or actor selection.

## Randomized policy-search budget

New loops default to **32 simulations on 75% of moves and 256 on 25%**:
`--simulations 32 --policy-simulations 256 --policy-search-probability 0.25`.
The cap is sampled before search from a separate per-game seeded RNG, independent
of position evaluation, search RNG consumption and worker scheduling. Expected
work is 88 simulations/move, not a wall-time guarantee.

Only expensive positions contribute policy cross-entropy. Cheap positions retain
outcome, final-score and ownership supervision, subject to sampling caps and natural
completion. SGFs retain real visits and a per-move `train_policy` mask; adoption
validates their correspondence. Logs report `samples` and `policy_samples` separately.

`--policy-search-probability 0` restores fixed-budget, all-policy self-play; `1`
searches/trains policy at the expensive budget every move. When enabled, cheap
simulations must be at least 32 and expensive simulations at least the cheap budget.
Existing loops missing these settings resume with randomization disabled. Budgets
remain frozen. Legacy/bootstrap games retain their original policy supervision.
See [randomized search](inference.md#randomized-policy-search-caps).

## Training and replay

`--warm-start-optimizer keep` is the default. SGD settings, architecture, feature
density and clipping inherit from the training parent; `--lr` may override learning
rate. Local epoch/step counters restart per job; cumulative
[network counters](training.md#network-names-and-cumulative-progress) continue.
Seeds and generated-row accounting are frozen; recovery never double-counts work.

The trainer's `best.pt` is a within-job diagnostic. The loop exports the **latest
completed epoch**, not a loss-selected promotion. Training always starts from the
latest candidate.

`--replay-iterations` (default 4) admits current/previous N−1 generated rounds.
`--replay-games` (default 256) limits unique labeled games, including older bootstrap
fallback data. Newest copies win duplicate canonical identities; weighted round-robin
chooses sizes. Supply `--bootstrap-data` explicitly; source manifests do not import
an original corpus. Ancestral train/validation assignments remain pinned across
windows. The selected corpus must contain both splits. Truncations are excluded;
each generation round needs a completed policy-labeled game. Rust validates moves,
policy, score and job/model identity. See [training](training.md).

Default `--sizes 9,13,19 --size-weights 9:2,13:2,19:1` drives generation, replay and
training/validation samples. The smooth generation cycle is `9,13,19,9,13`, continued
across boundaries; ten games are 4/4/2. Tiny budgets round, missing sizes renormalize
and insufficient data may prevent exact proportions. `samples_by_size` reports
actual counts. `uniform` restores literal cycling; zero weights exclude a size.

## Operating safeguards and interruption

Preserve weights/optimizer progress; resets and architecture changes are explicit.
Inspect losses, samples, SGFs, pass/game-length and completion statistics, throughput,
memory and disk. Pause for nonfinite values, invalid data, collapse or resource
problems. Loss alone is not strength; use separate evaluation to measure progress.
Explicit retention and recovery are in [model storage](model-store.md).
`--resume --migrate-storage` detaches legacy storage references without launching jobs.

`loop.pt` is an atomic orchestration checkpoint, not an alternate game format.
SGFs are authoritative. Resume adopts immutable snapshots, completed self-play and
published exports; regenerates only missing/invalid partial games into new attempts;
and resumes contiguous completed epochs with optimizer, holdouts and counters intact.
Partial epochs are not saved. Training/actor state is restored from run-owned files,
never interpreted as publication permission.

Ctrl-C interrupts the child process group and waits/escalates for cleanup. Inherited
OS locks prevent racing a live child. Never unlink `.loop.lock` or a live/malformed
`.train.lock` to bypass it. This is process-interruption safety, not end-to-end
power-loss or bitwise GPU reproducibility. Replay limits bound active selection,
not total disk use or the ancestral holdout ledger.
