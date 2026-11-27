# Restartable experimental neural self-play loop

`mise run loop` runs native self-play → growing training/validation windows → warm-start training →
ONNX export, with run-owned recovery. **Training does not run rating matches.**
It saves every round's model for separate evaluation under `artifacts/ratings/`.
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

For a unified, repo-tracked launcher with per-campaign recipes, frozen code/binaries
and inherited window games, use the [campaign runner](campaigns.md). The commands below expose the
lower-level loop directly.

## Start, resume, inspect

Requires a full training checkpoint and the CPU/CUDA setup in [inference](inference.md).
The supervisor requires POSIX process groups and `flock`.

```sh
mise run loop -- --checkpoint PATH/TO/checkpoint.pt --model-store artifacts/LINEAGE --bootstrap-data PATH/TO/window-games --iterations 2 --games-per-iteration 10 --train-epochs 1 --steps-per-epoch 20
# Use the dated path printed by the first command:
mise run loop -- --output runs/YYYYMMDD-HHMMSS-loop --resume
# Increase the TOTAL iteration target, not an additional count:
mise run loop -- --output runs/YYYYMMDD-HHMMSS-loop --resume --iterations 3
mise run loop-status -- --output runs/YYYYMMDD-HHMMSS-loop
```

New loops reserve `runs/YYYYMMDD-HHMMSS-loop` (UTC, collision-safe). Explicit
`--output` paths are literal; resume must name the existing output. Status is
read-only and reports roles, cumulative progress, metrics, phase, disk and live logs.

Use `mise run loop -- --help` or the [constants and formulas inventory](constants.md)
for defaults. New loops default to 32 concurrent games / batch limit 32 /
collection wait two ms; active slots are capped by `--games-per-iteration`.
Training batch size is 64; `--positions-per-epoch` defaults to
65,536, sampled uniformly without replacement from the eligible replay window
before feature loading, with a fresh seeded sample each iteration/epoch. Smaller
pools are exhausted; zero disables sampling. The default is one training epoch
per iteration (1,024 batches for a full sample). Per-game and step caps default
to zero; validation remains exhaustive and uncapped by this sampling budget. Feature
caches are required by default. See [inference batching](inference.md#cross-game-batching).
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

`--save-model-every 1` (the default) saves independent checkpoint and ONNX copies in
`artifacts/models/<family>/rNNNN/`, alongside small portable `metadata.json`. Original campaign files stay
intact. No window-game copy, hardlink, approval or rating match is created. Saving is
idempotent; resume repairs a missed save after a state-commit interruption without
retraining. `--save-model-every 0` disables it.

Training creates no `gate/`, `reference/`, `initial-reference/` or
`adaptive-ratings/` jobs and does not register models or update rating plots.
Evaluate saved models separately:

```sh
mise run rate-saved -- --model-store artifacts/LINEAGE --family cg5
mise run rate-saved -- --model-store artifacts/LINEAGE --resume
```

The standalone evaluator owns its protocol, 10+30-game champion challenges and
post-promotion history calibration: 50 games each against up to three opponents
(the defeated champion plus other available historical opponents, including random).
There is no separate mandatory random calibration batch.
Seven wins in ten permit continuation; 28 wins
in forty **automatically replace the evaluation champion**, initially the random
bot. Rated neural moves use 128 simulations by default. Publication is still a
separate user decision. The shared evaluator state/log and champion lineage live under `artifacts/ratings/`;
SGFs live under `artifacts/sgfs/YYYY-MM-DD/`, with no duplicate game ledger. See
[evaluation](evaluation.md). Ratings never affect training or actor selection.

## Randomized policy-search budget

New loops default to **100 simulations on 75% of moves and 500 on 25%**:
`--simulations 100 --policy-simulations 500 --policy-search-probability 0.25`.
The cap is sampled before search from a separate per-game seeded RNG, independent
of position evaluation, search RNG consumption and worker scheduling. Expected
work is 175 simulations/move, not a wall-time guarantee.

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

## Training and validation windows

`--warm-start-optimizer keep` is the default. SGD settings, architecture, feature
density and clipping inherit from the training parent; `--lr` may override learning
rate. Local epoch/step counters restart per job; cumulative
[network counters](training.md#network-names-and-cumulative-progress) continue.
Seeds and generated-row accounting are frozen; recovery never double-counts work.

The trainer's `best.pt` is a within-job diagnostic. The loop exports the **latest
completed epoch**, not a loss-selected promotion. Training always starts from the
latest candidate.

Training uses **KataGo's combined position window**, based on `d` after the
current self-play round:

```text
W = min(d, 250_000 + (0.4 * 250_000**0.35 / 0.65)
        * (max(d, 250_000)**0.65 - 250_000**0.65))
```

Round `W` up to an integer position. Select the **combined window first**, without
consulting train/validation labels: whole newest eligible games until `ceil(W)`
positions are reached, or the accessible eligible pool is exhausted. The final
game may overshoot the budget. All local rounds and explicit bootstrap games are
candidates; board sizes use weighted round-robin, newest-first within each size,
and canonical duplicates count once. Fixed game/iteration limits do not cap this mode.

**Split the selected window afterward**, using `validation_fraction` (default 2%)
and permanent game-level holdouts stratified by resolution. Existing assignments,
including inherited pool pins, never change. Only selected unseen games get new
assignments; there is no separately filled validation window and no pulling older
held-out games from outside the chosen window. Both splits must be nonempty; a
window lacking a usable holdout fails rather than leaking validation into training.
Corpora, budgets and splits are frozen per round for recovery. Validation subsets
may change, so validation loss is not a fixed-cohort benchmark. See the
[formula sample table](constants.md#katago-position-window-the-default).

KataGo is the only window policy; there are no fraction, fixed-game or
fixed-iteration overrides.

Supply `--bootstrap-data` explicitly; source manifests do not import an original
corpus. Missing ancestral SGFs cannot be sampled merely because their identities
appear in a checkpoint. The selected corpus must contain both splits. Truncations are excluded;
each generation round needs a completed policy-labeled game. Rust validates moves,
policy, score and job/model identity. See [training](training.md).

Default `--sizes 9,13,19 --size-weights 9:1,13:0,19:0` uses **size 9 only** for
generation, window selection and training/validation samples.
The former 40%/40%/20% mix (`9:2,13:2,19:1`) remains an explicit option and a
commented-out default in source. `samples_by_size` reports actual counts.
`uniform` restores literal cycling; zero weights exclude a size. Explicitly
selecting a different size also requires a positive weight for it.

## One board size per generation iteration

`--size-schedule iteration` cycles through `--sizes` in the supplied order, one
size per self-play iteration. It does **not** change the replay/training mix.
`--games-by-size 9:150,11:123,13:104,15:90,17:79,19:71` supplies positive game
counts for every size; if omitted, `--games-per-iteration` applies to each size.
Sizes must be unique and have positive training weights. The default `mixed`
schedule preserves the original weighted within-iteration generation behavior.

Seed ranges follow cumulative **planned game counts**, not iteration count times
a fixed count. Partial attempts keep their original size/seed/index on recovery;
the cycle and counts are frozen on resume. Replay continues to include all
positive-weight sizes, and `--size-sampling available` can ramp up training shares
as new sizes become available. See [cg13 mixed boards](mixed-boards.md).

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
power-loss or bitwise GPU reproducibility. Window limits bound active selection,
not total disk use or the ancestral holdout ledger.
