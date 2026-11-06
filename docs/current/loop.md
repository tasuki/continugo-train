# Restartable experimental neural self-play loop

`mise run loop` runs native self-play → bounded SGF replay → warm-start training →
ONNX export, with periodic experimental/reference matches and durable recovery.
It follows [KataGo's synchronous training pattern](https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md),
but **never automatically promotes a network**. Accepted publication follows the
[approval procedure](model-store.md#approval-operation).

## Three independent roles (v6)

New loops require score/ownership checkpoints and record `--score-weight` (default
0.5, the linear score fraction). Older loop controls are rejected, not migrated; keep historical campaigns
and their frozen runtime untouched. See [auxiliary supervision](auxiliary.md).

| Role | Behaviour | Durable pointer |
|---|---|---|
| Training | Latest completed candidate; weights/optimizer progress survive failed or inconclusive matches | `LATEST-TRAINING` |
| Experimental actor | Frozen model generating the next self-play batch; not accepted merely by being selected | `LATEST-EXPERIMENTAL` |
| Accepted | User-approved publication, or an existing historical accepted model; may be absent | `LATEST-ACCEPTED` |

Initialization backs up an unaccepted actor, not an accepted baseline. The loop
only reads the accepted head, refreshing it at startup and iteration boundaries.
Approval changes neither training nor actor selection.

`--actor-policy latest` is the default: after each candidate is trained, exported,
verified and archived, use it as the next experimental actor regardless of match
verdict. `--actor-policy gate` optionally holds the actor unless its head-to-head
match passes; rejected/inconclusive candidates still retain training progress.
With this optional policy, unevaluated iterations hold the actor; explicitly
zero gate games disables that actor guard. Neither policy publishes acceptance.
An actor checkpoint might already have been approved separately; its actor role
itself neither grants nor removes acceptance.

**Disabling actor gating does not require disabling evaluation.** Keep
`--actor-policy latest --gate-games 20` to measure head-to-head progress and update
ratings while always advancing the actor. Random-only results cannot distinguish
checkpoints once they consistently beat that baseline.

## Start, resume, inspect

Requires a full training checkpoint, inference dependencies and the CPU/CUDA setup
in [inference](inference.md). The task builds native binaries. The supervisor
requires POSIX process groups and `flock`.

```sh
# Small bounded example; zero matches does NOT accept anything:
mise run loop -- --checkpoint PATH/TO/checkpoint.pt --bootstrap-data PATH/TO/replay --iterations 2 --games-per-iteration 10 --train-epochs 1 --steps-per-epoch 20 --gate-games 0 --reference-games 0 --adaptive-max-games 0
# Use the dated path printed by the first command:
mise run loop -- --output runs/YYYYMMDD-HHMMSS-loop --resume
# Increase the TOTAL iteration target, not an additional count:
mise run loop -- --output runs/YYYYMMDD-HHMMSS-loop --resume --iterations 3
mise run loop-status -- --output runs/YYYYMMDD-HHMMSS-loop
```

New loops reserve `runs/YYYYMMDD-HHMMSS-loop` here (UTC, collision-safe). Explicit
`--output` paths are literal; resume must name the existing output. Status is
read-only and reports the three roles, cumulative progress, latest committed
metrics, current phase, available disk space and live training/self-play log tails.
Detailed warnings/matches remain in `loop.log` and stage logs.

Use `mise run loop -- --help` for defaults; these are setup budgets, not calibrated
strength settings. New loops and standalone neural self-play default to twelve
concurrent games / batch limit twelve / collection wait two ms; see
[inference batching](inference.md#cross-game-batching). Active slots are capped by
`--games-per-iteration` (default ten); that budget is unchanged. `--device cpu` selects CPU.
Play, training, and export are
separate supervised processes; export verification uses CPU.

Resume restores frozen configuration. Explicit conflicting overrides fail; the
total iteration target may increase, never decrease. For a clean boundary pause,
add `--stop-after-iteration N`: it exits after that round's training, scheduled
matches and archives, before launching round N+1, without lowering the saved total
target. The bound must not precede recorded work. It is invocation-only; resume
without it to continue toward the saved total. A new executable/protocol still
requires a fork, not an override of the frozen binaries. Keep the same absolute path
and compatible Python/rules/runtime versions. Self-play/feature executable hashes
are frozen even with matches disabled; arena/rating hashes are additionally
frozen when evaluation is enabled. They are checked on resume and around child
stages. Do not rebuild/replace active binaries; use frozen executable paths for
long experiments. Changed executables require a new loop/fork.

**v1–v5 control files and old policy/value-only models are not migrated or resumed.**
Start a new v6 loop from a newly trained score/ownership checkpoint. Preserve
historical campaigns with their frozen runtime. Historical accepted bundles stay
intact; the new runtime does not reinterpret or rewrite them.

A narrow resume exception enables previously disabled head-to-head evaluation:

```sh
# Existing latest-actor loop with reference evaluation; stopped after round N:
# Keep it paused while saving the new future evaluation setting:
.venv/bin/python -m continugo_train.loop --output PATH/TO/LOOP --resume --enable-head-to-head 20 --stop-after-iteration N
```

This requires a completed round boundary, `actor_policy=latest`, and unchanged
binaries/protocol. It enables only the game count, retaining the existing search,
move cap, evaluation interval and actor policy. It saves `before-head-to-head.pt`
and an audit entry in `loop.pt`; historical matches/decisions are not rewritten.
It does not backfill old checkpoints or establish strong rating confidence. For
that, use [adaptive rated matching](evaluation.md#adaptive-rated-matching) with an
explicit uncertainty/evidence target and hard game budget.

## Periodic evaluation, not automatic publication

`--evaluation-interval N` schedules candidate-vs-parent-actor and reference batches
(default 1). The initial actor gets a reference batch when enabled, as does every
scheduled candidate, even after a failed actor match. Other rounds record
`not-evaluated`. Zero `--gate-games` / `--reference-games` disables that match type.
See [evaluation](evaluation.md) for budgets, verdicts, ratings, and evidence handling.
Training and actor advancement do not wait for manual promotion review.

### Automatic adaptive rating precision

**New campaigns run adaptive precision evaluation by default** for every scheduled
candidate, in addition to the ordinary parent/reference diagnostics. Head-to-head
progress matches alone do not meet this stronger policy. To upgrade an older
saved loop, enable it explicitly at a completed boundary:

```sh
# New loops: already enabled; --adaptive-max-games 0 opts out for smoke tests.
# Existing latest-actor loop, paused after round N: enable it without starting N+1.
.venv/bin/python -m continugo_train.loop --output PATH/TO/LOOP --resume --enable-adaptive-ratings 600 --stop-after-iteration N
```

The resume operation requires a completed boundary and unchanged protocol. It
backs up `before-adaptive-ratings.pt` and audits the setting change in `loop.pt`.
It does not retroactively evaluate completed rounds or start training. Historical loops are not upgraded by this runtime. Within v6, saved settings,
including disabled evaluation, are not silently upgraded into expensive extra work. New loops need no opt-in.
Use `--adaptive-max-games 0` to explicitly disable only this precision phase; a
smoke run with no evaluation at all must also set gate/reference games to zero.

When enabled, scheduled rounds run ordinary parent/reference diagnostics, archive
the candidate's inference bundle, then invoke adaptive matchmaking **automatically**
before committing the round and advancing its actor. The target is the exact
candidate checkpoint ID, not whichever network was most recently registered.
Policy is frozen in these loop settings:

- `--adaptive-target-rd 25`, `--adaptive-min-pairs 60` (120 paired-color games).
- `--adaptive-min-opponents 2`, `--adaptive-min-opponent-pairs 10`.
- `--adaptive-batch-games 4`, `--adaptive-max-games 600` (additional cap **per scheduled candidate**).

Opponent selection and uncertainty interpretation follow
[adaptive rated matching](evaluation.md#adaptive-rated-matching). Each batch
updates the shared ledger and adapts its next opponent. A separate reserved seed
stream avoids reusing gate/reference/backfill seeds. Native children inherit both
the training-loop and evaluator locks. An interrupted round resumes its pending
adaptive batch without retraining, reexporting, replaying games or rerating evidence.

Evidence and `progress.json` live under `iteration-NNNN/adaptive-ratings/`; the
round's `adaptive_rating` receipt records precision progress and stop reason.
This evidence remains outside replay and in the durable ratings ledger. A budget
stop, or too few opponents, explicitly reports an unmet precision target; it does
**not** reject the latest actor or rewind training. Normal actor policy is unchanged.
The phase can add hours at a scheduled boundary; `--stop-after-iteration` includes
this phase too. Ordinary resume automatically retains the enabled policy.

## Randomized policy-search budget

New loops default to **32 simulations on 75% of moves and 256 on 25%**:
`--simulations 32 --policy-simulations 256 --policy-search-probability 0.25`.
The cap is sampled before each search from a separate per-game seeded RNG,
independent of position evaluation, search RNG consumption and worker scheduling.
The fraction is probabilistic, not an exact per-game quota. Expected search work
is 88 simulations/move, not a wall-time guarantee.

Only expensive positions contribute policy cross-entropy. Cheap positions retain
outcome, final-score and ownership supervision, subject to the same sampling caps
and natural-completion requirements. SGFs retain both budgets' real visits and a
per-move `train_policy` mask; replay/adoption validates their correspondence.
Phase logs report `samples` and the enabled subset `policy_samples` separately.

`--policy-search-probability 0` restores fixed-budget, all-policy self-play;
`1` searches/trains policy at the expensive budget on every move. When enabled,
the cheap budget must be at least 32 and the policy budget at least the cheap one.
Evaluation remains fixed-budget (`--gate-simulations`), never randomized.
Standalone self-play opts in explicitly; see [inference](inference.md#randomized-policy-search-caps).

Existing loops missing these settings resume with randomization **disabled**,
not the new default. Budgets remain frozen on resume; start a new loop to change
them, and keep old frozen executables intact. Legacy/bootstrap SGFs without a
`train_policy` field retain their original all-policy supervision; their old
visit counts are never relabeled as expensive searches.

## Training and replay

`--warm-start-optimizer keep` is the default. SGD settings, architecture, feature
density and clipping inherit from the training parent; `--lr` may override the
learning rate. Local epoch/step counters restart per job, while cumulative
[network sample/data counters](training.md#network-names-and-cumulative-progress)
continue across the lineage. Seeds and generated-row accounting are frozen per
iteration; recovery/reimports never double-count completed work.

The trainer's `best.pt` is a within-job validation diagnostic. The loop exports the
**latest completed epoch**, not a loss-selected promotion. Every next training
phase starts from the latest candidate, regardless of actor policy or match result.

`--replay-iterations` (default 4) admits current/previous N−1 generated rounds.
`--replay-games` (default 256) limits unique labeled games, including older bootstrap
fallback data. Newest copies win duplicate canonical-game identities; weighted
round-robin chooses sizes. Source checkpoint manifests do not implicitly import
the original corpus: supply suitable `--bootstrap-data`.

Whole-game ancestral train/validation assignments remain pinned across windows.
The selected corpus must contain both splits; insufficient data fails rather than
moving trained games into validation. Truncated actual games have no invented
outcome and are excluded. Each generation round needs at least one completed
policy-labeled game. Rust validates every move, policy, score and job/model identity.
See [training](training.md) for exact labeling, loading and warm-start semantics.

### 40/40/20 coverage

Default `--sizes 9,13,19 --size-weights 9:2,13:2,19:1` drives generation, replay,
actual training/validation samples and evaluation pairs. The smooth generation
cycle is `9,13,19,9,13`, continued across iteration boundaries. Ten generated games
are 4/4/2; twenty paired match games are 8/8/4. Both colors get the same size budget.

Tiny budgets round, missing sizes renormalize, and insufficient data may prevent
exact proportions. `samples_by_size` reports actual optimized/validated counts.
These are sample/game fractions, not wall-time fractions. Weights are configurable;
`uniform` restores literal cycling and zero weights exclude a size. All three
sizes remain covered by rule/feature/inference parity tests.

## Operating safeguards

Preserve weights/optimizer progress within a campaign; resets and network-size
changes are explicit decisions. Keep generating fresh games rather than fitting
one weak corpus indefinitely. Inspect losses, sample counts, SGFs, pass/game-length
and completion statistics, strength, throughput, memory, and disk. Pause for
nonfinite values, invalid data, behavioral collapse, or resource/recovery problems.
Loss alone is not strength and need not decrease across changing replay windows.

Store layout, backups, and recovery after losing `runs/` are in
[model storage](model-store.md).

## Interruption semantics

`loop.pt` is an atomic control checkpoint, not an alternate game format. SGFs are
the authoritative records. Stable iteration/attempt names and hashes let resume:

- Adopt immutable input snapshots, completed self-play SGFs and published exports.
- Regenerate only missing/invalid partial games into new attempts, keeping valid
  completed games and each frozen actor/seed assignment.
- Resume the newest compatible contiguous completed epoch; retain weights,
  momentum, holdouts and counters. A partial epoch is not saved; a failed first
  epoch gets a new attempt. Stale trainer locks are removed only for absent PIDs.
- Resume missing evaluation games without retraining, reexporting or rerating
  already committed evidence.
- Reuse durable training/experimental bundles and repair an actor-pointer commit
  gap without generating another batch. Never derive consent from these steps.

Ctrl-C interrupts the active child process group and waits/escalates for cleanup.
An inherited OS lock prevents another supervisor racing an in-flight child. Never
unlink `.loop.lock` or a live/malformed/inaccessible `.train.lock` to bypass it.
This is process-interruption safety, not end-to-end power-loss or bitwise GPU
reproducibility. Completed approval intents/receipts have additional fsync-backed
publication recovery, documented separately. Replay limits bound active selection,
not total archive disk use or the ancestral holdout ledger.
