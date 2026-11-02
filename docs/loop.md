# Restartable experimental neural self-play loop

`mise run loop` runs native self-play → bounded SGF replay → warm-start training →
ONNX export, with periodic experimental/reference matches and durable recovery.
It follows [KataGo's synchronous training pattern](https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md),
but **never automatically promotes a network**. See the agreed
[training and promotion policy](training-policy.md).

## Three independent roles (v5)

| Role | Behaviour | Durable pointer |
|---|---|---|
| Training | Latest completed candidate; weights/optimizer progress survive failed or inconclusive matches | `LATEST-TRAINING` |
| Experimental actor | Frozen model generating the next self-play batch; not accepted merely by being selected | `LATEST-EXPERIMENTAL` |
| Accepted | User-approved publication, or an existing historical accepted model; may be absent | `LATEST-ACCEPTED` |

Initialization exports and backs up an **unaccepted** experimental actor. No
accepted baseline is implicitly created. Starting/resuming a loop, disabling
matches, passing a match, or advancing an actor never grants promotion permission.
Approval is a [separate hash-pinned operation](training-policy.md#approval-operation).
It changes neither training nor actor selection. The loop only reads the accepted
head, refreshing its saved view at startup and iteration boundaries.

`--actor-policy latest` is the default: after each candidate is trained, exported,
verified and archived, use it as the next experimental actor regardless of match
verdict. `--actor-policy gate` optionally holds the actor unless its head-to-head
match passes; rejected/inconclusive candidates still retain training progress.
With this optional policy, unevaluated iterations hold the actor; explicitly
zero gate games disables that actor guard. Neither policy publishes acceptance.
An actor checkpoint might already have been approved separately; its actor role
itself neither grants nor removes acceptance.

## Start, resume, inspect

Requires a full training checkpoint, inference dependencies and the CPU/CUDA setup
in [inference](inference.md). The task builds native binaries. The supervisor
requires POSIX process groups and `flock`.

```sh
# Small bounded example; zero matches does NOT accept anything:
mise run loop -- --checkpoint PATH/TO/checkpoint.pt --bootstrap-data PATH/TO/replay --iterations 2 --games-per-iteration 10 --train-epochs 1 --steps-per-epoch 20 --gate-games 0 --reference-games 0
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

Defaults: CUDA, one iteration, ten self-play games, one training epoch capped at
20 batches, 20 head-to-head games and 10 reference games. These are setup budgets,
not calibrated strength settings. `--device cpu` selects CPU. Play, training and
export are separate supervised processes; export verification uses CPU.

Resume restores frozen configuration. Explicit conflicting overrides fail; the
total iteration target may increase, never decrease. Keep the same absolute path
and compatible Python/rules/runtime versions. Self-play/feature executable hashes
are frozen even with matches disabled; arena/rating hashes are additionally
frozen when evaluation is enabled. They are checked on resume and around child
stages. Do not rebuild/replace active binaries; use frozen executable paths for
long experiments. Changed executables require a new loop/fork.

**v1–v4 automatic-promotion loops are not migrated or resumed.** Their artifacts
remain available. Fork a v5 experiment from the latest full checkpoint and exact
replay; specify `--actor-checkpoint` if its previous actor differs from the latest
training checkpoint. `--incumbent-checkpoint` remains an alias for that flag, but
no longer means “accept this checkpoint.” Historical accepted bundles stay intact.

## Periodic evaluation, not automatic publication

`--evaluation-interval N` runs candidate-vs-parent-actor and candidate-vs-reference
batches every N iterations (default 1). The initial actor gets one reference batch
when reference games are enabled. Every evaluated candidate gets reference games,
including candidates failing the actor match. Unscheduled rounds are explicitly
`not-evaluated`; match verdicts are `passed`, `failed`, `inconclusive`, or `disabled`.
None means accepted. Reference matches are never an automatic promotion threshold.

Set `--gate-games 0` and/or `--reference-games 0` to disable those match types, not
to authorize publication. Periodic evaluation lets training/experimental actor
updates continue without human review after every phase. Matches still take time
when scheduled; approval itself does not block the learning loop.

See [evaluation](evaluation.md) for paired colors, conservative missing-outcome
bounds, per-size breakdown, natural draws, caps and ratings. Preserve all evaluation
SGFs outside replay. Ratings remain in `models/ratings/networks.csv` and
`matches.csv`, with immutable SGF evidence. Unapproved networks stay `accepted=0`,
even if well-rated or selected for self-play. `mise run ratings -- --accepted-only`
shows only user-approved or preserved historical accepted networks.

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

## Durable recovery

Default store: `models/` inside this repository, outside disposable `runs/`:

```text
models/
  LATEST-TRAINING
  LATEST-EXPERIMENTAL
  LATEST-ACCEPTED                 absent until approval, unless already historical
  training/<dated-name>/         checkpoint + exact replay for every candidate
  experimental/<dated-name>/     checkpoint + replay + frozen ONNX + match evidence
  accepted/<dated-name>/         only explicitly approved publications
  approvals/                    immutable user approval intents and receipts
  sgf-blobs/                    deduplicated store-owned SGFs
  ratings/                      both CSVs and authoritative evaluation SGFs
```

Training backup precedes export/evaluation; experimental backup precedes actor
selection. Bundles are write-once/read-only, hashed and never automatically pruned.
Accepted bundles also retain their exact approval record. New accepted publications
require explicit consent pinned to the full checkpoint hash, not a moving alias.

To fork after losing `runs/`, use `LATEST-TRAINING`'s checkpoint/replay and, if
different, `LATEST-EXPERIMENTAL`'s checkpoint as `--actor-checkpoint`, with a fresh
seed and output. Store pointers are convenience indexes across experiments; a live
loop's frozen parents are in `loop.pt`. Approval does not alter those parents.
See [model storage](model-store.md) for backup and strict offline continuation.

**ONNX alone cannot resume training.** Checkpoints preserve optimizer/BatchNorm,
RNG, options, cumulative progress, holdout history and corpus identities. Keep the
whole model store, source/rules and a compatible environment backed up off-machine.
`models/` is Git-ignored; destructive Git-clean commands can delete it.

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
