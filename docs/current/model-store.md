# Model storage, approval, and recovery

Training campaigns own their checkpoints, optimizer/RNG state, bootstrap/replay
SGFs, intermediate ONNX exports and loop controls in `runs/`. A campaign does
**not** need duplicate recovery bundles under `models/` to resume.

One shared `models/` store holds selected milestone snapshots, retained/published models and the
rating ledger/evidence. Do not create a model store for each campaign or place
one beneath `runs/` or `scratch/`. See [artifact organization](artifacts.md).

## Layout and protection

```text
models/
  STORE.pt                       ownership/format marker (metadata, not model weights)
  <campaign>-rNNNN/               checkpoint + both ONNX graphs + metadata.json
  retained/<UTC-dated-name>/      explicitly retained, unaccepted model bundles
  accepted/<UTC-dated-name>/      explicitly approved publications
  LATEST-ACCEPTED                 optional approved head
  approvals/                     immutable authorization intents and receipts
  ratings/networks.csv            shared network metadata and ratings
  ratings/matches.csv             authoritative rated-match commit ledger
  ratings/inference/<SHA>.pt      hash-pinned locators to run-owned checkpoint/graphs
  ratings/games/                  store-owned match SGFs + mandatory .nn.html sidecars
  ratings.png                    shared strength plot
```

New campaigns save a minimal model snapshot every tenth completed round and
calculate its rating with adaptive near-strength matches. They do not copy replay
or make a full recovery bundle per round. They never create `training/`,
`experimental/`, `LATEST-TRAINING` or `LATEST-EXPERIMENTAL`. Training and actor roles
are recorded in each campaign's `loop.pt`.

Only scheduled/evaluated networks and accepted publications enter the rating
registry. Inference locators contain paths, hashes and a cached cumulative training
game count, not weights or replay. Evaluation can snapshot available run-owned
exports or deliberately retained bundles. Deleting a campaign removes unretained
models from future opponent pools, but preserves ratings, plots and match evidence.
Changed model files fail verification rather than silently becoming new opponents.

Bundles are write-once/read-only and contain `checkpoint.pt`, exact
`replay/<SHA>.sgf`, both ONNX graphs, `recovery.pt`, available match evidence and a
README. Accepted bundles also embed approval. UTC-dated names include a sequence,
network name and short hash; the full checkpoint hash determines identity.
Publication reuses and verifies an existing identical bundle. There is no automatic full-recovery
retention or pruning; minimal milestone saving is separate. Read-only mode is not access control or a disk-loss backup.

All stored files are ordinary independent copies. Hardlinks are never created,
including between retained bundles or between an approval candidate and its
verification view. SGF evidence is copied directly to `ratings/games/`; there is
no shared `sgf-blobs/` pool. Before a new match is committed, its self-contained
[NN viewer](nn-viewer.md) sidecar is atomically published beside the SGF. Each
visualized move uses its own player's checkpoint; random/unavailable/unsupported
turns are skipped with reasons, never substituted with the opposing network.

## Milestone snapshots

```sh
# Backfill every completed tenth round in each existing campaign:
mise run save-models -- --runs runs
# Save another specific completed round explicitly:
mise run save-models -- --loop runs/CAMPAIGN/loop --iteration 15
```

Names identify the source campaign and its local round, for example
`models/20261106-142353-scratch-score-r0020/`. The four files are `checkpoint.pt`,
`encoder.onnx`, `policy.onnx` and `metadata.json`. JSON records campaign, round,
source loop, full checkpoint/graph hashes, cumulative progress and training-game
count. Writes are atomic, verified, idempotent, independent of the source files,
and never grant model approval. v5 historical graphs are copied as-is, not re-exported
with incompatible current code. Their frozen runtime remains necessary to use them.
Minimal snapshots do not copy replay: keep the campaign for exact training recovery,
or explicitly make a full recovery pack if needed.

New campaigns default to `--save-model-every 10 --evaluation-interval 10`, and saved
milestones are always evaluation boundaries. Ratings update after each committed
batch and the milestone save refreshes `models/ratings.png`. Moderate confidence
uses RD ≤ 75, 20 paired-colour seeds, two opponents with four pairs each, and a
360-additional-game cap per scheduled candidate. No all-pairs schedule is implied.
Explicitly disabled evaluation produces no invented rating.

To rate saved snapshots adaptively in a separate restartable background job:

```sh
mise run rate-saved -- --output scratch/YYYYMMDD-HHMMSS-ratings
# Resume the same frozen job without replaying committed games:
mise run rate-saved -- --output scratch/YYYYMMDD-HHMMSS-ratings --resume
```

This calibrates each model against the reference, then chooses near-strength
neural opponents until the precision/evidence target or budget is reached.
Incompatible schemas use separate compatible native runtimes, never cross-schema
games. Its default extra budget is 360 games per compatible cohort. It does not
reset the ledger itself; retire old ratings deliberately before requesting a
fresh start. New results, not archived ratings, determine the current chart.

## Explicit full-recovery retention

Retain a model **before deleting its campaign** if it should remain available:

```sh
# Latest completed round; --iteration N chooses an exact completed round.
mise run retain-model -- --loop runs/CAMPAIGN/loop
# Initial self-play actor instead:
mise run retain-model -- --loop runs/CAMPAIGN/loop --iteration 0
```

This copies only the requested checkpoint, replay and export into `models/retained/`
(or the campaign's configured store). It changes neither training/actor selection
nor acceptance and is safe to repeat. Unfinished rounds cannot be retained. Read
the bundle README for exact recovery context and limitations.

## Approval operation

**Every accepted publication requires explicit user approval of the full checkpoint
hash after judging it convincingly much stronger than the fixed random-but-not-eye-
filling reference.** A gate verdict, rating, loss or permission to train is not
approval. Present strength, board/color coverage, uncertainty and truncations for
review; there is no automatic numeric definition of "much stronger".

First explicitly retain the selected model. Only after specific user approval:

```sh
mise run promote -- --model-store models --candidate models/retained/BUNDLE --checkpoint-sha256 FULL_64_CHARACTER_HASH --approved-by USER --reason "Specific approval and reviewed strength evidence" --confirm-much-stronger-than-reference
```

The candidate must be an unchanged store-owned retained bundle. Checkpoint, replay,
graph provenance and eager/export parity are verified. Approver, reason and
reference-strength confirmation are mandatory. Approval covers one hash, never a
lineage or future candidate. It changes neither training nor actor selection.

A sealed, fsynced intent precedes publication, followed by a committed receipt.
Accepted bundles embed authorization. Only the latest committed approval may move
`LATEST-ACCEPTED`; retrying an older request cannot roll the head backwards. Locks
serialize approvals. Direct accepted publication without a matching intent fails.

Finish an already-authorized interrupted publication with:

```sh
mise run promote -- --model-store models --recover
```

Recovery cannot issue permission. Never hand-edit intents/receipts. Legacy
experimental candidates and historical accepted bundles remain readable without
fabricated retroactive approval. This is an operator workflow, not cryptographic
remote-user authentication; writable store/CLI access is inside the trust boundary.

## Resume and storage migration

Ordinary resume verifies run-owned files, not redundant global recovery copies:

```sh
mise run loop -- --output runs/CAMPAIGN/loop --resume
```

A storage-only migration for an existing v6 campaign can run **without starting
training or evaluation**:

```sh
mise run loop -- --output runs/CAMPAIGN/loop --resume --migrate-storage
```

It preserves `before-run-owned-storage.pt`, detaches legacy recovery references,
and indexes evaluated run-owned models. It does not change weights, optimizer,
seeds, search settings or existing match results. Old v1–v5 protocols are not
upgraded; preserve those historical campaigns and their frozen software.

Campaign-specific launchers may additionally pin Python sources. Their original
verification files are not rewritten by this migration; do not blindly relaunch
one after changing source. Explicitly validate a compatible runtime first.

## Restart learning after losing `runs/`

If nothing was retained or backed up, losing a campaign loses its trainable models.
**ONNX alone cannot resume training.** An explicitly retained bundle preserves the
checkpoint, optimizer/RNG state, corpus/holdouts and cumulative progress:

```sh
mise run loop -- --checkpoint models/retained/BUNDLE/checkpoint.pt --bootstrap-data models/retained/BUNDLE/replay --seed NEXT_UNUSED_SEED --warm-start-optimizer keep
```

Use `--actor-checkpoint OTHER_BUNDLE/checkpoint.pt` only when intentionally keeping
a different self-play actor. For strict offline continuation, copy `checkpoint.pt`
into a writable run and use `train --resume COPIED_CHECKPOINT --data BUNDLE/replay
--epochs NEXT_TOTAL_EPOCH`. Never train inside a sealed bundle. Compatible binaries,
feature semantics, rules and environment are still required; original loop-attempt
history is not reconstructed. Retention never grants approval.

## Backups

Back up campaigns and the whole model store off-machine. Preserve approvals, both
rating CSVs, inference metadata and SGF evidence; one saved model or retained
bundle does not contain the global ledger. Quiesce writers while copying CSVs;
use ordinary independent copies (`rsync -a`, without hardlink options). `runs/`, `scratch/` and `models/` are Git-ignored;
exclude retained work from destructive Git-clean commands. There is no automatic
backup/upload or garbage collector.
