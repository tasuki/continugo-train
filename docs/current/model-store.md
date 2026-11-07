# Model storage, approval, and recovery

Training campaigns own their checkpoints, optimizer/RNG state, bootstrap/replay
SGFs, intermediate ONNX exports and loop controls in `runs/`. A campaign does
**not** need duplicate recovery bundles under `models/` to resume.

One shared `models/` store holds deliberately retained/published models and the
rating ledger/evidence. Do not create a model store for each campaign or place
one beneath `runs/` or `scratch/`. See [artifact organization](artifacts.md).

## Layout and protection

```text
models/
  STORE.pt                       ownership/format marker
  retained/<UTC-dated-name>/      explicitly retained, unaccepted model bundles
  accepted/<UTC-dated-name>/      explicitly approved publications
  LATEST-ACCEPTED                 optional approved head
  approvals/                     immutable authorization intents and receipts
  ratings/networks.csv            shared network metadata and ratings
  ratings/matches.csv             authoritative rated-match commit ledger
  ratings/inference/<SHA>.pt      hash-pinned locators to run-owned checkpoint/graphs
  ratings/games/                  store-owned match evidence
  sgf-blobs/YYYY-MM-DD/<SHA>.sgf   deduplicated evidence and explicitly retained replay
  ratings.png                    shared strength plot
```

Loops initialize this store without copying any model or replay. They never
create `training/`, `experimental/`, `LATEST-TRAINING` or `LATEST-EXPERIMENTAL`.
Training and actor roles are recorded in each campaign's `loop.pt`.

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
Publication reuses and verifies an existing identical bundle. There is no automatic
retention or pruning. Read-only mode is not access control or a disk-loss backup.

Store SGFs are copies, never hardlinks to run-owned files. Hardlinks connect only
store-owned blobs, retained replay and rating evidence. Blobs use UTC first-retention
days and deduplicate across days; legacy flat blobs remain readable.

## Explicit retention

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
rating CSVs, inference metadata, SGF blobs and evidence; one retained bundle does
not contain the global ledger. Quiesce writers while copying CSVs; `rsync -aH`
preserves hardlink deduplication. `runs/`, `scratch/` and `models/` are Git-ignored;
exclude retained work from destructive Git-clean commands. There is no automatic
backup/upload or garbage collector.
