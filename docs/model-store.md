# Durable models, recovery and explicit acceptance

**ONNX alone cannot resume training.** Recovery bundles retain the exact full
checkpoint and replay SGFs. Training, experimental actor selection and accepted
publication are independent; see [policy](training-policy.md) and [loop](loop.md).

## Layout and protection

Default: this repository's **`models/`**, outside disposable `runs/`:

```text
models/
  README.md
  STORE.pt                      ownership/format marker
  LATEST-TRAINING                latest backed-up training candidate
  LATEST-EXPERIMENTAL            selected self-play actor, not acceptance
  LATEST-ACCEPTED                only explicit approval or preserved historical head
  training/<UTC-dated-name>/
    checkpoint.pt
    replay/<raw-file-SHA>.sgf
    recovery.pt
    README.md
  experimental/<UTC-dated-name>/
    checkpoint.pt
    encoder.onnx
    policy.onnx
    replay/<raw-file-SHA>.sgf
    evaluation/*.sgf             available candidate match evidence
    recovery.pt
    README.md
  accepted/<UTC-dated-name>/
    checkpoint.pt
    encoder.onnx
    policy.onnx
    replay/<raw-file-SHA>.sgf
    evaluation/*.sgf
    approval.pt                  embedded exact user authorization (new format)
    recovery.pt
    README.md
  approvals/
    SEQUENCE-APPROVAL_ID.pt       immutable authorization for one checkpoint
    SEQUENCE-APPROVAL_ID.receipt.pt
  sgf-blobs/YYYY-MM-DD/<SHA>.sgf  UTC-day buckets, immutable store-owned data
  ratings/networks.csv
  ratings/matches.csv
  ratings/games/*.sgf
```

`--model-store PATH` may select another location, but not disposable `runs/` or a
path overlapping loop output. Nonempty foreign stores without the ownership
marker are rejected. `LATEST-ACCEPTED` need not exist in a new experiment. These
pointers are indexes across experiments, not a global strength ranking: a live
loop freezes its own training/actor parents in `loop.pt`.

Training backup precedes candidate evaluation, retaining progress even if export
or matching fails. A frozen experimental pack precedes actor selection. Exporting,
backing up, choosing an actor or passing a match does **not** create acceptance.
Initial v5 checkpoints are experimental, not implicitly accepted baselines.

Names start with UTC `YYYYMMDD-HHMMSS`, a same-second sequence, readable
[network name](training.md#network-names-and-cumulative-progress) and a short
checkpoint hash. Full hashes determine identity; short names/counters do not.
Republishing reuses an existing verified pack and its original name/date. All
weights, optimizer/RNG state, exact corpus, holdout history and cumulative progress
are preserved. Recovery manifests hash each file; ONNX graphs retain matching
checkpoint/config/feature metadata.

Packs are write-once/read-only, staged and fsynced before atomic publication, and
never automatically pruned. Store SGFs are **copies**, not hardlinks to disposable
run-owned files. Hardlinks only connect store-owned blobs, bundle replay/evidence
and rating evidence. Deleting `runs/` cannot mutate or delete those retained SGFs.
New blobs are grouped by the UTC day they are first retained in this store, not
by an invented game date. A hash is reused across days: importing the same SGF
tomorrow does not create another blob. Bundle `replay/` and `evaluation/` paths
and `ratings/games/<SHA>.sgf` remain stable hardlinks, so no checkpoint, manifest
or CSV rewrite is needed for the daily blob layout. Legacy flat blobs remain
readable and are reused; all new copies use dated buckets.

Owners can chmod/delete local data deliberately; read-only mode is not access
control or a substitute for backups. On 2026-11-02 the user explicitly requested
removal of all old generated runs/model stores, including `cold1`; those artifacts
were deleted, not archived. A new campaign must bootstrap from scratch.

## Accepted publication is a separate manual operation

See the [approval procedure](training-policy.md#approval-operation). Direct
`archive.publish(..., "accepted", ...)` without a matching explicit intent fails,
even for an existing accepted pack. There is no automatic gate-disabled shortcut
or startup baseline publication. No new promotion is authorized until the user
judges the checkpoint convincingly much stronger than the reference and approves
its full immutable hash.

The approval intent records source pack/graph/checkpoint identities, approving
user, reason, UTC time and reference-strength confirmation. Accepted packs embed
it. File/manifest verification, export provenance/parity checks, writer locking,
immutable receipts and latest-receipt validation protect publication/retry paths.
Duplicate or concurrent requests do not mint multiple publications; retrying an
older approval cannot replace a newer accepted head.

A publication interrupted after explicit consent can be completed with:

```sh
mise run promote -- --model-store models --recover
```

This cannot issue new permission. It only finishes existing authorizations and
repairs accepted bundle/rating/head commit gaps. Neither approval nor its recovery
changes training or experimental actor pointers. Keep `approvals/` intact and do
not edit authorizations/receipts by hand. The workflow is an explicit operator
control, not remote-user cryptographic authentication.

Historical v1 recovery bundles remain readable, including old accepted baselines.
They are not rewritten to fabricate retroactive approval. New accepted bundles
use recovery format v2 with embedded authorization. Old v1–v4 automatic-promotion
**loop control files** are not resumed by the v5 supervisor; fork from their data
instead. A source checkpoint can be reused without granting a new acceptance.

## Ratings and evidence

`ratings/networks.csv` contains all registered networks; `accepted=1` identifies
manual publications or preserved historical accepted entries. Unaccepted training
and experimental packs register as `accepted=0`; actor updates never change it.
Ratings measure strength and remain separate from permission to publish.

`matches.csv` is the committed rating/evidence ledger; current numeric ratings
are reconstructed from its snapshots, not by inventing fresh observations.
Store-owned `ratings/games/` retains authoritative SGFs even after runs disappear.
Accepted/experimental packs include their available match evidence, but no single
pack preserves the global rating history. Keep both CSVs and all referenced SGFs.
Never edit ratings, processed hashes or audit snapshots by hand. See
[evaluation](evaluation.md) for static Glicko, protocol pools and censored results.

## Restart learning after losing `runs/`

Choose the packs named by `LATEST-TRAINING` and `LATEST-EXPERIMENTAL`. Fork into a
new output with an unused seed:

```sh
mise run loop -- --checkpoint models/training/TRAINING_BUNDLE/checkpoint.pt --actor-checkpoint models/experimental/ACTOR_BUNDLE/checkpoint.pt --bootstrap-data models/training/TRAINING_BUNDLE/replay --seed NEXT_UNUSED_SEED --warm-start-optimizer keep
```

Omit `--actor-checkpoint` only when intentionally using the training checkpoint
as the initial experimental actor. Neither form accepts it. `--incumbent-checkpoint`
is a compatibility alias for actor selection, not promotion. Recovery contexts
record prior settings/iteration and a suggested next generation seed; do not reuse
dead `runs/` executable paths from old configurations verbatim. Build compatible
binaries or deliberately fork a new search protocol.

For strict offline continuation, copy—not move—the checkpoint into a new writable
run directory and use the archived corpus:

```sh
mkdir -p runs/recovered-training
cp models/training/TRAINING_BUNDLE/checkpoint.pt runs/recovered-training/checkpoint.pt
mise run train -- --data models/training/TRAINING_BUNDLE/replay --resume runs/recovered-training/checkpoint.pt --epochs NEXT_TOTAL_EPOCH
```

The target must exceed the saved epoch. Source game paths may relocate: exact
hashes/replay provide recovery. Never train inside a sealed pack. Legacy checkpoints
retain unknown cumulative totals and original sampling defaults rather than
inventing history. GPU backward/changed batch/backend trajectories are not promised
bitwise reproducible; unchanged single-thread CPU continuation is tested.

A pack preserves learning state, not the original loop attempt history or software
installation. Keep code/rules/runtime versions compatible. Approval journal recovery
uses the original store identity; copying a pack off-machine preserves its embedded
audit, but does not grant permission to create new publications in another store.

## Backups still matter

Back up the **whole store** off-machine, including approvals, both rating CSVs,
immutable evidence and data blobs. Quiesce writers while copying the CSV pair;
`rsync -aH` can preserve store-owned hardlink deduplication. Local read-only files
do not protect against disk loss, root access or deliberate deletion.

`models/` is Git-ignored. `git clean -fdx` / `-fdX` can destroy it: explicitly exclude
it and keep backups. There is no automatic garbage collector or automatic upload.
