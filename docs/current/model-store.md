# Model storage, approval, and recovery

One shared store: `models/`, outside disposable `runs/`. Do not create a store
subdirectory for each training campaign; all campaigns share the same registry,
ratings and content-addressed recovery bundles. Run-specific files belong under
`runs/`. Only scheduled/evaluated networks enter the rating registry; intermediate
recovery backups are separate from registry entries. Model roles and actor
selection are defined in [loop](loop.md#three-independent-roles-v5).

## Layout and protection

```text
models/
  STORE.pt                       ownership/format marker
  LATEST-TRAINING                 latest backed-up training candidate
  LATEST-EXPERIMENTAL             selected self-play actor
  LATEST-ACCEPTED                 optional, explicitly approved publication
  training/<UTC-dated-name>/      checkpoint + exact replay + recovery manifest
  experimental/<UTC-dated-name>/  training contents + ONNX + available match SGFs
  accepted/<UTC-dated-name>/      experimental contents + embedded approval
  approvals/                     immutable intents and publication receipts
  sgf-blobs/YYYY-MM-DD/<SHA>.sgf   deduplicated store-owned SGFs
  ratings/                       global CSV ledger and match evidence; see evaluation
  ratings.png                    automatically updated global strength plot
```

Bundles contain `checkpoint.pt`, `replay/<raw-SHA>.sgf`, `recovery.pt`, and a
README. Inference bundles add `encoder.onnx`, `policy.onnx`, and available
`evaluation/*.sgf`; new accepted bundles also embed `approval.pt`.

`--model-store PATH` cannot overlap loop output or reside in disposable `runs/`.
Nonempty foreign stores without the ownership marker are rejected. Pointers index
experiments, not strength rankings; each live loop freezes its own parents.

Training backup precedes export/evaluation; experimental backup precedes actor
selection. Packs are write-once/read-only, staged and fsynced before atomic
publication, and never automatically pruned. Names combine UTC
`YYYYMMDD-HHMMSS`, a same-second sequence, the [network name](training.md#network-names-and-cumulative-progress),
and short checkpoint hash. Full hashes determine identity; republishing reuses
the original verified pack/name. Manifests hash files and ONNX provenance.

Store SGFs are copies, never hardlinks to run-owned files. Hardlinks connect only
store-owned blobs, bundle replay/evidence, and rating evidence. Removing `runs/`
cannot mutate retained SGFs. Blobs are bucketed by UTC first-retention day; hashes
are reused across days. Legacy flat blobs remain readable. Bundle and rating
SGF paths remain stable regardless of blob bucket.

This describes supported storage, not existing artifacts. Read-only mode is not
access control; owners can deliberately chmod/delete files.

## Approval operation

**Every accepted publication requires explicit user approval of the full checkpoint
hash, after judging it convincingly much stronger than the fixed random-but-not-eye-
filling reference.** There is no automatic numeric definition of "much stronger".
Present match evidence, board/color coverage, uncertainty, and truncations for review.
A gate result, rating, loss, waiting period, or permission to train is not approval.
Do not automatically delete or relabel accepted bundles.

Only after that specific approval:

```sh
mise run promote -- --model-store PATH/TO/STORE --candidate PATH/TO/STORE/experimental/BUNDLE --checkpoint-sha256 FULL_64_CHARACTER_HASH --approved-by USER --reason "Specific user approval and reviewed reference-strength evidence" --confirm-much-stronger-than-reference
```

The candidate must be an immutable experimental bundle owned by the store. The
command verifies checkpoint/replay/graph hashes, ONNX provenance, and eager/export
parity. Approver, reason, and reference-strength confirmation are mandatory. This
is an operator workflow, not remote-human cryptographic authentication: anyone
able to invoke it/write the store is inside the trust boundary.

A sealed, fsynced intent pins the source bundle, checkpoint, both graphs, and
approver/reason/time. Accepted bundles embed authorization; immutable intents and
receipts survive loss of `runs/`. Locks serialize approvals. Direct accepted
publication without a matching intent fails closed, including retries.

Authorization covers one hash, never an alias, lineage, or later candidates.
Only the latest committed approval may change `LATEST-ACCEPTED`; retrying an older
one is idempotent and cannot replace a newer head. Approval changes neither
training nor actor pointers. Ratings mark only the published checkpoint accepted.

To finish a publication interrupted after authorization:

```sh
mise run promote -- --model-store PATH/TO/STORE --recover
```

Recovery finishes existing authorizations and repairs publication/CSV/head gaps;
it cannot issue permission. Changed intents, graphs, or identities are rejected.
Never hand-edit intents or receipts.

Legacy v1 recovery bundles, including accepted baselines, remain readable without
fabricated retroactive approval. New accepted bundles use recovery v2 with embedded
authorization. Legacy automatic-promotion loop control files require a new fork,
not in-place migration; reusing data grants no acceptance.

## Restart learning after losing `runs/`

**ONNX alone cannot resume training.** Full packs preserve weights, BatchNorm,
optimizer/RNG state, corpus/holdouts, feature compatibility, and cumulative progress.
Choose `LATEST-TRAINING` and `LATEST-EXPERIMENTAL`, then fork with a fresh seed:

```sh
mise run loop -- --checkpoint models/training/TRAINING_BUNDLE/checkpoint.pt --actor-checkpoint models/experimental/ACTOR_BUNDLE/checkpoint.pt --bootstrap-data models/training/TRAINING_BUNDLE/replay --seed NEXT_UNUSED_SEED --warm-start-optimizer keep
```

Omit `--actor-checkpoint` only when intentionally choosing the training checkpoint
as actor. Recovery context includes prior settings and a suggested next seed; do
not reuse deleted executable paths. Build compatible binaries or deliberately fork
a new protocol. A bundle preserves learning state, not software or loop-attempt
history. Copying it off-machine does not authorize publication in another store.

For strict offline continuation, copy the checkpoint to a writable run and use the
archived corpus; never train inside a sealed pack:

```sh
mkdir -p runs/recovered-training
cp models/training/TRAINING_BUNDLE/checkpoint.pt runs/recovered-training/checkpoint.pt
mise run train -- --data models/training/TRAINING_BUNDLE/replay --resume runs/recovered-training/checkpoint.pt --epochs NEXT_TOTAL_EPOCH
```

The target must exceed the saved epoch. See [training](training.md#checkpoints-and-resume)
for relocation, supported legacy metadata, and reproducibility limits.

## Backups

Back up the whole store off-machine, including approvals, both rating CSVs, SGF
blobs, and evidence. No individual bundle preserves the global rating ledger;
see [evaluation](evaluation.md#ratings-storage). Quiesce writers while copying
CSVs; `rsync -aH` preserves hardlink deduplication.

`models/` is Git-ignored. Exclude it from `git clean -fdx` / `-fdX`. Local read-only
files do not protect against disk loss, root access, or deliberate deletion. There
is no automatic garbage collector or upload.
