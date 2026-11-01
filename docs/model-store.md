# Durable accepted models and training recovery

**ONNX alone is not a training checkpoint.** It contains inference graphs/weights
and compatibility/source metadata, but not the original SGD momentum, RNG, corpus,
split history or full trainable PyTorch state. There is no supported ONNX-to-trainer
resume path. The loop now preserves those separately alongside the graphs.

## Location and protection

Default location is **`models/` inside this repository**, separate from disposable
`runs/`. There is no sibling model directory:

```text
models/
  README.md
  STORE.pt                     store ownership/format marker
  LATEST-ACCEPTED               text: accepted/<dated-bundle-name>
  LATEST-TRAINING               text: training/<dated-bundle-name>
  accepted/YYYYMMDD-HHMMSS-0001-NAME-ID/
    encoder.onnx
    policy.onnx
    checkpoint.pt              exact original trainable checkpoint
    replay/<raw-file-SHA>.sgf   exact current corpus used by that checkpoint
    evaluation/*.sgf            accepted candidate's match records, when applicable
    recovery.pt                hashed file manifest + context/report
    README.md                  per-bundle recovery instructions
  training/YYYYMMDD-HHMMSS-0001-NAME-ID/  checkpoint/replay pack for every candidate
  sgf-blobs/                   immutable store-owned copies, shared via hardlinks
  ratings/networks.csv         all networks; accepted flag, progress, per-pool rating ±RD
  ratings/matches.csv          processed SGF hashes, protocols and rating-update history
  ratings/games/*.sgf          all rating evidence, including rejected candidates
```

`--model-store PATH` can select another disk, but not the loop output or `runs/`.
A nonempty foreign directory without the ownership marker is rejected rather than
clobbered. The store has its own writer lock. Dated bundles are **write-once**:
files are hashed and fsynced, staged directories are atomically published, and
model/replay directories are made read-only. Existing packs are checked, not
rewritten. The loop never automatically deletes models or recovery packs.

Names use publication time in **UTC**, with seconds to distinguish short jobs, a
four-digit sequence for same-second publications, a readable network name such as
`cg1-b4c32-s1248000-d550347`, and a 12-character checkpoint ID suffix. See
[training counters](training.md#network-names-and-cumulative-progress) for exact
`s`/`d` semantics; unknown legacy totals are explicitly named `unknown`. They sort by date/sequence within each category. Full SHA-256 remains the
identity in `recovery.pt`; short prefixes are never trusted as identity. Republishing
the same checkpoint reuses its original dated folder rather than giving an old
model a misleading new date. The full ISO timestamp is also saved in the manifest
and README. Stable filenames inside each bundle (`encoder.onnx`, `checkpoint.pt`,
`replay/`, etc.) are intentionally not timestamped.

SGF blobs are actual copies, **not hardlinks to disposable run files**. Only links
between files owned by the durable store are shared. Deleting a run cannot delete
or mutate its archived data. Each bundle is self-contained even if copied without
the shared blob directory. Read-only mode resists accidental edits/deletion, but
the owner can deliberately chmod/delete it; it is not an access-control guarantee.

Pointers are convenience indexes to the most recently selected publication of each
kind, not a cross-experiment strength ranking. The loop's accepted state is its
own `loop.pt` pointer. A newly selected initial checkpoint is a trusted baseline,
not a model proven stronger by a match. Historical v1 auto-accepted models can be
kept as historical baselines; that does not retroactively qualify them by gating.

The [rating tables](evaluation.md#two-csvs-no-database-or-rating-json-files) are
store-level **CSV files**, not a database or JSON state. `networks.csv` includes
all archived/evaluated networks; filter `accepted=1` for the accepted history.
`matches.csv` is the committed evidence/update ledger and rebuilds the current
numeric rating view after a crash. The initial trusted network can be unrated;
acceptance does not itself establish playing strength. Ratings do not replace the gate.

`ratings/games/` is self-contained via store-owned hardlinks and survives losing
`runs/`. It includes reference and gate records even for rejected candidates.
Accepted bundles also contain available evaluation SGFs, but a single bundle does
**not** preserve the global rating table/history. Back up both CSVs and the evidence;
quiesce writers when copying them as a pair. Never discard the match ledger just
to rebuild a view, and do not edit processed hashes or rating snapshots by hand.

## What survives loss of `runs/`?

An accepted bundle preserves everything needed to **continue learning from that
accepted checkpoint**:

- Weights and BatchNorm buffers, architecture and feature semantics/density.
- SGD momentum/settings, local epoch/step counters, cumulative lineage sample/data
  counters when known, RNG states and metrics.
- Full current corpus/split manifest and ancestral holdout ledger.
- Exact SGFs, allowing strict corpus validation after relocation.
- Both inference graphs, with matching source-checkpoint identity.

But training may have advanced beyond the last accepted model. To retain rejected
or inconclusive training progress too, preserve **training/** and `LATEST-TRAINING`,
not only accepted ONNX. Completed iteration candidates are backed up before gating;
in-flight/partial epochs are not durable candidate checkpoints.

The original loop's attempt/control history, exact future trajectory and software
installation are **not** reconstructed from one model pack. Keep the code/rules
repositories and a compatible environment. GPU continuation is not promised
bit-for-bit; unchanged single-thread CPU continuation from a relocated recovery
pack is tested against the original corpus after deleting the original run.

## Restart learning after losing a loop

Find the directories named by `LATEST-TRAINING` and `LATEST-ACCEPTED`, then use a
**new** output directory:

```sh
mise run loop -- --checkpoint models/training/TRAINING_BUNDLE/checkpoint.pt --incumbent-checkpoint models/accepted/ACCEPTED_BUNDLE/checkpoint.pt --bootstrap-data models/training/TRAINING_BUNDLE/replay --seed NEXT_UNUSED_SEED --warm-start-optimizer keep
```

Use a suitable next seed and budgets (`recovery.pt` stores the old configuration,
iteration and suggested next generation seed). Training starts from the newest
checkpoint, while self-play keeps the accepted model even if they differ. Omitting
`--incumbent-checkpoint` intentionally trusts the training checkpoint as the initial
accepted baseline; don't omit it when restoring rejected training progress.
Original paths inside the checkpoint manifest may no longer exist: the archiver
and loader can use the exact relocated `replay/` files instead.

For **strict offline continuation**, copy—not move—the checkpoint into a fresh
working run directory and resume it with the archived corpus:

```sh
mkdir -p runs/recovered-training
cp models/training/TRAINING_BUNDLE/checkpoint.pt runs/recovered-training/checkpoint.pt
mise run train -- --data models/training/TRAINING_BUNDLE/replay --resume runs/recovered-training/checkpoint.pt --epochs NEXT_TOTAL_EPOCH
```

The target epoch must exceed the saved epoch. Resume keeps optimizer/RNG/options
and the same corpus fingerprint; data paths may relocate. Never train directly
inside a read-only recovery bundle. Legacy checkpoints without board-mixture
options retain their original uniform sampling on strict resume; warm-start/new
loops default to the new 40/40/20 mixture.

Existing v1/v2/v3 loop control checkpoints are intentionally not resumed under the
v4 rating/evaluation protocol. Evaluation/rating executable fingerprints are also
frozen for an active v4 loop; changing them requires a fork. Fork from their newest immutable epoch checkpoint into
a new loop rather than silently changing a frozen experiment. Old files are not
overwritten. The earlier smoke runs and old sibling archive were explicitly deleted
at the user's request for a clean start. Historical benchmark numbers do not imply
those model/game files still exist.

## Real backups still matter

`models/` is Git-ignored to avoid committing binary/data archives. Normal source
work and removing `runs/` do not remove it, but **`git clean -fdx` / `-fdX` can target
ignored data**: explicitly exclude `models/` and keep a backup before cleanup.
Read-only mode does not replace a backup or make deletion impossible.

A local directory does not protect against **disk loss, ransomware, user root access
or deliberate recursive chmod/deletion**. Back up the
whole store (plus code/rules) to another device/service; for example, an operator
can use `rsync -aH` to preserve deduplication. These backups are not uploaded or
scheduled automatically. Do not delete/unlink live lock files or prune bundles
referenced by an active loop. There is no automatic archive garbage collector.
