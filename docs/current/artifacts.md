# Artifact organization

Keep three separate top-level locations:

- **`runs/`**: continuing training campaigns, including bootstrap data, self-play,
  checkpoints, optimizer/RNG state, replay, exports, metrics and resumable controls.
- **`scratch/`**: standalone tests, audits, benchmarks, profiling and debugging.
  Short experimental training sanity checks belong here too.
- **`models/`**: selected tenth-round model snapshots plus shared ratings and
  match evidence. Campaigns do not need these copies to resume. All files are
  independent ordinary copies, never hardlinks; there is no shared SGF blob pool.

The distinction is purpose, not whether training code executes. Use UTC
`YYYYMMDD-HHMMSS-description` names directly inside `runs/` and `scratch/`, without
extra category subdirectories. Both are Git-ignored and need their own backups.

Training and loop commands default to `runs/`; standalone export, neural self-play,
arena and pipeline-profile commands default to `scratch/`. Explicit `--output`
paths are literal. Native baseline self-play defaults to stdout.

```sh
mise run loop -- --checkpoint PATH/TO/CHECKPOINT --output runs/20261106-220000-size9-training
mise run neural-selfplay -- --model PATH/TO/MODEL --games 3
mise run pipeline-profile -- --positions PATH/TO/POSITIONS --model PATH/TO/MODEL --checkpoint PATH/TO/CHECKPOINT
```

New campaigns automatically save and rate each tenth round under
`models/<campaign>-rNNNN/`. Backfill existing campaigns with
`mise run save-models -- --runs runs`; select another completed round with
`--loop PATH/TO/LOOP --iteration N`. Minimal snapshots contain checkpoint, both
ONNX graphs and JSON provenance, but no training replay.

Use `mise run retain-model -- --loop runs/CAMPAIGN/loop` only when deliberately
preserving a full checkpoint/replay recovery pack before deleting its campaign. Otherwise deleting a run removes its
unretained models from future evaluation pools; rating history and evidence stay
in `models/`. See [model storage](model-store.md).

## Historical campaigns

Eight campaigns were restored from trash; four with fewer than 20 completed rounds
were subsequently returned to trash. Five remain in `runs/`. Restoration does not
restart or make historical formats compatible with current software. Old model
stores remain in trash, unmerged.

`runs/20261106-142353-scratch-score/` is a training-from-scratch campaign, not an
ad hoc test. It remains in `runs/` and is now **stopped at the user's request**.
Its original hash-pinned launcher/verification and pre-migration source/state are
preserved; do not bypass those runtime checks to restart it.

See the [folder-cleanup receipt](../history/2026-11-06-artifact-organization.md) and
[model-storage cleanup](../history/2026-11-06-run-owned-models.md). Historical logs,
JSON and checkpoint provenance may retain former absolute paths; they were not
rewritten for organization changes.
