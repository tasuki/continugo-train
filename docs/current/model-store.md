# Artifact storage

Training owns its checkpoints, replay and intermediate exports under `runs/`.
The durable store is **`artifacts/`**, not the former `models/` root.
For this evaluation, only **cg5** is included; cg1–4 remain outside the active store.

```text
artifacts/
  models/cg5/r0001/ ... r0100/
  ratings/
  sgfs/YYYY-MM-DD/
```

Each model directory contains exactly `checkpoint.pt`, `encoder.onnx`, `policy.onnx`
and `metadata.json`. Evaluation references these files directly: no second model
copy, evaluation-specific model tree, job directory or compatibility-cohort directory.
The original training files remain in the run; one explicitly saved durable model
copy is intentional, copying it again for evaluation is not.

## Model metadata

Example for `models/cg5/r0001/metadata.json`:

```json
{
  "family": "cg5",
  "round": 1,
  "network_name": "cg5-b4c32-s18000-d22112",
  "progress": {
    "training_games": 441,
    "samples_seen": 18000,
    "data_positions": 22112
  },
  "source": {
    "run": "20261108-154636-schema5-scratch",
    "iteration": 1
  }
}
```

No hashes, format version, file manifest or machine-specific absolute paths are
stored here. Ratings, champion status and evaluation budgets are separate mutable
concerns, not model metadata. Native model/export identity checking still operates
internally when loading a checkpoint or running a match.

Saving uses independent files, atomic directory publication and idempotent checks
against the training source. There are no hardlinks. Saved files are read-only to
resist accidents; owners can explicitly make them writable for maintenance.

```sh
# The loop saves every tenth round by default, without evaluating it.
# Explicitly save every completed round of cg5 for the retrospective ladder:
mise run save-models -- --loop runs/20261108-154636-schema5-scratch/loop --every 1
```

`--save-model-every 0` disables automatic saving. Saving does not register ratings,
run games, change champion or publish a model. Minimal saved models do not copy
training replay; keep the source run for exact recovery.

## Ratings and SGFs

The [shared evaluator](evaluation.md) uses `ratings/evaluator.pt` and retains
`ratings/evaluator.log` (gitignored). `networks.csv` is a rebuildable summary;
`champions.json` records the automatic champion lineage. SGFs are the authoritative
game records; there is **no `matches.csv`**. Each game and its `.nn.html` report live
once under `sgfs/YYYY-MM-DD/`, named Black-vs-White with the shared pair seed.
Refreshing `ratings/ratings.png` never removes dated plots.

## Approval operation

Champion qualification automatically changes the **evaluation champion only**.
It grants no permission to publish an officially accepted model, and does not
change the training actor. Publication remains a separate explicit user decision;
its future presentation/UI is not defined by the ratings storage layout.

The existing explicit retention/publication tools remain separate, opt-in recovery
operations. Their sealed full-recovery bundles include replay and authorization;
no such bundles or publications are created by saving or rating cg5.

## Recovery and backups

Resume the shared evaluator with `mise run rate-saved -- --resume`. Frozen runtime,
search budget and seed reservations remain in its single checkpoint. Completed
SGFs are adopted rather than replayed. `mise run ratings` can discard the rating
cache and rebuild the summary from the actual SGFs.

Training resume continues to use run-owned files, not evaluator model copies.
Changing generation/feature binaries still requires a compatible new campaign.
ONNX alone cannot resume training: keep the training checkpoint and its source run.

Back up `runs/` and `artifacts/` off-machine. Both are gitignored. Stop active
writers before copying checkpoint/state files. There is no automatic upload or
backup service. The former layout is preserved separately under `scratch/` for
migration recovery; it is not an active model/rating source.
