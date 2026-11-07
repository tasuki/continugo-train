# Run-owned model storage; campaign stopped

User requested removing mandatory per-round recovery copies from `models/`, then
explicitly requested stopping the current campaign and resolving/removing its
related TODOs. No campaign was restarted.

## Campaign status and provenance

`runs/20261106-142353-scratch-score/` was interrupted cleanly through its supervisor
with SIGINT. The supervisor and loop child exited. It has **21 completed rounds**
(1,050 generated games in completed rounds, plus its initial bootstrap/training).

`storage-migration/` inside the campaign preserves the original Python source,
launcher, plan, verification, source snapshot, pre-pause and paused loop controls,
and original supervisor command. The loop also preserves
`before-run-owned-storage.pt`. Weights, optimizer/actor checkpoint identities,
seeds, config, native executable fingerprints and iteration completion states
were compared after migration and remained unchanged.

The original launcher's pinned verification/plan were not rewritten to bypass
source checks. It must not be relaunched blindly under changed Python code.
The campaign README records the stop; no restart is planned.

## Storage behavior

- Checkpoints, replay, exports and training/actor selection remain in campaigns.
- Loops no longer create or verify global training/experimental recovery copies
  or move their global latest pointers.
- `--resume --migrate-storage` detaches old v6 archive references, backs up the old
  control file, verifies run-owned artifacts and indexes already-registered
  evaluated models. It launches no jobs, does not import historical networks into
  a replacement ledger and does not upgrade old v1–v5 training protocols.
- Both retained v6 campaigns were storage-migrated without restarting training.
  The three historical v5 campaigns and their frozen software were left alone.
- Rating opponents use hash-pinned run-owned checkpoint/graph locators, or
  deliberately retained/accepted bundles. Missing unretained models are excluded
  from future opponent pools; modified artifacts fail verification.
- Rating history, match SGFs and cached training-game counts survive deletion of
  a run, so plots no longer require redundant archived checkpoints.
- `mise run retain-model -- --loop PATH/TO/LOOP [--iteration N]` explicitly retains
  one completed model under `models/retained/`, without acceptance or actor changes.
  Approval still requires the exact hash and explicit user permission; legacy
  experimental candidates remain supported.

## Existing store cleanup

All **44** redundant bundles (22 training + 22 experimental) belonged to the cg4
campaign. Each manifest was verified, and every checkpoint, ONNX and replay/match
SGF hash had an identical run-owned file before cleanup.

The following were moved intact to `/home/vita/data/.Trash-1000/files/`, with
individual `.trashinfo` records, using the suffix `20261106-234430`:

- `models/training` → `continugo-train-models-training-20261106-234430`
- `models/experimental` → `continugo-train-models-experimental-20261106-234430`
- `models/LATEST-TRAINING` → `continugo-train-models-LATEST-TRAINING-20261106-234430`
- `models/LATEST-EXPERIMENTAL` → `continugo-train-models-LATEST-EXPERIMENTAL-20261106-234430`

Device/inode/size/mtime inventories matched after each rename. The campaign's
`storage-migration/model-backups-trash.json` records exact destinations. No
campaign model or shared match evidence was deleted. The original store README
was preserved before replacing its obsolete instructions.

The SGF blob pool was also checked against all shared match evidence. Its 1,433
unreferenced replay blob paths (19,019,967 bytes) had identical run-owned SGFs and
remaining recovery-bundle hardlinks in trash. Those paths were moved, preserving
their day/hash layout, to
`continugo-train-models-unused-sgf-blobs-20261106-234835` in the same trash, with a
`.trashinfo` record. All 440 match-evidence blobs and their ledger hardlinks remain
in `models/`. The campaign's `storage-migration/unused-replay-blobs-trash.json`
records that partial blob-pool cleanup. No automatic garbage collector was added.

A second storage-only resume after removal verified that those directories and
pointers were not recreated. Five registered models remain available from run-owned
sources. The shared plot was successfully regenerated for four rated networks.
No model was explicitly retained or approved as part of this cleanup.

## Defaults, documentation and TODO cleanup

Standalone export, neural self-play, arena and profiling now allocate dated output
under `scratch/`; training/loop defaults stay in `runs/`, and explicit paths remain
literal. Model/approval stores are rejected beneath either disposable directory.
CLI help, examples and current storage documentation were updated.

Removed the now-implemented output-default TODO and the stale campaign-review item
that incorrectly described an old campaign as still running. Removal of the latter
records retirement of that task, **not** a claim of measured playing strength.
Unrelated future performance/integration work remains in `docs/todo.md`.

## Validation

- Complete Python suite: **210 tests passed**.
- Focused storage suite including an additional replacement-ledger regression:
  **9 tests passed**.
- Ruff lint/format and `git diff --check` passed.
- Real stopped-campaign storage-only migration, local artifact verification,
  unchanged training/config/native fingerprints, opponent discovery and ratings
  plot regeneration passed after the old global copies were trashed.

Logs: `scratch/20261106-234430-storage-cleanup/`. Tests cover interruption/resume,
run-owned evaluation, explicit retention/approval, missing/corrupt model sources,
legacy storage detachment, unavailable opponents, durable rating history/plots,
foreign/disposable-store guards and separated output defaults. Rust gameplay and
training/search algorithms were not changed.
