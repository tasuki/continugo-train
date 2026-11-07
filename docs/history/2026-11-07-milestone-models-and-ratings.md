# Tenth-round model snapshots and fresh adaptive ratings

User requested saving every tenth round from every surviving campaign, making
that automatic for future campaigns, and colouring/connecting each campaign's
rating trajectory. The user explicitly rejected hardlinks, discarded existing
ratings, and requested adaptive near-strength opponents rather than a round robin.
No training campaign was restarted.

## Saved models

Fourteen minimal snapshots were backfilled without training, exporting new graphs,
changing checkpoints or modifying campaign controls:

| Campaign | Saved local rounds |
|---|---|
| `20261104-151420-size9-training` | 10, 20, 30 |
| `20261104-215404-pass-balanced` | 10, 20, 30 |
| `20261105-092434-leaky-value` | 10, 20, 30 |
| `20261105-214158-score-ownership-size9` | 10, 20, 30 |
| `20261106-142353-scratch-score` | 10, 20 |

Directories use `models/<campaign>-rNNNN/`. Each contains independently copied
`checkpoint.pt`, `encoder.onnx`, `policy.onnx`, and readable `metadata.json`.
Metadata pins campaign, local round, full file hashes, network progress, cumulative
unique training-game count, and source runtime/config/protocol context. Checkpoint
and ONNX hashes match the committed run-owned originals. Repeated backfill verifies
existing snapshots without rewriting them; changed sources/name conflicts fail.
Snapshots do not include replay or compatible native binaries and do not grant
approval. Historical v5 exports are copied as-is, not reinterpreted by a current
PyTorch model loader.

## Automatic future behavior

New campaigns default to `--save-model-every 10 --evaluation-interval 10`.
Milestones trigger reference calibration and adaptive evaluation even if a separate
evaluation interval was chosen. Finished models are atomically saved, and the
rating plot is refreshed. Resume repairs a missed save after completion without
retraining or replaying rated games. Explicit smoke-test settings can disable
saving/evaluation; old learning/search/protocol settings remain frozen.

Default moderate precision is RD ≤ 75, at least 20 complete colour-swapped seed
pairs, and at least two opponents with four pairs each. Matches use batches of
four games and near-strength/Fisher-information selection, adapting after each
batch rather than precomputing all pairs. The extra cap is 360 games per scheduled
future candidate. Unmet evidence/precision is explicitly reported at the budget
or when compatible opponents are insufficient. Rating confidence is not approval
or a guaranteed 95% confidence interval.

## Fresh background evaluation

Old `models/ratings/`, `models/ratings.png`, and the now-obsolete `models/sgf-blobs/`
were moved intact to filesystem trash with individual `.trashinfo` records:

- `continugo-train-old-ratings-20261107-081140`
- `continugo-train-old-ratings.png-20261107-081140`
- `continugo-train-old-sgf-blobs-20261107-081140`

No historical rating CSV or match result was imported into the new ledger.
The active evaluation job is under:

`scratch/20261107-081140-fresh-milestone-ratings/`

- `run.log`, `run.pid`, `command.json`: supervisor status and invocation.
- `runtime/`: independent copy of the validated Python implementation, isolated
  from subsequent source edits.
- `evaluation/schedule.pt`: frozen restartable schedule/policy, not a pair bracket.
- `evaluation/cohort-NN/`: reference calibration state and immutable model views.
- `evaluation/cohort-NN/adaptive/progress.json`: adaptive precision/evidence progress,
  once that phase starts.
- `old-ratings-trash.json`: exact retired-artifact locations.
- `legacy-preflight/`: two separate bounded real native games used only for runtime
  validation, not imported as rated results.

There are two compatible feature-schema cohorts: nine policy/value-only snapshots
and five score/ownership snapshots. Each cohort uses the newest compatible
campaign's frozen native runtime, fixed 32-simulation evaluation, and size-9 play.
Incompatible schemas never play each other. Legacy v1 rating protocols are validated
without fabricating a score-weight field; newer v2 protocols retain their frozen
score utility. Separate runtime groups share the fixed random reference, but
cross-protocol comparisons still require caution.

All 14 models receive ten fresh reference games each before adaptive refinement.
The adaptive extra cap is 360 per cohort: **860 games maximum**, not a requirement
to exhaust the budget. It stops early when the moderate evidence/precision goals
are met. The reference-first pass gives every campaign visible calibration points
before refining either cohort. The job remains running in the background; no
claim that final confidence is already met is implied by saving or calibration.

The supervisor was cleanly interrupted once to install the final validated plotting
and calibration-order implementation, then resumed with its original seeds,
committed results and valid partial-attempt SGFs. Completed games were not replayed;
the log records adoption and the updated PID. Use `run.pid`, not an old copied PID.
To resume after another clean stop, use the recorded command with
`PYTHONPATH=scratch/20261107-081140-fresh-milestone-ratings/runtime` and `--resume`.
Never bypass an active OS lock.

## Plot and independent-file storage

`models/ratings.png` updates after committed batches. Campaigns get stable distinct
colours, using saturated hues before lighter variants; lines connect rounds only
within a campaign. The x-axis remains ancestral unique training games, excluding
validation and repeated replay/epochs. Error bars remain ± rating deviation. No
point or strength estimate is invented for an unscored model.

All model snapshots and SGF evidence are ordinary independent files. Evidence goes
directly into `models/ratings/games/`; new writes never create `sgf-blobs/` or any
hardlinks. Full-recovery publication and approval verification also use copies.
Every file currently under `models/` was checked to have link count one. Retired
artifacts in trash were not rewritten or destroyed.

## Validation

- Complete Python suite: **217 tests passed**.
- Plot-specific regression suite: **8 tests passed** after the final palette update.
- Ruff lint/format and Git whitespace checks passed.
- Fourteen saved checkpoint/graph sets verified against run records, with independent
  inodes, preserved hashes, readable provenance and idempotent backfill.
- Real legacy native preflight: two paired SGFs passed strict v1 protocol validation.
- Real background v2 calibration generated and committed fresh games and plots.
- Tests cover automatic save/evaluation boundaries, crash/resume repair, v5 backfill,
  corruption/name conflicts, no-hardlink evidence/publication, legacy protocol CSV
  round trips, adaptive stopping/resume and per-campaign colours/line ordering.

Logs are preserved alongside the background job. Rust gameplay, network architecture,
training objectives and the stopped campaigns' checkpoints were not changed.
