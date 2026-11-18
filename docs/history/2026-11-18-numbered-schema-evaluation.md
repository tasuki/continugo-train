# Schema-5/schema-6 evaluation compatibility — 2026-11-18

## Scope and contracts

Restored schema-5 extraction/inference alongside schemas 4 and 6 so cg10 can
challenge the existing cg9 champion. Schema 5 is explicitly named `schema5`,
not described as a moving “current” or “legacy” format. Fresh training and tensor
sidecars still default to schema 6; schema-5 checkpoints cannot initialize a
schema-6 training job.

- `src/features/schema5.rs`: schema-5 names and coarse extraction, recovered from
  Git `71f8daf00bcf98775f0f3333d74b03ad5d59a622`, the parent of the schema-6 change.
  Preserves twelve planes, eleven exact fields, uncapped count normalization,
  0.30-stone-radius bridges, spatial liberty maps and the exact splitting bit.
- `src/features/schema6.rs`: named schema-6 constants/scaling; fourteen planes,
  nine fields, 0.40-radius bridges and scalar saturation at 100 remain unchanged.
- Shared raster primitives take the explicitly selected bridge width. Model
  metadata selects each arena participant's extractor independently. Schemas 5
  and 6 both retain their geometry-proposal policy; schema 4 retains complete cells.
- Python extraction, checkpoint export/viewing and rated reports understand the
  separate numbered contracts. Old schema-5 footprint-column metadata still fails.

## Verification from Git, not campaign binaries

Rebuilt feature executables directly from Git sources: `71f8daf` for schema 5,
`6e30107` for schema 4/6. No comparison against cg9's frozen executable was used.
The external rules dependency was the same checkout for reference and new builds.

`tests/fixtures/feature-schema-history.json` records the Git revisions, SGFs,
action subsets and SHA-256s of exact coarse/exact/legality/action/ownership bytes.
45 cases span schemas 4/5/6, board sizes 9/13/19, openings and dense/capturing game
positions. All arrays match bit-for-bit; schema-4 and schema-6 extraction is unchanged.
The regeneration/build receipts and source trees are in ignored
`scratch/20261118-063955-schema5-history/`.

Validation:

- 144 Rust tests passed, 15 opt-in diagnostics ignored; clippy passed.
- 108 targeted Python tests passed, including input parity, numbered-schema
  metadata checks, training/cache handling, native inference, viewer, shared ratings
  and evaluator migration tests.
- CPU mixed-schema arena integration test passes with both colors.
- Real cg10-r0010/schema-6 vs cg9-r0060/schema-5 CUDA smoke: two four-ply,
  two-simulation diagnostic games, not submitted for ratings.
- Updated one stale inference-test expectation from 32/32 to the existing 12/12
  native batching defaults; no batching settings were changed.

## Shared evaluator upgrade and cg10 recovery

`rate-saved --upgrade-runtime DIR` explicitly upgrades the evaluator's four
executables. It verifies unchanged search/match settings, backs up evaluator and
champion registry metadata under `artifacts/ratings/runtime-upgrades/`, and records
old/new protocols and hashes. It preserves the champion, ratings, SGFs, matches,
trials, seeds and family cadence. Ratings refresh selects each SGF's protocol from
that audited history using its recorded arena hash, then retains all existing
player/search/visit/cap checks. Unknown arena hashes are rejected; the disposable
parse cache records the authorized protocol set. A different runtime cannot replace one in a
pending champion trial; resuming the identical runtime is a no-op.

`include_family` validates every model's own supported contract and input widths;
the match's policy version, target format and raster density must agree, but the
feature schema need not. This does not relax the training checkpoint contract.

Cg10's existing training runtime/binaries/caches were not modified. A separate
snapshot lives at
`runs/20261117-164438-cg10-schema6/evaluation-runtime/`, with its manifest in
`evaluation-runtime.json`. The local campaign supervisor redirects only rated
subprocesses to that snapshot. Training and self-play continue using their original
frozen code and binaries.

The supervisor resumed to the already-approved 100-round total. It is backfilling
rounds 5 and 10 against the existing shared champion before continuing interrupted
round 12, then evaluates every five rounds. Both real mixed-schema challenges use the unchanged evaluation budget of 128
simulations. Round 5 finished 5–5 against cg9-r0060, so did not pass the 7/10
screen; the registry marked it complete and preserved the champion. Round 10 was
confirmed running next. No champion reset or unrequested hyperparameter change was made.

The first attempt completed those ten games, then exposed the historical-protocol
refresh issue described above. Its retry adopted the same SGFs rather than replaying
them. Evaluator Python revision 2 is frozen separately at
`evaluation-runtime/python-revision2/`; native executables did not change between
attempts. All 41 champion/ratings tests were rerun successfully after this fix,
including migration to a genuinely different arena hash, old-SGF reconstruction,
unknown-runtime rejection and idempotent resume of a pending trial.
