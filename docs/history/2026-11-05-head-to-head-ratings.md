# Replace random-only measurement with rated head-to-head evidence

**Current policy:** adaptive matching, not the initial round-robin. The fixed
schedule was retired after 26 games at the user's request; do not resume its
unplayed matchups. See the adaptive continuation at the end of this receipt.

The user noticed the newest three entries in `models/ratings/networks.csv` all
had 20 wins, zero draws/losses and the same 431.1 ± 75.8 rating. Inspection found
**240 recorded games, all against `random-no-eye-fill-v1`, zero neural-vs-neural
matches**. The campaign used `actor_policy=latest` and `gate_games=0`. Always
advancing the actor does not require disabling head-to-head measurement; the
random-only schedule had saturated and could not compare these checkpoints.

## Future scheduled evaluation

The leaky-value campaign was already paused at local round 30 (target remains
60), with no live training process. With the user's approval to fix evaluation:

```sh
.venv/bin/python -m continugo_train.loop --output runs/20261105-092434-leaky-value/loop --resume --enable-head-to-head 20 --stop-after-iteration 30
```

This saved `gate_games=20`, keeping `actor_policy=latest`, interval ten, 32 search
simulations, size 9 and the existing 1,024-move neural match cap. Reference games
remain enabled. No new training round was started; the actor/checkpoint, completed
iteration evidence, frozen binaries and total target were retained. The exact
pre-change control file is `loop/before-head-to-head.pt`; `config_changes` records
that the setting takes effect after iteration 30. Resume the campaign normally
when the user requests further training. New scheduled candidates will have
rated matches against their frozen parent actors without gating actor advancement.

## Backfill tournament (launched, not yet final)

Added `continugo_train.round_robin` / `mise run round-robin`, reusing the existing
arena supervisor's SGF validation, interrupted-game adoption, process locking,
hash-pinned model records and idempotent ratings ingestion. The source loop's
settings are read-only; inference graphs and checkpoints are verified and copied
into the evaluation run, without copying training replay or publishing models.

Started on CUDA at 20:00:49 UTC, with fresh seed 9,200,000:

```sh
.venv/bin/python -u -m continugo_train.round_robin --output runs/20261105-200049-rated-round-robin --loop runs/20261105-092434-leaky-value/loop --latest 4 --games 20 --seed 9200000
```

Four checkpoints, in registry order:

| Samples | Full checkpoint identity |
|---|---|
| 732000 | `581c83e7fb4d470be41b003e0edb396141b222db82665a093e29a2500eca7383` |
| 652000 | `69cab5431dfe7b899858b9cddf72035bf997b7fea944bd8361f42e2a2741f151` |
| 572000 | `79f776a011e48792c45e486a253d3cfda3458afab2fac5565a2f131209411fd2` |
| 492000 | `ab421f4602ac0daba1c9dca0b9c11f283f0fd981991a5ecc290721d900d62a06` |

Each unordered pair plays 20 games (ten same-seed color pairs), six matchups and
**120 planned games**. Budget: 32 simulations per side per move, size 9, maximum
root visits, existing frozen native binaries, 1,024-move safety cap. Natural draws
are draws; capped games are unknown, never invented labels.

The detached supervisor PID receipt is `runs/20261105-200049-rated-round-robin.pid`;
its log is the adjacent `.log`. Each completed matchup updates the shared CSV
ledger and rating plot. The process lock is inherited by the native arena child.
If interrupted, first confirm the supervisor/child have stopped, then resume with:

```sh
mise run round-robin -- --output runs/20261105-200049-rated-round-robin --resume
```

Do not call an unfinished tournament a final ranking. SGFs and per-match reports
remain under its evaluation directory; committed results are authoritative in
`models/ratings/matches.csv`. Training and accepted publication remain unchanged.

## First committed head-to-head result

The first matchup completed naturally: **s732000 lost 5–15 to s652000**, with no
draws or truncations. All 20 games were ingested as event
`79bab50e4a8e0f98d957346659ae42b5304f4d4184d865d266eb1b16395ebcfc`.
Their ratings changed from identical 431.1 ± 75.8 to:

- s732000: **346.4 ± 55.0** (40 total games, 25 wins / 15 losses).
- s652000: **515.8 ± 55.0** (40 total games, 35 wins / 5 losses).

This is a measured regression against that opponent at this budget, not a final
ranking across the whole pool. Matchup two (s732000 versus s572000) is running;
five matchups remain. The remaining ratings will update as their games finish.

Validation: all 169 Python tests passed, including all-pair ratings ingestion,
idempotent resume, partial-game recovery, unchanged source-loop state, live-lock
and changed-graph rejection, seed overflow checks, and safe boundary-only enabling
of future head-to-head matches. Ruff check/format passed.

## User-requested adaptive matching (supersedes fixed tournament)

The user preferred suspected similarly rated opponents, reselection after results,
and substantially more confidence than the initial twenty-game batches. Sent
SIGINT to the verified round-robin supervisor, waited for it and its native arena
child to exit, then ran:

```sh
mise run round-robin -- --output runs/20261105-200049-rated-round-robin --resume --ingest-only
```

This launched **no new games**, retained the first 20-game batch unchanged and
ingested six completed games from its interrupted second matchup as event
`dd305bcc3fe293ef734c9fa277c1f4ccbc7798542339e6ca7d03b7308c380d71`.
All 26 finished games are retained; no outcomes were discarded or manufactured.
The round-robin's remaining schedule is retired.

Started the new adaptive evaluator at 20:20:21 UTC:

```sh
.venv/bin/python -u -m continugo_train.adaptive_ratings --output runs/20261105-202021-adaptive-ratings --loop runs/20261105-092434-leaky-value/loop --latest 3 --target-rd 25 --min-pairs 60 --min-opponents 2 --min-opponent-pairs 10 --batch-games 4 --max-games 600 --seed 9400000
```

The three formerly tied checkpoints are the precision targets. All eleven
registered neural checkpoints form the compatible, snapshotted opponent pool.
After each **four-game batch**, update ratings and choose the next pair using
expected Glicko variance reduction. Close strength, opponent certainty, remaining
target uncertainty and opponent diversity matter; truncated matchups are penalized.
No random opponents or precomputed all-pairs schedule are used.

Every target must satisfy **RD ≤ 25**, **60 complete distinct color/seed pairs**
(120 neural games), and **two opponents with at least ten pairs each**. Existing
compatible evidence counts, including the salvaged games. Unknown outcomes and
incomplete/repeated-color seed pairs do not satisfy the evidence floor. This is
an explicit model-based precision/evidence policy, **not a guaranteed 95% CI** or
a forced ordering of close models. The hard cap is **600 additional games**;
exhausting it without meeting all conditions is reported as a failed precision
target, not as confidence achieved.

Initial standings after salvage were s732000 371.8 ± 51.6, s652000 515.8 ± 55.0,
and s572000 387.0 ± 67.5. The first adaptive matchup selected s732000 versus
s572000 (predicted first-model score 0.479); it finished **1–3** and committed four
more games. The next selection changed to s652000 versus s572000 (predicted 0.648).
This demonstrates actual result-driven reselection, not just a different fixed
schedule. These are intermediate standings, not final confidence claims.

Log: `runs/20261105-202021-adaptive-ratings.log`. PID receipt: adjacent `.pid`.
Live precision/evidence counts: `runs/20261105-202021-adaptive-ratings/progress.json`.
If interrupted, confirm the supervisor/child are stopped and resume only this run:

```sh
mise run rate-models -- --output runs/20261105-202021-adaptive-ratings --resume
```

The source training loop stays paused at round 30, with the same actor, checkpoint,
optimizer, binary hashes and total target. Its ordinary future 20-game parent
matches remain enabled as progress diagnostics; they do not claim to meet this
stronger adaptive precision policy.

Validation after adaptive implementation: **180 Python tests passed**; Ruff
check/format and diff whitespace checks passed. Tests cover near-strength matching,
changed-opponent selection after ratings move, uncertainty and truncation penalties,
opponent-diversity requirements, complete paired-seed counting, all stopping
conditions, budget exhaustion without a confidence claim, interrupted-batch resume,
idempotent ingestion, policy freezing and unchanged training control state.

## Automatic adaptive evaluation for future scheduled candidates

The user asked whether the fix also covered future checkpoints. At that point,
only ordinary 20-game head-to-head diagnostics were automatically scheduled; the
stronger adaptive backfill was a separate run. This gap is now fixed.

Added a loop-integrated adaptive phase after each scheduled candidate's ordinary
matches and immutable inference archive, before round completion/actor commit.
It pins the **exact candidate checkpoint ID**, avoiding registration-order mistakes
when a parent actor is registered later. Opponents adapt after each four games;
resuming a pending evaluation does not retrain, reexport or repeat committed games.
The native arena child inherits both the parent-loop and evaluator process locks.
The adaptive SGFs stay outside training replay and enter the shared rating ledger.

All precision/evidence settings are explicit frozen loop configuration fields.
At a completed boundary, applied to this campaign with:

```sh
.venv/bin/python -m continugo_train.loop --output runs/20261105-092434-leaky-value/loop --resume --enable-adaptive-ratings 600 --stop-after-iteration 30
```

Verified the pre-change control snapshot at `loop/before-adaptive-ratings.pt` and
compared before/after: training checkpoint/state, actor, accepted publication, all
30 completed iterations and target 60 are unchanged. The audit trail records
`adaptive_max_games: 0 -> 600` after iteration 30. Training remains paused. The
currently running standalone backfill was not interrupted or reconfigured.

On ordinary future resume, **rounds 40, 50 and 60** automatically evaluate their
new candidates to RD ≤ 25, 60 complete color/seed pairs, and two opponents with ten
pairs each, or report a clearly unmet target at the **600-additional-game cap per
candidate**. A distinct high-bit seed stream avoids gate/reference/backfill seeds.
A too-small opponent pool is explicitly reported as insufficient rather than
inventing confidence or blocking latest-actor advancement. Neither a low rating
nor an unmet precision target rewinds training or changes `actor_policy=latest`.
At this stage new loops still opted in with `--adaptive-max-games 600`; the next
section supersedes that setup default. Old loops are not silently upgraded into
expensive extra evaluation.

Validation: **186 Python tests passed**. Added actual loop-to-evaluator integration
coverage, explicit-target selection, older-v5 disabled-default compatibility,
future-boundary enabling, automatic interruption/resume without retraining, dual
lock inheritance, insufficient-opponent handling and unchanged latest-actor
advancement even when the precision target remains unmet.

## Adaptive precision is now the default for all new campaigns

The user clarified that the operating policy must cover future campaigns, not
just this saved campaign. Changed `LoopConfig.adaptive_max_games` from 0 to **600**.
New loops now automatically run adaptive evaluation on scheduled candidates with
RD ≤ 25, 60 color/seed pairs, and two opponents with ten pairs each; no extra
enabling flag is needed. Latest-actor selection remains the default and unchanged.

`--adaptive-max-games 0` is an explicit opt-out. Smoke tests and the documented
no-evaluation example now explicitly disable adaptive games as well as gate and
reference games. Existing saved loops restore their saved policy; missing adaptive
fields in older v5 control files explicitly migrate to budget zero rather than
silently inheriting the new enabled setup default. This campaign already has the
enabled policy, and the running adaptive backfill was not interrupted.

Added an integration regression test that **omits** the adaptive-budget CLI option
and verifies the new campaign automatically executes adaptive matching and saves
budget 600. Old-loop compatibility and explicit smoke-test opt-out remain covered.
All **187 Python tests passed**; Ruff checks/format passed. README and current loop
documentation now describe adaptive precision as the new-campaign default.
