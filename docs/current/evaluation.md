# Evaluation and ratings

Training saves models; the **one shared evaluator** rates them independently.
For this campaign only **cg5 rounds 1–100** are included. cg1–4 are not imported.

Ordinary arena/rated games default to fixed **komi 7** for both colour-swapped games
and write root `KM[7]`. Explicit cross-lineage comparisons may choose another fixed
komi (see below). Training variation never applies to ratings. Historical results
remain unchanged; the frozen arena binary hash distinguishes the new scoring
protocol. See [komi](komi.md).

## Artifact layout

```text
artifacts/
  models/cg5/r0001/
    checkpoint.pt
    encoder.onnx
    policy.onnx
    metadata.json
  models/cg5/r0002/ ... r0100/
  networks.csv
  champions.json
  ratings.png  # generated, gitignored
  ratings/
    evaluator.pt
    evaluator.log
  sgfs/2026-11-10/
    cg5-r0001-vs-random-seed97000000.sgf
    cg5-r0001-vs-random-seed97000000.nn.html
    random-vs-cg5-r0001-seed97000000.sgf
    random-vs-cg5-r0001-seed97000000.nn.html
```

Black is first in each game filename. A shared seed produces two games with swapped
colors, so both filenames are needed. Each model and each SGF is stored **once**.
There are no evaluator model copies, job/cohort directories, or `matches.csv`.
Small hidden store/lock files provide process safety; they are not extra model stores.

`evaluator.log` is retained and gitignored along with the local artifact store.
The evaluator never removes dated PNGs when refreshing `ratings.png`.

## Champion policy

The initial champion is **random**, fixed at rating **100** with deviation zero.
By default, actual rounds **5, 10, 15, …** are evaluated, in chronological order.
A family-specific override can evaluate/challenge every saved model without
resetting the shared champion or revisiting another family's skipped rounds:

```sh
mise run rate-saved -- --resume --include-family cg6 --family-evaluation-every 1
```

`--include-family` appends compatible saved models to the existing queue; repeat
it as new models are saved. The cadence override is persisted and audited for
that family only. cg6 currently requests every-round challenges.

All other rounds skip **everything**, including random calibration and neural
reports. An off-cycle partial batch is abandoned; its completed SGFs are kept,
but no more games or reports are generated for it. Completed historical results
from earlier policies remain valid and are not removed. Off-cycle skips are derived
from the cadence, not recorded as per-network entries in `champions.json`.
The registry retains decisions, champion history and recovery metadata.
`policy_changes`, `runtime_changes` and `family_evaluation_every` live only in
`evaluator.pt`, not in `champions.json`.

1. Each eligible candidate plays **10 games against the reigning champion**.
   At least **7 wins** permit continuation.
2. It plays **30 more games against that same frozen champion**, with fresh paired seeds.
   At least **28 wins across the full 40 automatically replace the champion**.
3. **Only after promotion is committed**, the new champion plays **50 games each
   against up to three history opponents**: the defeated champion, plus up to two
   other available, previously evaluated opponents nearest its rating. Random is
   eligible for these remaining slots, not a separate mandatory calibration batch.
   The defeated champion gets a fresh history batch in addition to the challenge.
   These extra games calibrate ratings; they do not veto promotion or reset it.

`POLICY` in `rate_saved.py` is the single source of the six policy defaults.
`win_rate_required: 0.7` applies to both the screen alone and the combined
screen + extension: with 10 + 30 games, the thresholds are 7/10 and 28/40.
Every evaluated round challenges the champion; there is no separate challenge cadence.
The historical store's existing games and promotion history remain unchanged.

Match reports contain counts, scores and truncation bounds, not a generic
50% acceptance threshold or passed/failed label. Only the 7/10 and 28/40 win
requirements above decide challenge continuation and champion replacement.
Historical reports remain evidence; legacy `acceptance_rate` config is ignored.

All neural turns use **128 simulations** by default, including random calibration
and historical matches. The random bot is direct random play, not MCTS. This budget
is half cg5's expensive policy-training budget of 256. Evaluation never randomizes
its search budget. New shared evaluators cap both neural matches and direct-random
reference matches at **512 moves**. Existing evaluators retain their frozen caps.
A cap is a safety limit, **not** a forced-scoring rule.

Draws and truncations are **non-wins for promotion**. The denominators stay 10 and
40. Ratings give natural draws half-points; truncations have no outcome and do not
update skill. This is a practical filter, not a claimed confidence level; the small
screen can reject genuine improvements. All planned games are played.

Champion selection never changes the training actor or publishes an accepted model.
Publication remains a separate explicit user decision; see [model storage](model-store.md#approval-operation).

## Numbered schemas and evaluator runtime upgrades

A match may pair schema-4, schema-5, schema-6 and schema-7 models. Each model's metadata
selects its own channel names, scalar normalization, geometry and exact-field
width. Policy version, target format and raster density must agree. Schema-5
inputs are recovered from Git commit `71f8daf` (not a frozen campaign binary);
schema-6 inputs remain unchanged. Schema 7 adds three recent-move planes after
occupancy, with unchanged schema-6 geometry/exact fields. Regression fixtures
compare schemas 4/5/6 against Git history and preserve their bits under schema 7.

The shared evaluator can be upgraded explicitly without resetting the champion,
ratings, completed trials or SGFs:

```sh
mise run rate-saved -- --resume --upgrade-runtime PATH/TO/NUMBERED-SCHEMA-BINS \
  --include-family cg10 --family-evaluation-every 5
```

The directory must contain `continugo-train`, `features`, `arena` and `rating`.
An upgrade refuses an outstanding champion evaluation; finish it with its frozen
runtime first. It backs up evaluator/registry metadata under
`ratings/runtime-upgrades/`, records old/new executable fingerprints/protocols,
and preserves all search and match budgets. Existing SGFs keep their recorded
runtime identities. Ratings refresh validates each SGF against the matching arena
protocol recorded in the evaluator checkpoint's audited upgrade history; unknown arena hashes
and changed search/match budgets are rejected. The disposable parse cache is keyed
by that authorized protocol set as well as the file signatures. Repeat the same
directory safely: an identical upgrade is a no-op (including during a pending trial). Evaluation code must also support the numbered schemas; changing executable
paths alone is insufficient.

## Commands and recovery

```sh
# Save every completed cg5 candidate, not just tenth-round milestones:
mise run save-models -- --loop runs/20261108-154636-schema5-scratch/loop --every 1

# Create the shared evaluator, or resume its frozen queue:
mise run rate-saved -- --family cg5 --simulations 128
mise run rate-saved -- --resume

# Rebuild the rating summary from the SGFs:
mise run ratings -- --model-store artifacts/20261122
mise run plot-ratings
```

`--model-store PATH` is required to select a lineage artifact root; there is no
default lineage. There is
no per-study `--output` directory. One inherited OS lock serializes the evaluator
and its arena children. The queue, frozen runtime/protocol, seed reservations,
partial-match progress and disposable rating cache live in `ratings/evaluator.pt`.
A restarted evaluator adopts completed SGFs and plays only missing games. It does
not retrain, export or copy models. Seeds are reserved before launching a match;
a crash may waste seeds but cannot silently reuse them. Promotion decisions and
moving champion head commit together in `champions.json`. Policy changes are audited;
already-completed results and the existing champion are preserved when the cadence changes.

Native arena files are renamed to Black-vs-White names as they are adopted, including
a clean interruption. Invalid partial files are kept with an `.incomplete-N` suffix;
they are not rated or treated as draws. Neural reports use each mover's own model;
random turns have no neural model. Native game/model identities remain checked by
the engine internally; public model metadata and filenames contain no content hashes.

## Reusable cross-lineage comparisons

Prepare a frozen schedule without playing any games:

```sh
mise run compare-lineages -- prepare --name 20261122-vs-20261111-k0 \
  --lineage-a 20261122 --lineage-b 20261111 --latest 3 --games 40 --komi 0
```

This selects the latest three **promoted neural champions** in each lineage,
chronologically, and schedules the Cartesian product: nine pairings, 360 games.
Each pairing uses 20 fresh paired seeds, alternating candidate Black/White by game
ordinal. Board mix, search budget, device and move caps come from lineage A's
existing evaluator. Default komi is 0; supply another fixed value explicitly.
Current numbered-schema inference supports each model's own input schema;
legacy score heads need no compensation adjustment at komi 0. Original
separate-head schema-4/5/6 models are supported read-only, without re-export or
weight migration. Their score output is a point estimate: score utility uses
zero predictive deviation, not a fabricated learned uncertainty. The viewer
uses their actual separate value, score and ownership heads.

Preparation reserves fresh shared-root seeds, fingerprints model references,
and copies/fingerprints a private executable runtime under
`artifacts/ratings/comparisons/NAME-runtime/`. Models are never copied or
re-exported. The restart checkpoint is `artifacts/ratings/comparisons/NAME.pt`.
Preparation also loads every model with native `arena --check-models` and verifies
that each checkpoint is supported by the HTML viewer. Load-only checks exit before
creating output or generating games. Preparation does not alter champions, train
anything, or start games. After approval:

```sh
mise run compare-lineages -- run --name 20261122-vs-20261111-k0
# Repeat the exact same command to resume after interruption.
```

Run refuses changed models/runtime and concurrent evaluators for those lineages.
Already adopted games are verified; interrupted attempts adopt completed files
and play only missing ordinals. SGFs live **once** in
`artifacts/sgfs/YYYY-MM-DD/Black-vs-White-seedN.sgf`, with `.nn.html` reports
rendered from each mover's own checkpoint. Date directories are frozen per job
when it starts. Finished games are not rescored or moved between stores.

After each completed pairing, ratings are fitted **jointly** from all historical
lineage SGFs plus all authorized shared-root SGFs. This refits both lineages,
including old opponents—not merely the six selected participants. Shared games
must match a frozen manifest's players, seed, search/runtime, caps and komi;
historical games retain their recorded rules/results and are revalidated by a
compatible **historical** inspector, also copied and fingerprinted at preparation.
Preparation probes one archive per recorded arena runtime, trying the current
inspector and then available source-campaign inspectors. Override discovery with
`--historical-inspector LINEAGE=PATH` (repeatable). This matters for `20261111`:
its older board-point units must not be rescored with the current runtime.
Pruned historical players are identified from archived
SGFs and the existing network registry, without resurrecting model artifacts.
The disposable parsing cache is `artifacts/ratings/joint-cache.pt`.

The canonical summary is **`artifacts/networks.csv`**, with an additional `lineage`
column. There is one all-games fit per refresh, even before any cross-lineage
matches exist. Ordinary rated batches also refresh this same fit. Per-lineage
`networks.csv` files are filtered views of it, with identical ratings, deviations
and game counts—not independent baseline fits or extra evidence. The evaluator's
rating cache also uses this fit for rating-proximity opponent selection.

Plot membership is unchanged: `artifacts/ratings.png` shows promoted champions;
each lineage's `ratings.png` shows all its evaluated iterations. Every lineage's
plot is refreshed because new evidence can change old opponents' ratings.

Frozen campaigns retain their copied Python implementation. Working-tree changes
do not hot-update an active evaluator. Migrate only at a checkpointed campaign
boundary, preserving evaluator state, champion decisions, frozen match jobs and
SGFs; do not edit runtime fingerprints underneath a running campaign.
To rebuild explicitly:

```sh
mise run ratings -- --artifacts-root artifacts
```

**Statistical limitation:** the fit pools historical komi-0/komi-7 games and new
comparison games into one strength per model. It does not estimate separate
komi-specific strengths or colour effects. Colour-swapped pairs balance playing
colour; they do not eliminate possible komi-dependent model strength. No promotion
or champion history is changed by these games or by the joint fit.

## Champion head-to-head table

`artifacts/champion-wins.csv` is rebuilt alongside `ratings.png`, both after rated
batches and by `plot-ratings`. Rows and columns start with `random`, followed by
neural champions ordered by lineage name and then promotion order. Labels show
only the model name (e.g. `cg11-r0055`), without the lineage. Cell `(A, B)` counts A's recorded wins against B across both colours;
cell `(B, A)` counts the reverse. Draws and truncations add no wins. Blank means
no recorded pairing (or the diagonal); zero means games exist but no wins.
Both lineage-local SGFs and shared `artifacts/sgfs/` games are included.
These are raw SGF counts, not fitted ratings or win percentages. Different komi
or runtime settings are not separated in this table.

`artifacts/champion-wins.html` is an ignored, self-contained offline view generated
with the CSV. Open it directly in a browser: the diagonal is black, headers stay
visible while scrolling, and hovering or keyboard-focusing a cell highlights its
mirror and shows both champions' win counts. Cells display row wins divided by
both champions' combined wins (decisive games only), coloured pink–neutral–green
with 50% neutral. Draws and unfinished games are excluded; no decisive games
means a dash, not an invented percentage. The CSV retains raw win counts.
Click a cell (or press Enter/Space when focused) to list that pairing's games
beneath the matrix, alphabetically by filename. SGF and available neural-report
links open alongside project-relative paths; both lineage-local and shared-root
games are included, even draws and unfinished games. Refresh the browser after
a ratings update to load new counts and links.

## Ratings, not another game ledger

**SGFs are authoritative.** `networks.csv` is a derived summary:

```csv
network,rating,deviation,games,wins,losses,draws,truncated
```

It contains readable names such as `cg5-r0001` and `random`, not hashes or absolute
paths. Existing rows keep their order when ratings are refreshed; newly observed
networks are appended at the bottom in first-evidence (paired seed) order, with
random first. A rebuild without an existing CSV uses that same evidence order.
It can be regenerated by scanning `sgfs/`. The evaluator checkpoint caches
unchanged file signatures and validated results only to accelerate parsing; it is not a
second authoritative game ledger. `mise run ratings` discards that cache and rebuilds from SGFs.

The shared backend jointly fits all completed games with a Bradley–Terry logistic
model (400 rating points = 10:1 odds), fixing random at 100. Each observed opponent
pair gets half a win and half a loss of smoothing, keeping unbeaten records finite
without pulling individual networks toward random. A 5–5 record against a sole
opponent therefore gives exactly that opponent's rating, with greater uncertainty;
a 6–4 record gives a difference of `400 log10(6.5/4.5)` points. Components with no
path to random have an unidentified absolute origin: only their mean gets a normal
prior (100 ±350), not their relative strengths. Draws count as half a win;
unfinished games contribute counts, not outcomes.
Every refresh refits old opponents too. Identical records against the same opponents
therefore give identical ratings, regardless of seed, color or result order.
Deviations are marginal posterior standard deviations from the inverse joint Hessian,
including uncertainty in opponents. These are approximate, not calibrated confidence
intervals. There is no time decay. A net with no completed evidence retains its broad prior.
Deleting/changing cached files invalidates parsing from the changed prefix; a modified
file must still pass authoritative SGF validation. Legacy sequential and individual-prior rating caches are ignored.
Champion promotion remains a separate decision, unaffected by the rating fit.

`ratings.png` uses a linear **`d`: cumulative generated positions in the training
lineage** axis and the same rating scale (random=100). It reads stored progress,
never substitutes game counts or `s`, and omits networks with unknown `d`.
Existing counters are historical baselines; no ancestral reconstruction is required. All evaluated rounds are connected in order; champion
lineage checkpoints are solid circles in their campaign colour and non-champions are same-sized, grayer dots at 40% opacity. Error bars show **± rating
deviation, not a 95% confidence interval**. Validation and truncated games count
on the x-axis; repeated training-window selection/epochs do not increment `d`. Only models with actual evidence are plotted; skipped rounds do not receive invented
ratings or dots. Existing results remain plotted, including completed games from
an abandoned partial batch, with uncertainty reflecting their actual evidence.
