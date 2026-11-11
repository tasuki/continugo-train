# Evaluation and ratings

Training saves models; the **one shared evaluator** rates them independently.
For this campaign only **cg5 rounds 1–100** are included. cg1–4 are not imported.

## Artifact layout

```text
artifacts/
  models/cg5/r0001/
    checkpoint.pt
    encoder.onnx
    policy.onnx
    metadata.json
  models/cg5/r0002/ ... r0100/
  ratings/
    networks.csv
    champions.json
    evaluator.pt
    evaluator.log
    ratings.png
    ratings.2026-11-10.png
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
from earlier policies remain valid and are not removed.

1. Each eligible candidate plays **10 games against the reigning champion**.
   At least **7 wins** permit continuation.
2. It plays **30 more games against that same frozen champion**, with fresh paired seeds.
   At least **28 wins across the full 40 automatically replace the champion**.
3. **Only after promotion is committed**, the new champion plays **10 games against
   random** for rating calibration. Rejected challengers do not play random.
   Initial bootstrap against random already supplies the ten reference games;
   these are reused, never duplicated. No later result resets the champion to random.
4. A new champion plays **10 games each against up to two other available, previously
   evaluated models** near its rating. The defeated champion is excluded: it already
   supplied forty games. These extra games calibrate ratings; they do not veto promotion.

Match reports contain counts, scores and truncation bounds, not a generic
50% acceptance threshold or passed/failed label. Only the 7/10 and 28/40 win
requirements above decide challenge continuation and champion replacement.
Historical reports remain evidence; legacy `acceptance_rate` config is ignored.

All neural turns use **128 simulations** by default, including random calibration
and historical matches. The random bot is direct random play, not MCTS. This budget
is half cg5's expensive policy-training budget of 256. Evaluation never randomizes
its search budget. The neural move cap is 1024; reference games allow 2048 moves.
A cap is a safety limit, **not** a forced-scoring rule.

Draws and truncations are **non-wins for promotion**. The denominators stay 10 and
40. Ratings give natural draws half-points; truncations have no outcome and do not
update skill. This is a practical filter, not a claimed confidence level; the small
screen can reject genuine improvements. All planned games are played.

Champion selection never changes the training actor or publishes an accepted model.
Publication remains a separate explicit user decision; see [model storage](model-store.md#approval-operation).

## Commands and recovery

```sh
# Save every completed cg5 candidate, not just tenth-round milestones:
mise run save-models -- --loop runs/20261108-154636-schema5-scratch/loop --every 1

# Create the shared evaluator, or resume its frozen queue:
mise run rate-saved -- --family cg5 --simulations 128
mise run rate-saved -- --resume

# Rebuild the rating summary from the SGFs:
mise run ratings
mise run plot-ratings
```

`--model-store PATH` selects another artifact root; default `artifacts/`. There is
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

## Ratings, not another game ledger

**SGFs are authoritative.** `networks.csv` is a derived summary:

```csv
network,rating,deviation,games,wins,losses,draws,truncated
```

It contains readable names such as `cg5-r0001` and `random`, not hashes or absolute
paths. It can be regenerated by scanning `sgfs/`. The evaluator checkpoint caches
unchanged file signatures and rating states only to accelerate replay; it is not a
second game ledger. `mise run ratings` discards that cache and rebuilds from SGFs.

The shared backend's static Glicko update has no time decay or inactivity inflation.
Same-seed color pairs form simultaneous update batches. Rebuilding from the same
SGFs produces the same summary. Deleting/changing cached files invalidates replay
from the changed prefix; a modified file must still pass authoritative SGF validation.
If a net has no completed evidence, show its broad prior rather than invent outcomes.

`ratings.png` retains the linear cumulative-unique-training-game axis and the same
rating scale (random=100). All evaluated rounds are connected in order; champion
lineage dots are saturated and non-champions are grayer. Error bars show **± rating
deviation, not a 95% confidence interval**. Validation games are excluded from the
x-axis, and replay/epochs do not count a game twice. Only models with actual evidence are plotted; skipped rounds do not receive invented
ratings or dots. Existing results remain plotted, including completed games from
an abandoned partial batch, with uncertainty reflecting their actual evidence.
