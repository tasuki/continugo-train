# Recipe-driven campaigns

Campaign recipes live in their ignored campaign directories under `runs/`.
Only common orchestration is committed, in `python/continugo_train/campaign.py`.

From the repository root:

```sh
mise run campaign -- runs/20261114-cg9-geometry/cg9.json --prepare-only
mise run campaign -- runs/20261114-cg9-geometry/cg9.json
# Resume, optionally increasing the TOTAL target:
mise run campaign -- runs/20261114-cg9-geometry/cg9.json --iterations 30
```

The first invocation builds current native binaries and freezes the Python package,
recipe and binaries into the recipe's output directory. `runtime.json` records
SHA-256 fingerprints and source revision. The supervisor re-execs that frozen
package; children use it too. Subsequent invocations verify rather than overwrite
it. Changed recipes/runtime require a new campaign. `--prepare-only` builds and
prepares window games/checkpoints but launches no games or training. Exact-compatible
seed caches are shared by symlink (keep those ancestral caches available). Missing
or extractor-incompatible caches are rebuilt in the new campaign, never rewritten
or relabeled in historical campaigns. Preparation reports progress on stdout;
redirect stdout/stderr to `<output>/run.log` for unattended runs.

The launcher passes an absolute `CONTINUGO_VIEWER_BUNDLE` path to the frozen
runtime so rating reports can find the shared viewer outside the campaign tree.
By default it uses `../continugo/frontend/viewer/dist/continugo-viewer.js`; build
that bundle with `mise run //frontend:build-viewer` in `../continugo`, or set
`CONTINUGO_VIEWER_BUNDLE` to override it. This also applies when resuming older
frozen campaigns; their runtime files do not need modification.

A recipe pins a completed source round/checkpoint and gives only family, output,
iteration/rating cadence and loop-setting overrides. The runner inherits the
parent's search/training configuration, model, SGD momentum, cumulative counters,
D4 setting and ancestral split ledger. New forks adopt the default one-third cumulative
training-window policy, even when the parent used a fixed window; a recipe may
explicitly set `training_window_fraction: 0` for legacy fixed windows. See [training/validation windows](loop.md#training-and-validation-windows).
It copies the round's ordered training/validation windows,
validates game hashes/caches and changes checkpoint family identity only.
`launch.json` records effective settings, seed counts and checkpoint/runtime hashes.
No optimizer reset or automatic publication is performed.

The common supervisor handles locks, signal interruption, status, window ordering,
cache propagation, runtime verification and optional synchronous shared-champion
challenges before the next generation. `rating_every: 0` disables those challenges;
saving cadence is a separate loop option. Existing shared ratings retain their
frozen evaluation protocol. Resume after termination with the same command; do not
run an old campaign launcher concurrently. Logs/status are under the recipe's
output directory, including `run.pid`, `status.json`, `loop/loop.log` and stage logs.

## Iteration timing

New campaigns automatically maintain **`timing.png`** beside `metrics.png` (both
under the campaign directory and `loop/`). Stacked bars show self-play and feature
caches, window preparation, training including validation/checkpoints, arena games,
evaluation/reporting, export, model saving, and other orchestration overhead.
Arena is subtracted from the evaluator total, so it is not counted twice; reporting
includes joint rating fits, plots and neural-viewer generation.

Stage durations are appended to `loop/iteration-NNNN/timing.jsonl` using monotonic
clocks. Failed/interrupted work counts; downtime between resumes does not. Timing
is diagnostic, not a training checkpoint or game ledger. No command/recipe option
is needed to enable it. A newly frozen runtime includes the recorder automatically.

For existing frozen campaigns, generate an ad-hoc chart without stopping training:

```sh
mise run plot-timing -- --run runs/20261114-cg9-geometry
# Exclude reconstructed historical estimates:
mise run plot-timing -- --run runs/EXAMPLE --measured-only
```

Old rounds without journals are **hatched estimates**, using process-log birth and
last-write timestamps plus native arena-game durations from the shared evaluator
log. They can include pauses within a reused log; unknown setup/selection/saving
and between-process gaps are omitted. They are not exact total wall-clock times.
Filesystem birth times must be available (GNU `stat`); unknown spans are omitted.
Faded bars indicate partial measured rounds. The current cg9 chart is generated
this way; its already-running frozen supervisor is not modified or interrupted.

## cg9

`cg9.json` forks cg8 round 5 (the latest completed round) with current code/binaries,
including default schema-5 `geometry-v2-frontier` proposals (budget 16).
It retains cg8's size-9, 100 games/round, 1,000-game combined training/validation window, exhaustive one-epoch
training, D4, batch size 2, jigo-as-loss and randomized 100/400 search (25% expensive
policy positions). It uses seed 99100000, saves every round and requests a shared
champion challenge every fifth round. Initial target: 20 rounds. No historical
campaign is modified by preparing a new fork.

For continuation after completed round 35, cg9's runtime/config is explicitly
upgraded to the one-third cumulative training-window policy. Round 36 onward uses
separate, permanently pinned training/validation pools; completed round corpora are
unchanged. The run remains paused after round 35 until explicitly resumed.
