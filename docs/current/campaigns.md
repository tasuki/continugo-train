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
prepares replay/checkpoints but launches no games or training. Exact-compatible
seed caches are shared by symlink (keep those ancestral caches available). Missing
or extractor-incompatible caches are rebuilt in the new campaign, never rewritten
or relabeled in historical campaigns. Preparation reports progress on stdout;
redirect stdout/stderr to `<output>/run.log` for unattended runs.

A recipe pins a completed source round/checkpoint and gives only family, output,
iteration/rating cadence and loop-setting overrides. The runner inherits the
parent's search/training configuration, model, SGD momentum, cumulative counters,
D4 setting and ancestral split ledger. It copies the round's ordered replay window,
validates game hashes/caches and changes checkpoint family identity only.
`launch.json` records effective settings, seed counts and checkpoint/runtime hashes.
No optimizer reset or automatic publication is performed.

The common supervisor handles locks, signal interruption, status, replay order,
cache propagation, runtime verification and optional synchronous shared-champion
challenges before the next generation. `rating_every: 0` disables those challenges;
saving cadence is a separate loop option. Existing shared ratings retain their
frozen evaluation protocol. Resume after termination with the same command; do not
run an old campaign launcher concurrently. Logs/status are under the recipe's
output directory, including `run.pid`, `status.json`, `loop/loop.log` and stage logs.

## cg9

`cg9.json` forks cg8 round 5 (the latest completed round) with current code/binaries,
including default schema-5 `geometry-v2-frontier` proposals (budget 16).
It retains cg8's size-9, 100 games/round, 1,000-game replay, exhaustive one-epoch
training, D4, batch size 2, jigo-as-loss and randomized 100/400 search (25% expensive
policy positions). It uses seed 99100000, saves every round and requests a shared
champion challenge every fifth round. Initial target: 20 rounds. No historical
campaign is modified.
