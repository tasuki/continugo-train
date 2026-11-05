# Shared model store, global ratings, and live plots — 2026-11-05

At the user's request, replaced per-training model-store placement and split
rating tables with one shared `models/` store and one rating per checkpoint.
Evaluation settings/fingerprints remain match provenance, not separate rankings.

## Migration and recovery

Safely interrupted the live leaky-value loop via SIGINT and waited for its
inherited child lock to release. Backed up the complete original store, all three
loop control snapshots, and the active campaign plan under:

`runs/20261105-101542-global-store-migration/`

Moved the original store's contents directly into `models/`. Updated store paths
in all three loop controls and the active campaign plan; checkpoint bytes, SGFs,
sealed recovery manifests, model identities, optimizer state and frozen binaries
were not rewritten. Historical paths inside immutable provenance remain historical.
No accepted networks or approvals existed, and none were created.

The old CSVs are also preserved in `models/ratings/backups/before-global-*/`.
Validated retained SGF hashes and replayed original committed batches on one scale,
deduplicating SGFs globally. The resulting registry contains **8 rated neural
networks plus one fixed random reference**, backed by **180 unique arena games**.
The checkpoint evaluated under both old tables now has a single combined rating.
Recovery-only intermediate checkpoints were removed from the registry, not deleted
from the immutable backup archive. New intermediate archive publications do not
register networks; evaluation scheduling does.

Verified all committed active-loop artifacts and runtime hashes, then restarted
`runs/20261105-092434-leaky-value/run.py --iterations 10`. Resume adopted 48 completed
self-play games from the interrupted attempt and generated only the remaining two.
Local iteration 2 completed with training and actor progress intact, and iteration
3 started. No reset, reseeding, new iteration budget, or automatic acceptance.

## Plots

- `models/ratings.png`: cumulative unique **training** games from the complete
  ancestral split ledger, excluding validation and counting replay/epochs once.
  The true log axis starts at 100. A separate, narrow zero panel keeps the random
  bot's star fully visible. Error bars show rating deviation, not confidence bounds.
- Training metrics refresh after each durable epoch save. Loop trainers locate
  their enclosing campaign automatically; the active campaign writes
  `runs/20261105-092434-leaky-value/metrics.png`.
- Completed self-play batches refresh the campaign's `draws.png`.
- Newly committed arena batches refresh the shared ratings plot. Reimports do not
  rerate or replot unchanged evidence. Images are atomically replaced; rendering
  failures warn without aborting training or rolling back ratings.

`mise run plot-ratings` and `mise run plot-metrics -- --run PATH` remain available
for manual regeneration. Setup now installs the plotting extra.

## Validation

All **162 Python tests** passed, including global deduplication/migration,
registry filtering, plot axes, automatic plot timing, fault isolation, saved-log
visibility, and the existing inference/recovery/approval tests. Ruff check and
format checks passed. The resumed real campaign also generated fresh metrics
plots and archived its next checkpoint directly into the shared store.
