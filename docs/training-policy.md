# Training and promotion policy

Agreed with the user on 2026-11-02. This is the intended operating policy, not a
claim that the current supervisor implements it. This policy supersedes the
previous plan to rely on automatic gating or a daily promotion limit alone.

## Three independent model roles

1. **Latest training checkpoint:** weights, optimizer state and cumulative progress
   continue across training phases, including rejected/inconclusive evaluations.
2. **Experimental self-play model:** an explicitly unaccepted checkpoint used to
   generate fresh games. It may advance during the experiment without approval
   of an official promotion. Actor selection/health checks must be explicit and
   logged; an experimental regression need not rewind training or optimizer state.
3. **Accepted network:** a separately selected, trusted publication. Experimental
   actor updates, checkpoint backups and exports are **not promotions**.

Keep these roles separate in saved state and recovery. Training and experimental
self-play must not wait for a promotion review or remain permanently frozen on the
last accepted model. An accepted network may be absent in a fresh experiment.
Existing historical accepted bundles must not be deleted or silently relabeled.

## Every promotion requires explicit user approval

- **No promotion until the network is convincingly much stronger than the fixed
  random-but-not-eye-filling reference bot AND the user explicitly approves it.**
  This reference is not a trivial opponent: its eye-avoidance heuristic is useful.
- No automatic numeric threshold for "much stronger" has been agreed. Present
  completed match evidence, board/color coverage, uncertainty, truncations and
  comparisons with historical checkpoints for review; do not invent a cutoff.
- Passing a head-to-head gate, falling loss, a favorable rating, elapsed time,
  starting a run or exceeding a daily waiting interval is **not authorization**.
- Approval applies to a specific immutable checkpoint identity (full hash), not
  a moving `latest.pt` alias, a whole lineage, or future candidates.
- Without that approval, do not publish an accepted bundle, mark a network
  accepted in ratings, change the accepted pointer, or implicitly accept a new
  experiment's initial checkpoint. Backups stay unaccepted training artifacts.
- Approval and the checkpoint identity must be durably recorded so restarts,
  retries, concurrent processes or loss of `runs/` cannot create new permission.
- A future one-per-day cap would only be an additional restriction; it could
  never replace approval. No pending candidate becomes accepted just by aging.

Evaluation measures progress; it is not permission to promote. A weak reference
score does not by itself stop training. Manual review adds negligible computation;
strength matches are the expensive part and can be scheduled periodically rather
than after every training phase.

## Long-run operating intent

Continue the **fresh `cold1` lineage from the search-calibration experiment**, not
the pass-collapsed original overnight lineage. Preserve weights/optimizer progress
and keep adding fresh self-play games, rather than repeatedly resetting or fitting
one bootstrap corpus indefinitely. Network-size changes remain manual.

Plan a resumable week of training initially, extending to two weeks after review
if healthy. Check after the first day and periodically thereafter. There is no
expectation of beating the reference quickly on this hardware. Short diagnostic
timeouts are not a learning criterion and must not silently interrupt a longer
campaign. No week-long campaign has been launched merely by documenting this plan.

Inspect training and held-out losses, actual sample counts, pass/game-length and
natural-completion statistics, SGFs, occasional strength matches, throughput,
memory and disk space. Loss alone is not proof of strength and need not decrease
monotonically as the corpus changes. Pause for nonfinite values, invalid data,
behavioural collapse or resource/recovery problems—not merely slow improvement.

Preserve the existing 40/40/20 board mix, shared authoritative rules, real search
visits, natural outcome labels, holdout history and durable recovery. Never turn
capped games into draws or synthetic labels. Keep evaluation evidence out of replay.

## Implementation status — not implemented yet

The current [`mise run loop` supervisor](loop.md) still couples its self-play actor
to the accepted model, automatically promotes passing candidates, and publishes
its initial checkpoint as accepted. **Do not launch it unattended under this
policy yet.** `--gate-games 0` is not a manual-approval mode; it bypasses gating
rather than preventing promotion. Standalone arena matches do not promote models.

Before the long campaign:

- Separate training, experimental-actor and accepted identities, including an
  experiment with no accepted model. Record actor provenance in generated SGFs.
- Make acceptance fail closed without explicit approval, including initialization,
  resume, archive publication, rating flags and all gate-disabled paths.
- Add a hash-pinned, auditable approval/publication operation. It must not reset
  training, silently switch the actor, or authorize a different/newer checkpoint.
- Test actor advancement without acceptance, withheld/failed gates, continued
  optimizer progress, startup/restart/recovery, duplicate approvals and races.
- Keep progress inspection and checkpoint/replay recovery usable throughout a
  long run. Preserve old experiments rather than silently migrating their policy.
