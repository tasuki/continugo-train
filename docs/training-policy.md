# Training and promotion policy

Agreed with the user on 2026-11-02 and implemented by the v5 loop and separate
manual approval operation. This policy supersedes the previous plan to rely on
automatic gating or a daily promotion limit alone.

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

## Implemented loop behaviour

The [v5 supervisor](loop.md) separates all three roles. It starts with no accepted
network unless the store already has a historical/approved head, which it only
reads. Initial actors and subsequent candidates are backed up unaccepted. Default
`--actor-policy latest` advances experimental self-play independently of match
verdicts; optional `--actor-policy gate` guards only that actor selection.
`--gate-games 0` disables the experimental match without publishing acceptance.
`--evaluation-interval N` schedules periodic strength checks, not promotion.

Each iteration freezes separate training and actor parents. SGFs carry the exact
actor checkpoint identity. Training and actor recovery packs preserve all exact
data, hashes, holdouts and optimizer progress independently of accepted publication.
`mise run loop-status -- --output PATH` is read-only and reports roles and progress.
Old v1–v4 automatic-promotion loops must be forked, not silently resumed/migrated.

## Approval operation

**Only run this after the user explicitly approves the specific full checkpoint
hash and judges the network convincingly much stronger than the reference.**
Permission to implement software, start training, or inspect results is not that
approval. The assistant must not issue this command on its own initiative.

```sh
mise run promote -- --model-store PATH/TO/STORE --candidate PATH/TO/STORE/experimental/BUNDLE --checkpoint-sha256 FULL_64_CHARACTER_HASH --approved-by USER --reason "Specific user approval and reviewed reference-strength evidence" --confirm-much-stronger-than-reference
```

The candidate must be an immutable experimental recovery bundle owned by that
store. The command checks its full checkpoint hash, replay/inference file hashes,
ONNX provenance and eager/export parity. User identity/reason and qualitative
reference-strength confirmation are mandatory. There is intentionally no invented
numeric threshold for "much stronger". This is an explicit operator-approval
workflow, **not cryptographic authentication of a remote human**; anyone with
permission to invoke this command/write the store is inside the trust boundary.

A sealed, fsynced approval intent pins the source bundle, exact checkpoint and
both ONNX graphs, approved-by/reason/time and explicit strength confirmation.
Accepted bundles embed that approval. Store-owned intents and publication receipts
survive removal of `runs/`; writer locks serialize concurrent approvals. Direct
accepted archive publication without a matching intent fails closed, including
reuse of an already published bundle. Other checkpoint hashes need new approval.

Only the latest committed approval may change `LATEST-ACCEPTED`; retrying an old
approval is idempotent and cannot roll it back over a newer accepted head. Approval
changes neither `LATEST-TRAINING` nor `LATEST-EXPERIMENTAL`, and never grants
permission for later candidates. Ratings mark only the explicitly published
checkpoint accepted; old historical accepted metadata remains readable.

If interrupted after authorization but before complete publication:

```sh
mise run promote -- --model-store PATH/TO/STORE --recover
```

Recovery may finish **only previously recorded explicit authorizations** and repair
their publication/CSV/head commit gaps. With no intents it accepts nothing. It
cannot issue fresh permission. Changed intents, graphs or checkpoint identities
are rejected rather than silently substituted. Preserve and back up the entire
store, including `approvals/`, both rating CSVs and SGF evidence; never edit these
files or receipts by hand.

Tests cover all startup/gate-disabled paths, advancement after failed matches,
optional actor holding, continued optimizer/replay progress, periodic evaluation,
read-only status, lost-run recovery, real export verification, changed artifacts,
duplicate and concurrent approval processes, and publication/pointer commit gaps.
A week-long campaign is still a separate launch decision, not an effect of approval
or of running tests.
