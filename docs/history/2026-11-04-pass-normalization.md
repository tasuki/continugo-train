# 2026-11-04 pass normalization and round-30 handoff

User request: correct pass normalization, finish the existing protocol through
**round 30**, then restart from that checkpoint with corrected inference. Do not
reset learned weights, optimizer progress, or replay.

## Correction

Previously, inference returned global hierarchical weights for only the selected
nonempty cells and pass. Search normalized their union, dropping omitted
placement mass but retaining all pass mass. The resulting pass probability could
be far larger than the network's global coarse probability.

Corrected native inference keeps `P(pass)` from the whole coarse softmax and
allocates **all** `1-P(pass)` to selected nonempty cells, proportional to their
coarse weights and exact within-cell legal probabilities. With no selected legal
placement, it still falls back to pass only. Both strata are computed stably in
log space, including extreme logits. Cell/coordinate budgets, legality, pass
proofs, value evaluation and sampled-search importance correction are unchanged.

The training objective and `PolicyValueNet.forward` remain unchanged: global
coarse supervision and complete target-cell local supervision. No architecture,
feature, checkpoint or ONNX graph migration is required. The correction changes
execution semantics; the new native arena fingerprint creates a separate rating
pool. It does not authorize publication or make historical ratings directly
comparable across protocols.

## Tests and before/after inspection

- All Rust tests and Clippy passed, including three new normalization tests.
- **147 Python tests passed**, including native CPU/CUDA and batched parity,
  checkpoint/replay recovery, and round-boundary stopping.
- Native oracle checks cover board sizes 9/13/19, selected-coordinate
  probabilities, original pass mass and total policy mass one.
- An explicitly 10%-pass model stays at 10% with both 1+1 and 32+8 cell budgets.
  A no-legal-placement position still yields probability-one pass.
- The boundary-stop test verifies that all work for the selected round commits,
  no subsequent round starts, the saved total target remains unchanged, and an
  ordinary resume can later continue.

Same empty-board CPU diagnostic, same frozen graphs and seed:

| Actor | Original coarse P(pass) | Old effective P(pass) | Corrected P(pass) |
|---|---:|---:|---:|
| Initial, five bootstrap epochs | 9.83% | 74.22% | 9.83% |
| After round 25 | 92.72% | 99.31% | 92.72% |

Selected cells and value predictions matched. The corrected probabilities sum to
one. Raw reports are in the new run's `before-after.json`. These are inference
checks, **not strength measurements**. The round-25 network already has a large
learned pass bias; removing amplification does not promise immediate disappearance
of opening draws. No opening-pass ban or extra feature plane was added.

## Active handoff

Old campaign: `runs/20261104-151420-size9-training/`.
Corrected continuation: `runs/20261104-215404-pass-balanced/`.
Shared durable store: `models/20261104-151420-size9-training/`.

At setup, round 26 was in progress. The launcher was interrupted cleanly and
resumed with its **original frozen native binaries**, preserving completed work,
plus the new invocation-only `--stop-after-iteration 30` bound. It must finish
round 30's four epochs, scheduled reference batch, archives and actor commit,
then exit before creating round 31. Its saved total target remains 60. The
run-owned original launcher now always supplies that boundary bound.

The new run's `run.py` supervises the original launcher to that boundary and
checks that exactly 30 iterations completed. It then verifies the corrected
read-only binaries against `verified.pt` and forks from the durable round-30
training pack (`checkpoint.pt` plus its exact `replay/`). A failed stage or
missing verification receipt stops the handoff rather than starting a fresh or
unverified model. Old artifacts are retained, not deleted or overwritten.

Corrected loop configuration inherits the old search/training/replay settings:
**30 remaining rounds**, 50 new games/round, four epochs/round, optimizer `keep`,
latest actor, 1,000 batches/epoch, size 9 only, 32 simulations, 1,024 self-play cap,
20 reference games initially and every ten rounds, reference cap 2,048, no actor
gate. Counters, holdout ledger and optimizer/BatchNorm state survive through the
normal checkpoint warm-start path. Its bootstrap fallback is the round-30 replay,
not a regenerated cold-start corpus. Its first self-play seed is **2,101,500**,
continuing the old planned generation seed range without reuse.

Local corrected rounds 1–30 correspond to **global rounds 31–60**. Local
reference rounds 10/20/30 correspond to global 40/50/60; an additional initial
reference batch measures the same round-30 weights under corrected inference.
The additional initial batch belongs to the new protocol pool. No score/eye
feature or acceptance-policy change is bundled into this experiment.

## Operation

```sh
# Status of the supervised boundary/handoff:
tail -f runs/20261104-215404-pass-balanced/run.log
# Original-stage output while it finishes through round 30:
tail -f runs/20261104-215404-pass-balanced/finish-old.log
# After the corrected loop begins:
mise run loop-status -- --output runs/20261104-215404-pass-balanced/loop

# Resume the handoff/continuation after interruption, with no other launcher live:
.venv/bin/python -u runs/20261104-215404-pass-balanced/run.py
```

Do not launch the old supervisor separately while the handoff is running. The
handoff uses inherited process locks and supervised process groups. Both sets of
native binaries remain frozen. Actual round-30 and corrected-run results should
be appended when available; this report records the armed handoff, not a claim
that it has already completed.
