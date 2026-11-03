# 2026-11-02 training campaign — historical results

Results snapshot: **2026-11-03 23:01 UTC**. The launcher was interrupted with
SIGINT at **2026-11-03 23:08:41 UTC**, and all campaign processes exited. At the
user's request, its run directory, training/experimental bundles, replay/SGF blobs,
rating CSVs and rating-game evidence were then deleted. No accepted models were
present. This report preserves the summary; follow-up work is in [todo.md](../todo.md).

Run directory: `runs/20261102-140324-training/`. The launcher logged its start at
2026-11-02 14:11:14 UTC. Training started from random weights, without reusing the
[overnight pilot](2026-11-01-overnight-pilot.md).

## Configuration

- Source commit: `83ff92fe4409f36eff9594c73e3f73b5a9e8776c`.
- Shared rules commit: `2afb7821b9cc4a0f24cab0e18b5b2a747a83896b`.
- Schema-4 network: 80,099 parameters, 32 channels, four residual blocks,
  12 coarse planes and 10 exact move fields. FP32 CUDA; SGD lr 0.01, momentum 0.9.
- Size-9/13/19 mix: 40/40/20. Search: 32 simulations/move, PUCT 1.5,
  widening `2 * sqrt(visits)`, independent exploration 0.25, 128 proposal attempts.
- Neural inference: six concurrent games, batch limit six, 2-ms collection wait,
  four feature workers, one ONNX helper thread, 64 cached contexts.
- Bootstrap: 500 fresh unguided MCTS games on four CPU workers. No bootstrap
  eye-avoidance guidance was enabled.
- Initial training: five epochs, 1,000 batches / 2,000 optimized positions per
  epoch, batch size two, two Rust feature-loader workers.
- Planned loop: 60 rounds of 50 fresh neural games, four training epochs/round,
  1,000 batches/epoch, optimizer kept, replay capped at 512 games over 16 rounds.
- Latest experimental actors advanced independently of acceptance. Initial
  reference evaluation plus candidate/reference matches every ten rounds were
  configured. Play/actor-match caps were 1,024 moves; reference cap was 2,048.
  Accepted-model publication required separate explicit approval.

Native binaries were frozen under `bin/`. The single launcher/resumer was
`run.py`, with stage commands in `run.log` and loop transitions in `loop/loop.log`.

## Completed work at the snapshot

Bootstrap finished at 14:16:53 UTC: **500 naturally completed games**, 62,363
moves, seven draws, no truncations. Canonical-game deduplication excluded five
copies, leaving 495 unique games: 396 training / 99 validation, with 49,832 /
12,521 eligible positions respectively.

Initial training finished at 15:08:04 UTC. It optimized 10,000 positions across
five epochs; the final epoch's validation policy CE was **11.047012**, value MSE
**1.012718**, total **12.059731**. The loop used the final completed epoch rather
than the within-job validation-best epoch.

Five loop rounds had completed training/export/actor advancement. Round six had
finished self-play and saved two of its four training epochs. The latest committed
actor was therefore round five, checkpoint SHA-256
`bb04c37680530dea91324037559985f6e9dbe17cc9feb861cd38f37df86a74ac`.

### Neural self-play

Times are each batch's whole-run wall timer, not sums of concurrent game latencies.

| Round | Games | Naturally completed | Moves | Wall seconds | Moves/s | Draws | Truncated |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 50 | 49 | 14,850 | 20,793.208 | 0.71 | 0 | 1 |
| 2 | 50 | 50 | 15,164 | 9,879.099 | 1.53 | 0 | 0 |
| 3 | 50 | 49 | 13,141 | 19,192.975 | 0.68 | 0 | 1 |
| 4 | 50 | 50 | 13,474 | 19,068.373 | 0.71 | 0 | 0 |
| 5 | 50 | 50 | 10,716 | 14,551.939 | 0.74 | 5 | 0 |
| 6 | 50 | 50 | 13,947 | 13,477.405 | 1.03 | 0 | 0 |

Total: **300 games, 81,292 moves, 298 natural completions, five draws, two
truncations**; self-play batches consumed 96,962.999 s (~26.93 hours), excluding
training/export/evaluation. Truncated games did not receive invented outcomes.
These different trajectories are not controlled before/after speed comparisons.
The five round-five draw records each logged two moves; no full draw replay audit
was performed for this report.

### Training diagnostics

| Checkpoint | Validation policy CE | Validation value MSE | Validation total |
|---|---:|---:|---:|
| Initial epoch 5 | 11.047012 | 1.012718 | 12.059731 |
| Round 1 epoch 4 | 10.783701 | 1.062600 | 11.846300 |
| Round 2 epoch 4 | 10.658411 | 1.073155 | 11.731566 |
| Round 3 epoch 4 | 10.461752 | 1.004392 | 11.466144 |
| Round 4 epoch 4 | 10.318357 | 0.943939 | 11.262297 |
| Round 5 epoch 4 | 10.253131 | 0.981772 | 11.234903 |
| Round 6 epoch 2 (partial job) | 9.854640 | 0.974350 | 10.828991 |

Replay corpora change between rounds, so these are diagnostics, not a fixed-set
learning curve or evidence of playing-strength improvement. Value MSE remained
near one. Through round-six epoch two, completed epochs had optimized 54,000
positions in total, counting repeat exposures rather than unique positions.

Round five used 410 training / 102 validation games with 80,307 / 19,259 eligible
positions. Each epoch optimized 2,000 positions and validated 800; actual size
counts were 800/800/400 and 320/320/160 respectively. Training throughput was
4.01–4.03 positions/s, with **434–437 s loader wait versus ~61 s compute/transfer**
per epoch. Validation reached ~4.18 positions/s. Peak live CUDA allocation was
~495 MiB training / 145 MiB validation; parent peak RSS reached ~1,500 MiB,
excluding child processes.

## Strength and observed behavior

The initial actor's reference batch scored **0 wins / 9 losses / 0 draws / 1
truncation** against `random-no-eye-fill-v1`. This measured the initial actor,
not the later round-five model. The ten-round evaluation interval had not yet
produced a later scheduled strength match at this snapshot. No automatic
promotion occurred.

The user observed continued filling of the network's own eyes and judged progress
insufficient to justify the current compute cost. This is a qualitative observation,
not a measured eye-error rate or proof that no skills transferred. Longer games
and lower policy loss alone did not establish useful strength.

The proposed redirection was a fresh size-9 curriculum with temporary bootstrap
small-eye guidance, plus investigation of selective coarse-to-fine policy scoring.
Neither change was implemented in the measured run.

## Performance breakdown

Round five took **14,551.939 s (~4.04 hours)**:

| Stage | Seconds | Share of whole run |
|---|---:|---:|
| Exact coordinate features, including legality/tactical analysis | 9,394.407 | 64.6% |
| Coarse features and rule preparation | 2,349.882 | 16.1% |
| GPU dense policy scorer | 2,336.980 | 16.1% |
| GPU encoder | 231.571 | 1.6% |
| Remaining work | ~239.099 | 1.6% |

There were **136,113 full-policy queries and 339,815 value-only queries**.
Full policies evaluated all million coordinate rows: approximately **136 billion
candidate analyses** in this round alone. The context cache recorded 101,360 hits
and 374,568 misses. The measured bottleneck was CPU feature preparation, not the
coarse GPU encoder.

Code inspection showed that exact analysis constructs hypothetical child positions
and computes connection/capture summaries that feature extraction consumes before
discarding those positions. Internal profiling had not yet isolated allocations,
validation, and repeated scans, so their individual costs were not measured.
Queue/request totals include overlapping waits across concurrent callers and must
not be added to the whole-run wall time.

## Evidence locations

All paths below are relative to the run directory and may disappear with cleanup:

- `run.py`, `run.log`, `bootstrap.log`.
- `initial-training/training.log`.
- `loop/loop.log`.
- `loop/initial-reference/attempt-0001/arena.log` and its SGFs.
- `loop/iteration-000N/selfplay/attempt-0001/selfplay.log` and its SGFs.
- `loop/iteration-000N/train-attempt-0001/training.log` and epoch checkpoints.

This page retains the summary independently of the deleted artifacts. At the
actual stop, round-six epoch three had finished its training phase but had not
saved a completed epoch checkpoint; epoch two remained the latest saved epoch.
