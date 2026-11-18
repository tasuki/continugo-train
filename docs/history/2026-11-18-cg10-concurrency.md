# cg10 concurrency pilot and defaults — 2026-11-18

With explicit user approval, new neural self-play, loop and pipeline-profiler
jobs default to **32 active games / inference batch limit 32**, retaining the
2 ms collection wait. Random/MCTS defaults are unchanged. Saved loops restore
their existing settings; cg10 remains paused after round 22 with its frozen
12/12 runtime/configuration unchanged.

Artifacts: `scratch/20261118-cg10-concurrency/`. Frozen cg10 round-22 model and
binary, CUDA GTX 1080, six-core i5-8400, six feature threads, one ONNX host thread,
64 cached contexts. Each case generates 64 size-9 games capped at 32 moves, same
seeds, reduced 32/128 search caps with 25% expensive searches, no training-cache
creation. Detailed stage profiling is enabled uniformly. This is an early-game,
reduced-budget throughput pilot, not a strength or full-campaign comparison.

| Games / batch limit | Moves/s | Relative to mean 12/12 baseline |
|---|---:|---:|
| 12/12, first | 9.125 | .998x |
| 24/24 | 9.804 | 1.072x |
| 32/32 | 10.118 | 1.106x |
| 32/12 | 9.223 | 1.009x |
| 12/12, repeat | 9.165 | 1.002x |

GPU utilization rises from about 25% to 28%; CPU from 2.7 to 3.0 cores.
Encoder mean batch rises from 9.79 to 25.77, but encoder elapsed time only falls
from about 62.6 to 60.0 seconds. Coarse features fall from about 105 to 94 seconds.
Larger batches help modestly; concurrency alone, retaining batch 12, barely helps.
The alternating CPU feature / GPU execution pipeline remains the larger issue.
Pipelining has not been implemented by this defaults change.
