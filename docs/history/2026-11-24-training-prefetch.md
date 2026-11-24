# Complete CPU-batch prefetch — 2026-11-24

## Adopted change

Training and validation use one bounded CPU producer to prepare complete batches
while the trainer consumes the previous batch. Existing sample workers remain
parallel; augmentation, resolution bucketing and collation move off the GPU
consumer thread. Sampling order, augmentation RNG, losses, SGD ordering, precision
and CUDA synchronization are unchanged. No new tuning flag or checkpoint format.

The queue holds one complete batch; at most one additional batch is in preparation
or awaiting enqueue. Step caps bound production exactly. Early consumer exit and
producer errors unwind nested loaders and close Rust readers before returning.

## Measurement

Scratch artifacts: `scratch/20261124-training-prefetch/`, including source
snapshot, scripts, fingerprints, telemetry, per-case JSON and `REPORT.md`.

Hardware: six-core i5-8400, GTX 1080, Torch 2.7.1+cu118, FP32 CUDA, four Torch CPU
threads and six sample workers. Real cg11 iteration-27 checkpoint, optimizer and
manifest; size 9, schema 7, batch size 64, D4 augmentation. Original cg11 feature
extractor and sidecars were read-only; no historical cache/run was rewritten.

Initial scratch experiment, two repeats in reverse order, 128 steps / 8,192
positions after eight warmup steps:

| Mode | Wall s | Positions/s | Batch wait s | GPU mean | Peak host RSS MiB |
|---|---:|---:|---:|---:|---:|
| Baseline | 46.53 | 176.1 | 28.43 | 36.9% | 1,345 |
| One queued batch | 24.14 | 339.4 | 3.97 | 73.1% | 1,475 |
| Two queued batches | 23.88 | 343.1 | 3.64 | 72.5% | 1,498 |

One batch reduced wall time by 48.1%; two added little benefit. A 256-step
confirmation reduced wall time from 91.30 to 47.04 s. GPU tensor peak allocation
was unchanged within each matched comparison. Stage instrumentation is inclusive
and overlaps across threads; its summed times are not a wall-time breakdown.

Fresh comparison using the **production implementation**, with prefetch bypassed
only in the scratch baseline process:

| Mode | Steps | Wall s | Positions/s | Batch wait s | Peak host RSS MiB |
|---|---:|---:|---:|---:|---:|
| Baseline | 128 | 46.15 | 177.5 | 28.07 | 1,349 |
| Production prefetch | 128 | 24.38 | 336.0 | 4.28 | 1,433 |

This is 1.89× throughput / 47.2% less wall time for the measured training segment,
not the entire self-play/training/evaluation campaign.

## Validation

- Production full CUDA epoch: **65,536 positions / 1,024 steps in 184.68 s**,
  354.9 positions/s, 22.77 s batch wait. Full held-out validation: **5,227 positions
  in 10.34 s**. Peak host RSS 1,425 MiB. No prefetch thread remained afterward.
  These are full-phase smoke results, not a paired full-epoch speedup measurement.
- Separate scratch audit: all tensors in all 16 batches match byte-for-byte.
  Timed cases match first-four full payload hashes and all index/visit ordering
  hashes. Production real-feature tests compare every batch field exactly with
  and without prefetch, including D4 and serial/parallel sample loading.
- Queue bounds, empty/exhausted sources, exact step cap, early consumer close,
  producer exceptions and consumer failure with nested loaders are covered.
- 82 Python tests passed across prefetch, training, cache, augmentation, pipeline
  profiling, loop recovery/export and native inference integration.
- CUDA optimizer updates are **not promised bitwise identical**. Even repeated
  baseline audits differed with deterministic algorithms enabled. After 16 steps,
  max parameter differences were 4.22e-5 for repeated baseline and 1.31e-4 for
  prefetch vs baseline; including BN running buffers, 8.44e-5 and 1.56e-3. Identical
  payloads establish ordering/augmentation parity, not identical future trajectories
  or playing-strength improvement.

No cg12 campaign was launched. Existing frozen runtimes remain unchanged.
