# Quick training batch-size screen — 2026-11-11

## Findings

**64 was fastest in this small cached-data test**, at 380 positions/s versus
145 for batch 2 (**2.62×**). Batches 128 and 256 fit easily but did not improve
throughput. Batch 32 was within 6% of 64. This supports trying **64**, not blindly
choosing the largest batch that fits. No defaults or campaign settings changed.

Artifacts/scripts/raw logs: `scratch/20261111-233000-training-batch-size/`.
No campaign resumed, no checkpoints retained, no source SGFs/caches modified.

## Actual cached training path

GTX 1080 (8 GiB, desktop attached), Torch 2.7.1+cu118, CUDA FP32 eager,
current 4-block/32-channel network (81,766 parameters), four Torch threads,
six loader workers, required existing feature caches. Fresh seed-20261111 model
and SGD optimizer per trial: lr .01, momentum .9, weight decay 1e-4, gradient
clipping 5; current default loss weights. Full forward/backward/optimizer path.

Use exactly **512 fixed size-9 positions** from the previous experiment's frozen
corpus, resolving them to already-cached original SGFs and checking source hashes.
That frozen corpus contains only size-9 games; the pipeline result is not a mixed-
board estimate. No global shuffle, uniform exhaustive sampling. All batch sizes
divide 512, with no partial batches. Run ascending then descending batch sizes,
two trials each, **6,144 total position exposures**. All weights discarded.

| Batch | Positions/s¹ | Peak allocated GPU MiB² |
|---:|---:|---:|
| 2 | 145.2 | 34.8 |
| 16 | 333.0 | 160.7 |
| 32 | 359.2 | 300.0 |
| 64 | **379.7** | 592.2 |
| 128 | 368.3 | 1,141.5 |
| 256 | 358.5 | 2,281.9 |

¹ Total positions / summed phase wall time across both trials; includes loader
startup/wait, transfers and metrics. Timed phases total about 21 seconds; excludes
model initialization and process startup. Small, hot-cache pilot, not full-epoch
or cold-storage throughput. At 256 each trial contains only two optimizer steps.
² Torch allocated memory, not total device usage or reserved allocator memory.

Loader wait averaged about .83–.89 seconds per phase for batches 32–256, versus
phase totals around 1.35–1.45 seconds. The pipeline is increasingly loader-bound;
extra GPU capacity alone does not remove that bottleneck. Ascending/descending
results agree on 64 as fastest, but small nearby differences need longer tests.

## Compute-only cross-check

Stock synthetic benchmark, diameters 110/76/52 (resolutions 38/54/78), batch
2/16/32/64/128/256, one warmup and three measured calls per stage, four threads.
Random inputs/targets generated outside timing. Table reports full training-step
median positions/s (not inference throughput):

| Batch | Diameter 110 | Diameter 76 | Diameter 52 |
|---:|---:|---:|---:|
| 2 | 230 | 249 | 212 |
| 16 | 670 | 923 | 580 |
| 32 | 767 | 1,020 | **621** |
| 64 | 905 | **1,068** | 619 |
| 128 | 912 | 952 | 548 |
| 256 | **959** | 997 | 524 |

All cases fit; batch-256 peak allocated full-step memory was approximately
3,058 / 2,932 / 4,745 MiB respectively. These synthetic selected-cell action
counts differ from actual training batches; do not substitute these rates for
corpus throughput.

The first unmodified synthetic attempt failed at batch 64 with a nonfinite or
masked policy-target error after earlier random-target optimizer steps. Its log
is preserved as `synthetic.log`. The successful rerun uses the scratch wrapper
`compute.py` to set SGD lr=0: backward, momentum and optimizer operations still
execute, but random-target training cannot move weights into instability. This
is a compute benchmark only, not evidence of learning behavior. The real-data
pipeline test used normal lr=.01 and completed successfully.

Reproduce from repository root:

```sh
PYTHONPATH=python .venv/bin/python scratch/20261111-233000-training-batch-size/compute.py \
  --device cuda --diameters 110,76,52 --batch-sizes 2,16,32,64,128,256 \
  --warmup 1 --iterations 3 --threads 4
PYTHONPATH=python .venv/bin/python scratch/20261111-233000-training-batch-size/pipeline.py
```

## Interpretation

Memory is not the only constraint: throughput plateaus, convolution efficiency
varies with shape, and loading costs matter. **128/256 are viable but were not
faster for the actual cached size-9 workload.** This test measures execution
speed only. It does not compare validation progress, playing strength, or suitable
learning rates. Batch 64 makes 32× fewer updates than batch 2 per data pass;
changing batch size in a real run requires attention to optimization and any
batch-count-based epoch caps. The stopped four-architecture comparison remains
stopped and unchanged.
