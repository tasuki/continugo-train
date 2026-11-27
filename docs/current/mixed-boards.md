# cg13: mixed odd board sizes

Cg13 forks **cg12 round 60**, the latest completed round at preparation time,
keeping model weights, BatchNorm, SGD momentum, progress and permanent holdouts.
Cg12 and its frozen runtime are not modified. The new recipe is
`runs/20261127-cg13-mixed-boards/cg13.json`; recipes live in ignored run directories.
Its pinned source SHA-256 is
`e3cbc0ffa4aaf260f4c76a30e5eb625576970059a4cf3570640da936c607f5d8`.

## Generation and training

Generate one size per iteration, cycling **9 → 13 → 17 → 11 → 15 → 19**.
All concurrent self-play games in an iteration have the same spatial resolution.
Game counts use approximately `round(150 * 9 / size)`, the agreed compromise
between constant positions (inverse area) and constant independent game counts:

| Size | Games per iteration |
|---|---:|
| 9 | 150 |
| 11 | 123 |
| 13 | 104 |
| 15 | 90 |
| 17 | 79 |
| 19 | 71 |

This supplies more positions on larger boards if game length scales with area;
it does not promise equal generation time or equal generated positions. The move
safety cap is raised from 512 to 2048 so larger-board games have room to finish.
Truncated games remain excluded from training. Other search settings inherit cg12.

Train after **every iteration**, including the first; no six-wave bootstrap is
required. Use all six sizes in the replay window, targeting equal training-position
shares of the existing **65,536 rows per round**. Cap each size by available rows,
redistribute deficits, and never repeat a row within a round. Missing sizes are not
invented. As their pools fill, their shares converge to one-sixth each. Permanent
whole-game validation assignments remain pinned; validation is exhaustive over
selected positive-weight holdouts, not quota-balanced.

Keep logical optimizer batches at **64**. Set `microbatch_size_9: 64` to bound
physical batches by raster area: at widths 38/46/54/62/70/78, the physical limits
are 64/43/31/24/18/15, respectively. These are initial spatial heuristics, **not
measured optimal VRAM limits**. Gradients accumulate within each logical batch,
then clip and step once. Policy and auxiliary losses retain their respective
logical-batch denominators. BatchNorm still sees the smaller physical batches.
See [training](training.md#availability-capped-mixtures-and-gpu-microbatches).

Track actual sampled positions, available positions and replay exposure fractions
per size, plus physical batch counts. No within-round repetition does not prevent
reusing a scarce pool across successive rounds; inspect those exposure fractions.

## Recipe

The local recipe can be recreated as follows:

```json
{
  "family": "cg13",
  "iterations": 60,
  "lineage": "20261122",
  "output": "runs/20261127-cg13-mixed-boards",
  "source_loop": "runs/20261124-164042-cg12/loop",
  "source_round": 60,
  "parent_sha256": "e3cbc0ffa4aaf260f4c76a30e5eb625576970059a4cf3570640da936c607f5d8",
  "rating_every": 5,
  "overrides": {
    "sizes": [9, 13, 17, 11, 15, 19],
    "size_schedule": "iteration",
    "games_by_size": "9:150,11:123,13:104,15:90,17:79,19:71",
    "size_weights": "9:1,11:1,13:1,15:1,17:1,19:1",
    "size_sampling": "available",
    "batch_size": 64,
    "microbatch_size_9": 64,
    "positions_per_epoch": 65536,
    "steps_per_epoch": 0,
    "positions_per_game": 0,
    "validation_positions_per_game": 0,
    "seed": 13100000,
    "max_moves": 2048
  }
}
```

Build/freeze and prepare inherited checkpoints/caches **without generating games
or training**:

```sh
mise run campaign -- runs/20261127-cg13-mixed-boards/cg13.json --prepare-only
```

Start/resume only when requested:

```sh
mise run campaign -- runs/20261127-cg13-mixed-boards/cg13.json
```

The 60-round target and every-fifth-round separate ratings retain cg12's cadence;
ratings never change training actors or automatically approve publication.
