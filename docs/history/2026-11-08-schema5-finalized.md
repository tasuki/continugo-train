# Schema 5 finalized

**Finalized by explicit user approval on 2026-11-08**, independently of starting
a training run. Earlier reports describing schema 5 as a draft are historical;
this approval supersedes that status.

The frozen contract is documented in [current features](../current/features.md):

- **Twelve coarse planes:** friendly/enemy antialiased occupancy, connected group
  size, liberty-center unions, liberty-footprint unions, then four borders.
- **Eleven exact fields:** legality, connected groups/stones, captured
  groups/stones, resulting group size, `local_liberty_footprint_stones_cap3`,
  then four borders.
- Local footprint geometry uses fixed existing obstacles, not child-group
  liberties or capture removal. Its 128×128 raster uses 16 samples per physical
  radius, calibrates one isolated center to one disc, and saturates at three.
  Certified saturation probes and raster early exit preserve the clipped result.
- Policy remains `cells-v1`; targets remain `final-area-v1`. Schema-4 models keep
  their original feature definitions. Incompatible earlier draft tensors or
  weights are not silently converted.

Further changes to feature meanings, order, count, scaling or raster definitions
require schema 6 or later and matching checkpoint/ONNX metadata. Input-preserving
implementation optimizations may retain schema 5. Network-capacity experiments
with the same feature contract do not themselves change the feature schema.

Updated source comments, README and current documentation to say finalized,
removing the former "finalize when the next training run starts" condition. The
stable preview now says finalized and records `schema5_finalized: true`, while
retaining the original schema-4 predictions and historical input alternatives.
All 70 frame payloads were checked unchanged during this label/metadata update.

Finalization changes no tensor values, model weights, rules, training targets,
wire version or network parameter shapes. It starts **no training, self-play,
rating or benchmark run**.
