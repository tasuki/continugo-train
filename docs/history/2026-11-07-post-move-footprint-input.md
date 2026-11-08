# Draft schema 5: post-move footprint-union input

Added `resulting_liberty_footprint_stones` as exact column 6, before borders (eleven columns
in total). For each legal candidate, authoritative rules produce the child
position; the resulting friendly connected group's exact liberty spans are
swept by the physical stone radius and their union area is numerically integrated.
The scalar is in physical stone-disc units, without logarithmic scaling or
clipping. Illegal candidates get zero. No eye reward, forced maximization,
move filter, policy normalization, or training target changed.

Schema 5 remains an unreleased draft. Fresh schema-5 checkpoints use eleven
exact fields; earlier ten-field draft checkpoints/packets are rejected rather
than padded. Schema-4 checkpoints, exports, extraction and viewing retain the
original twelve coarse planes and ten exact fields. Native scorer batches and
viewer action-table columns now follow the model/frame's schema.

## Preview

Updated the same disposable preview:

`scratch/20261107-144658-schema5-input-preview/schema5-preview.nn.html`

The new field was extracted at **all 1,635,525 original diagnostic coordinates**
across 70 frames (936,000 legal candidates). Existing coarse variants and original
schema-4 policy/critic/context outputs were preserved; candidate feature vectors
were extended without changing their predictions. No original game, checkpoint,
or rated sidecar was changed. The report prominently states that the original
model did not consume the added feature.

The directory includes the pre-update backup, regeneration script, per-frame
exact-tensor hashes/timings in `exact-footprint-manifest.json`, a Chrome screenshot
`exact-footprint-1920.png`, and a browser check recording eleven loaded exact maps,
fifteen candidate-table columns, working navigation and no JavaScript exceptions.

## Cost and checks

This is not a cheap border-style input: every legal candidate adds child-group
construction, liberty-span extraction and numerical union-area integration.
Preview regeneration used four concurrent extraction clients and took about
61.6 seconds; summed server exact-extraction wall times were 238.3 seconds.
These are diagnostic timings, not isolated benchmarks or playing-strength results.

A subsequent isolated paired comparison used the same 32 complete diagnostic
cells (23,328 coordinates) at four plies of this size-9 game. Each entry is the
median of three serial Rust-server extractions, alternating schemas 4 and 5;
shared exact features were checked bit-for-bit after dropping the new field.
Replay, transport, neural execution and search are excluded. Schema 4 supplies
the existing-ten-field exact baseline; schema 5 supplies current coarse timings.

| Ply | Legal coordinates | All 12 coarse planes | Existing 10 exact fields | Added footprint cost |
|---|---:|---:|---:|---:|
| 0 | 23,328 | 0.027 ms | 3.07 ms | 7,769 ms |
| 42 | 14,321 | 2.41 ms | 151.91 ms | 1,646 ms |
| 84 | 3,433 | 1.45 ms | 13.79 ms | 882 ms |
| 126 | 11,992 | 1.43 ms | 64.25 ms | 946 ms |

The added column accounts for 91–99.96% of coarse-plus-exact extraction time in
these snapshots. Exact extraction is 11.8–65.0 times slower in the nonempty
positions, and 2,529 times slower on the empty board where the original facts
are exceptionally cheap. Added cost per legal candidate is about 79–333 µs.
These are position-dependent measurements, not universal overhead factors.
Raw paired measurements and the script are in `footprint-cpu-comparison.json`
and `benchmark_exact_footprint.py` in the preview directory. This is a CPU
geometry/integration bottleneck, not the cost of the 16 added neural weights.

Coarse-profile area/capturability counters remain zero for schema 5 because they
exclude exact move-row extraction. Synthetic neural benchmarks preallocate exact
features and likewise do not measure this cost.

Rust release tests passed (101 tests, seven explicit benchmarks ignored), along
with clippy and Python lint/format checks. The six directly affected Python test
modules passed all 60 tests; native inference passed all 16 tests, including
PyTorch/ONNX agreement and legacy policy execution. The inference oracle now
extracts complete selected/probe cells instead of integrating a million child
positions: within-cell denominators are unchanged. A broader Python acceptance/loop
sweep was intentionally stopped after extended runtime; no full-suite result is
claimed. Regression coverage includes connections
of multiple groups, exclusion of unrelated groups, capture-created liberties,
one isolated remaining eye liberty, illegal rows, schema-specific widths,
legacy shared-column equality, and serial/parallel bit agreement. No continuing
training campaign or strength evaluation was started.
