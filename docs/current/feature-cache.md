# Training-feature sidecars

SGFs are authoritative. An optional, disposable binary sidecar caches all eligible
training positions for one SGF, independently of model architecture, seed, split,
optimizer and epoch. This avoids repeated Rust feature extraction during offline
comparisons. It is not a full-domain inference cache.

```text
attempt-0001/
  game-000001-size9-seed123.sgf
  features/
    game-000001-size9-seed123.cfeat
```

## Build and consume

```sh
PYTHONPATH=python .venv/bin/python -m continugo_train.feature_cache \
  --data PATH/TO/SGFS --workers 4 --feature-binary target/release/features

PYTHONPATH=python .venv/bin/python -m continugo_train.train \
  --data PATH/TO/SGFS --output scratch/YOUR-TIMESTAMP-train \
  --feature-cache require --device cuda
```

Cache modes (`--feature-cache`):

- `off` (default): unchanged live extraction.
- `prefer`: use compatible sidecars; fall back to live extraction only when missing.
  Stale/incompatible/corrupt caches are errors, not silent fallbacks.
- `require`: every requested position must be cached; sample loading starts no Rust
  feature server. Initial corpus inspection still validates SGFs using Rust.

Cache mode can be changed on resume without changing sampling or training targets.
Frozen campaign runtimes do **not** acquire cache support automatically; they must
remain frozen. Updated offline training can explicitly use the caches.

The builder validates existing sidecars fully and skips rebuilding them. Use
`--rebuild` explicitly to replace incompatible or damaged sidecars. It never edits
SGFs. One writer lock per sidecar, temporary sibling files and atomic rename prevent
partial publication; errors clean up the writer's temporary file/lock. Following a
hard process termination, remove a stale `.cfeat.lock`/temporary file only after
verifying no builder is alive. Inputs must remain immutable during building/training.
Caches themselves must also remain immutable during training.

## Container version 1

- Prefix: `CGCACHE1`, little-endian u64 footer offset, SHA-256 of footer bytes.
- Independently zlib-compressed (level 3) existing `CGFEAT02` packets, one per ply.
- Small JSON footer: SGF SHA-256, validated Rust game metadata, feature schema,
  feature names, density, target format, extractor binary SHA-256, compression
  identifier, and index entries `(ply, offset, compressed bytes, raw bytes, raw SHA-256)`.

The JSON is only metadata/indexing, never tensor storage or an alternate game record.
All arrays remain lossless FP32/integer data in their existing packet representation.
Record bounds, index coverage, compressed stream length/termination and checksums
are validated. There is no quantization or changed policy/label interpretation.
Exact extractor-binary hashing is deliberately conservative: even an otherwise
compatible rebuild requires explicit cache rebuilding with that binary.

The loader keeps at most eight sidecars open per worker, decompresses only requested
positions, and preserves the existing plan order, shuffling, batching, policy masks
and game-disjoint splits. It does not preload the whole corpus or store model outputs.

This first version repeats ownership/static fields inside each packet. Deduplication
and specialized packing can be added after measurement; preserving exact existing
packets makes the first implementation easy to validate. Disk use scales with
positive-target cell coverage, not just the number of raster cells.

## Measurements

See [the network-size experiment](../history/2026-11-11-network-size-comparison.md)
for the scratch equality/size/throughput pilot and real-run additive cache build.
Short hot-cache benchmarks do not guarantee full-corpus throughput or GPU saturation;
decoding, collation, transfers and tiny batches can become the next bottlenecks.
