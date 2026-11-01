# ONNX export and native neural MCTS

The trained network guides native Rust self-play through one **CPU/CUDA ONNX
interface**, using the same model and features. CUDA is the default for the
`neural-selfplay` and `infer` tasks; CPU remains available for tests, debugging and
machines without a GPU. Queries are currently synchronous and unbatched. Browser
integration is still pending. This is not a claim of competitive playing strength.

## Run

`mise run setup` installs the inference extra. For an existing environment:

```sh
.venv/bin/python -m pip install -e '.[inference]'
mise run setup-gpu
mise run export -- --checkpoint runs/offline-training/best.pt --output runs/onnx-initial
mise run neural-selfplay -- --model runs/onnx-initial --games 3 --sizes 9,13,19 --seed 200 --simulations 8 --max-moves 64 --inference-threads 4 --output runs/neural-initial
```

Use fresh export/game directories. `neural-selfplay` supplies `--engine neural`
and defaults to `--device cuda`. To use CPU, pass `--device cpu`; it needs no
`setup-gpu` step. The actual search and inference are **in-process Rust/native
code**. Python only selects the runtime and library paths before launching Cargo.
Baseline `mise run selfplay` needs no inference runtime.

### Both backends, without changing the training environment

`setup-gpu` installs the official **ONNX Runtime GPU 1.20.1 CUDA 11.8 build** and
**cuDNN 8.9.6** into `.venv/ort-gpu/`. It reuses CUDA 11 libraries from the existing
PyTorch cu118 installation. It does **not** replace PyTorch, its cuDNN 9.1, or the
CPU ONNX Python package (1.22.1). Installation currently targets Linux; other
platforms can supply their own compatible native library.

The Rust binding targets **API 1.20**, which also works with the newer CPU 1.22.1
runtime. This older ABI is deliberate: official CUDA 11.8 GPU wheels stop at 1.20,
while newer GPU wheels target newer CUDA versions. This configuration supports the
local Pascal GTX 1080 and driver 550 without a driver or training-stack upgrade.
See the [official CUDA compatibility table](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

An explicit CUDA request **fails** if the runtime lacks CUDA or provider setup
fails; there is no automatic whole-session fallback to CPU. Shape/control
operators may still run on CPU, but tests inspect runtime profiles to require
convolutions, matrix multiplies and GridSample on CUDA. FP32 is used with TF32
disabled, heuristic convolution selection, and a 2-GiB arena limit per CUDA session
(not a limit on total process/device usage). CPU thread spinning stays disabled.
MCTS, rule-derived feature extraction, normalization and sampling remain on CPU.

For a standalone release binary, set `ORT_DYLIB_PATH` or pass `--ort-library`:

```sh
export ORT_DYLIB_PATH="$(.venv/bin/python -m continugo_train.export --runtime-path)"
cargo run --locked --release -- --engine neural --device cpu --model runs/onnx-initial --sizes 9 --simulations 8
```

Both the standalone binaries and task launcher default to CUDA. Use
`--device cpu` explicitly for CPU inference. For a custom GPU runtime, supply
`--device cuda --ort-library PATH` and its dependency library paths. The
`ORT_DYLIB_PATH` environment variable is also respected. Merely pointing at an
arbitrary `.so` is not supported. No backend silently switches to an unguided
baseline if model loading or inference fails.

## Model bundle and verification

Export writes two graphs atomically as one new directory:

- `encoder.onnx`: `coarse [B,12,H,W]` → projected policy context, pass logit and
  side-to-move value. Batch and spatial axes are dynamic.
- `policy.onnx`: context + `exact [B,N,10]` + shared int64 action IDs `[N]` → raw
  placement logits `[B,N]`. Batch, spatial and candidate axes are dynamic.

The split allows scoring coordinate chunks without rerunning the ResNet. Default
chunks are 65,536 coordinates. Both graphs embed identical feature definitions,
density, architecture, source checkpoint SHA-256 and epoch. There is no model
metadata sidecar. Native loading rejects missing/incompatible metadata or mixed
encoder/scorer bundles. Feature semantics and the existing compatibility marker
are unchanged. The checkpoint remains the trainable artifact; ONNX is inference-
only. Both graphs use standard opset 17, including `GridSample`; browser runtime
operator support still needs an actual integration test.

Each export checks both graphs against eager PyTorch at all three default
resolutions, varying batch size and candidate count, including boundaries and
neighboring coordinates. Tests additionally compare native full-domain inference
against PyTorch using real nonempty Rust features at sizes 9/13/19, legality
masking, chunk boundaries, pass and value-only queries.

One-position native diagnostics (stdout is a JSON numeric report, not a game
record; no file is written):

```sh
mise run infer -- --model runs/onnx-initial --sgf FILE --ply 1
# Cheap leaf-value path; no million-coordinate extraction/scoring:
mise run infer -- --model runs/onnx-initial --sgf FILE --ply 1 --value-only
# CPU backend, or repeated warm measurements (timers exclude warmup):
mise run infer -- --device cpu --model runs/onnx-initial --sgf FILE --ply 1 --warmup 1 --iterations 3
```

The launcher sets the runtime environment. Diagnostic `--actions` only selects which
raw logits are printed; the normal diagnostic still computes the full policy.
Illegal scores print as null rather than invalid JSON infinity. To inspect actual
operator placement, set `CONTINUGO_ORT_PROFILE` to an existing directory before a
native diagnostic. ONNX Runtime writes encoder/scorer execution profiles there;
these are optional backend diagnostics, not game records or SGF sidecars.

## Feature extraction and context caching

Neural inference defaults to `--feature-threads 4 --cache-positions 64`:

- A bounded, caller-owned **Rayon CPU pool** extracts disjoint coordinate chunks
  using one immutable shared-rules prepared position. Row order and facts are
  bit-for-bit identical to serial extraction. `--feature-threads 1` is the serial
  control. This pool is separate from ONNX `--inference-threads` / diagnostic
  `--threads`; CPU helper threads for CUDA still default to one.
- A first value query already produces projected spatial context, pass logit and
  value. These small **host-side encoded contexts** are cached. Later value
  queries or policy expansion reuse them, skipping coarse raster/area/capture
  extraction and the ResNet. Raw coarse rasters need not be retained separately.
- Keys use exact immutable **Game equality**, including board, stones and full
  record/turn/pass history. The cache belongs to one frozen model/backend/density,
  and reconfiguration clears it. There is no cross-model cache or approximate
  position matching. Terminal queries are rejected before lookup.
- LRU eviction bounds entries to 64 by default, with an additional **32 MiB spatial
  payload ceiling**. Game keys, scalar data and allocator overhead are separate,
  bounded by entry count. Oversized contexts are not cached. `--cache-positions 0`
  disables the cache.
- **Dense move features and million-action logits are never cached.** Policy
  expansion still extracts fresh exact move facts; there is no incremental update
  across different boards or duplicated Python rule logic. MCTS separately keeps
  its small sampled candidate reservoirs as before.

Only the neural self-play and `infer` executables use **mimalloc**, a scalable Rust
allocator: the short-lived vectors created by authoritative move validation caused
system-allocator contention under parallel workloads. Without addressing that,
fixed-position speedups did not translate to whole games. This changes allocation,
not game rules or feature semantics. The offline feature exporter remains serial
per process (the training loader already has multiple processes).

Stderr separates coarse/move extraction time and reports actual encoder positions,
cache hits/misses/evictions, current entries and spatial payload bytes. Logical
full-policy/value query counts include cache hits. Warmup retains the cache, so
`infer --warmup 1 --iterations 3` may report zero measured encoder calls for a
repeated position. Add `--cache-positions 0` to benchmark uncached work; also set
`--feature-threads 1` for a serial baseline. See [benchmarks.md](benchmarks.md).

## Search semantics

The existing unguided `--engine mcts` and direct `--engine random` remain available.
Neural search uses the same authoritative rules, PUCT, progressive-widening budget,
side-to-move backup signs, terminal scoring and visit-count move sampling. It has
no eye-avoidance filter, random rollout evaluator or handcrafted value bonus.
Terminal outcomes override the network. Nonterminal leaf values come from the
network; the full dense policy is deferred until a node is actually expanded.
Most new leaves are never expanded, so computing a million logits for each would
waste most of the inference budget. Dense policy vectors are discarded after
candidate sampling; nodes retain only a small candidate reservoir.

### Huge-action-space adaptation

At an expanded node, Rust extracts/scores all million coordinates, masks them
using shared-rule legality, and computes a stable full-policy normalizer including
pass. **Training and inference do not replace this with a visited-action softmax.**
The subsequent finite-candidate search is an approximation, not full-million-action
PUCT. Its sampled priors follow importance-sampling principles (see
[Sampled MuZero](https://arxiv.org/abs/2104.06303)), adapted to separate the exactly
enumerated pass action from the coordinate stratum:

1. Let `P` be total placement probability and `r(a) = p(a)/P` its conditional policy.
2. Draw a bounded reservoir from `q(a) = epsilon/L + (1-epsilon)*r(a)`, where `L`
   is the legal-coordinate count. Default `epsilon = 0.25`; configure with
   `--independent-exploration` in `(0,1]`.
3. Draw **with replacement**, merging duplicates and retaining multiplicities.
   Candidate weight is `multiplicity * r(a)/q(a)`. Rejecting duplicates instead
   would change the proposal law and invalidate that correction.
4. Progressive widening admits candidates in first-sampled order. Their PUCT
   priors share the full placement mass `P` in proportion to these weights; pass
   retains its full-policy mass. If all admitted weights underflow to zero, their
   placement mass is shared uniformly. This remains a finite-sample approximation.
5. Each newly admitted coordinate gets a real leaf evaluation, even when its
   neural prior is tiny. Uniform proposals can therefore discover undervalued moves.

Simply normalizing raw logits over a handful of coordinates plus pass would make
pass dominate: each coordinate represents one of roughly a million possibilities,
whereas pass is already an entire discrete option. Conversely, sampling from the
policy and then weighting directly by it would count the policy twice. The
stratified importance correction avoids both mistakes. No claim is made that
these initial search settings are optimal; strength testing must guide tuning.

## Records, measurements and limits

PB/PW identify `neural-<checkpoint SHA-256>`. Root JSON comments contain seed,
search settings and model identity; move JSON comments contain actual pre-move
root visits. Completed games have authoritative scored RE; truncated games have
no outcome label. No game sidecars or alternate JSONL records are emitted. Rust's
normal SGF inspector/replay validates neural records exactly like baseline ones.

Stderr reports game time/moves/s plus full-policy calls, value-only calls, feature
extraction, encoder and dense-scorer time. Model loading/startup is outside game
runtime. Native peak memory is bounded by transient full-domain features/logits,
not a million-action distribution retained for every tree node. See
[benchmarks.md](benchmarks.md) for the measured three-game run and memory caveats.

Full-domain extraction is still expensive despite the CPU pool and host context
cache. Batched queries across independent games, retaining encoder context on GPU
(instead of host round trips), and skipping unnecessary
coordinate work are follow-up throughput work. GPU inference alone cannot remove
the CPU extraction cost; do not infer production self-play speed from this milestone.
The iterative self-play/training loop is also still pending: fixed-corpus offline
`--resume` intentionally rejects new games, so a new replay corpus needs a separate
warm-start training contract, not an override of that safety check.
