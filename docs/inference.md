# ONNX export and native neural MCTS

The trained network can now guide native Rust self-play. The tested backend is
**ONNX Runtime 1.22.1 on CPU**, FP32, one synchronous query at a time. Training can
still use CUDA; exporting a CUDA-trained checkpoint does not require GPU inference.
GPU providers, batched search and browser integration are **not implemented/tested**
here. This is a working integration, not a claim of competitive playing strength.

## Run

`mise run setup` installs the inference extra. For an existing environment:

```sh
.venv/bin/python -m pip install -e '.[inference]'
mise run export -- --checkpoint runs/offline-training/best.pt --output runs/onnx-initial
mise run neural-selfplay -- --model runs/onnx-initial --games 3 --sizes 9,13,19 --seed 200 --simulations 8 --max-moves 64 --inference-threads 4 --output runs/neural-initial
```

Use fresh export/game directories. `neural-selfplay` supplies `--engine neural`
and locates the CPU runtime library from the Python wheel. The actual search and
inference are **in-process Rust/native code**; Python only locates the library
before launching Cargo. Baseline `mise run selfplay` needs no runtime library.

For a standalone release binary, set `ORT_DYLIB_PATH` or pass `--ort-library`:

```sh
export ORT_DYLIB_PATH="$(.venv/bin/python -m continugo_train.export --runtime-path)"
cargo run --locked --release -- --engine neural --model runs/onnx-initial --sizes 9 --simulations 8
```

The dynamic Rust `ort` binding is pinned to the ONNX Runtime 1.22 API. Use a
compatible shared library with its dependencies installed; merely pointing at an
arbitrary `.so` is not supported. There is no silent switch to a baseline evaluator
if model loading or inference fails. Thread spinning is disabled so idle runtime
workers do not contend with Rust feature extraction.

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
cargo run --locked --release --bin infer -- --model runs/onnx-initial --sgf FILE --ply 1
# Cheap leaf-value path; no million-coordinate extraction/scoring:
cargo run --locked --release --bin infer -- --model runs/onnx-initial --sgf FILE --ply 1 --value-only
```

Set the runtime environment as above. Diagnostic `--actions` only selects which
raw logits are printed; the normal diagnostic still computes the full policy.
Illegal scores print as null rather than invalid JSON infinity.

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

Full-domain extraction and CPU scoring are still expensive. GPU inference,
parallel/batched queries and skipping unnecessary coordinate work are follow-up
throughput work; do not infer production self-play speed from this milestone.
The iterative self-play/training loop is also still pending: fixed-corpus offline
`--resume` intentionally rejects new games, so a new replay corpus needs a separate
warm-start training contract, not an override of that safety check.
