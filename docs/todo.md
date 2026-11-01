# ContinuGo AI — Training Plan

Build an AlphaZero/KataGo-style AI training system.

Use [KataGo's self-play training](https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md) as the primary reference. Follow its established approaches to MCTS, self-play, training, evaluation, and network scaling wherever applicable, adapting them to ContinuGo.

## Technology and repositories

- **Rust:** Game simulation, MCTS, self-play, and feature extraction. Use the existing ContinuGo Rust rules which live in `../continugo` .
- **Python + PyTorch:** Neural network architecture and training.
- **ONNX:** Export models for native and browser inference.
- **Records:** Use self-contained SGF through the shared codec: `SZ` for board size, `PB`/`PW` for players, and `RE` for the scored result (one decimal place). Reserve JSON comments for seeds/search settings, move-policy targets, and optional profiling; no JSON sidecars or JSONL game records.
- Develop experimental components in `continugo-train`. Move stable, reusable components into `../continugo` (upon my confirmation).
- Maintain a single implementation of the rules and rule-derived features. The existing rules are authoritative; preserve their semantics when optimizing implementation.

## Neural network architecture

Train a single **ResNet with policy and value heads**, supporting the discrete stone sizes provided by the existing rules, corresponding to approximately 9×9 through 19×19 Go boards.

**Coarse ResNet:**
- Scale the input resolution with stone diameter, keeping local stone formations at approximately the same scale.
- Input planes include friendly/enemy stone positions, group sizes, group liberty area matching the UI highlight (in stone-disc units), exact one-move capturability by the opposing color on the unchanged board, and directional distances to board borders. Do not include move-history planes or exactly-one-integer-liberty flags, and do not treat area as a handcrafted moves-to-capture estimate.
- Start small, perhaps **32 channels and 4–6 residual blocks**, with configurable architecture.
- Increase network capacity as training progresses, guided by playing strength relative to computational cost.

**Value head:**
- Use global pooling to handle variable spatial dimensions.
- Predict the position's expected outcome.

**Cheap dense policy head:**
- Combine spatial features from the ResNet with exact rule-derived features for each potential move.
- Features include legality, connected groups, captures, resulting group status, and relevant geometric information.
- Use a lightweight shared scoring network to evaluate all **1,000,000 coordinates plus pass**, without expensive full-resolution convolutions.
- Preserve exact tactical distinctions between neighboring coordinates.
- Consider learned group embeddings later, but don't implement them initially.

## Training

Follow KataGo's approach wherever applicable, with these ContinuGo-specific considerations:

- **Bootstrap training from unguided MCTS self-play** (`--engine mcts`): uniform priors over admitted candidates, neutral nonterminal leaf values, and actual outcomes at terminal nodes. There is no trained network or random rollout evaluator yet. Record root visit counts as policy targets and completed-game outcomes as value targets.
- **Direct random play** (`--engine random`) uses no tree search and supplies no search-policy targets; retain it for rules/performance diagnostics. Its narrow eye-avoidance filter rejects only an enclosed, single-friendly-group pocket that one stone would completely fill; allow larger pockets, captures and connections. Pass only after an exhaustive search finds no acceptable placement. Keep this filter separate from MCTS; do not add heuristic-guided MCTS or handcrafted value evaluations.
- Replace the bootstrap priors/value evaluator with neural guidance in MCTS, retaining PUCT and progressive widening to handle the enormous action space.
- Maintain sufficient independent exploration to discover moves the current policy undervalues.
- Train across the supported discrete stone sizes, with good coverage of 9×9, 13×13, and 19×19 equivalents.
- Batch positions with similar input resolutions efficiently.
- Train the policy from MCTS search results and the value from game outcomes.
- Iterate through self-play, training, model export, and evaluation.

## Performance and experimentation

Prioritize **playing strength per unit of computation**, particularly during self-play.

Benchmark the main costs independently:

- Rust feature extraction across the million-coordinate board.
- Coarse ResNet inference and training.
- Dense policy evaluation.
- Batched MCTS inference and overall self-play throughput.

Keep the architecture configurable to compare network sizes, input features, and policy-head designs.

If evaluating all million coordinates per position proves too expensive, investigate more selective approaches without sacrificing the ability to discover unexpected moves.

## Implementation priorities

1. Build the iterative neural self-play training loop, including warm-start training on a changing replay corpus (separate from fixed-corpus offline resume).
2. Improve neural self-play throughput before scaling runs: parallel/batched queries, native GPU inference, and reducing full-domain feature/scoring work without losing independent exploration or correct policy normalization. The current native backend is synchronous CPU; dense policy is already deferred at value-only leaves.
3. Evaluate playing strength across board sizes and scale the network as justified.
4. Move stable reusable components into the main ContinuGo repository.

**Guiding principle:** Keep the initial implementation simple and performant. Reuse KataGo's proven training methods, while concentrating original engineering effort on ContinuGo's unusual action space, exact geometric rules, and efficient neural policy representation.
