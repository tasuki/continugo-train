# Neural game microscope

Generate a **single self-contained HTML file**, with embedded PNG images, inline
JavaScript/CSS, and a downloadable embedded source SGF. Open it directly in a
browser: no server, Internet, image directory or Python installation is needed to
view it. Use **← / →**, Home/End, the slider, or the navigation buttons. Click any
image to enlarge it. The dashboard targets a large desktop screen (1920×1080 or
larger); smaller screens reflow and scroll.

## Existing training or rated game

```sh
mise run nn-view -- --sgf GAME.sgf --checkpoint MODEL/checkpoint.pt
# Different networks playing Black and White:
mise run nn-view -- --sgf GAME.sgf --checkpoint BLACK/checkpoint.pt \
  --white-checkpoint WHITE/checkpoint.pt
```

Default output is `GAME.nn.html`, beside the source game; `--output NEW.html`
chooses another path. Existing reports are never overwritten. Checkpoint hashes
are checked against neural identities in generated SGFs; both colors must match.
For an unannotated SGF, the supplied checkpoint is the explicitly chosen analysis
network. A checkpoint trained after a game is not interchangeable with the frozen
network that played it.

## Generate alongside new games

```sh
mise run neural-selfplay -- --model MODEL --games 1 --sizes 9 \
  --nn-html --nn-checkpoint MODEL/checkpoint.pt

mise run arena -- --candidate CANDIDATE --incumbent INCUMBENT --games 2 \
  --nn-html --nn-checkpoint CANDIDATE/checkpoint.pt \
  --nn-opponent-checkpoint INCUMBENT/checkpoint.pt
```

In arena, `--nn-checkpoint` always means the **candidate**, regardless of its color;
`--nn-opponent-checkpoint` means the incumbent. Against `--reference-random`, omit
the opponent checkpoint. **Random turns are skipped**, never analyzed with the
opponent's network. Navigation preserves the original move numbers.

Native `--nn-html` requires neural play and a real output directory (not stdout).
It runs an additional CPU diagnostic replay **after** saving each SGF, without
changing search, RNG streams, training targets, or ratings. It is deliberately
off by default: complete-cell rule extraction, extra neural evaluation and image
encoding add work and potentially large output files. Failures preserve the SGF.
Training self-play remains opt-in. **All newly committed rated games always get
`.nn.html` sidecars in `models/ratings/games/`**, including supervisor/manual imports,
without requiring an arena flag. A rated ledger commit waits for atomic report
publication. Re-importing a game repairs a missing/stale sidecar without re-rating.
Reports use each mover's own hash-matched checkpoint. Random, unavailable, and
unsupported-architecture turns are omitted and listed with reasons. If no turns
are supported, the sidecar says so and still embeds the original SGF; it does not
invent heads or borrow the opponent's network. Historic policy/value-only
checkpoints remain unsupported by this viewer, not migrated into new networks.

Generation needs the normal Python environment and an up-to-date Rust feature
binary: `mise run setup` / `mise run build-features`. The checkpoint is necessary
because production ONNX exports omit ownership. Standalone native binaries find
`.venv/bin/python` and `target/release/features` under this checkout; override with
`CONTINUGO_PYTHON` / `CONTINUGO_FEATURE_BIN` if needed. The offline command also has
`--feature-binary` and `--threads` (CPU, default one).

## What is shown

Each page is a **pre-move position**, relative to the player about to move:

- Authoritatively replayed physical board, with a cross at the move played.
- Every coarse input plane, named and scaled: twelve in either schema, with
  different meanings. Draft schema 5 shows antialiased occupancy, connected group
  size, both liberty-center and liberty-footprint unions, and borders. It has no
  scalar liberty-area or capturable-in-one planes. Legacy schema 4 keeps its actual
  point-sampled inputs and flags; its uncapped scalar liberty area is shown in raw
  stone-disc units, not clipped to one.
- Coarse cell logits and global softmax probabilities, plus separate pass mass.
- Diagnostic selected-cell coverage, exact legal raw logits, conditional
  `P(move | cell)`, and redistributed final placement probabilities.
- All 10 exact coordinate input features, including legality, connections,
  captures, resulting group size and border geometry.
- Outcome value, expected points percentage, normalized score and stone-unit
  score, their score-weighted head utility, final labels when available, and
  policy-training eligibility.
- Ownership prediction, validated final ownership and prediction error when the
  game has a scored natural result. Truncated games never acquire fake targets.
- The actual shared pooled-trunk input to the value/score MLPs, their hidden
  activations, a spatial policy-context RMS summary, and expandable individual
  projected policy-context channels (the inputs sampled by the exact policy).
- Recorded MCTS visit heatmap, pass visits and totals, distinct from NN priors.
- Expandable top diagnostic-policy action table, with exact feature values.
- Checkpoint and SGF SHA256 provenance.

Signed maps use blue/neutral/orange; nonnegative maps dark/bright. Every image
shows its range; probability/visit intensities use a documented log display.
Exact-coordinate images are **256×256 display bins**: probability and visit mass
sum within a pixel; features and logits average. Checkerboard pixels have no
evaluated coordinates, not necessarily illegal coordinates. The network itself
still receives exact coordinate rows, not those display averages.

### Important scope distinction

This is an FP32 PyTorch **checkpoint replay**, not a live inference/search trace.
Rust supplies all rules, replay, geometry, input planes, exact move features and
labels. Each position scores **complete top-ranked coarse cells plus cells
containing the played move or positive recorded visits**. Local probabilities
normalize over every legal coordinate in each such cell. As in native inference,
placement mass is redistributed over selected nonempty cells while preserving
global coarse pass probability; if no legal coordinate is selected, pass is one.

Live search instead uses independently seeded exploration cells and can propose
additional whole-board actions, evaluate many child positions, and optionally
use rollout critics. Its exact cell selection and tree backups are **not**
reconstructed here. The report labels its diagnostic scope and shows the actual
recorded root visits separately. Ownership is auxiliary, never a search input;
the displayed head utility is not an MCTS Q-value. Small ONNX/PyTorch floating
point differences are possible, as covered by export parity checks.
