# Nonterminal pass / critic audit — 2026-11-06

User observed increasingly early passes, explicitly including developed midgames,
not merely opening passes. Read-only audit of the active cg4 campaign; no search,
training, frozen Python, checkpoint, optimizer or campaign settings were changed.
Artifacts: `scratch/20261106-182938-pass-audit/`.

## Data and pass evidence

Snapshot: 500 bootstrap games and the 50 committed games in each of rounds 1–11
(1,050 games total). Round 12 was active and is excluded. Rust/shared SGF codec
and full authoritative replay validate moves and labels; Python does not parse
SGF/rules. Generated records are not modified. `inputs.json`, `games.json` and
`summary.json` retain inputs and per-game/move diagnostics.

Exclude the final two ending passes. In round 11 there are 118 other passes on
boards with at least ten stones (2.36/game); 95 occur while currently ahead on
raw board area and 23 while behind. **87.3%** are the most-visited action. These
passes are followed by later placements; switching sampling to argmax would not
remove most of them. Board area is a diagnostic, not a nonterminal training label
or an oracle proving a pass strategically bad.

Example: round-11 game 3 (seed 10,100,502), pre-move ply 55: **41 stones**, pass
**157/256 visits**, followed by **49 more placements**. Other developed-board
examples appear in `developed-pass-cases.json` and `fixed-cases.json`.

The trend is not monotonic: median game length is 80.5 / 37.5 / 21 / 22.5 /
8.5 / 10.5 / 24.5 moves in rounds 5–11. Round 10 contains 11 empty-board draws
out of 50 games; round 11 rebounds to three. Median first passes in those rounds
are 44.5 / 18.5 / 5 / 10 / 3 / 3 / 8. Do not describe this as smooth irreversible
collapse or mix empty-board draws with nonterminal passes.

## Fixed-position network probes

CPU PyTorch inference on eight failure-conditioned developed-board cases, four
currently ahead/four behind, with unchanged inputs across checkpoints 0–11.
Pass priors can be very large: for the actor that generated round 11, one
39-stone position has coarse pass probability **0.994**. The mean pass prior on
this selected set is 0.674 for checkpoint 10 and 0.227 for checkpoint 11: it
fluctuates, rather than increasing every checkpoint. These are selected witnesses,
not an unbiased population estimate. `probes.json` records critic and prior values.

The network has no previous-pass feature, so it cannot directly distinguish
otherwise identical board/side-to-move states with different immediate-ending
options. Also, at a newly visited leaf the neural estimate can fall below the
exact utility achievable by a terminal pass. These are structural issues worth
testing, not established explanations for all observed passes.

A separate diagnostic-only evaluator wrapper lower-bounded leaf utility by an
available terminal-pass utility, without forcing actions or altering priors. Four
selected positions, one paired seed, budgets 32/256, native CPU inference, same
actor checkpoint: very small changes, e.g. pass visits 17→16 and 41→40 in one
case; the two ahead cases were unchanged. This **does not fix the main symptom**
on this tiny sample. It is not an actual-game RNG replay, a training intervention,
a strength match or a general causal result. `search-probes.json` retains details.

## Critic learning on fixed held-out games

The user's hypothesis that the outcome/score heads have learned little is
particularly supported **in the middle of games**, not equally at all stages.

Choose 32 games deterministically by game ID from checkpoint-11's inherited
validation manifest (all selected games are neural self-play, not bootstrap).
Use the first/middle/last eligible positions: **96 fixed inputs**, evaluated on
CPU across checkpoints 0–11. The complete-game split ledger remains untouched;
these are held out across the training lineage. `critic-references.json` pins
paths, file hashes, game IDs and plies. Stages are move-count proxies, not a
strategic endgame oracle. Labels are actual completed self-play results and
scores, not perfect-play/counterfactual action values.

Checkpoint 11:

| Fixed stage (32 positions each) | Outcome MSE | Best constant-mean MSE | Score MSE | Best constant-mean MSE |
|---|---:|---:|---:|---:|
| First eligible / empty-board | 1.055 | 0.984 | 0.186 | 0.167 |
| Middle eligible | **1.119** | **0.902** | **0.183** | **0.147** |
| Last eligible | 0.618 | 0.527 | **0.043** | **0.115** |

Constants are fitted to each held-out subset's own mean, an optimistic descriptive
baseline rather than a separately trained predictor. Outcome/score midgame
correlations are **0.031 / -0.105**; neither head beats its constant baseline
there. The late score head does have useful signal (correlation 0.897, R² 0.623),
so claiming both heads have learned nothing would be inaccurate. Empty-board
inputs are identical and intrinsically cannot predict individual stochastic game
outcomes; these should not dominate interpretation of overall errors.

Across all 96 positions, outcome MSE is 0.931 vs constant 0.993 (R² 0.063), score
MSE 0.137 vs constant 0.166 (R² 0.172). Score prediction std is 0.151 vs target
std 0.407. This is weak overall prediction with stronger late score recognition,
not constant/dead heads. Replay-mixture training logs alone cannot establish
learning progress; the fixed-input report avoids that comparison problem.

## Next investigation, not applied

Focus on critic learning/calibration and a controlled fixed-data critic-only
fit/overfit test before more search, another reset, a pass ban or a move-count
threshold. Establish whether the critics can fit real positions at all, then
separate supervision/loss interference, optimization, feature aliasing and model
capacity. Weak critics plus pass-heavy search targets can form a reinforcing
policy/self-play loop, but this audit does not prove a single cause. No active
campaign changes have been made.
