# Spatial search, fragmented visits and excessive passing — 2026-11-09

Discussion following the [round-70 branch pilot](2026-11-09-r70-branch-pilot.md)
and [exact terminal-pass follow-up](2026-11-09-r70-terminal-pass-pilot.md).
The user asked for useful architectural intuition and a durable summary of the
situation and possible solutions, not another large controlled training campaign.

**This is an important search-design concern, not merely evidence that the
network needs more training.** We have a concrete position where a profitable
capture survives four short continuations, the immediate critic recognizes it,
yet ordinary search makes pass the most-visited individual action. This warrants
revisiting the search/policy interface. It does not establish that one mechanism
explains every opening pass, endgame pass or eye fill.

This document records the discussion and candidate remedies. No remedy is
implemented or selected here; no production code, ongoing campaign, weights,
optimizer, replay or accepted publication is changed.

## How the current design arose

The three often-described approaches are compatible layers, not necessarily
mutually exclusive alternatives:

1. **Hierarchical spatial policy:** factor a move probability into a coarse
   region probability and a conditional coordinate probability.
2. **Sampled-action MCTS:** propose a small set of concrete moves and search it.
3. **Progressive widening:** introduce more searchable children as visits grow.

Git history shows incremental construction rather than a fresh comparison of
complete search designs:

- **2026-10-30, `a582b78`:** initial unguided MCTS already has flat coordinate
  children and progressive widening.
- **2026-11-01, `71c510e`:** neural inference and sampled coordinate candidates
  feed that search, retaining progressive widening.
- **2026-11-04, `63b06e2`, “Calculate exact analysis only for selected cells”:**
  coarse/fine cell policy replaces full-domain exact scoring to reduce feature
  extraction/scoring cost. The search structure is essentially retained; the
  commit changes only six lines in `src/search.rs`.

The coarse/fine work solved a computational problem. It did not make MCTS itself
hierarchical or establish that flat coordinate visit counts are the best final
recommendation for a finely discretized spatial action space.

## What happens now

At a policy-expanded game position:

1. The coarse head scores all cells plus pass. Inference selects **32 top cells
   and 8 exploratory cells**.
2. Exact legality/features and fine-network scores are computed for every
   integer coordinate in those cells. Within-cell softmaxes combine with coarse
   cell weights to produce coordinate probabilities.
3. Pass keeps its global coarse probability. Selected nonempty cells share the
   remaining placement probability. The earlier selected-cell normalization bug
   that artificially inflated pass was [fixed separately](2026-11-04-pass-normalization.md).
4. A finite coordinate reservoir is sampled with replacement from a mixture of
   **75% conditional placement policy and 25% uniform exploration within the
   selected legal domain**. Duplicate coordinates retain multiplicity through
   importance weights. Reservoir draws are approximately **12 at a 32-simulation
   budget** and **33 at 256**.
5. Progressive widening admits candidates under approximately `2*sqrt(visits)`.
   Each newly admitted coordinate gets a real evaluation. Candidates come from
   the pre-sampled reservoir; this is not indefinite fresh proposal generation.
6. PUCT allocates further visits among **individual concrete coordinates and
   pass**. Every coordinate has its own value statistics and visit count.

There are **no MCTS cell nodes or cell-level visit budgets**. “A cell received 20
visits” means only that its coordinate visits sum to 20 after search. The fine
network does not redraw a coordinate inside that cell on each of those visits.

Evaluation selects a maximum-visit coordinate/pass. Self-play samples individual
root visits at **temperature one**. Training aggregates placement visits into
coarse-cell targets and also learns their conditional within-cell distribution.
The campaign trains policy only on expensive **256-simulation** positions;
cheap **32-simulation** positions still train the critics. The recent probes
therefore directly characterize cheap play/evaluation, not measured 256-visit
policy targets at this witness.

## Evidence from round 70

Pinned model: `cg5-b4c32-s570000-d177833`, SHA-256
`b3fd8e849cd18b5074ed3441bb8f53c7f89cd3ff69a130d91992d0907638c305`.
Full source positions, rules validation and caveats are in the two pilot notes.

The first pilot's 13-stone capture comparison was inconclusive: later tactical
swings overwhelmed the initial difference. Its immediate child critic still
recognized the capture, and two of four root searches did not instantiate a
13-stone capture at all.

The cleaner follow-up used a different game, **Black to move at ply 84**, after
White passed. Black can end the game immediately at exact **+67.47**, or play
**B[rtrG] / (903,916)** and capture eight White stones. Four frozen-network
continuations after the forced capture naturally finished at:

**+83.52, +88.56, +89.71, +82.57** — gains of **15.10–22.24**, mean **18.62**.

Ordinary 32-visit root searches produced:

| Seed | Pass visits | All eight-stone captures together | Capture children | Largest single capture visits |
|---|---:|---:|---:|---:|
| 7101 | 6 | 8 | 4 | 2 |
| 7102 | 6 | 6 | 4 | 3 |
| 7103 | 5 | 10 | 4 | 3 |
| 7104 | 5 | 11 | 5 | 3 |

**Pass is most visited in all four**, despite capture alternatives in every
search. The coarse pass prior is **24.47%**. The exact terminal-pass utility is
**0.82059**, whereas the immediate capture child predicts **0.96741**, using the
actual equal outcome/score mixture and the same Black-relative perspective.

Thus an explanation consisting only of “the critic cannot see the capture” is
inadequate for this witness. However, these immediate estimates are not the
backed-up root-child Q values after search; those were not instrumented.

## Why this is concerning

### One pass versus many versions of a useful move

Many nearby coordinates can accomplish essentially the same tactical purpose.
The search nevertheless treats them as separate actions. Their search budget
and visit counts can fragment while pass retains one bucket. For example:

```
capture coordinate A: 3 visits
capture coordinate B: 3 visits
capture coordinate C: 2 visits
capture coordinate D: 2 visits
pass:                 5 visits
```

Maximum-visit recommendation chooses pass even though the capture family has
more combined visits. Ordinary grid Go also has several good moves, but a fine
coordinate discretization creates many more near-duplicates of one intention.

### Exploration becomes actual play

PUCT visits measure both exploitation and exploration; they are not pure
endorsements. A probable but lower-valued pass can keep receiving exploratory
visits. At a small budget those visits are a substantial fraction of the root
policy. Temperature-one self-play turns that mass into actual passes, and
exploratory visits in policy-enabled searches can also become supervision.
A pass-heavy prior, short searches, pass-containing targets and future self-play
can therefore reinforce one another. The strength of that feedback in this
campaign has not been isolated.

### Important distinction: argmax fragmentation versus sampling

**Simply summing visits into cells does not, by itself, reduce temperature-one
pass probability.** Sampling a cell in proportion to summed visits, then a
coordinate in proportion to its visits inside that cell, gives exactly the same
coordinate/pass distribution as flat sampling. In the follow-up, actual pass
probability is **5/32–6/32**, not 100%, despite pass being the argmax every time.

The discussion's phrase “pass wins” must not conflate those two facts. Aggregated
recommendation can change argmax selection; changing sampled play requires
changing search allocation, target construction or the final sampling rule.
There is also no measured claim here that pass receives fabricated total prior
mass: current normalization explicitly preserves it.

## Possible solutions

### A. Make the search genuinely hierarchical

Allocate search between **pass and spatial cells/regions**, then allocate within
a chosen region to concrete coordinates. Keep statistics at both levels so a
family's evidence can influence coarse allocation and final recommendation.
The existing coarse/fine network is a useful proposal model for this direction.

This must be a real allocation change, not post-hoc grouping followed by the
same proportional sampling. Design decisions include how region Q represents
its coordinates, how exploration is divided between levels, and how coarse and
fine training targets correspond to the resulting search policy. A virtual
cell choice is **not a game move**: it must not change player, flip backup signs,
advance the record or consume a game ply before a coordinate is applied.

Limits: raster cells are not strategic equivalence classes. Equivalent captures
can cross cell boundaries; one cell can contain tactically very different moves.
Region aggregation must not hide a narrow useful coordinate or average it away.
This is an architectural option, not a proven automatic cure.

### B. Keep flat search, but reduce redundant coordinate competition

Use more diverse/stratified proposals: fewer nearly identical coordinates per
region initially, with local refinement where search finds promise. Alternatively,
use spatial clusters or representatives to share proposal budgets or to construct
a hierarchical final recommendation.

Limits: close coordinates can have different legality, captures and survival.
Capturing the same enemy group, or producing the same current area score, does
not prove two resulting states interchangeable. Avoid unsound state merging or
removing exact tactical distinctions. Shared proposal budgeting is less drastic
than claiming identical game states/values.

### C. Separate exploration allocation from the final move policy

Retain exploratory PUCT visits but construct a recommendation/policy target that
better reflects estimated action quality rather than raw small-budget counts.
Options include exploration-aware target pruning, a regularized value-based
root policy, or a dedicated finite-budget root policy-improvement method such as
Gumbel/sequential-halving search adapted to sampled actions.

These are alternatives to blanket temperature-one play, not justification for
blind maximum-Q selection: a coordinate evaluated once can be optimistically
wrong. Sampled priors, importance weights, uncertainty and terminal exact values
need consistent treatment. Purely lowering temperature or taking visit argmax is
**not enough** for the demonstrated case: pass already wins visit argmax.

### D. Retune how the existing small budget is spent

Lower exploration pressure, narrow widening, reserve more visits for comparing
existing promising actions, or selectively extend search when an exact terminal
pass competes with a higher-valued continuation. This is a smaller change than
rebuilding the tree hierarchy.

Trade-offs: narrower search can miss moves; lower exploration can lock in a bad
prior; extra search costs time and does not universally reduce passing. Earlier
[32/256 investigations](2026-11-06-pass-search-budgets.md) already found that more
visits can sharpen a bad pass preference. The current witness has only been
probed at 32 visits, so it does not establish how any of these changes behaves
at 256 or beyond.

### E. Do not treat a symptom mask as the underlying solution

A pass penalty, opening pass ban or permanent eye-fill filter could make games
look better without improving spatial decisions. Passing can genuinely be best;
eye-like placements can be tactically necessary. Likewise, adding previous-pass
input has not been established as a remedy for this failure: search already knows
the exact terminal-pass option, and the immediate critic recognizes the capture.
Another feature reset or simply fitting more of the same targets does not
address the observed allocation/recommendation mismatch by itself.

## Direction from this discussion

Treat the search/policy interface as a priority design issue. The useful next
work is to choose and implement a coherent handling of spatial alternatives and
exploration versus recommendation, rather than repeat ambiguous continuation
pilots or assume all failure lies in critic learning. The frozen round-70
witnesses provide cheap implementation checks, not a demand for another large
training campaign before taking the architectural concern seriously.

Keep the current successful facts intact: exact legality, authoritative terminal
scoring, side-to-move signs, pass-preserving normalization and bounded feature
work. Preserve the ongoing run's frozen implementation; any changed search must
be a separate diagnostic/new runtime, not a replacement inside that campaign.

A change making these witnesses sensible would still not establish overall
playing strength, eliminate rare reversals or explain every early pass/eye fill.
The concern is concrete and important; the preferred remedy remains open.
