# Why the old value-MSE curve looked better — 2026-11-06

Read-only comparison requested by the user, using the trashed but recoverable
`runs/20261105-214158-score-ownership-size9/` (restored from trash)
and current cg4. Nothing is restored into the current store or used for training.
Artifacts: `scratch/20261106-190000-old-critic-comparison/`.

## Matched training progress

Both runs use 50 games / four epochs per round, batch two, at most 1000 steps per
epoch, eight train/four validation positions per game, same learning rate,
momentum, decay, clipping, leaky-ReLU, and **identical architecture and feature
specification**. At round 11 both have optimized **98,000 rows**.

| Metric at round 11 | Old cg2 | New cg4 |
|---|---:|---:|
| Training outcome MSE | 0.6923 | 0.9944 |
| Logged held-out outcome MSE | **0.6068** | **0.9153** |
| Logged held-out score MSE | 0.0450 | 0.1246 |
| Held-out value prediction std | 0.6456 | 0.2206 |

The old curve really is lower at equal update counts, not merely because it ran
31 rounds versus the new snapshot's 11. Different replay/holdout mixtures mean
those losses alone do not isolate learning quality or optimization.

## Substantial generation changes

The archived source confirms that old search forced a terminal winning pass and
filtered a losing pass while other candidates existed. It even filtered a first
pass if a hypothetical cooperative second pass would lose. These restrictions
mechanically delay loss-conceding passes and tend to extend games toward crowded,
already-decided positions. They are not correct score-maximizing rules and should
not be restored merely to improve a loss curve.

Old bootstrap had eye guidance, 32 simulations everywhere, 30,658 eligible rows
before two canonical duplicate games were excluded. New bootstrap omitted eye
guidance, used randomized 32/256 caps, and had 17,405 eligible rows. Policy was
trained on all old positions versus only expensive new positions. **Both critics
still train on all sampled positions in the new run**; the mask does not quarter
their number of supervised rows.

Other changes: old search utility was ~76.9% outcome / 23.1% score; current is
50%/50% linear blending. Score-MSE training weight was raised **0.25→1.0** while
outcome weight stayed 1. This changes shared-trunk optimization as well as future
data. The effect of that loss-weight change is not isolated here.

## Cross-corpus diagnostic

Take 32 held-out games from each round-11 manifest, deterministically by game ID,
and probe first/middle/last eligible positions (96 per corpus). Exclude any game
assigned to training in any compared checkpoint's inherited split ledger. Original
paths in the old manifest are resolved into trash without rewriting records.
Each corpus is extracted by its **own frozen Rust binary**, so the old numeric
SGF policy format is read authoritatively rather than converted in Python.
Matching architecture/feature specs permit CPU PyTorch evaluation by either model.
References retain source paths, hashes, game IDs, plies and stage proxies.

This is a new fixed cohort, different from the earlier pass audit's 96 inputs;
compare numbers within this table, not across the two cohort selections.

| Probe corpus | Old round-11 model, value MSE | New round-11 model, value MSE |
|---|---:|---:|
| Old held-out games | **0.665** | 0.715 |
| New held-out games | 0.929 | **0.853** |

The old model's headline advantage does **not** transfer to current games. Even
the old round-31 model (258,000 optimized rows) scores 1.040 on the new corpus.
Different self-play policies define different future outcome distributions, so
cross-corpus evaluation is a diagnostic, not a controlled strength comparison or
proof of a single cause.

Selected old held-out games average **120.59 moves**, new ones **42.59**.
Stage-proxy outcome MSE:

| Corpus/model | First eligible | Middle eligible | Last eligible |
|---|---:|---:|---:|
| Old corpus / old round 11 | 1.141 | 0.844 | **0.011** |
| Old corpus / new round 11 | 1.037 | 1.029 | **0.080** |
| New corpus / old round 11 | 1.061 | 1.178 | 0.549 |
| New corpus / new round 11 | 1.002 | 1.014 | 0.543 |

Thus much of the easy recognition is on old very-late boards, not broadly strong
midgame outcome prediction. This is not just predicting a constant: the old
late subset's target mean is -0.5, constant-mean MSE 0.75, versus model MSE 0.011.
The current model also recognizes those old late positions well. Score targets
are broader on the new corpus (early score variance ~0.184 versus old ~0.082),
which also makes raw score-MSE curves non-comparable.

## Interpretation

The strongest measured explanation is **changed data difficulty and game-ending
behavior**, with shared-trunk loss weighting and policy supervision as additional
unisolated changes. This supports a controlled fixed-data critic fit/loss-weight
ablation, not reinstating incorrect pass filters, equating low self-play loss
with strength, or claiming the current critics receive fewer value labels.
The active campaign is unchanged.
