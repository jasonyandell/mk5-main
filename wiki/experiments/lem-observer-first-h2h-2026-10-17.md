---
title: "Lem observer: first head-to-head games"
kind: experiment
first_seen: 2026-10-17
last_updated: 2026-10-17
status: complete
---

## Question and registration

Does the frozen [[lem-observer-corpus-squeeze-2026-10-17]] observer earn better
game outcomes than its untrained twin, random legal play, and the earlier
[[lem-observer-data-scale-2026-10-17]] observer? Jason requested the first
head-to-head step toward improving players. Prediction accuracy alone does
not establish any of these comparisons.

Registered before the measured games: expect the trained model to beat the
untrained and random arms; predict a positive improvement over the 50k
predecessor, with that prediction unconfirmed unless the paired 95% interval
excludes neutral. A self-match must be exactly neutral. No reward updates or
checkpoint selection happen in this experiment.

## Fixed protocol

- Candidate: 107,392 parameters, 500k corpus, validation-selected step 200,000.
- Opponents: 50k corpus validation-selected step 18,500; the candidate's saved
  untrained twin; uniform legal play seeded with the match seed.
- Each comparison: 256 games (128 paired seeds), seed 20261017; both
  partnerships use two copies of their policy. A 32-game candidate self-match
  checks the paired neutral baseline.
- Existing [[arena]] engine, both `Bid30Bidder`, pip declarations, first to
  seven marks. The bidder is left of the shaker and rotates between hands.
  The bidder chooses its strongest pip trump using the existing rule. This
  is a play comparison under a fixed bidding procedure, not learned bidding,
  Nello, or a full-menu tournament rating.
- Same input: original own hand, public bid-30 contract, relative actor/tile
  prefix, ending with the current actor. No features, legal-action list,
  value estimates, identities or match scores are appended.
- Act by the highest legal tile logit; exact ties choose the lowest tile ID.
  Legality is an output constraint computed from the actor's own hand and
  public trick. This adapter is part of the measured player.
- MPS float32, two CPU threads, pooled paired halves. Deterministic argmax
  has no wall-clock thinking limit. Random draws and batch configuration
  are pinned. All heavy runs use `scripts/cpu-slot` with its 300-second cap.
- Primary outcome: game win rate, neutral 50%. Secondary: mark margin per
  game, neutral zero. Percentile bootstrap resamples whole paired game
  seeds, 10,000 draws, seed 42. Hands and team-swapped halves are not treated
  as independent samples. Intervals concern these frozen policies, not
  training-seed uncertainty.
- Every recorded hand is replayed through `scenario.record`; legality and
  banked totals must agree with the arena. Physical deals must be absent
  from the entire 500k source corpus, including its earlier test recordings.
  Complete games, hashes, model provenance and reports are preserved.

Implementation: `lem/observer/play.py`, `lem/observer/h2h.py`. Tests compare
every live prefix to its training counterpart in all nine supported declarations,
perturb hidden hands, exercise variable history lengths and legal-output
selection, and check deterministic replay and paired self-match neutrality.

## Result

The registration and implementation were frozen at `38d3285b` before the games.
All four arms completed at that clean commit. The 500k observer's outcomes:

| Opponent | Games | Game wins, 95% paired interval | Mark margin/game, 95% paired interval |
|---|---:|---:|---:|
| 50k observer, best step 18500 | 256 | **57.03% [52.73, 61.72]** | **+0.609 [+0.367, +0.852]** |
| Saved untrained twin | 256 | **89.06% [85.55, 92.58]** | **+3.227 [+2.980, +3.465]** |
| Uniform legal | 256 | **87.11% [83.20, 90.63]** | **+3.141 [+2.879, +3.398]** |
| Itself | 32 | **50.00% [50.00, 50.00]** | **0.000 [0.000, 0.000]** |

Neutral baselines are 50% wins and zero margin. All three positive-direction
predictions held with their individual, unadjusted intervals excluding neutral;
the self-match was exactly neutral. The improved predictive specimen also plays
better against its predecessor in this setting. This is not a causal estimate
of data size: training updates and source mixture differ too. It does not establish
an ordering against stronger planners or a diverse partner/opponent population.

Across 800 games and 8,448 hands, every recorded play was legal and the public
referee reproduced every score. No evaluated physical deal was in the 500k
source corpus. Candidate checkpoint SHA256:
`99fd106a15ae468dca9c02426541816f1121a2cdef0cc7b96fd37f2122f31d8d`.
The candidate and predecessor both kept their validation-selected weights;
there was no head-to-head-based checkpoint selection or weight update.

The four jobs consumed 3.702e11 CPU instructions, 25.6 CPU seconds and 29.9
wall seconds in total, including source loading and audits. MPS inference
work is not included in CPU instructions. These are descriptive costs, not
policy thinking budgets or a comparison between machines.

[Frozen evidence, d92c5c9](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/d92c5c9ebe7b3568f5f8e081c14885131a2e1790/lem/observer-first-h2h-2026-10-17)
preserves all game records, manifests, reports, cost records and check logs
(793,758 bytes). A fresh download of the predecessor-match replay matched
SHA256 `48575f7ffe64d1d4b23dd41c178d4d631adb3b25487f6dac522e9a1258dc15b9`.
`scripts/check --fast` passed all 47 Lem tests and the wiki, conflict and
license gates. The implementation is [PR #211](https://github.com/jasonyandell/mk5-main/pull/211).

## What this prepares

[[lem-nanogpt-foundations]] describes the improvement loop. These games establish
the starting player; playing them does not update its weights. Future reward
training needs separate training, selection and final evaluation deals. The
checkpoint archive remains immutable for native-neuron formation studies.

These are homogeneous partnerships. Mixed partners, persistent opponent
encounters, diversity and payoff cycles remain the broader league work in
[#182](https://github.com/jasonyandell/mk5-main/issues/182). The model observes
within-hand history; it has no cross-hand memory. No native-unit interpretation
is performed by this matchup.
