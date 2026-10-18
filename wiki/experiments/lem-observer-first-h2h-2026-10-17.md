---
title: "Lem observer: first head-to-head games"
kind: experiment
first_seen: 2026-10-17
last_updated: 2026-10-17
status: active
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
