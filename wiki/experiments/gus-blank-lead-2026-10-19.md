---
title: "Gus blank-suit lead confirmation"
kind: experiment
first_seen: 2026-10-19
last_updated: 2026-10-19
status: complete
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question and frozen test

Does layer 1/unit 65 of [[gus-tiny-gpt]] show a positive response to the current
blank-suit lead across two different lead contexts, after accounting for public
phase, role, held blanks and earlier blank plays? The candidate came from
exploratory cards in [[gus-native-cooldown-2026-10-19]], after the full Cammarata
reread. This study freezes that interpretation before measuring fresh hands.

512 new physical deals are reserved before collection, with development and
confirmation panels of 256 each. Reference play uses G0 in the focal and partner
seats and the same pinned Zeb opponents. The original parent, cosine-final
checkpoint and one untrained twin are frozen. No player training, input change,
search or field change. All 606,794 previous exclusions plus these 512 roots
become 607,306 future training exclusions, including unused/unsupported rows.

The two required contrasts are:

1. With a nonblank suit trump, compare following a double-blank lead with
   following other suits.
2. With blanks trump, exclude every double-blank lead, then compare following
   another blank-containing trump lead with following other suits.

This distinguishes a printed zero from a suit. For example, 4–0 leads fours
with fives trump, but leads the blank trump suit with blanks trump. Labels come
only from the original public prefix and its viewer's own tiles. The game
engine independently checks current lead, remaining hand and public score.

## Measurement and scope

Signed pre-GELU activation is scaled by that coordinate's standard deviation
on the old microscope's discovery rows. Each contrast uses the same frozen
additive ordinary least-squares model: an intercept, current blank-lead flag,
one-hot declaration/trick number/slot/bidder, remaining own blank count, held
double-blank flag, and blank count/double-blank flag in completed earlier tricks.
There is no feature selection, penalty search or neuron fitting. Each row has
equal weight; 2,000 paired bootstrap resamples keep complete physical roots
together and refit the same regression. These are adjusted associations, not
causal effects or nonparametric elimination of all confounding.

Both contrasts in both panels must have at least ten positive and ten negative
physical roots and an identifiable target coefficient. Cosine-final's adjusted
target coefficient must be at least 0.5 discovery standard deviations; its
paired excess over the standardized untrained coordinate must be at least 0.25.
Both 95% lower bounds must exceed zero. These are conjunctive requirements;
no secondary result substitutes for a failure. Coverage/identification failure
is inconclusive and does not permit automatic sampling expansion.

The original checkpoint and cosine-final unit 193 are secondary controls.
Both activation tails, pre/post distributions, literal prefixes and raw outputs
are retained. Examples are the first available root/decision of each class,
selected without consulting activation. Upper/lower cards are descriptive only.
No exclusive neuron meaning, global cleanliness improvement, new causal steering
result or playing-strength claim follows from this test.

## Result: one component replicates; the broader test is inconclusive

The combined registered criterion did not pass. Following a double-blank lead
under nonblank trump strongly predicts this unit's preactivation in both fresh
panels after the specified adjustment. The broader blank-suit hypothesis remains
unconfirmed because the independent development panel lacks enough non-double
blank-trump examples.

| Registered contrast | Development | Confirmation |
|---|---|---|
| Double blank led; another suit trump | +3.398 SD, 95% [3.198, 3.586]; component passes | +3.171 SD, 95% [2.906, 3.423]; component passes |
| Other blank tile led; blanks trump | Inconclusive: 5 positive/9 negative roots | +2.002 SD, 95% [1.431, 2.214]; component passes, 10 positive/13 negative roots |

The ordinary contrast has 94 and 82 positive roots. Its paired excess over the
standardized untrained coordinate is +3.314 [3.029, 3.600] and +3.012
[2.713, 3.309] SD. Original-parent coefficients are +3.188 and +3.018 SD, so the
response clearly predates cooldown. Cosine control unit 193 is near zero in
this ordinary contrast (+.077 and +.124 SD, both intervals include zero).
Different per-model normalization prevents treating these coefficients as a
cleanliness improvement comparison.

The development blank-trump contrast has only 39 rows from nine roots. Some
bootstrap samples cannot identify the target coefficient, and maximum row
leverage reaches one. Its stored pseudoinverse estimates and percentile
quantiles are therefore not valid confidence evidence for the hypothesis.
The confirmation component cannot substitute for this missing replication.
The control unit also responds in that confirmation subset (+.844 SD,
95% [.261, 1.279]); no exclusive blank-suit meaning is established.

The frozen bid-30 schedule yields only nine blank-trump roots in development
and thirteen in confirmation; declarations are not sampled uniformly. More
unselected hands would be an inefficient way to ensure these near misses.
A separately registered conditional sample could investigate that question,
but no expansion was run and this study's verdict stays unchanged.

Context checks on the same reference histories: cosine-final raw next-tile NLL
is .3690/.4006 in the two panels, versus original .4175/.4694; agreements are
86.16%/85.04% versus 84.32%/82.14%. These describe G0 reference moves, not teacher
optimality, unit cleanliness or wins. Pre-GELU and post-GELU tails are archived
separately; a low preactivation is not a negative output vote.

## Execution and evidence

Recovery root: `scratch/lem/gus-blank-lead-v1`. Source cf37ae36; three scripts
frozen before public registration. Eight collection segments are capped at
45 seconds cooperative/90 seconds outer; replay/manual-forward audit at
180 seconds, summary at 120 seconds. All heavy stages use `cpu-slot`. Completed
immutable units resume without duplication; failures require inspection rather
than automatic retries or budget growth.

All five executed stages passed: three collection segments, audit and summary.
All 512 hands and 3,584 prefixes replayed; all 10,752 manual model-row checks
matched stored pre/post activations and logits bit-identically. Four full units
replayed in reverse order with identical bytes. Independent row-level least
squares matched point estimates and the first twenty bootstrap estimates per
contrast/panel within 3.3e-12. There were no operational failures or cap increases.
Total stage cost was approximately 2.27e12 retired instructions; 104.5 CPU seconds
and 104.7 wall seconds are descriptive. Existing weights are referenced by hash
and prior archives rather than uploaded again.

[Public registration](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/06b1f628e8246cbee02cb5a4b0ec36b10b6e8ef9/lem/gus-blank-lead-2026-10-19/registration)
was freshly verified before collection: six files, five archive members;
registration SHA256 `f0070e482ba505a2a8cdad0162d5b19959c8a8b2cafbe06ec5da3e5317b2b84b`.

The boundary refresh found Lem #207 and Book #121 still awaiting replies and
Silas #166 open. Existing prefix replay and native capture suffice for this
question; no new collaborator request or unmerged capability is required.

[Verified evidence](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/259f8940608bc92c8bca3da424bab4183024cb08/lem/gus-blank-lead-2026-10-19/evidence):
eight files, 547 archive members, 11,154,190 bytes, with fresh anonymous
file/member SHA256 verification. Original failures of statistical coverage are
preserved; no operational failure occurred. The archive includes the frozen
protocol, scripts, exclusions, complete reference records, engine/manual-forward
receipts, regression diagnostics, both poles and literal first-row examples.
Supplementary probabilities are deterministic renderings of saved logits and
legal supports; they do not introduce new tests or example selection.
