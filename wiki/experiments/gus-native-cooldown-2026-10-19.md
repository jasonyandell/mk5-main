---
title: "Gus: reading native units across the saved cooldown"
kind: experiment
first_seen: 2026-10-19
last_updated: 2026-10-19
status: complete
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Direction after the full reread

Jason requested a fresh reading of Cammarata before another experiment. The
entire official [essay text](https://already-clean.goodfire.com/essay.md), all
943 lines including appendices and references, was fetched and read. Its SHA256
is recorded in the public registration; the article remains at its publisher.

The essay distinguishes learning failures from instrument failures. Residual
connections and LayerNorm are explicit architectural examples; the improved
model-training regularization recipe is unspecified. Visualization regularization
is separate. Native mechanisms, both activation poles, full context, downstream
effects and development over training supply complementary evidence. These are
working hypotheses and methods, not a guarantee that a particular schedule
produces clean neurons. [[clean-neurons-42]] owns the broader thesis.

[[gus-tiny-gpt]] already has pre-LN residual blocks, GELU and AdamW decay.
[[gus-modern-learning-recovery-2026-10-18]] already compared matched constant-rate
and cosine continuations, with improved forecasting for cosine. Its saved
formation checkpoints make a native comparison possible without more training.
The preceding [[gus-public-score-probe-2026-10-19]] did not establish a linear
score-readout advantage; that result does not require imposing score supervision
before inspecting what the original net actually computes.

## Registered exploratory microscope

Eight coordinates were chosen before measurement: four without replacement per
layer, PCG64 seed 421019. Zero-based coordinates are layer 0 units 44, 86, 113, 236 and
layer 1 units 65, 70, 86, 90. These are native 256-wide MLP coordinates, not fitted
probe directions or dictionary entries.

Eight frozen snapshots: original parent, architecture-matched untrained twin,
and constant/cosine continuation steps 250, 1000, 2000. The continuations share the
original training comparison's starting weights, optimizer and minibatches.
This follows later refinement, not initial formation from random initialization.
Matching coordinates does not assume their meanings stay fixed.

The first 256 already-reserved public-score roots supply 1,792 original focal
prefixes. First 128 are discovery, next 128 a replication panel. Both are held out
from model training but reused from another diagnostic; they are not new
independent confirmation of a chosen semantic explanation or playing strength.
No games, training, population change, new inputs or root reservations occur.
All 606,794 existing physical-root exclusions remain. Reference hands use fixed
G0 focal/partner and pinned Zeb opponents.

For every unit/snapshot/panel, cards retain the six strongest upper-tail and
lower-tail examples from distinct hands, six median-near examples, raw
pre/post-GELU quantiles, complete decoded prefixes and raw/legal outputs.
The upper tail can contain negative values: card labels positive/negative denote
upper/lower tails, not a guaranteed sign. GELU's strongly negative preactivations
can be almost silent; negative preactivation is not a negative downstream vote.

Original, both final continuations and the untrained twin receive actor-site
pre-GELU shifts of −1, −0.5, 0, +0.5, +1 discovery standard deviations. The same rows
receive a control intervention at unit (u+128)%256 in the same layer, using that
unit's own scale. Earlier token sites, weights and inputs remain unchanged.
These are local interventions on a fixed history, not deployed player variants.

## An exploratory lead, not a certified name

Layer 1/unit 65's six strongest discovery cards in both original and cosine-final
Gus have double-blank leading the current trick. The second panel also contains
4-0 leads when blanks are trump. A **blank-suit context response** is therefore
a candidate explanation; merely detecting token 0-0 is too narrow. Turn, bidder,
declaration and held-tile correlations still need matched near misses.

After viewing the cards, summing raw probability over blank-containing tiles
provides a descriptive output check. On each of the six positive cards in both
panels, increasing this unit by one standard deviation increases that mass in
the original, constant-final and cosine-final models. The same-row control is
smaller or inconsistent. This summary was chosen after inspection and is not a
registered semantic confirmation. Raw mass includes unavailable tiles; several
rows have only one legal action, so their legal distribution cannot change.
The pattern already exists before cooldown and does not establish cleaner units.

One illustrative discovery position, row 19:0, has fives trump and left bidding 30.
Own seven: 3-3, 4-0, 4-1, 4-4, 5-0, 5-1, 6-0. Public play: left 0-0, partner 2-0, right 3-0.
Legal replies are 4-0 and 6-0. In cosine-final Gus:

| Shift to layer 1/unit 65 | P(4-0), legal | P(6-0), legal |
|---|---:|---:|
| −1 discovery standard deviation | .433290 | .566710 |
| No shift | .428306 | .571694 |
| +1 discovery standard deviation | .423790 | .576210 |

The effect is small and smooth, with no preferred-move change. The constant
continuation reverses the direction between these two tiles: a broad response
pattern is not a permanent precise output role. This example was selected after
inspection, not as a representative or favorable outcome certificate.

Layer 0/unit 113 offers another lead: its strongest discovery cards have 5-4 led
with fours trump. Other coordinates show phase/position structure or unresolved
patterns. None receives a certified semantic label. Original-to-cosine
preactivation correlations across the eight coordinates range .9629–.9930 on
these rows; this describes coordinate continuity, not preserved meaning.

Raw next-tile NLL to these fixed G0 reference hands is .4252 originally,
.4485 after constant continuation and .3732 after cosine. This differs from the
original teacher-validation distribution. It supplies task context, not a win
rate or cleanliness metric. No global clean-share estimate or player promotion.

## Audit, bounds and publication

All 14 controller stages completed: eight capture units capped at 90 seconds,
four intervention units capped at 120 seconds, audit 180 seconds and summary 90 seconds.
Each heavy stage used `cpu-slot`; each completed model unit is resumable and
immutable. No automatic retries or cap increases. Source 36f315c8; scripts frozen
before registration; no production model implementation changed.

The audit checked all 256 original immutable record hashes, exact zero-dose
behavior, hook cleanup and 96 separate explicit-forward checks. A supplementary
reverse-order audit verified all 640 stored intervention batches, 18 rows each,
bit-identically against the explicit forward implementation. Its source and
receipt are archived alongside the cards. Native capture files, original
prefixes, small-dose outputs, controls, formation cards and descriptive readout
are retained. Original full model assets remain in their prior public archives.

- [Verified public registration](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/6dd8b3cce1dfc636460ea1dfa046d4772b35bfa0/lem/gus-native-cooldown-2026-10-19/registration): 6 files, 5 members; registration SHA256 `a7121cbd5585c0dda06329bd5bfc4eba6567ca58e42768b9d2a47917715e6160`.
- [Verified evidence](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/6e63f720b40155489ba02538044d8ee586f83715/lem/gus-native-cooldown-2026-10-19/evidence): 8 files, 79 members, 152,183,639 bytes; fresh anonymous file and member SHA checks passed. Archive order, timestamps and ownership are normalized.

The boundary refresh found no new Lem #207/Book #121 replies and Silas #166 still open.
No reping or new verifier was needed. This scratch microscope adds explicit
GELU card/intervention semantics, not a completed `lem.probe` transformer adapter.

A useful next question is whether the candidate tracks the **current** blank-suit
lead rather than any earlier blank, early turn, declaration alone or held tile.
Freeze that hypothesis and lawful matched near misses before fresh measurement.
That study is not selected or executed by this exploratory readout. A broader
cleanliness comparison needs evidence beyond activation correlation or loss.
