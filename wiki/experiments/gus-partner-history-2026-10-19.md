---
title: Gus partner-history attention paths
kind: experiment
first_seen: 2026-10-19
last_updated: 2026-10-19
status: complete
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question

A bounded interpretation study for [[gus-tiny-gpt]], following the completed
[[gus-leave-one-out-2026-10-19]] comparison and Jason's learning-first charter.
Does information carried from earlier partner tile tokens affect the current
legal-move distribution? The registered study is complete, audited and publicly verified. The prespecified
material-sensitivity criterion passed on both splits. No training or new
playing-strength claim.

The original input tokens stay unchanged. A temporary hook attenuates selected
attention-key columns in every head and query, either in all layers or the last
layer. All-layer knockout prevents earlier layers from relaying the selected
tile identities through other tokens. Residual paths at those selected positions
remain, but cannot transmit to other positions through blocked attention edges.

## Prespecified comparison

256 fresh physical deals, split 128 development and 128 confirmation. One full
reference hand per deal: G0 in both focal and partner seats, pinned Zeb opponents.
The splat parent and the already-selected RLOO checkpoint are evaluated on these
same histories, alongside three architecture-matched untrained twins. No model
selection, changed partner population, new inputs or search inside the player.

Eligible decisions precede public settlement, have more than one legal move,
and follow at least one complete trick. Masks select only tile tokens in complete
tricks; each relative seat contributes the same number of keys. Current-trick,
actor, metadata, own-hand and query tokens stay intact. Controls: zero-strength
sham, left/right/self cuts, half-strength partner attenuation and last-layer-only
partner knockout. Token data never changes; external legal support stays fixed.

Primary: per-root mean total variation of the parent's legal probabilities under
partner knockout, then equal-weight mean across eligible roots. Material
sensitivity requires mean greater than .01 and bootstrap95 lower bound above
zero in both splits. 20000 physical-root resamples; decisions within a hand are
not independent samples. Secondary: argmax changes, recorded-action NLL,
other-seat contrasts, random twins, RLOO and descriptive attention mass.
The worked example is the first eligible decision in the first eligible
development root, not the prettiest confirmation example.

Eight resumable 45-second segments, each with 90-second outer cap, one process;
audit180s and summary90s. No automatic retry or budget growth after failure.
Source, runtime, scripts, weights, field and exclusions freeze before execution.
All newly reserved roots become future training exclusions.

## What the instrument can and cannot establish

The method adapts attention knockout from
[Geva et al. (2023), section5](https://aclanthology.org/2023.emnlp-main.751/),
which studied factual recall in language models. Its published empirical findings
do not transfer automatically to 42. Last-layer-only cuts can miss information
already transported by earlier layers; the all-layer condition cuts those paths.

Attention pictures alone do not identify a causal mechanism; see
[Jain and Wallace (2019)](https://aclanthology.org/N19-1357/) and the methodological
qualification in [Wiegreffe and Pinter (2019)](https://aclanthology.org/D19-1002/).
The instrument is separately tested for exact sham parity, cleanup, causal masking,
probability renormalization, batch isolation and synthetic information-path cuts.

A changed prediction is evidence of sensitivity to these paths. It also removes
ordinary bookkeeping about played tiles. It does not establish partner-style
inference, intent, conventions, coordination, neuron names or stronger play.
These off-distribution cuts can cause generic damage. Equal counts do not match
seat content or recency; untrained twins differ in prediction sharpness. A small
effect can reflect redundant information. Half-dose results show sensitivity to
intervention severity, not a guarantee of monotonic or semantic response.

This net, its original own-seven/public-history inputs, this fixed behavioral
field and these attention paths: not all of 42. The `clean-neuron` tools-first
principle applies, without importing the field-net vocabulary, clean-share floor
or its row-interval adjustment into this different statistic.

## Capability boundary

Lem207 and Book121 requests remain unanswered at the20:30UTC refresh; no repeated
request. Silas166 remains open. Existing full-hand replay verifiers suffice;
this small attention instrument is separate from a native-neuron concept audit.
See [[gus-book-silas-tools]]. Cross-hand partner learning and whole games remain
separate proposals.


## Completed readout

The parent's partner-path sensitivity replicated: development TV .09510
95% [.08256,.10857]; confirmation .09663 [.08514,.10887]. Both exceed the
registered .01 mean threshold, with positive lower bounds. TV is half the sum
of absolute probability changes: approximately9.7% probability mass moves among
legal choices on confirmation. It is not a win-rate improvement.

There were323 eligible decisions in106/128 development roots, and306 in112/128
confirmation roots. Each eligible root receives equal weight after averaging its
eligible decisions. The other38 roots remain recorded and reserved; no replacements.
Intervals bootstrap physical roots, not individual decisions.

| Parent intervention | Confirmation mean TV | Root-bootstrap95% | Changed preferred move, hand-weighted |
|---|---:|---|---:|
| Sham | 0 | [0,0] | 0% |
| Half partner attenuation | .04305 | [.03799,.04837] | 5.34% |
| Partner, all layers | .09663 | [.08514,.10887] | 14.99% |
| Partner, last layer only | .05621 | [.04886,.06442] | 9.67% |
| Seat1, all layers | .07709 | [.06755,.08682] | 11.06% |
| Seat3, all layers | .07760 | [.06800,.08791] | 12.10% |
| Own earlier plays, all layers | .11708 | [.10405,.13068] | 18.35% |

Primary partner-cut preferred-move change95% interval is[10.86%,19.42%].
The all-layer and last-layer effects are different interventions in a nonlinear
network; their difference is not an isolated amount of information relayed
through earlier layers. The synthetic instrument test establishes that such
indirect paths are possible and that the all-layer cut blocks them.

Untrained partner-cut TV floors, same inputs and masks: seed1729 .001333
[.001239,.001428], seed1730 .001494 [.001382,.001611], seed1731 .001251
[.001148,.001359]. The trained-minus-three-twin mean difference is .09528
[.08379,.10750]. This calibrates the instrument, but cannot isolate learning
of partner semantics from changed scale and prediction sharpness.

Secondary partner-minus-average-other-seat TV is .01929 [.00840,.03055],
consistent with the development contrast .01361 [.00177,.02616]. The own-play
cut is larger still. Equal key counts control intervention size, not all game
information or temporal-position differences; these are not semantic rankings.
Selected RLOO has .09797 [.08597,.11073] partner-cut TV on confirmation, a similar
sensitivity profile. No training comparison or improvement follows from this.

Recorded G0-action NLL change under partner knockout is +.02009 nats
[-.01372,+.05334] on confirmation. The preferred moves change, but this study
has not established that those changes harm imitation, much less actual winning.

## A literal input and output

Prespecified example: first eligible decision in development root0, physical
seed2695861795, viewer/bidder seat0, bid30 with fours trump. Original hand:
22,31,32,41,54,64,66. First trick: self54, seat1 33, partner44, seat3 40.
Second trick so far: partner00, seat3 30; Gus is to act. The intervention blocks
only paths from the earlier **44** tile token. The partner's current-trick00,
actor roles, own initial hand and every literal input token remain intact.

| Legal tile | Intact probability | Half attenuation | Partner knockout |
|---|---:|---:|---:|
| 22 | 2.00% | 1.50% | 1.08% |
| 31 | 10.67% | 10.43% | 9.89% |
| 32 | 59.69% | 51.91% | 44.85% |
| 41 | 20.31% | 24.60% | 28.39% |
| 64 | 6.53% | 10.70% | 15.02% |
| 66 | 0.81% | 0.87% | 0.77% |

Gus still prefers32, which G0 played in the reference recording. This illustrates
how a probability splat can change substantially without changing argmax. It
establishes no best move and names no concept represented by the earlier44.
The lawful current history is fixed; the intervention changes computation,
not what happened at the table.

## Receipts and next question

Source b18d2382;170 Lem tests and all four affected suites green. Registration
[8451d5e](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/8451d5e071e42a87654a91f1147e7a22205bb33e/lem/gus-partner-history-2026-10-19/registration)
was freshly anonymously verified before collection. Frozen scripts and endpoints
were unchanged.605642 previous physical deals were excluded, including the
previous896 reserved roots and39 potential tournament deals. All new256 roots
are future training exclusions.

The controller used2/8 collection segments, then its registered audit and summary;
all four stages exited0. Collection cost1.188e12 instructions,55.8s CPU and wall
descriptively; audit1.554e11 instructions,8.4s CPU and wall. These are instrument
costs on this host, not player thinking budgets or cross-machine speed claims.
All256 full hands/1792 focal decisions replayed; all629 eligible intervention
outputs recomputed exactly; eight complete units regenerated in reverse order
with byte identity. Model parameters were never updated.

[Verified evidence57ac08e9](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/57ac08e9dbf1115f9f86ccb315b1e86d916c858d/lem/gus-partner-history-2026-10-19/evidence):
9 files,21975821 bytes,290 archived members; every file and member freshly
verified by SHA256. Includes literal inputs, legal supports, full reference
hands, logits/probabilities, attention mass, random twins, pinned model assets,
recovery receipts and the worked example. Original model exports and canonical
tournament records required no updates or repeated games.

The actionable finding is that earlier partner-tile paths already matter to
Gus. A next study can ask whether history helps infer **behavioral differences**
between explicitly defined partners, after separating ordinary tile bookkeeping.
That is a proposed learning question; no partner roster, input change or new
training recipe is selected by this readout. The completed fixed-field reward
studies still have no confirmed playing-strength promotion.
