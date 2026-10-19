---
title: Gus partner-history attention paths
kind: experiment
first_seen: 2026-10-19
last_updated: 2026-10-19
status: active
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question

A bounded interpretation study for [[gus-tiny-gpt]], following the completed
[[gus-leave-one-out-2026-10-19]] comparison and Jason's learning-first charter.
Does information carried from earlier partner tile tokens affect the current
legal-move distribution? Instrument tests pass; fresh-data execution waits for
public registration and verification. No training or new playing-strength claim.

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
