---
title: Gus public-score linear probe
kind: experiment
first_seen: 2026-10-19
last_updated: 2026-10-19
status: active
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question and scope

Can a small linear readout recover the already-public partnership score margin
from [[gus-tiny-gpt]]'s frozen internal representation? This follows
[[gus-partner-history-2026-10-19]]: earlier partner-tile paths affect predictions,
but that may include ordinary game-state bookkeeping. A concrete public-state
question precedes attributing partner-style inference or choosing a new roster.
Implementation tests pass; collection and fitting await verified public registration.

The label is banked points for the viewer's partnership minus opponents' banked
points, reconstructed from the original causal prefix alone. Exact symbolic
replay already computes the answer. No hidden hand, future score, success label
or extra input token is needed. This differs from the future-success regression
in [[gus-value-baseline-2026-10-18]]. Gus receives no gradient or new weights.

A probe predicts a property from a frozen representation, following
[Alain and Bengio (2016)](https://arxiv.org/abs/1610.01644). This study uses ridge
regression for a numerical target, not their classifiers. A successful readout
establishes accessibility to that readout, not that the network uses the quantity
causally. [Hewitt and Liang (2019)](https://aclanthology.org/D19-1275/) motivate
controls for probe learning capacity; this study's shuffled trajectories are a
fitting control, not a reproduction of their word-type control tasks.

## Frozen comparison

896 fresh physical deals:512 fitting,128 development,256 confirmation. One greedy
reference hand per deal, G0 focal player and partner, pinned Zeb opponents.
Original own-seven/public-history inputs, bid30/pip declarations and deterministic
field budgets stay fixed. All605898 earlier reserved physical deals are excluded.
All896 new roots become future training exclusions.

Six focal decisions after the first completed trick per hand form the main
probe rows. Forced decisions and positions after public contract settlement are
included because the target is a current factual state. The original seven
focal decisions and full hand remain in the audit. Discretionary unsettled
positions are a secondary slice; they do not replace the registered endpoint.

Primary features: final normalized64-dimensional actor-position vector of the
pretrained parent. Secondary features: each block's residual, three pinned
architecture-matched untrained twins,26 metadata indicators, and166 metadata +
own-hand + actor-by-played-tile indicators. The latter is an orderless baseline;
it does not retain trick grouping. Metadata names bidder, declaration, completed
tricks, current trick slot and current leader. None receives the computed score.

Each readout fits squared error of margin/42 plus .001 ridge on standardized
nonintercept coefficients. Standardization uses only3072 fitting rows; intercept
is unpenalized, standard-deviation floor1e-8. Same fixed penalty, no tuning,
no clipping of linear predictions. Parent readout has65 fitted parameters.
A constant fit-mean baseline and a parent readout trained on permuted six-score
trajectories (within bidder-team strata) calibrate simpler explanations.

Prespecified criterion: parent point-MSE at least10% below metadata and the
arithmetic mean of three random-readout MSEs, with paired95 upper deltas below
zero against both, in development and confirmation. All six rows of a physical
root stay together;20000 paired root-bootstrap draws per split. The same fitted
readouts reach both panels: independent data, not independently refitted models.
No confirmation selection or deployment promotion. Report RMSE/MAE in42 points,
per-trick slices, bag baseline, layer results and shuffled-label control separately.

## Engineering and limits

`lem/observer/state_probe.py` reconstructs public labels and baseline features
from tokens; tests cover relative seats, malformed prefixes and independent
augmented-least-squares equivalence. Existing native capture and immutable lazy
records are reused. Actual engine scores independently check every token-derived
label. All features are recomputed, with eight complete units rechecked in reverse
order. Fitting is independently audited by augmented least squares.

Eight45-second collection segments,90-second outer cap, one process; data audit
180s, fit90s, independent fit audit90s, summary90s. Frozen source, scripts, field,
weights and runtime; failures or cap exhaustion require inspection, not retries
or budget growth. Publication batches registration and completed evidence.

Probe success can reflect what the probe learned from informative features.
Failure need not mean score information is absent or unused. Different controls
have different dimensions and information; none defines a universal semantic
floor. This is not a native-unit naming or causal score-intervention study.
A next actor experiment could test auxiliary public-state learning if warranted;
no such loss, player change or partner population is selected here.

Lem207/Book121 remain unanswered at the21:30UTC refresh; Silas166 is open.
Existing public replay and capture suffice. No repeated request or core change.
The `clean-neuron` tools-first principle applies with this history-conditioned
scope; field-net clean-share thresholds and interval adjustments do not transfer.
See [[gus-book-silas-tools]] for capability boundaries.
