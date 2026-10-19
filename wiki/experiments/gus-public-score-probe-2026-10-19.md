---
title: Gus public-score linear probe
kind: experiment
first_seen: 2026-10-19
last_updated: 2026-10-19
status: complete
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question and scope

Can a small linear readout recover the already-public partnership score margin
from [[gus-tiny-gpt]]'s frozen internal representation? This follows
[[gus-partner-history-2026-10-19]]: earlier partner-tile paths affect predictions,
but that may include ordinary game-state bookkeeping. A concrete public-state
question precedes attributing partner-style inference or choosing a new roster.
The registered comparison is complete, audited and publicly verified. The
predicted advantage of the final pretrained representation failed on both panels.

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
No confirmation selection or deployment promotion. Report RMSE/MAE in game points,
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


## Completed readout

The registered criterion **failed on development and confirmation**. A linear
score readout from the final pretrained representation did not beat the metadata
or matched untrained-feature controls. There is no changed endpoint, selected
replacement layer, new actor or playing-strength claim.

On the 256 untouched confirmation deals, all 1536 probe rows contribute six
rows per root. The target is the current **within-hand banked point margin**,
not a match score or future outcome. RMSE is in game points; smaller is better.

| Fitted readout | Confirmation RMSE | Root-bootstrap 95% interval | Mean absolute error |
|---|---:|---|---:|
| Parent final representation (primary) | 14.51 | [13.60,15.43] | 11.31 |
| Phase/role metadata | 14.10 | [13.35,14.83] | 11.17 |
| Mean of three random-feature errors | 14.03 | [13.30,14.77] | — |
| Orderless public-input bag | 12.99 | [12.22,13.76] | 10.15 |
| Parent first-block residual | 13.82 | [13.11,14.52] | 10.72 |
| Parent second-block residual | 14.89 | [13.97,15.82] | 11.68 |
| Parent fitted to permuted score trajectories | 17.06 | [16.06,18.05] | 13.65 |
| Constant fit mean | 16.47 | [15.60,17.33] | 13.31 |

The random row is sqrt(mean MSE of three separately fitted random probes), not
an ensemble prediction. Their individual RMSEs are 13.99,14.06 and14.05 points.
All probes use the same 3072 fitting rows and fixed ridge penalty; none was
chosen using confirmation. The bag has166 features versus the parent's64,
and retains different information; it is a secondary baseline, not a new player.

Confirmation paired parent-minus-control MSE differences (point-squared units):
metadata +11.91,95%[-8.10,31.92]; mean random error +13.65,[-5.03,32.36]. These
intervals do not establish that the parent is worse, but they do not support the
registered advantage. Parent-minus-bag is +41.79,[17.34,66.40], a secondary
comparison; its development interval included zero. The first-block readout
also has a lower point estimate without becoming the selected endpoint.
Linear accessibility need not improve monotonically with depth, and this is not
proof that a later layer destroys information.

Development parent RMSE is14.53 [13.19,15.87], versus metadata14.26 and random
mean14.18. In both panels the parent improves over constant and shuffled-label
controls, but that alone is insufficient because structured controls do as well
or better. A weak baseline would have encouraged an unsupported positive claim.

The primary question includes forced/post-settlement states: exact running score
is a stricter bookkeeping target than winning a settled contract requires.
The prespecified discretionary-unsettled confirmation slice contains662 rows:
parent RMSE12.27, metadata9.69, random twins10.09/10.13/9.96, bag9.71. Those are
descriptive slice estimates from the same globally fitted readouts; no new fit
or confirmation choice is made from them.

## Worked input and readout

The example was fixed before execution: development root512, physical seed
1994624181, focal decision index3. Gus is the bidder/viewer, bid30, sixes trump.
Original hand:32,40,41,52,55,62,63. Completed public tricks:

| Trick | Plays in order, relative to viewer | Winner | Points |
|---|---|---|---:|
| 1 | self55, left54, partner53, right64 | right | 21 |
| 2 | right00, self40, left50, partner10 | right | 6 |
| 3 | right61, self62, left66, partner60 | left | 1 |

The opponents have banked28, the viewer's partnership0: exact margin **−28**.
Current trick: left33, partner21, right43; viewer to act. The contract is already
set and32 is the viewer's sole legal move. Those facts make this a score-reading
example, not evidence of a strategic choice. The actual parser receives only the
same original metadata, own seven and public actor/tile prefix that Gus receives.

The frozen parent plus fitted linear readout says−25.54; metadata−6.88, bag−14.28,
untrained twins−7.10/−6.84/−7.62. This happens to be a comparatively good parent
example. It was not chosen for its error and does not overturn the negative
aggregate result. The full literal tokens, representations, weights and all
readout outputs are in the evidence.

## Interpretation and possible next work

The previous partner-history test establishes that selected history paths affect
move probabilities. This probe does not establish a superior easy-to-read current
score representation in the final vector. Neither result proves that Gus lacks
score information, cannot count, or fails for this reason when playing. The score
could be encoded nonlinearly, elsewhere, or at a granularity this readout misses;
the small readout and this fitting regime are part of the test.

The quantity is fully expressible in the original inputs: deterministic public
replay computes it exactly, with no hidden hand or future action. A useful next
candidate is explicit auxiliary public-state learning under the same clean inputs
and fixed table, then separately measuring representation and verified play.
That would test a hypothesis about learning, not repair a proven bottleneck.
No auxiliary loss, partner population, nonlinear probe or next recipe is selected
by this failed confirmation. Native-unit naming and causal score use remain open.

## Verification and publication

Source0ff3d2f5;173 Lem tests and all four affected suites green. Registration
[2fa7fa1f](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/2fa7fa1f4796520958146c918050b418121cb314/lem/gus-public-score-2026-10-19/registration)
was freshly verified before collection. All three frozen scripts and thresholds
remained unchanged. Four of eight collection segments completed896 hands;
data audit, fitting, independent fit audit and summary all exited0.

All6272 original focal rows were replayed and checked against engine banked
scores. All feature arrays reproduced exactly, and eight full units regenerated
in reverse order with identical record bytes. The nine fitted readouts passed an
independent augmented-least-squares audit; maximum prediction discrepancy was
8.21e−12 game points. No Gus parameter was updated. Literal readout coefficients,
normalization, predictions and fitting-label permutation are retained.

[Verified evidence59033be5](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/59033be58422b452284d4c7b90bde1494ed5a675/lem/gus-public-score-2026-10-19/evidence):
10 files,38950883 bytes,946 archived members; every fresh anonymous file and
archive member matches SHA256. Includes all roots, source pins, full hands,
clean inputs, score labels, native features, readouts, predictions and audit/cost
receipts. Existing player exports and canonical tournament records were not
rewritten. All896 new reserved roots join the605898 prior training exclusions,
for606794 total. No completed previous study was rerun.

Completion capability refresh found no new Lem207/Book121 reply and Silas166
still open. Existing public replay and capture were sufficient; no repeated
request or new core capability was needed.
