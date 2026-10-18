---
title: "Gus frozen benchmark — stable games, static decisions and native representations"
kind: experiment
first_seen: 2026-10-18
last_updated: 2026-10-18
status: complete
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question and scope

Can [[gus-tiny-gpt]] compare every new checkpoint against the same recorded
positions and rich representations, instead of spending every short evaluation
on different games? Jason requested stable seed games with known outcomes and
public frozen sets. The resulting benchmark combines a fast static regression
panel with a reusable paired-game schedule. Repeated measurement reduces
comparison noise; repeated selection makes the panel development data.

A known outcome is a fact about the recorded sequence of actions. Changing a
move can change all subsequent information and choices, so a different model's
agreement with the sequence is not its win rate. Static action loss, observed
forecast scores, exact restricted endgame regret, representation drift, and
freshly played game outcomes remain distinct measurements.

## Frozen panel

The first version preserves all 16 duplicate pairs from
[[gus-splat-pretraining-2026-10-18]] and [[gus-rlvr-silas-2026-10-18]], explicitly
marked reused regression. Two additional G0-versus-Zeb duplicate pairs use
seeds **1890259532** and **1753916461**, selected by the namespace
`gus-frozen-anchor-v1` before their outcomes were known. Both reference pairs
split one game each. They are anchors for future comparison, not evidence of
50% playing strength.

The complete panel has **18 duplicate pairs, 36 games, 397 hand recordings,
217 distinct played physical deals and 11,116 decision rows**. Every recorded
play, trick total, make/set result, mark, cumulative score and game stopping
point is independently replayed. Silas and the Python public-record referee
also agree at every decision.

For each pair, the audit sidecar freezes all **13 possible initial deals and
contracts** in the first-to-seven, one-mark game, including hands the reference
game never reached. There are **234 reserved physical deals** in total.
`reserved-hands.npy` is an exclusion source for future training; reserving only
previously played hands would allow a future challenger to encounter trained-on
hands after its game lasts longer. The whole schedule is checked against the
previous training, development, test and feasibility exclusions.

All four seats use the common `Bid30Bidder`: fixed bid30 and best pip trump.
G0 uses CPU float32, one forward pass and legal argmax. Zeb retains its pinned
n160, f16 deterministic budget, posterior sampling, uniform rollout setting,
margin tie rule, seed0 and field weights. Wall time caps execution, not thought.
The two new pairs took about 3.7 and 4.2 seconds of play, each under the one-minute
outer cap. Their measured times are descriptive, not scheduling guarantees.

## What a row contains

`question` contains only the acting seat's original seven tiles, its seat,
bidder, declaration, bid30 and the public actor/tile prefix. `tokens` reproduces
the existing 46-token encoder exactly. The legal output set comes from that
seat's remaining hand and the public led suit; it is never appended to the
input. Full physical hands live only in separate `audit/` records.

The same row also preserves the reference action, behavior identity, public
banked points, current trick, leader, inferred void exclusions and remaining
sizes, plus recorded hand/game outcomes with an explicit trajectory-only scope.
The extra fields are diagnostic labels, never model inputs. Stable pair, game,
hand, physical-deal and decision IDs let later checkpoints use identical rows
and let uncertainty respect correlated observations.

The immutable manifest pins tokenizer, row order and checksum, source and
physics identities, participant/checkpoint/field hashes, exclusions and every
raw game/schedule. Original G0 weights and Zeb field weights are packaged as
portable, checksum-checked reference assets.

## Exact answers where they are available

**471 unsettled two-choice endgame rows** also receive the exact verifier from
[[gus-rlvr-silas-2026-10-18]]. It enumerates legal-consistent hidden hands and
uses exact fractions for the equal mixture of lowest-legal, highest-legal and
uniform-legal continuation styles. Every candidate value and the complete tied
best set are retained. This is the separately named finite model
`e2e41679106a11be7d6922ac04543bde0151bf98120e1f598c86a2d52ce6ffa7`.

These certificates support actual optimal-answer accuracy and rational regret
within that model. They do not pretend that the three simple styles are the
Zeb/Gus opponents who generated the full-game traces. Uncovered positions have
no counterfactual labels; no win is copied onto an unplayed action. The same
information-state action is evaluated across hidden worlds, preserving the
[[strategy-fusion-fence]].

## Static evaluation and interpretation

`lem/observer/benchmark/evaluate.py` accepts a checkpoint and policy head. Each
run stores every raw tile logit/probability, legal probability, chosen tile,
last-actor residual before and after final LayerNorm, each layer's residual and
MLP pre/post activations, and each attention head's weights over the causal
prefix. Joint models also store all 28×43 outcome probabilities. Arrays share
stable row IDs and are retained, not reduced to a dashboard average.

The report separates raw and legal NLL, reference-action agreement, impossible
probability, entropy, the exact endgame subset, and splat scores against the
recorded action's eventual result. Unsettled discretionary positions receive
their own slice so forced/post-settlement plays cannot hide decision changes.
Paired checkpoint deltas and intervals resample complete duplicate pairs.

The `clean-neuron` skill can read these native tensors at identical inputs,
compare formation or matched interventions, and build a vocabulary for what
changed. A residual-coordinate shift or linear-readable signal alone is not a
causal neuron explanation. Static comparisons describe the reference histories;
actual playing strength requires rerunning the pinned schedule or fresh
confirmation games.

## First fixed-panel pass

The first pass evaluates G0, the initial joint splat net, GRPO and the cosine
continuation on exactly the same rows. Capturing the full representations and
computing reports took **2.9–4.4 seconds per checkpoint** on one CPU thread,
including compressed output. All 3,192 recorded G0 actions match its frozen
policy exactly. A new run of anchor 0 reproduces both complete game records,
including every action and outcome, byte-for-byte in the deterministic game
serialization.

The behavior columns below use the **4,895 unsettled discretionary decisions**.
The exact columns use the 471 covered endgames. Of those, 287 have tied best
answers: uniform legal sampling already obtains **80.47% exact-answer accuracy**
and **5.336 pp rational regret**. Reference agreement is imitation of the named
recorded policy, not optimal-move accuracy.

| Checkpoint and deployed head | Raw tile NLL ↓ | Reference-action agreement | Exact-answer accuracy | Exact regret ↓ |
|---|---:|---:|---:|---:|
| G0, tile | 1.012 | 63.11% | 84.50% | 4.501 pp |
| Initial joint net, tile | 1.080 | 55.91% | 81.53% | 4.827 pp |
| Initial joint net, splat | 1.080 | 56.51% | 86.62% | 3.330 pp |
| GRPO, tile | 1.471 | 59.80% | 85.14% | 4.156 pp |
| Cosine continuation, splat | 1.034 | 49.01% | 86.84% | 3.496 pp |

The GRPO-versus-parent regret change is −0.671 pp, with paired 95% interval
[−2.226,+0.710] pp: this panel does not resolve a regret gain. Its raw illegal
probability on unsettled discretionary states rises from 2.46% to 17.55%, while
the output legal mask still prevents illegal moves. That is a concrete
retention diagnostic outside the narrow endgame curriculum. Comparing the
cosine continuation with its parent using the **same splat head** gives regret
change +0.165 pp [−1.167,+1.551] pp, also unresolved. The initial joint model's
splat and tile rows share one captured representation archive; selecting the
head changes the output decision rule without another network pass.

The full reports retain behavior-specific slices, paired differences and
2,000-draw duplicate-pair bootstrap intervals. These intervals describe the
fixed regression sample; they do not make its reused games fresh confirmation.

## Reproduction and publication

Implementation: `lem/observer/benchmark/{build,evaluate,h2h}.py`.
`build.py` produces deterministic compressed rows and verifies the full referee
replay before freezing them. Corrupt rows, illegal actions, mismatched scores,
partial pairs, wrong schedules, overlapping reserved deals and altered assets
fail closed. `evaluate.py` can compare later checkpoints against the preserved
G0 representation archive. The paired-game wrapper reuses the fixed schedule
and verifies that the reference side still matches its pinned identity.

[Frozen panel and all four representation archives](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/df8277f60d2c95ce16dfca9473b8eed58a696c40/lem/gus-frozen-benchmark-2026-10-18):
119 files, 410,609,771 bytes, public CC0. Immutable revision
`df8277f60d2c95ce16dfca9473b8eed58a696c40` contains the clean rows, complete audit
records, all potential schedules, portable reference assets, prior exclusions,
all model captures, comparison reports and exact implementation snapshots.

[Independent download/replay and quality-gate receipts](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/e86cd3184644693283e701a6a7262c558161c724/lem/gus-frozen-benchmark-2026-10-18/verification)
at `e86cd3184644693283e701a6a7262c558161c724` confirm all 119 files were freshly
downloaded anonymously and matched SHA256. The downloaded games were replayed
and all 11,116 clean rows independently reconstructed; all 234 potential deals
were checked against the 588,030 prior exclusions. Every verification receipt
was itself freshly downloaded and hash-checked.

Source implementation: `cbc47a836698196fbe7e5489f0287f5a65de71d2`.
`scripts/check --fast`: GREEN, 102 Lem tests passed; wiki, conflict-marker and
license suites passed. No model is promoted by this work. Future training must
exclude `panel/reserved-hands.npy`; future evaluation should keep this version
unchanged and publish any expanded panel under a new identity.
