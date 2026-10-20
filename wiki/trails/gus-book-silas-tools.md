---
title: "Gus: connecting the book, Silas and Lem's instruments"
kind: trail
first_seen: 2026-10-18
last_updated: 2026-10-20
status: active
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## The division of work

[[gus-tiny-gpt]] learns from lawful forecasts and verified outcomes. The book
supplies strategic questions and their assumptions; [[silas]] makes distinctions
and verifiers executable; [[lem-the-teacher]] supplies instruments for reading
the learned representation. Prediction, interpretation and playing strength
remain separate claims. The current training choice is owned by
[[gus-success-probability-planning]].

## Start with the question and its quantifiers

Read [the compact mathematical account](../../docs/math/walt.md) before choosing
a label. A seat must make the same choice in worlds it cannot distinguish.
Future observations are averaged under the named other-player model; future
learner choices respect its information. Counting legal hidden deals does not
count future play outcomes or specify other players' choice probabilities.

| Question | Existing route | Authority and limit |
|---|---|---|
| Did this completed hand make or set? | [[gus-full-hand-outcomes-2026-10-18]] | Independently replayed terminal reward; sampled trajectories. |
| Which endgame action succeeds more often under G0/Zeb? | [[gus-fixed-field-curriculum-2026-10-18]] | Exact rational values under its named finite model; uniform legal root belief omits earlier behavioral likelihood. |
| How does a new policy fare after a frozen position? | [[gus-lazy-rlvr-smoke-2026-10-18]] and the success-probability decision | Append immutable paired continuations; changed actions require recomputed play. Seeds, policy weights and runtime pins travel together. |
| Does this information state guarantee make/set? | [[silas]], `certain-make` / `certain-set` | Viewer information-consistent policy against every other seat's legal choices, including the partner; budget refusal is not failure. |
| Can the existing representation predict success? | [[gus-value-baseline-2026-10-18]] | Learned forecast, not a reward authority or proof of stronger play. |

The fixed-field verifier is deliberately narrow: two legal root choices, small
enumerated hidden support, and a forced later learner move. It holds each root
action constant across worlds. Its certificate is not a general full-hand
optimal-play claim. Sources: `lem/observer/rlvr/fixed_field.py`, `fixed_data.py`.

## Silas capabilities available to this work

Main exposes legal-world enumeration, rational beliefs, conditioning, witnesses,
queries and information-respecting policies through Rust and `scenario_core`.
Policy readings include specified seats' action likelihoods. `ReadingPredicate`
guards, textual `(reading ...)` declarations and `crisis-read.silas` are built;
the detailed gap-1 section in [GAPS.md](../../silas/GAPS.md) and the
[README's record-as-evidence section](../../silas/README.md) describe them.
Reading another player's knowledge and a self-referential sender/receiver model
remain different, unfinished capabilities. Consistency likelihood and uniform
choice likelihood are different behavioral assumptions.

The constructive `WorldSpace` / `OpeningSpace` / `PreparedFiber` API and
`silas construct` remain on open [PR #166](https://github.com/jasonyandell/mk5-main/pull/166),
not main at `4ddbb310`. Existing Gus exact verifiers use enumeration already
on main; they do not depend on merging that constructor. Construction of worlds
does not supply a general future-record constructor. The reading-policy coverage
diagnostic has a separate open defect [#164](https://github.com/jasonyandell/mk5-main/issues/164).

## Read the book and the net without imposing a vocabulary

[[book-assumptions]] identifies what each experiment can express: within-hand
history, partner conventions, opponent response, auction context and match
utility are distinct dimensions. The current fixed-bid, fixed-partner Gus task
does not test cross-hand adaptation or the joint invention of conventions.
[[book-authority]] governs book comparisons; ties are not rule agreement, and
choosing the same tile is not yet evidence of the same reason.

The companion skills are [`clean-neuron`](../../.claude/skills/clean-neuron/SKILL.md)
and [`book-interp`](../../.claude/skills/book-interp/SKILL.md). Book concepts are
opening questions, not an answer key for a net's units. For an unnamed unit,
check tools, vocabulary, data/learning and coherent generalization before claiming
superposition. Use both poles, held-out examples, same-vocabulary untrained and
shuffled controls, and report task learning alongside any clean-share statistic.

## What plugs into Gus today

The `lem.probe` kit supports concepts, cards/readers, fidelity, formation, width
families, ablation and steering. Read [[lem-naming-2026-10-17]] and
[[lem-causal-2026-10-17]] for their completed findings and limitations;
matched-row steering remains open in [#205](https://github.com/jasonyandell/mk5-main/issues/205).
Its current loader (`lem/probe/net.py`) expects the integer ReLU field net's
`.fnet` / trainer `.npz`, not Gus's transformer checkpoint.

Gus already records row-aligned native residuals, MLP pre/post activations,
attention to the prefix and output distributions in
[`benchmark.evaluate.capture`](../../lem/observer/benchmark/evaluate.py).
[[gus-frozen-benchmark-2026-10-18]] owns those stable row identities. The missing
bridge is an explicit transformer adapter into the concept/card/causal workflow.
GELU units, attention and sequence dependence need their own intervention
semantics and controls; field-net floors and interval adjustments do not transfer
automatically. No such adapter is claimed by the existing captures.

## Consequence for the next experiment

[[gus-ppo-value-2026-10-19]] completed the matched PPO/value comparison without
a fresh playing gain. Its online critic forecasts worse than a simple prior
before fitting new batches, despite the earlier pooled representation result.
Jason's subsequent winning/drift steering selects
[[gus-unanchored-outcomes-2026-10-19]]: keep GRPO and compare teacher-anchoring
losses with outcome-only selection. This avoids combining critic repair with
the new factor; it does not repeat a longer exposure budget.
Keep the G0 partner, pinned Zeb opponents, original clean tokens and fresh
confirmation discipline. The frozen panel is development evidence, not training
material. Book/Silas concepts can later explain where behavior changes; they do
not replace verified rewards or require expanding inputs now.

Silas's session confirmed these availability boundaries on 2026-10-18. Requests
for the Lem bridge and Book's current executable distinctions are recorded in
[Lem #207](https://github.com/jasonyandell/mk5-main/issues/207#issuecomment-6258785752)
and [Book #121](https://github.com/jasonyandell/mk5-main/issues/121#issuecomment-6258785965).
These are requests, not completed desk endorsements or adapter work.

The `gus-training` skill incorporates this refresh into the research loop:
before a new experiment, after a completed comparison or at a capability blocker,
read related progress and pending replies, verify available implementations,
then state whether the new evidence changes the lever or its measurement.
An unchanged short resume segment does not require repeating the full refresh.


Completion refresh for [[gus-unanchored-outcomes-2026-10-19]]: the latest Lem
and Book comments are still the requests linked above, with no new replies;
Silas PR #166 remains open and unmerged. No additional message was sent. Existing
verifiers and native captures supported the completed evaluation. Neither
teacher-anchor removal nor permission to drift established a greedy gain;
the secondary sampled-policy signal warrants independent confirmation, not a
new information contract or opponent change. Canonical tournament publication is now verified; the original service-quota
failure remains preserved in the full research archive.


[[gus-sampled-confirmation-2026-10-19]] independently tested the sampled-policy
lead and did not replicate it. All10,240 hands replayed; no promotion. The
completion refresh found no new Lem/Book replies and SilasPR166 still open.
Existing verifiers sufficed; no new capability changes the next comparison's
controls. Selection of another learning-signal comparison follows the readout
steering window, not a switch of the completed endpoint.

The19:00UTC boundary refresh found those same pending replies and open Silas PR.
[[gus-leave-one-out-2026-10-19]] uses existing verified full-hand returns to test
a parameter-free baseline; no new capability or information contract is required.

The completed [[gus-leave-one-out-2026-10-19]] comparison established no player
gain. Completion refresh again found no Lem/Book reply and SilasPR166open;
no new capability or scope change. The saved formation/native captures remain
available for the separate transformer bridge.

[[gus-partner-history-2026-10-19]] completed a bounded causal attention-path
diagnostic on fixed reference histories. Earlier partner-tile paths materially
affect legal probabilities on both splits. The study separates this sensitivity
from unestablished partner-style inference or playing improvement. See its
literal worked hand and fixed-table evidence before proposing a partner roster.

[[gus-public-score-probe-2026-10-19]] completed a frozen-representation probe for
current public score. The final pretrained vector did not support a better
linear readout than metadata and untrained controls. This is a scoped negative
accessibility result, not evidence that score information is absent. Exact public
replay supplied labels without hidden-world queries or a changed partner roster.

[[gus-native-cooldown-2026-10-19]] follows the complete fresh Cammarata reread
with a bounded native-unit microscope on saved constant/cosine formation.
The blank-suit response candidate is exploratory and predates cooldown; no
cleanliness improvement or stronger player is established. Raw pre/post-GELU
cards and independently audited local interventions now support a narrower
mechanism question before further training.

[[gus-blank-lead-2026-10-19]] tests that candidate on fresh fixed-table histories.
The adjusted double-blank lead association replicates, but the broader blank-suit
test is inconclusive because one panel lacks non-double blank-trump coverage.
The player and inputs stay unchanged; no cooldown cleanliness gain follows.

[[gus-first-look-2026-10-19]] then attenuates current-lead tile attention on
matched rows reused from those panels. The U65 preactivation response is dominated
by layer-0 routing in both panels. Downstream output movement is not evidence of
unit-specific mediation, new semantics or a playing gain. The 48 prefixes and
607,306 root exclusions remain reused/unchanged; the full audited bundle is
publicly verified.

[[gus-route-rescue-2026-10-20]] separates the layer-0 final actor query from the
bundle of earlier queries on those same 48 prefixes. The U65 raw-preactivation
contrast is concentrated in zero-based head1 at the final query in both panels.
Restoring the baseline U65 post-GELU scalar does not reduce raw output distance
more than U193 in both panels; this prespecified prediction fails across-panel
replication. The scalar patches differ greatly in magnitude and projection
direction, so neither result identifies exclusive semantics, per-edge
importance, or unit mediation.

[[gus-value-routing-2026-10-20]] attempted a registered same-position value vs
routing decomposition at that final query. The numeric/nonfinite audit stopped
on original-parent confirmation row `292:6` before cosine-final was measured.
Only 42 original-parent row records were accepted; the failing condition and
its numerical error were not saved. Its single primary prediction therefore
remains untested, with no scientific conclusion from the stopped instrument.

[[gus-value-numerics-2026-10-20]] is a separately scoped fixed-row numerical
diagnostic on that original-parent row. One float32 condition narrowly crosses
the prior strict numeric audit; in the same case the float64 path difference is
near machine precision. This supports local precision/order sensitivity, but
the original run's unsaved failing condition remains unknown and its primary
prediction unevaluated. No whole-study or playing claim follows.

[[gus-u65-formation-2026-10-20]] follows the registered fixed-coordinate
trajectory from splat initialization through 200,000 steps on those same 48
prefixes. The dose-0.5 tile-minus-actor route-sensitivity endpoint shifted
negative in both reused panels, as predicted. This tracks one final-selected
coordinate and local intervention across checkpoints; it does not show
monotone formation, preserved semantics, fresh confirmation, output mediation
or a playing gain.

The 2026-10-20 contingent-teacher boundary refresh found Lem
[#207](https://github.com/jasonyandell/mk5-main/issues/207) closed with its final
overbake interpretation, Book #121 still open without the requested reply, and
Silas PR #166 still open. No repeated request was sent. Existing exact holdings,
lawful seat-policy wrappers and replay instruments can support a bounded
extension, but `fixed_field.solve` still rejects a second discretionary learner
choice. The supplied continuation v3 paper gives the information-consistent
policy calculus, not a ready-made generic teacher. The selected direction in
[[gus-success-probability-planning]] therefore calls for an independently checked
information-cell solver and direct decision labels, preserving the original
encoder and fixed field.
