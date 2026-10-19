---
title: "Gus: connecting the book, Silas and Lem's instruments"
kind: trail
first_seen: 2026-10-18
last_updated: 2026-10-19
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
