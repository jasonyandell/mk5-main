---
title: What You Do Past Belief
kind: topic
first_seen: 94d8646
last_updated: 94d8646
status: active
---

## The reframe

After [[belief-bayes-ceiling]] showed that belief is at the information limit, the remaining 42-craft is not about *knowing better* — it is about *acting well given unresolvable uncertainty*. The user framed this as "the heart of the game." Human experts call it reading the table, signaling, playing the percentages (94d8646, PRACTICALITIES §22).

## Extractable analytics (not yet run)

Every decision in the existing corpus has a `q_per_world` tensor — [M, 7] oracle Q values per world × legal action. Three derived quantities are computable with pure indexing, no new training:

1. **Outcome-variance**: `q_per_world[:, action_taken].std()` — how much the chosen action's result depends on which hidden world is true.
2. **Action-choice fragility**: per-world argmax action count — how many distinct plays the oracle would pick across M worlds. 1 = cleanly determined; 3+ = fog-of-war call.
3. **Belief-limited high-impact decisions**: join (1) and (2) with §21's per-decision belief sharpness. Decisions where belief is ~33% (pure prior) AND fragility is high AND outcome-variance is high are the defining moments of 42.

Estimated effort: one afternoon. No new model training. Output: a ranked (game, decision) list sorted by "drama," plus a per-trick-pos distribution of where strategic uncertainty concentrates (94d8646).

## The research question

π_me is trained on `argmax(marginal E[Q])`. At belief-limited decisions the marginal argmax commits to one meta-strategy — "play for the mode." Real players use several: **mode** (highest mean), **signal** (informs partner at slight E[Q] cost), **hedge** (minimizes downside across worlds), **gamble** (catastrophic in most worlds, wins big in one). Training data for all of these is already in the oracle's per-world tensor — no LAMIR or CFR+ required (94d8646).

This is noted as a future direction, not blocking anything. See PRACTICALITIES §22.

## Links

[[belief-bayes-ceiling]] [[belief-co-train]] [[lamir1-ceiling]] [[student-distillation]] [[gus]] [[expected-q-value]]
