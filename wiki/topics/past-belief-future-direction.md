---
title: What You Do Past Belief
kind: topic
first_seen: 2026-04-22
last_updated: 2026-07-13
status: complete
---

## The reframe

After [[belief-bayes-ceiling]] showed that belief is at the information limit, the remaining 42-craft is not about *knowing better* — it is about *acting well given unresolvable uncertainty*. The user framed this as "the heart of the game." Human experts call it reading the table, signaling, playing the percentages (94d8646, PRACTICALITIES §22).

## Extractable analytics — run three days later

Every decision in the existing corpus has a `q_per_world` tensor — [M, 7] oracle Q values per world × legal action. Three derived quantities are computable with pure indexing, no new training: outcome-variance, action-choice fragility, and belief-limited high-impact decisions (join belief sharpness with fragility and outcome-variance).

This ran three days later (`76355ac`, `gus/analysis/drama_atlas_findings.md`): drama
(oracle-disagrees-and-belief-is-blind) fraction is 26.2% of all decisions; the opening lead
alone accounts for 64.9% of drama decisions and 62% of all drama decisions in the corpus.
See [[gus-drama-atlas]] for the full quadrant analysis and findings.

## The research question

π_me is trained on `argmax(marginal E[Q])`. At belief-limited decisions the marginal argmax commits to one meta-strategy — "play for the mode." Real players use several: **mode** (highest mean), **signal** (informs partner at slight E[Q] cost), **hedge** (minimizes downside across worlds), **gamble** (catastrophic in most worlds, wins big in one). Training data for all of these is already in the oracle's per-world tensor — no LAMIR or CFR+ required (94d8646).

This is noted as a future direction, not blocking anything. See PRACTICALITIES §22.

## Links

[[belief-bayes-ceiling]] [[belief-co-train]] [[lamir1-ceiling]] [[student-distillation]] [[gus]] [[expected-q-value]] [[gus-drama-atlas]]
