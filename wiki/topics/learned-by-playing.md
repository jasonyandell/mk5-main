---
title: Learned by Playing, Not Drilling
kind: topic
first_seen: 2026-04-10
last_updated: 2026-04-11
status: superseded
---

## Overview

"Learned by playing, not drilling" is the key insight from Stage 0 v1 of [[lem]]. The observation: Q&A-format drilling on trump membership produced 100% token accuracy on the training corpus but did not transfer to narration-context reasoning, where the model still called 4-4 and 6-4 trumps under fives-trump — the same error it made before fine-tuning (lem/OVERVIEW.md @ 24ae55a).

## The observation

The [[rules-adapter]] Stage 0 v1 training fixed hand-state tracking completely: the model correctly reads "remaining: 6-2, 6-1" from the narration instead of pulling from the initial hand. But trump membership — a compositional rule ("fives trump + 6-4 contains a 4 → 6-4 is trump") — did not transfer. Drilling correct answers in Q&A format established the pattern but not the reasoning that generalizes to novel surface forms (lem/OVERVIEW.md @ 24ae55a).

## Frontier explanation

> "Trump rules learned by playing, not drilling. Humans learn trump rules the same way: not from flashcards, but from playing hands and getting it wrong until it clicks."

(lem/OVERVIEW.md @ 24ae55a)

## Why this motivates Stage 1

[[star]] Stage 1 will encounter trump errors naturally through play: if the model calls a non-trump a trump and makes a bad play, [[k1-grading]] will reject the trace. [[r1-rationalization]] will then reveal the bot's correct action and ask the model to reason backward to why. This is the correction mechanism that drilling lacked — negative feedback tied to concrete consequences in the game (lem/OVERVIEW.md @ 24ae55a).

The claim is that trump membership will close through practice in Stage 1 rather than requiring further Stage 0 drilling (lem/OVERVIEW.md @ 24ae55a).

## First evidence

Prediction (ingest 2, 24ae55a): trump rules and other compositional rule applications will be learned through [[star]] play-and-correction, not through Q&A drilling.

Observation (ingest 9, efad16e): 10 STaR iterations on narration-context play moved K1 pass rate from 30% (Stage 0 adapter baseline) to a plateau of 36–42% (best: 42% at iters 5 and 7). This is a 12-point absolute improvement, consistent with the model internalizing at least some rules through graded practice.

The plateau is also informative: learning by playing has diminishing returns once the easy wins (likely the most egregious rule errors) are absorbed. The remaining gap likely requires either more diverse training data or a qualitatively different signal. [[scratchpad-validation]] — re-enabling engine-verified fact checks once the scratchpad format is bootstrapped — is the leading candidate for breaking through the plateau (lem/OVERVIEW.md @ efad16e).

5 more iterations (10–14, 908773a) did not break 42%; plateau confirmed at 38–41%. At that point the hypothesis was that K1 grading itself had a structural ceiling.

That hypothesis was contradicted by ingest 13 (8c1bb14). A better Stage 0 adapter broke through: [[kerry-curriculum]] lifted the plateau to ~43% avg / 46% peak; v3 ([[trump-drilling]] added) reached 48% peak. The 42% ceiling on v1 was a curriculum-quality ceiling, not a K1-structural ceiling.

**Revised claim:** playing-based learning is complementary to targeted drilling, not a replacement for it. Drilling raises the rules-comprehension floor; playing improves above that floor. The plateau of K1-only STaR depends on how high the floor is. "Each curriculum round raises the floor." (8c1bb14) The original title of this page requires a nuance: drilling alone is insufficient (v1 showed this), but drilling combined with the right curriculum structure (Kerry + trump drills) does transfer — and then playing can take it further. The interaction between drilling and playing is the real story (8c1bb14).

## Recurring theme

This observation may generalize: factual state-reading (what dominoes remain?) transfers from Q&A to narration context; compositional rule application (which dominoes are trump given this declaration?) may require game-grounded correction to transfer. A pattern worth tracking across future stages (lem/OVERVIEW.md @ 24ae55a).

## Inapplicable to the successor

[[burl]] does not test this thesis at all — it doesn't drill rules into weights in the
first place, so there is nothing for "playing" to correct. Rules facts come from the
[[engine]] tool at inference time. The drilling-vs-playing question this page tracks was
never resolved by LEM (it ended mid-plateau, see [[lem]]) and became moot once the project
pivoted to a weights-don't-hold-facts architecture. See [[lem-to-burl-handoff]].

## Links

[[rules-adapter]] [[star]] [[k1-grading]] [[r1-rationalization]] [[lem]] [[scratchpad-validation]] [[kerry-curriculum]] [[trump-drilling]] [[experiments/second-gemma-contact]] [[experiments/star-10-iterations]] [[experiments/stage-0-progression-star]] [[sources/24ae55a]]
