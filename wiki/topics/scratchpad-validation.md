---
title: Scratchpad Validation (attempted and shelved)
kind: topic
first_seen: 380f3fa
last_updated: 78ba940
status: retired
---

## Overview

Scratchpad validation is a stricter grading strategy for [[star]] traces in [[lem]] Stage 1. Rather than grading only on E[Q] outcome ([[k1-grading]]), the approach requires the model to fill a structured scratchpad and validates each claim against engine ground truth before accepting a trace for training (380f3fa).

## Scratchpad structure

The model was prompted to fill four fields before choosing a play:

- **HAND** — the narrator's remaining dominoes (must match engine state exactly)
- **VOIDS** — which suits each opponent is known void in
- **COUNTS** — which count dominoes have been played / are still live
- **PLAY** — the chosen domino

Only the completed scratchpad trace is submitted to grading (380f3fa).

## Grading categories

Scratchpad validation introduced a superset of plain K1 grades:

| Grade | Condition | Fate |
|---|---|---|
| `valid_pass` | Facts correct AND E[Q] beats bot | Train as-is (gold) |
| `valid_fail` | Facts correct BUT E[Q] fails K1 | [[r1-rationalization]] |
| `invalid` | Any scratchpad fact wrong | DISCARD |
| `illegal` | Chosen play not in legal moves | DISCARD |
| `parse_fail` | Cannot extract a play | DISCARD |

The rationale for discarding `invalid` traces extends the [[decisions/discard-illegal-traces]] "poison" principle: a correct move arrived at via hallucinated game-state is still corrupted reasoning. Training on it would teach the model to reason from wrong facts (380f3fa). The iteration-summary wandb metric was renamed from `pass_rate` to `valid_pass_rate` at this frontier to match the stricter grading ([[sources/34775ca]]).

## Result

On the first iteration with scratchpad validation active, 64.5% of traces graded `invalid`. Only 5 traces passed to training — insufficient for a meaningful LoRA step. The model had never seen the scratchpad format, so it could not produce one reliably (b12fcec, 78ba940).

The project briefly relaxed to hand-only validation (HAND must match; COUNTS logged but not enforced) in b12fcec, then reverted entirely to simple K1 in 78ba940.

## Key lesson

Format-bootstrapping must come before fact-validation. A model cannot be expected to produce a new structured output format AND satisfy factual-accuracy constraints on that output in a single training step. The format must be learned first — e.g., via a Stage 0 comprehension corpus or dedicated format-training pass — before validation can be enforced. This is a generalization of [[learned-by-playing]]: the model needs exposure to the format before it can be held accountable to it.

## Code status

The scratchpad validation logic and the enriched narration dataset v2 (with ground-truth HAND/VOIDS/COUNTS fields) are retained in `lem/gemma_star/star_loop.py` behind flags and are available for re-enabling once the model has learned the scratchpad format (78ba940, 5946c94).

## Links

[[star]] [[k1-grading]] [[r1-rationalization]] [[decisions/discard-illegal-traces]] [[learned-by-playing]] [[sources/380f3fa]] [[sources/78ba940]]
