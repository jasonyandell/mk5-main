---
title: Flexible Grader for Free-Form Responses
kind: decision
first_seen: 2026-04-13
last_updated: 2026-07-13
status: complete
---

## Decision

Grade LLM comprehension by extracting facts from free-form responses, not by rigid pattern matching. Implemented in `lem/gemma_star/grade_offline.py`.

## Specifics

Per question type in [[game-context-qa]]:

| Type | Grading method |
|---|---|
| `legal_moves` | Find numbered lists of dominoes anywhere in the response |
| `is_trump` | Find yes/no anywhere in the response |
| `where_is` | Find trick numbers anywhere in the response |
| `what_beats` | Check domino overlap between model's claim and ground truth |
| `count_status` | Extract count domino disposition from response text |

## Why

Rigid pattern matching ("Your answer must start with 'Legal moves:'") fails when the model gives a correct answer in a different format. The rigid grader produced `legal_moves` at 0%; the [[flexible-grader]] raised it to 70% — same model, same adapter, same responses, different grading logic.

This mattered concretely: the earlier "~40% overall" result for [[v4-adapter]] was a grader artifact, not a model failure. See [[stage-0-v4-comprehension-eval]] and [[1d3e1b7]].

## Generalizable principle

For knowledge evaluation of LLMs, the eval should parse what the model *means*, not what format the model chose. This is especially true for models learning to answer without rigid templates — the model may know the correct answer but express it differently than the template expects.

## Relation to prior decisions

Complements [[discard-illegal-traces]]: discard when the model is clearly wrong (illegal move — unambiguously bad); accept when the model is clearly right even if the format is unfamiliar. Both decisions distinguish signal from noise in the training and evaluation loops.

## Related pages

[[stage-0-v4-comprehension-eval]] · [[v4-adapter]] · [[rules-adapter]] · [[game-context-qa]] · [[3c33e86]]
