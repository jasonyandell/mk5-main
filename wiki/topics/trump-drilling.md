---
title: Trump-Membership Drilling (Stage 0 v3)
kind: topic
first_seen: 2026-04-11
last_updated: 2026-04-11
status: superseded
superseded_by: game-context-qa
---

## Overview

Trump-membership drilling is the targeted Q&A extension added to the [[kerry-curriculum]] to form Stage 0 v3. Five drill types attack the stubborn 6-4-under-fives error that persisted through [[first-gemma-contact]], [[second-gemma-contact]], and [[third-gemma-contact]] (601f622).

## Five drill types

| Type | Example question |
|---|---|
| `is_trump` | "Is 6-4 trump when fives are trump?" (No — pips 6, 4; no 5) |
| `list_trumps` | "List all trumps when fours are trump" |
| `which_trumps` | Batch classification of multiple dominoes under a given declaration |
| `trump_or_follow` | "Did they follow suit or play a trump?" |
| `count_trump` | "Is the 5-5 also a trump when doubles are declared?" |

5,000 examples, all engine-generated ground truth. Combined with the 15k [[kerry-curriculum]] examples to produce the Stage 0 v3 corpus of 20k examples total. Trained 200 steps on B200 (601f622).

## Outcome

v3 STaR results (5 iterations from [[v3-adapter]] base, 8c1bb14):

| Iter | Pass | Illegal |
|---|---|---|
| 0 | 44% | ~13% |
| 1 | 42% | ~13% |
| 2 | **48%** | ~13% |
| 3 | 47% | ~13% |
| 4 | 38% | ~13% |

Peak 48% at iter-2 is a new high water mark (previously 42% on v1, 46% on Kerry). Illegal rate stable at ~13% — the drilling did not cut illegal rate further (Kerry's 12% already achieved the floor) but did raise peak pass rate. Best adapter: `star-iter2` (8c1bb14).

## Stage 0 progression summary

| Stage 0 version | Avg pass | Peak | Illegal |
|---|---|---|---|
| v1 (3.5k Q&A) | ~37% | 42% | 33% |
| Kerry (15k) | ~43% | 46% | 12% |
| v3 (20k + trump drill) | ~44% | 48% | 13% |

"Each curriculum round raises the floor." (8c1bb14)

Superseded two days later by [[game-context-qa]] (Stage 0 v4, 4729dad), which discarded the
flashcard structure entirely — the five drill types above do not carry forward past v3.

## Links

[[rules-adapter]] [[kerry-curriculum]] [[v3-adapter]] [[third-gemma-contact]] [[stage-0-progression-star]] [[first-gemma-contact]] [[second-gemma-contact]]
