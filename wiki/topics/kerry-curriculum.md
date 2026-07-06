---
title: Kerry Newberry Curriculum
kind: topic
first_seen: f8cdbe7
last_updated: 43009a4
status: superseded
superseded_by: trump-drilling
---

## Overview

The Kerry Newberry Curriculum is the Q&A corpus structure used for [[rules-adapter]] Stage 0 v2 ("Kerry"). It is modeled on Kerry Newberry's Learner's Guide to [[texas-42]] and engine-scaled to 15,000 examples (f8cdbe7).

## Structure

Four sections with deliberate weighting:

| Section | Topic | Examples | Weight |
|---|---|---|---|
| A | Getting to Know the Dominoes | 2,250 | 15% |
| B | Understanding the Game | 3,000 | 20% |
| C | Following Suit | 6,000 | 40% |
| D | Who Wins the Trick? | 3,750 | 25% |

**Section A** covers pip recognition, count domino identification, and doubles.

**Section B** covers bidding basics, trick structure, and hand scoring.

**Section C** (heaviest, 40%) covers legal-move derivation, void inference, and trump membership under every declaration. It deliberately generates tricky hands where suit/trump interaction is non-obvious — e.g., 5-3 when fives are trump cannot follow threes because 5-3 is a trump, not a three. This is the exact stumbling block seen in [[experiments/first-gemma-contact]] and [[experiments/second-gemma-contact]] (f8cdbe7).

**Section D** covers trick resolution under every declaration.

## Rationale for weighting

Section C matches precisely the failure mode identified in second-contact evaluation: trump membership under non-obvious hands. Heavy weighting there is deliberate — this is the hardest part for human learners and for 2B models alike. The original v1 corpus spread weight more evenly across 7 categories; v2 concentrates signal where the model needs it most (f8cdbe7).

## Results

Evaluated in [[experiments/third-gemma-contact]] against the same probe prompt used in first and second contact (43009a4):

| Dimension | v1 adapter | v2 Kerry adapter |
|---|---|---|
| Hand tracking | Correct | Correct (maintained) |
| Led suit | — | Correct |
| Void recognition | — | Correct ("you hold no fours") |
| Trump non-membership | Wrong (called 6-2/6-1 trump) | **Fixed** ("no fives, no trump") |
| Trump membership (6-4 under fives) | Wrong | Still wrong (narrowed to this one case) |
| Strategic reasoning depth | Shallow | Dramatically deeper |
| Final answer | Legal, correct | Legal, correct |

The 6-4 under fives error remains. All other trump-membership cases are now handled correctly. This narrows the remaining rule gap to a single edge case (43009a4).

Superseded same-day-ish by [[trump-drilling]] (601f622), which adds 5k targeted
trump-membership Q&A on top of this 15k corpus to close the remaining error. Frozen
mid-arc: this page does not describe the later v4–v10 curriculum rounds (see
[[rules-adapter]] for the full progression).

## Links

[[rules-adapter]] [[kerry-adapter]] [[learned-by-playing]] [[texas-42]] [[experiments/third-gemma-contact]] [[experiments/first-gemma-contact]] [[experiments/second-gemma-contact]]
