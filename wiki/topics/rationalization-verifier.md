---
title: Rationalization Verifier (6 engine checks)
kind: topic
first_seen: 2026-04-17
last_updated: 2026-07-13
status: complete
---

## Overview

The rationalization verifier is a filter applied to model-generated rationalizations before they are accepted as training data. Each rationalization is run through 6 engine-derived checks. Traces that pass all 6 are safe to train on regardless of prose style; traces that fail any check are discarded (b857299).

## The 6 checks

| Check | What it verifies |
|---|---|
| **domino validity** | Every domino name the model references is a real domino in the set |
| **references-visible** | Claims about played dominoes match the actual trick history |
| **hand claims** | Claims about the narrator's remaining hand match the true hand |
| **trump declaration** | The model's stated trump matches the actual declaration |
| **trump membership** | Every "X is trump" claim is engine-true |
| **action match** | The chosen play matches what the model is supposed to explain |

(b857299)

## Rationale

It is acceptable for the model to produce creative reasoning chains, but only if the facts it reasons about are true. A rationalization that arrives at the correct play by reasoning through hallucinated game-state is still a corrupted trace. The 6 checks make hallucinations irrelevant for training: any trace that passes is factually grounded, regardless of how it was generated (b857299).

This extends [[decisions/discard-illegal-traces]] — "reasoning from impossible states is poison" — from the action level to the full reasoning chain: every factual claim within the reasoning must be engine-verified (b857299).

## Usage in v10

In Stage 0 v10 ([[v10-adapter]]), 500 decisions were scouted with the v9 adapter. The verifier accepted 331 (66% pass rate). These clean rationalizations were joint-trained with v9 comprehension data (upweighted 10×) in one SFT pass. See [[r1-rationalization]] "Rationalization-SFT bootstrap" section (0c7392f).

## Relationship to flexible grader

The [[decisions/flexible-grader]] extracts facts from free-form comprehension answers. The rationalization verifier is the analogous mechanism for play-reasoning outputs: both accept free-form text but check specific claims against engine ground truth (b857299).

## Links

[[r1-rationalization]] [[v9-adapter]] [[v10-adapter]] [[decisions/discard-illegal-traces]] [[decisions/flexible-grader]]
