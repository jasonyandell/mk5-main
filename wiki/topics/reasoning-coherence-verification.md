---
title: Reasoning-Coherence Verification (the bottleneck)
kind: topic
first_seen: 2026-04-19
last_updated: 2026-04-20
status: superseded
---

## Overview

Reasoning-coherence verification is the act of checking that a model's reasoning *chain* is internally consistent and grounded in the game state — not just that its final answer is correct (which K1 checks) and not just that each intermediate fact is true (which the [[rationalization-verifier]] checks). It asks: do the model's conclusions follow from its stated premises? (ceca203, 0545342)

## Why it matters for Burl

[[burl]] aims to teach tool-orchestrated reasoning via [[star]]. If the training corpus contains traces that are locally fact-correct but globally incoherent — where the model states correct facts and then draws unrelated conclusions — STaR trains the model to reproduce that incoherence. The K1 gate only checks whether the final play was good; it cannot detect whether the reasoning that produced the play was valid (ceca203).

## Identified as the iter-5 bottleneck

The [[candlewax]] E2 null result and the broader pattern of not getting past iter-3-rules 90% both trace back to this gap. The environment-shape levers (candlewax hints, spike_drivers, what_would_change_my_mind) are validated at their firing sites, but the model's policy — breadth-first alternative evaluation rather than depth-probing — persists across training iterations because coherence is not being filtered (ceca203, 0545342).

Without coherence verification, adding more training data may amplify existing incoherent reasoning patterns rather than correcting them. This was confirmed in the candlewax spike: STaR iter 2 plateaued without a reasoning verifier (0545342).

## Scope and deferral

A real reasoning-coherence verifier requires a multi-week subproject: designing what "coherence" means for tool-orchestrated traces, building the checking infrastructure, and validating it against human judgment.

Never built, because the LLM-as-reasoner program it gates was retired ([[candlewax]],
0545342), not because the subproject was merely postponed.

The [[candlewax-spike]] pivoted precisely away from LLM-as-reasoner because this verification is missing. It substituted structured prediction blocks (winner_seat, count_to_my_team, count_to_opponents) verified by post-commit engine simulation — a partial, domain-specific coherence check for that specific pipeline.

## Relationship to prior verification work

The [[rationalization-verifier]] checks 6 engine-derived facts: domino validity, references-visible, hand claims, trump declaration, trump membership, action match. These are necessary but not sufficient for coherence — a trace can pass all 6 checks and still reason incoherently ("the 5-5 is trump, so I should avoid using trumps, therefore I play the 5-5").

Coherence verification extends fact-checking to the logical structure of the argument (ceca203).

## Links

[[rationalization-verifier]] [[candlewax-spike]] [[burl]] [[star]] [[experiments/iter5-e2-candlewax-null]] [[candlewax]]
