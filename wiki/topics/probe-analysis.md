---
title: Probe Analysis (Gus interpretability)
kind: topic
first_seen: 2026-04-21
last_updated: 2026-04-21
status: active
---

## Overview

Probe analysis runs counterfactual and attention-based interpretability experiments on a trained [[gus]] student to verify that it has internalized real game structure — not argmax lookup. PRACTICALITIES §19 documents a six-probe receipt on v3-10k against a nightmare hand (seed 900000, blanks declaration, 5 of 7 trumps held by opponents including the 0-0 boss) (245918d).

## Six probes

**1. Embedding structure**: doubles, count dominoes, and high-pip dominoes cluster distinctly in the student's learned embedding space. The model has organized the domino space by game-relevant categories without being explicitly told to (245918d).

**2. Attention patterns**: multi-layer reasoning is principled. Layer 0 anchors on the declaration token (DECL); middle layers scan the narrator's hand (MINE tokens); later layers converge toward the committed action. CLS evolves across layers toward the chosen action (31f0ec3, 245918d).

**3. Counterfactual V deltas match oracle E[Q] deltas within 0.5 Q-pts**: swapping one domino in the hypothetical hand and re-running V_head produces delta estimates that track the oracle's E[Q] differences to within 0.5 Q-points on the nightmare hand. V_head has internalized real game value, not surface pattern matching (245918d).

**4. Trumpness-gated sensitivity**: the 6-6's impact is conditional on declaration. In trump declarations: V delta +17 to +22 (large). In fours (where 6-6 is not trump): V delta ≈ 0. In a declaration where 6-6 displaces a trump boss: V delta −28. The student respects trump declaration semantics (245918d).

**5. Hand-level threats and boons**: all five opponent-held trumps are correctly identified as threats. The 0-0's location alone is worth a 26 Q-pt swing — "if right-opponent has 0-0 we're sunk" is literally what the V deltas say. Consistent with the game-theory of trump boss positioning (245918d).

**6. Oracle agreement on counterfactuals**: counterfactual sensitivity is confirmed against oracle-computed E[Q] deltas. Direction and magnitude match. The student's V_head is not just correlated — it agrees quantitatively (245918d).

## Threats

**Initial strategy-fusion diagnosis retracted**: an early read of the 1-1→6-6 swap delta suggested strategy fusion (PIMC-trained bias bleeding into V). Corrected in the writeup: the asymmetry is bilateral-swap asymmetry + depth-vs-breadth saturation, not strategy fusion. The oracle confirmed the direction (245918d).

The corrected interpretation: swapping 1-1 out and 6-6 in changes multiple strategic dependencies simultaneously; the asymmetry reflects genuine game-theory, not training artifact (245918d).

## Boons

Counterfactual V swap is a **usable interpretability tool** — the delta between V_head on the actual hand and V_head on a modified hand gives a grounded sensitivity estimate without oracle calls. Promotable to `gus/eval/` if threat-boon analysis becomes a recurring diagnostic (245918d).

The probe results collectively support the conclusion: Gus has internalized real game structure, not surface correlations. The [[student-distillation]] approach from variance-free oracle labels succeeded in teaching the model to reason about game position (245918d).

## Links

[[gus]] [[student-distillation]] [[v-pi-decoupling]] [[consistency-regularizer]] [[regret-eval]] [[belief-propagation-gap]] [[expected-q-value]]
