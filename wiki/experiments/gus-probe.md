---
title: Gus Interpretability Probes (v3-10k)
kind: experiment
first_seen: 2026-04-21
last_updated: 2026-07-13
status: complete
---

## Summary

Six-probe interpretability receipt (PRACTICALITIES §19) on v3-10k against a nightmare
hand. Counterfactual V sensitivity matches oracle E[Q] deltas within 0.5 Q-pts. Gus
has internalized real game structure, not argmax lookup. (commit message @ 245918d)

## Setup

- **Adapter**: v3_consistency_10k
- **Hand**: seed 900000, blanks declaration, 5 of 7 trumps held by opponents including
  the 0-0 boss — a genuine worst-case hand for the student

## Probes and findings

### Probe 1 — embedding structure

Doubles cluster (avg cos +0.047 vs non-doubles −0.022), counts cluster (+0.037 vs
−0.021), high-pip families are tighter. Categorical concepts (doubleness, countness,
magnitude) live in the raw embedding; relational concepts ("6-6 protects 6-4") are
contextual — encoded in the transformer layers, not the embeddings. (commit message @ 245918d)

### Probe 2 — attention evolution

CLS attention across the 6 encoder layers follows a readable reasoning chain: layer 0
anchors on the declaration (weight 0.65), layers 1-2 survey the big non-trump 6-4,
layers 3-4 reconsider toward 1-1, layer 5 concentrates on 1-1 at 0.26. Final π_me
probability on 1-1: **0.91**. Multi-step reasoning, not one-shot argmax lookup.
(commit messages @ 31f0ec3, 245918d)

### Probe 3 — counterfactual hand swaps

Swapping 1-1 → 0-0 raises V +11.7 Q-pts (top trump is gold). Swapping 1-1 → 6-6
**drops V −5.3 Q-pts** — superficially counter-intuitive since 6-6 is a "bigger" card.
(commit message @ 245918d)

### Probe 4 — oracle verification of the counterfactual

The 3.3M-param oracle ran on both counterfactual deals (18.9s on MPS). Oracle
ΔE[Q_max] = −5.83; Gus ΔV = −5.34. **Agreement within 0.5 Q-pts.** The
counterfactual is correct game theory, not a model artifact: in a defensive hand, 6-6
displaces 1-1 (which the opponent then holds), and the opponent's gain exceeds P0's
gain. An initial "strategy-fusion leakage" diagnosis was wrong; bilateral-swap
asymmetry plus depth-vs-breadth saturation in the 6-suit fully explains the delta.
(commit message @ 245918d)

### Probe 5 — per-domino impact atlas for 6-6

238 bilateral swaps across 20 eval games:

| Declaration | Mean ΔV | Verdict |
|---|---:|---|
| Sixes / doubles / follow-me-8 (trump) | +17 to +22 | Always helpful |
| Fives / threes / twos / ones | +2 to +4 | Mixed; catastrophe if displaces trump boss |
| Fours | −0.09 | Essentially neutral |
| Worst case: 6-6 displaces 5-5 in fives | −27.67 | Trump-boss displacement |

6-6's value is entirely determined by trumpness of the declaration and which card it
displaces. Gus has not learned "big card = good" — it learned the context-dependent
value function 42 actually has. (commit message @ 245918d)

### Probe 6 — hand-level threats and boons

Conditioned on the stored [[joint-world-tensor]], no new forward passes needed. For
game 0 decision 0, baseline E[Q] = −11.15:

| Top boon (partner holds) | Q uplift |
|---|---:|
| 0-0 (top trump) | +16.6 |
| 6-0 (high trump) | +12.5 |

| Top threat (opp holds) | Q drop |
|---|---:|
| R-opp holds 0-0 | −9.4 — "if R-opp has the boss we're sunk" (literal) |

0-0 alone contributes a **26-Q-pt swing** based on location. All five top threats and
all five top boons are trumps — the hand's outcome is determined by trump distribution
before the first card hits the table. (commit message @ 245918d)

## Oracle agreement

Despite the nightmare hand, V_head matches the ground-truth oracle to within 0.5 Q-pts.
This is the strongest interpretability result in the Gus replay: the student is doing
principled inference, not table lookup. The [[student-distillation]] approach from
variance-free oracle labels succeeded in teaching the model to reason about game position.

## Conclusion

Counterfactual V sensitivity is a usable interpretability tool — grounded sensitivity
estimates without oracle calls; promotable from `scratch/` to `gus/eval/` if
threat-boon analysis becomes a recurring diagnostic.

## Links

[[gus]] · [[gus-line]] · [[regret-eval]] · [[v-pi-decoupling]] · [[joint-world-tensor]] · [[student-distillation]] · [[gus-v3-consistency-full-run]]
