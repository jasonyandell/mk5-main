---
title: Gus Interpretability Probes (v3-10k)
kind: experiment
first_seen: 245918d
last_updated: 245918d
status: active
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

### Embedding structure

Doubles, count dominoes, and high-pip clusters form in the learned token embeddings.
Structural game concepts (trump membership, pip value categories) emerge without
explicit supervision. (commit message @ 245918d)

### Attention patterns

Multi-layer attention evolves with principled progression:
- DECL token anchors early layers (declaration context)
- MINE scan in middle layers (own-hand reasoning)
- Action commit in final layers (decision formation)

CLS evolves across layers toward the chosen action. (commit message @ 31f0ec3)

### Counterfactual V deltas match oracle within 0.5 Q-pts

Swapping individual dominoes in the hypothetical hand and measuring V_head response:
student V deltas track oracle E[Q] deltas to within 0.5 Q-pts. The model "knows"
what each domino is worth in context. (commit message @ 245918d)

### 6-6 impact is trumpness-gated

| Context | 6-6 swap delta |
|---|---|
| Trump declarations | +17 to +22 Q-pts |
| Fours (non-trump) | ~0 |
| Displacing a trump boss | −28 Q-pts (catastrophic) |

The student correctly treats 6-6 as context-dependent — not uniformly valuable. (commit message @ 245918d)

### Hand-level threats and boons

All five threats are trumps. 0-0 location alone accounts for a 26-Q-pt swing:
"if right-opponent has 0-0 we're sunk" is literally quantified in V_head response.
(commit message @ 245918d)

### Strategy-fusion correction

Initial diagnosis of a "strategy fusion" pattern on a 1-1→6-6 swap was incorrect.
Corrected to bilateral-swap asymmetry + depth-vs-breadth saturation. Oracle confirmed
the direction of the correction. (commit message @ 245918d)

## Oracle agreement

Despite the nightmare hand, V_head matches the ground-truth oracle to within 0.5 Q-pts.
This is the strongest interpretability result in the Gus replay: the student is doing
principled inference, not table lookup.

## Conclusion

Counterfactual V sensitivity is a usable interpretability tool — promotable from
`scratch/` to `gus/eval/` if threat-boon analysis becomes a recurring diagnostic.

## Links

[[gus]] · [[topics/regret-eval]] · [[topics/v-pi-decoupling]] · [[experiments/gus-v3-consistency-full-run]]
