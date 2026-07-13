---
title: Gus Shine Analysis — Where the Student Is Perfect
kind: experiment
first_seen: 2026-04-21
last_updated: 2026-04-21
status: active
---

## Summary

Mirror of blunder forensics. Characterizes the 73% of held-out decisions where the
student is already optimal (regret < 0.1 Q-pt). Reveals a two-stage routing heuristic
that eliminates detector work on ~80% of decisions at negligible cost. (commit message @ 7a9c720)

## Setup

- **Adapter**: `v2_voids_3000g_big`
- **Dataset**: 560 held-out decisions
- **Perfect threshold**: regret < 0.1 Q-pt
- **Script**: `gus/eval/shine_analysis.py` (691 lines)
- **Breakdown axes**: trick, declaration, legal count, oracle E[Q] spread

## Key findings

### Perfect-decision composition (410 of 560)

| Category | Spread range | Count | Share |
|---|---|---|---|
| Dead-ties | < 0.5 | 242 | 59.0% |
| Moderate | 0.5-5 | 64 | 15.6% |
| Sharp-and-perfect | ≥ 5 | 104 | 25.4% |

59% of perfects are dead-ties where any legal play is roughly equivalent. But 25.4%
(104 decisions) are genuinely high-spread positions where the student made the right
call — real skill, not luck. These span all 10 declarations. (commit message @ 7a9c720)

### Distribution flip: perfect vs blunder by decision index

- **End-game** (dec 24-27): 100% perfect, 0 blunders
- **Early-mid leads** (dec 0, 4, 8): 70% of all blunders concentrated here

### Zero-inference routing heuristic

Trust the student unconditionally when:
- `legal_count ≤ 2` **OR** `decision_idx ≥ 22`

Effect: covers 450/560 decisions (≈80%) at ≤ 2% blunder rate. Only ~110
"wide-choice mid-game" decisions need the detector at all — 4× reduction in detector
workload. (commit message @ 7a9c720)

## Implication for capacity/data allocation

- End-game training examples are wasted regret-reduction budget; student already achieves
  0 regret there without targeted effort.
- The sharp-and-perfect 104 decisions prove the shared encoder has genuine mid-game
  capability — the blunder tail is not from a globally weak model, but from specific
  high-spread early decisions.
- Two-stage routing (cheap pre-filter → expensive detector on residual ~110 decisions)
  is the right production shape, cheaper than running the full GBM on all 560.

## Links

[[gus]] · [[topics/regret-eval]] · [[experiments/gus-blunder-detector]] · [[experiments/gus-router-pilot]]
