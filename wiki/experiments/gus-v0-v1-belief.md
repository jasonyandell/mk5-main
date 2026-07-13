---
title: Gus v0/v1 Belief Heads
kind: experiment
first_seen: 2026-04-20
last_updated: 2026-04-20
status: active
---

## Summary

Two iterations of belief-only student on the 100-game corpus (seeds 0-99 train,
900000-900019 eval). v0 MLP confirms infrastructure works but overfits severely.
v1 transformer shows the right per-decision shape and establishes that the ceiling
is data, not architecture. (commit messages @ c04bda3, 8dbf7f3)

## v0 — MLP (c04bda3)

- **Features**: 183-dim flat state vector (decl + player + decision_idx + 6×28 domino masks:
  my_hand, played_total, played_by_seat×4)
- **Model**: StateEncoder (MLP) + BeliefHead only; ~200K params
- **Result**: peak eval 34.6% (chance floor 33.3%), train reaches 100% — severe overfit
- Smoke-tested on 1-game tire-kick corpus; real evaluation awaited 100-game corpus

## v1 — Transformer (8dbf7f3)

Replaces the bag-of-masks MLP with a transformer over tokenized play sequences.

**Token layout** (33 tokens, fixed):
- CLS + DECL + 7 MINE + 24 PLAY

**Channels per position** (5): token / type / trick / pos / player_rel

The transformer can attend across plays and infer voids — impossible with v0's flat features.

**Results** (100-game corpus, seeds 0-99 train, 900000-900019 eval):

| Model | Eval top-1 | Train top-1 | Notes |
|---|---|---|---|
| v0 MLP | 34.6% | 100% | Severe overfit |
| v1 transformer | 37.5% | 74% | Overfits less; right shape |

**Per-decision breakdown (v1)**:

| Decision | Unseen dominoes | Eval top-1 | Interpretation |
|---|---|---|---|
| 0 | 21 | 33% | No info yet — chance floor |
| 10 | ~13 | 42% | Mid-game learning |
| 22 | ~4 | 45% | Late-game sharpening |
| 26 | 1 | 75% | Near-certain (1 domino left) |

Late-game belief IS learning. The aggregated 37.5% is dragged down by the mandatory
chance floor at decision 0 where no observations exist yet. (commit message @ 8dbf7f3)

## Verdict

Ceiling is data, not architecture. The per-decision shape is correct; 100 games is
insufficient to push mid-game belief above chance. The full 4-head student with
[[topics/dense-q-supervision]] (3400× denser signal per decision) is the right next step —
see [[experiments/gus-4head-baseline]].

## Links

[[gus]] · [[topics/student-distillation]] · [[joint-world-tensor]] · [[experiments/gus-4head-baseline]]
