---
title: "Q-Value Models Supersede Policy Models"
kind: decision
first_seen: 2026-01-17
last_updated: 2026-01-18
status: active
---

## Decision

Train and ship Q-value models — models that predict [[expected-q-value]] per candidate action —
rather than policy models that directly predict a move distribution. Documented as policy
during the 2026-01-17 posterior-weighting work in [[eq-genesis]] (era 3).

## Evidence at the time

The Q-value catalog promoted the same day (`b3a7e91`, 2026-01-17) carried two checkpoints:

| Model | q_gap | q_mae | Accuracy |
|---|---|---|---|
| `domino-qval-large-3.3M` | 0.071 | 0.94 | 79% |
| `domino-qval-small-816k` | 0.094 | 1.49 | 75% |

A Q-value model gives a continuous, inspectable score per legal action — it can be probed,
compared against the [[argmax-q-ceiling]], and averaged over sampled worlds directly (the
mechanism [[expected-q-value]] is built on). A policy model collapses that structure into a move
distribution up front, discarding the per-action value information the rest of the pipeline
(grading, ceiling analysis, later [[candlewax]] work on multi-modal outcome shapes) depends on.

Schema v2 (`cc59c61`, 2026-01-18) formalizes the Q-value-centric representation: `e_logits` →
`e_q_mean`, `e_var` → `e_q_var`, with `docs/EQ_STAGE2_TRAINING.md` (491 lines) as the canonical
spec, superseding `docs/EQ_MVP.md` the same day.

## Links

[[eq-genesis]] [[expected-q-value]] [[argmax-q-ceiling]]
