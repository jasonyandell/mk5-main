---
title: Gus π_opp Training (Schema v2)
kind: experiment
first_seen: 2026-04-22
last_updated: 2026-07-13
status: complete
---

## Summary

Training a dedicated opponent policy head (PiOppHead) on Schema v2 corpus data.
Frozen v3 trunk + new head reaches 68.6% oracle accuracy. NaN loss bug found and fixed.
π_opp is a real side product of the LAMIR-1 work. (commit messages @ 93859a0, b4e8ecd, 8106f01)

## Setup

- **Architecture**: frozen v3_consistency trunk + new `PiOppHead(state_emb, seat_id) → logits[7]`
- **Seat embedding**: 3-way (L-opp / partner / R-opp relative to decision player P)
- **Training target**: `oracle_softmax_per_seat[rel_seat]` from Schema v2 corpus — legal-masked CE
- **Data multiplier**: each batch item generates 3 training pairs (one per opponent seat)
- **Optimizer**: AdamW on head weights only; cosine LR schedule
- **Script**: `gus/train/train_pi_opp.py`

## Schema v2 fields (dcd9365)

Three new fields exposed by the dataset loader when corpus has Schema v2:

| Field | Shape | Content |
|---|---|---|
| `oracle_softmax_per_seat` | `[4, 7]` | π_opp training target per seat |
| `legal_mask_per_seat` | `[4, 7]` | Legal action mask per seat |
| `voids_per_seat` | `[4, 24]` | Flattened from [4, 3, 8] |

Backwards compatible — v1 corpora load unchanged, fields simply absent from item dict.

## NaN loss bug and fix (b4e8ecd)

Illegal slot log_probs are `-inf` after legal-masked log_softmax. Multiplying by a
zero target gives IEEE `0 × (-inf) = NaN`, poisoning the entire gradient. Fix: zero
out illegal slot log_probs before the dot-product with the target distribution. One
line; the fix is correct by construction (illegal slots contribute 0 to the loss).

## Result

**π_opp oracle accuracy: 68.6%** — the head predicts the correct oracle-argmax opponent
action on 68.6% of held-out decisions. (commit message @ 8106f01)

This is a real capability: a head trained supervised on oracle softmax targets that
models how each opponent seat plays, conditioned on observable state + seat identity.
Useful as a side product for LAMIR-1 rollout quality independent of the V/Q leaf issue.

## Links

[[gus]] · [[gus-lamir1-piopp]] · [[lamir1]] · [[joint-world-tensor]]
