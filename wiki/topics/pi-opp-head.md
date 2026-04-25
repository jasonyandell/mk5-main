---
title: π_opp Head
kind: topic
first_seen: 93859a0
last_updated: b4e8ecd
status: active
---

## What it is

`PiOppHead` is a lightweight adapter head trained to predict what each opponent seat will play given the current game state. It takes the frozen trunk's `state_emb` plus a 3-way seat embedding (L-opp / partner / R-opp relative to the decision player) and outputs logits over the 7 legal domino slots. The head has 1,879 parameters; the trunk is fully frozen during training (93859a0).

## Why it was needed

[[lamir1]] rollout variants (lamir1 / lamir1-qleaf) simulate 1-3 opponent plays between the decision and the leaf. Earlier modes used rotated π_me as a proxy for opponent play — a poor fit because π_me was trained from the decision player's frame, not a seat-relative opponent frame. Oracle top-1 accuracy for rotated π_me as π_opp is ~55%. A dedicated π_opp head reaches **68.57% oracle top-1**, a meaningful improvement (MORNING4_STATUS @ b42669a).

The motivation was that better opponent simulation might fix the lamir1 look-ahead failure observed in G8 (all rollout variants worse than direct π_me). The result confirmed it did not — the bottleneck is the leaf evaluator, not opp simulation quality (b42669a).

## Training setup

- **Corpus**: Schema v2 1000-game diverse-seed corpus (`corpus_v2_train_*_d0-9.pt`), which includes `oracle_softmax_per_seat [4, 7]` and `legal_mask_per_seat [4, 7]` (dcd9365)
- **Loss**: legal-masked cross-entropy against `oracle_softmax_per_seat[rel_seat]`; each batch item generates 3 training pairs (one per opp seat)
- **Optimizer**: AdamW on head weights only, cosine LR schedule
- **Epochs**: 20
- **Result**: 68.57% oracle top-1 accuracy; adapter saved as `gus/adapters/v3_10k_piopp.pt`

## Schema v2 loader changes

`JointWorldFullDataset` and `JointWorldFullIterable` were extended to expose three new fields when the corpus has Schema v2 fields (dcd9365):
- `oracle_softmax_per_seat [4, 7]` — π_opp training target
- `legal_mask_per_seat [4, 7]`
- `voids_per_seat [4, 3, 8]`

Backwards compatible: v1 corpora load unchanged.

## NaN bug and fix

**Bug**: During training, loss went NaN. Root cause: illegal slot log-probs are `-inf` after legal-mask softmax. Multiplying by the target weight (even 0.0) produces `0 * (-inf) = NaN` per IEEE 754 (b4e8ecd).

**Fix**: Zero out illegal slot log-probs explicitly before the dot-product with the target, so the NaN-producing multiplication never occurs. One-line fix in `gus/train/train_pi_opp.py`.

## Result in rollout

`lamir1-piopp` mode (1a1a324) uses the π_opp head for opp steps and Q_head as the leaf evaluator. On the 560-decision held-out eval:

| mode | regret | bot-match |
|---|---:|---:|
| lamir1-piopp | 2.268 | 62.10% |
| lamir1-qleaf (rotated π_me) | 2.006 | 64.29% |
| direct π_me (baseline) | **0.551** | 76.07% |

Better opp simulation made rollout outcomes slightly worse. The bottleneck is not opp quality — it is the scalar noise in the distilled leaf evaluator (b42669a). See [[lamir1-ceiling]].

## Links

[[lamir1]] [[lamir1-ceiling]] [[student-distillation]] [[gus]] [[expected-q-value]]
