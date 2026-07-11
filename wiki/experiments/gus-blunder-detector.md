---
title: Gus Blunder Detector (oracle-feature + student-feature)
kind: experiment
first_seen: f90682c
last_updated: 5373223
status: active
---

## Summary

Two iterations of a small GradientBoostingClassifier that predicts whether [[gus]] will
blunder (regret > 8 Q-pts) at a given decision, trained from state features. Oracle-feature
detector (AUC 0.926) establishes the ceiling; student-feature detector (AUC 0.839) is the
deployable version. Receipt 13 in PRACTICALITIES.md is the key summary. (commit messages
@ f90682c, 5373223)

## v1 — Oracle-feature detector (f90682c)

- **Training**: 6k decisions from 3000g corpus; 3k disjoint test decisions
- **Features**: oracle E[Q] vector per decision (includes oracle spread, oracle E[Q] std)
- **Dominant features**: `oracle_spread` (max − min of legal E[Q]) + `oracle_eq_std`
  account for 77% of feature importance — both proxy for decision difficulty

**Results**:

| Flag rate | Recall of blunders | Regret (oracle replacement) |
|---|---|---|
| 15% | 80% | — |
| 25% | 99% | — |
| ~20% | — | 1.23 → 0.46 |

ROC-AUC: 0.926. Approaches the 0.5-1.0 Q-pt teacher-noise floor.

## v2 — Student-feature detector (5373223)

Deployable version using only features derivable from the student's own outputs at
inference — no oracle required.

**Feature set (28 dims)**:
- Policy: `pi_peak`, `pi_entropy`, `pi_argmax_margin`
- V_head: scalar
- Q_head (K=20 sampled worlds): per-action mean/std/min/max; legal-action spread; std of per-action means
- Consistency gaps: V_head vs policy-expected-Q, V_head vs Q_mean_chosen
- Meta: `decision_idx`, `trick_num`, `trick_pos`, `player_rel`, legal count, declaration one-hot, voids count
- Belief: max prob, entropy, high-confidence count

**Results vs v1**:

| Version | Features | ROC-AUC | PR-AUC |
|---|---|---|---|
| v1 oracle | Oracle E[Q] vector | 0.926 | 0.29 |
| v2 student | Student outputs only | 0.839 | 0.15 |

At 20% flag rate with oracle-argmax replacement: baseline regret 1.13 → 0.49 (57% reduction).

**Top feature**: `pi_peak` (importance 0.19) — the student's own policy confidence.
When π_me is uncertain, trigger fallback. Intuitive and deployable.

## Receipt 11 (109f9e1)

Arena: 1.39 Q-pt decision regret → 18pp game-level win-rate gap. Student team wins 52%
vs all-oracle 70%. Real but meaningfully weaker player.

## Receipt 12 (109f9e1)

Naive ensembling hurts regret. Voting/softmax-average over 8 adapters boosts bot-match
(71.3%) but increases regret (1.43-1.55) vs best single (1.39). Averaging dilutes the
confident-right adapter on sharp decisions. Oracle-per-decision ceiling: 0.36 regret
(74% reduction) — massive latent adapter diversity. Router, not ensemble.

## Receipt 13 (109f9e1)

Student-feature blunder detector at ROC-AUC 0.839. At 20% flag rate: 1.13 → 0.49 regret.
Key weak link: Q_head spread underperforms oracle spread as blunder proxy — Q_head trained
on one world per forward pass is too noisy. Fixes: multi-world variance regularization
during training, or K=50+ worlds at inference.

## Conclusion

Detect-and-route is deployable from student-only features. ~0.5 Q-pt regret is achievable —
the practical floor for vanilla distillation. See [[experiments/gus-router-pilot]] for the
end-to-end validation.

## Links

[[gus]] · [[topics/regret-eval]] · [[topics/v-pi-decoupling]] · [[experiments/gus-router-pilot]] · [[experiments/gus-scaling-ladder]]

## Audit (2026-07-07)

Two-pass audit against code and artifacts; no issues found.

- Cheap next probe: rerun v2 with K=50 sampled worlds at inference (the receipt-13 fix) — no retraining, a flag change in `gus/eval/blunder_detector_student.py`.
- The v1/v2 baseline regrets differ (1.23 vs 1.13) because they use different eval slices.
- Status is `active`; if [[gus-router-pilot]] closed this line, the page may deserve `status: closed`.
