---
title: LAMIR-1 Ceiling
kind: topic
first_seen: 2026-04-22
last_updated: 2026-04-22
status: superseded
---

## The finding

After building the full [[lamir1]] harness and evaluating 8 inference variants on 560 held-out decisions, none beat the 0.551-regret direct π_me baseline. This is the LAMIR-1 ceiling: depth-1 look-ahead with distilled heads does not improve on argmax policy play (b42669a, PRACTICALITIES §20).

## Complete inference ladder

Evaluated on `corpus_eval_20.pt` (560 decisions, v3_consistency_10000g.pt adapter). Lower regret = better.

| mode | regret | bot-match | notes |
|---|---:|---:|---|
| direct π_me (baseline) | **0.551** | 76.07% | winner |
| q-bootstrap | 0.679 | 72.50% | depth-1 Q_head, world-conditioned |
| v-bootstrap | 1.645 | 66.96% | depth-1 V_head, world-blind |
| lamir1-qleaf | 2.006 | 64.29% | full rollout + Q_head leaf |
| lamir1 | 2.094 | 60.36% | full rollout + V_head leaf |
| lamir1-piopp | 2.268 | 62.10% | trained π_opp rollout + Q_head leaf |
| lamir1-piopp + Fix 6 | 2.350 | 62.50% | Bug 6 "fix" made it worse |

q-bootstrap (+23% regret vs direct) is the best look-ahead result because it avoids full rollout and uses the initial-deal world assignment without depletion. Full rollout variants are uniformly worse — compounding errors across 1-3 opp steps overwhelm any leaf signal (b42669a).

## Root cause

Kubíček & Lisý 2025 explicitly warn that a value function trained by distillation cannot serve as a look-ahead leaf evaluator. The scalar noise in the distilled V/Q heads is large enough to flip argmax at decision boundaries. π_me trained directly on argmax preserves action ordering; V_head trained on marginal oracle E[Q] does not.

**V_head is architecturally world-blind**: empirically confirmed std=0.000 across 200 world samples for the same decision. V_head takes only `state_emb` with no world encoder input — it cannot differentiate between world hypotheses and cannot serve as a look-ahead leaf.

The paper's fix is the T×T **multi-valued-states matrix value function** — a table of expected values under pairs of strategy transformations, requiring full CFR+ retraining. This is months of work (b42669a, MORNING4_STATUS).

## What Bug 6 revealed

Fix 6 (zeroing played-domino rows from `world_assign` before the leaf Q_head call) was believed to correct a stale-assignment artifact. It made every rollout mode slightly worse. The training convention in `dataset_seq_world` preserves the original `world_assignment` as the `played_mask` advances — "correcting" the assignment broke an invariant the Q_head relied on. The bug was in the hypothesis, not the model (2c380a6, b42669a).

## Side product

The [[pi-opp-head]] reached 68.57% oracle top-1 accuracy — substantially better than the rotated π_me proxy (~55%). Even so, `lamir1-piopp` performed worse than `lamir1-qleaf`. The bottleneck is the leaf evaluator, not opp simulation quality. π_opp is a real trained artifact; it just doesn't fix the ceiling (b42669a).

## Pivot options documented

Four directions identified at the ceiling (MORNING4_STATUS @ b42669a):

1. **Accept depth-1 ceiling**: ship q-bootstrap as a secondary inference mode. Use it as a second opinion when π_me entropy is high (small lift, low risk).
2. **Train a look-ahead-compatible V-head**: train V on expected Q under sampled opp play rather than marginal oracle E[Q]. Closer to the paper recipe, medium effort.
3. **Implement LAMIR paper faithfully**: multi-valued states + CFR+ solver. Research-grade, months of work.
4. **Bridge-AI / BMCS recipe**: Move to PPO self-play (OVERVIEW Move 1). The 68.6% π_opp head and world-conditioned Q_head are raw materials for a Bridge-AI-style player. Gus does not have to be a LAMIR clone.

## Option 4 taken

This page is the pivot-decision record. The project took **option 4**: self-play/value-native,
no CFR+, which became [[w42-jud-v1]] and [[jud]] — not a further LAMIR refinement, and not
option 2's look-ahead-compatible V-head. Neither `jud` nor `champion` cite this page back; that
missing forward link was the most consequential status mismark found in the era-6 gus-family
audit (2026-07-06). This page is now superseded, not active frontier — the LAMIR-1 track is
closed history, and Gus's decision-time look-ahead question was answered by abandoning
look-ahead in favor of `jud`'s different mechanism.

## Links

[[lamir1]] [[pi-opp-head]] [[student-distillation]] [[gus]] [[expected-q-value]] [[regret-eval]] [[w42-jud-v1]] [[jud]]
