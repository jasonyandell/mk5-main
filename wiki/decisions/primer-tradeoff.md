---
title: "Primer Trade-off: Vocabulary vs Tool-Use Breadth"
kind: decision
first_seen: b8116b5
last_updated: 789e14d
status: active
---

## The trade-off

Prepending the LEM rules primer + a 42-aware framing block to Burl's prompts:

**Buys:** 42-specific vocabulary in traces. Partner, team, offense/defense, count, bid — 5–11 mentions per trace, up from 0 at spike v2. This vocabulary is critical for [[star]]: the training corpus must contain domain terms for the adapter to internalize them.

**Costs:** bot-match regression (88.9% → 70%), `eq_outcome_distribution` usage collapses (15 → 2 on 10 decisions), prompt size grows 4× (16s → 55–65s warm decision time).

## Why the trade is forced

Without 42 vocabulary in traces, STaR's kept corpus won't teach the adapter to reason in domain terms. The adapter would learn tool-call syntax without semantics — a format win with no content.

## Iter-0 result weakens the trade

The iter-0 adapter trained on the primer-heavy corpus reproduced Layer 1 Gemma's eq-shy pathology rather than internalizing the vocabulary AND recovering spike-v2-level tool-use breadth. The corpus shape mirrored the base's pathology — the adapter learned "Layer-1 Gemma." See [[experiments/burl-iter0-eval]].

This weakens (but does not invalidate) the trade. The vocabulary was in the corpus; the question is whether iter-0 training was sufficient to both internalize vocabulary AND recover distribution-tool usage.

## Direction at this frontier

Trim or remove the primer for iter-1; keep the 42-aware framing block (provides vocabulary at tiny cost without the 4× prompt size penalty). Re-harvest STaR corpus on the lighter prompt, retrain, eval. Open.

## Generalizable principle

When a pretext modifier buys one thing and costs another, the kept corpus shape matters more than the raw rollout metric. Phase 2's 54% K1 with `eq_outcome_distribution` at 8/50 is a lower-signal corpus than spike v2's distribution implied. The shape of what the model does in training — not just whether it passes K1 — determines what the adapter learns.

## Related pages

[[burl]] · [[burl-iter0-adapter]] · [[star]] · [[experiments/burl-phase1-primer]] · [[experiments/burl-phase2-starcorpus]] · [[experiments/burl-iter0-eval]] · [[tool-orchestration]] · [[sources/b8116b5]] · [[sources/789e14d]]
