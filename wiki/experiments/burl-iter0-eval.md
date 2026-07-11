---
title: "Phase 4: Burl iter-0 Adapter Eval (Regressed)"
kind: experiment
first_seen: 789e14d
last_updated: 789e14d
status: active
---

## Summary

[[burl-iter0-adapter]] evaluated on the same 10 held-out decisions as Move 3 and the Move 4 spike. Result: 60% bot-match — down 10pp from Layer 1 and 29pp from spike v2. Clear negative signal. The analysis is sharp: the corpus inherited Layer 1's pathology; the primer is too long; one catastrophic tail call accounts for most of the mean regression; syntax is fine — this is a judgment regression, not a format failure.

([burl/eval/run_move4_spike.py @ 789e14d](../sources/789e14d.md))

## Results

| Metric | spike v2 | Layer 1 | burl-iter0 |
|---|---|---|---|
| bot_match_rate | 88.9% | 70% | **60.0%** |
| p_eq_geq_bot (K1) | 88.9% | 70% | 60.0% |
| mean_eq_delta | −1.92 | −3.00 | −3.33 |
| empty_tool_rollout | 0% | 20% | 10% |
| eq_outcome_dist | 15/10 | 2/10 | 2/10 |
| legal_rate | 100% | 100% | 100% |

Cost: $0.43 of $0.50 cap.

## Four-point analysis

**1. Corpus shape mirrored Layer 1's pathology.** Phase 2's 27 K1 wins were harvested from the primer+framing base, which was already eq-shy and `is_legal`-heavy. The adapter learned "Layer-1 Gemma" baked into weights.

**2. Primer is too long.** `mean_tokens_in` ≈ 40K chars per decision; attention spent on rules text rather than decision. `is_legal` (lightest-cognitive tool) dominates; `eq_outcome_distribution` stays at 2/10.

**3. One catastrophic tail call.** Decision 6: dom 27 vs bot 9, Δ−24.81. Without that single decision, iter-0 mean tracks Layer 1. The mean regression is largely one outlier.

**4. Syntax is fine — judgment regressed.** Tool-call format works. Gemma emits structurally valid `<|tool_call>` envelopes with the right schema. The failure is choosing badly given correct mechanics.

## Infrastructure: vLLM-LoRA blocker and fix

vLLM 0.19 rejects `Gemma4ForConditionalGeneration` for LoRA inference. Fix via `hf_overrides` forcing `Gemma4ForCausalLM` at load time. No HF-generate fallback needed. Endpoint retains vLLM path for both base and LoRA inference.

## Outlook

iter-1 plan: drop the full primer, keep the 42-aware framing block (provides vocabulary at tiny cost), re-harvest STaR corpus on the lighter prompt, retrain, eval. If the primer was the main regression cause, iter-1 should recover toward spike v2's ~88%.

## Audit (2026-07-07)

Two-pass audit against code and artifacts; no issues found.

- Commit 789e14d's message says "Phase 2's 50 K1 wins"; the page's 27 is correct (27/50 per [[burl-phase2-starcorpus]]) — the commit message has the typo.
- Raw eval artifacts (`burl/eval/results/move4_iter0_eval/`) were gitignored, so per-decision numbers (e.g. decision 6 Δ−24.81) trace only to the commit message.

## Related pages

[[burl-iter0-adapter]] · [[burl]] · [[gemma-4-e2b]] · [[tool-orchestration]] · [[primer-tradeoff]] · [[burl-phase1-primer]] · [[burl-phase2-starcorpus]] · [[burl-move3-base]] · [[burl-move4-native-spike]] · [[789e14d]]
