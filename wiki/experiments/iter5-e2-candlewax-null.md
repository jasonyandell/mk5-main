---
title: "iter-5 E2: Candlewax-Aware EQ Didn't Help"
kind: experiment
first_seen: 2026-04-19
last_updated: 2026-04-19
status: active
---

## Summary

[[candlewax]]-aware `eq_outcome_distribution` return fields (bimodality, modes, suggested counterfactuals) were implemented and validated at the tool surface, but did not change model behavior in live rollouts. The bottleneck is not tool-surface legibility — it is reasoning-coherence.

([burl/experiments/iter5_e2_candlewax_eval_writeup.md @ ceca203](../sources/ceca203.md))

## Setup

- Candlewax fields added: `distribution_shape`, `modes`, `gap_between_modes`, `suggested_counterfactuals` (see [[sources/1efb9c5]])
- Smoke: 8/10 plays returned non-unimodal shapes with populated rationales
- Live rollout T12 (base Gemma, N=10, `--enable-rules-tools`): 0 `eq_outcome_distribution` calls — rules-tools preamble crowds it out
- E3 rollout (N=500 decisions, no `--enable-rules-tools`): 98 `eq_outcome_distribution` calls, 74 non-unimodal, 53 mixed-mode — still **0 `conditional_outcome` calls**

## Result

Null at the behavioral level. Making bimodality legible at the tool surface did not cause the model to probe distributions with `conditional_outcome`.

## Interpretation

The model prefers breadth-first alternative-play evaluation ("try another play") over depth-probing ("probe this play's uncertainty"). Three traces show Gemma considering `conditional_outcome` in thought prose then declining in favor of trying a different play. The information is available; the policy does not use it.

This is a [[reasoning-coherence-verification]] gap, not a tool-surface legibility gap. Distinguishing "did the model read the bimodal signal and reason from it correctly?" from "did it produce the right commit?" requires a verifier that is a multi-week subproject.

## Related pages

[[candlewax]] · [[reasoning-coherence-verification]] · [[candlewax-spike]] · [[burl]] · [[conditional-outcome-structural-nonuse]] · [[sources/ceca203]] · [[sources/1efb9c5]]
