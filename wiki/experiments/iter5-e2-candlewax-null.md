---
title: "iter-5 E2: Candlewax-Aware EQ Didn't Help"
kind: experiment
first_seen: ceca203
last_updated: ceca203
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

## Audit (2026-07-07)

Two-pass audit against code and artifacts; 1 correction applied in place and independently re-verified; second pass amended the follow-up list.

- Open follow-up: stack candlewax with the EQ-gate's `tool-nudge` variant on the stubborn T12 non-match decisions to force `eq_outcome_distribution` exposure under rules-as-tools (writeup follow-up #2; never run).
- Open follow-up: tune the 0.04 prominence threshold on an N=50 stability sweep — the current knee came from a 10-play smoke (writeup follow-up #4).
- Raw E3 traces (`burl/eval/results/e2|e3`) are not in the repo; the "three traces" claim is commit-attested ([[ceca203]], [[1efb9c5]] both state it), not artifact-attested.
