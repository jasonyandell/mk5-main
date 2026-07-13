---
title: "Source digest: 1efb9c5 — candlewax-aware eq_outcome_distribution return"
kind: source
first_seen: 1efb9c5
last_updated: 1efb9c5
status: active
---

## Commit

- **SHA:** 1efb9c54def5b0597db1d26a51de18e2620487b3
- **Date:** 2026-04-19
- **Author:** Jason Yandell

> feat(burl): candlewax-aware eq_outcome_distribution return (ITER4_PLAN §2 E2)
>
> Extends OutcomeDistribution with: distribution_shape, modes,
> gap_between_modes, suggested_counterfactuals. Bimodality legible at
> tool surface. E3 rollout (N=500): 98 eq calls, 74 non-unimodal,
> 53 mixed-mode, still 0 conditional_outcome calls.
>
> Next candidate: ITER4_PLAN §2 Candidate C (what_would_change_my_mind).

Adds four new fields to `OutcomeDistribution` so bimodality is legible at the tool surface without parsing an 85-bin PDF. The `suggested_counterfactuals` field runs `conditional_outcome` on top-5 candidate dominoes × 3 seats × both modes. Live E3 rollout (N=500) showed 0 `conditional_outcome` calls despite the hints — upstream blocker is the model's breadth-first policy, not tool-surface legibility. See [[experiments/iter5-e2-candlewax-null]].

## Related pages

[[candlewax]] · [[burl]] · [[experiments/iter5-e2-candlewax-null]] · [[conditional-outcome-structural-nonuse]] · [[sources/7321952]]
