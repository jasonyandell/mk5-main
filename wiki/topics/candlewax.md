---
title: Candlewax (bimodal/multimodal outcome distributions)
kind: topic
first_seen: 1efb9c5
last_updated: 0545342
status: active
---

## Overview

A "candlewax" distribution is one where the E[Q] outcome PDF from `eq_outcome_distribution` is multi-peaked. The expected value (mean) is a poor summary because actual outcomes cluster around two or more distinct modes — e.g., "land the bid with 35 points" vs "get set and lose 30 points." The mean of these two spikes may be near zero, which would misleadingly suggest the decision is neutral (1efb9c5).

## Detection

The `eq_outcome_distribution` tool surfaces bimodality explicitly via four fields added in 1efb9c5:

- `distribution_shape`: `"unimodal"` / `"bimodal"` / `"multimodal"`
- `modes`: list of `{center, mass}` per peak
- `gap_between_modes`: float quantifying separation

In the E3 rollout set (N=500 decisions, 98 `eq_outcome_distribution` calls), 74 returned non-unimodal shapes; 53 had mixed-mode geometry (ceca203).

## Tool-surface companions

Three tools make candlewax distributions actionable for [[burl]]:

**`suggested_counterfactuals`** — hypothetical probe. Runs `conditional_outcome` on top-5 candidate dominoes × 3 non-self seats × both modes at N=5. Returns the top-2 counterfactuals with action-shaped rationale strings ("collapses the disaster tail," "confirms the winning scenario"). Cost-bounded; disabled inside `conditional_outcome` to avoid recursion (1efb9c5).

**`spike_drivers`** — empirical complement. For bimodal/multimodal distributions, reports which (seat, domino) assignments are empirically over-represented in each mode's worlds. Answers: "when this play lands in the disaster branch, who tends to be holding what?" Repackages the same information in Gemma's bid-satisfaction vocabulary ("partner has d5 → you win; partner has d14 → you lose"). Reliable at N≥50; coarse at N=10 (b0952a2).

**`what_would_change_my_mind`** — meta-tool. Returns the top-K unseen-world assumptions that most shift E[Q] of a given play, ranked by |shift|. Surfaces probe candidates directly rather than requiring the model to formulate them from the bimodal hint (7321952).

## Iter-5 E2 null finding

Surfacing bimodality legibly at the tool surface did not change model behavior. In live rollout (base Gemma, N=500, trimmed primer), the model's natural policy remained breadth-first alternative-play evaluation — it would consider `conditional_outcome` in thought prose, then decline in favor of trying another play. The environment-shape lever is validated at its firing site; the blocker is Gemma's policy, not the tool design (ceca203).

This is the candlewax E2 null result. See [[experiments/iter5-e2-candlewax-null]] (ceca203).

## Candlewax spike (multimodal investigation)

The [[candlewax-spike]] (`burl/candlewax_spike/`) pivoted away from LLM-as-reasoner because [[reasoning-coherence-verification]] is the bottleneck — reasoning chains are locally fact-correct but globally incoherent in ways that STaR cannot filter without a verifier subproject. The spike uses image-rendered PDFs + post-commit simulation + K1 soft-margin gate instead (0545342).

Receipts from the spike: image-as-alignment rescues small models (Haiku decision flipped 13→21 by image), v7 adapter beats base by +15% bot-match on 33 held-out examples (0545342).

## Links

[[burl]] [[tool-orchestration]] [[zeb]] [[forge]] [[reasoning-coherence-verification]] [[experiments/iter5-e2-candlewax-null]] [[candlewax-spike]]
