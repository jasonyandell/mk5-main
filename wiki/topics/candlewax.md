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

The browser visualizer runbook [[eq-browser-visualizers]] is the visual companion:
`eq_pdf_discs.html` shows the 85-bin PDF shape and win-threshold mass for each
candidate action, while `eq_surface_3d.html` and `eq_game_journey.html` make
near-tie and high-uncertainty E[Q] surfaces inspectable before changing policy.

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

## Also called (concordance)

The same object wears different names per project era, with near-zero co-occurrence
before 2026-07-06 — a documented cause of missed prior art. **Candlewax ≡ bimodal /
multimodal outcome PDF ≡ the "melted blob" of [[jud]]'s vocabulary ("un-melt the eq
blob") ≡ "mixed-mode geometry."** The phenomenon predates the April name by ~3.5
months: `forge/analysis/report/11_imperfect_info.md` (2026-01-06) measured it at
founding — the same P0 hand swings −42 → +40 across opponent configurations, only 11%
of hands are stable, and 53% of oracle outcome variance is within-hand (opponents'
cards). The 85-bin per-action PDFs (the "melted candlewax discs",
`eq_pdf_discs.html`) were rendered by 2026-01-24.

## The wall, stated precisely (Jason, 2026-07-06)

> "I saw eq, I said sure I could distill it. but for what purpose? no idea what to do
> with distilled melted candlewax."

**Distillation was never the wall; consumption is.** eq's output is this multi-peaked
object; a distilled copy is the same object, cheaper. The unanswered question — the
actual brick wall the project hit in early 2026 and has been answering ever since — is
what decision procedure *deserves* this shape, given that its whole message is that
every scalar collapse (mean, argmax, single threshold) discards the decision-relevant
structure. Distilling doesn't answer that; it makes the unanswered question run faster
(measured: the Feb 2026 [[full-teacher-eq-experiment|full-teacher capacity-ceiling null]],
`6081420`). Every era since
is a successive consumer hypothesis: LLM-as-reasoner (retired, 0545342), this page's
tool surface (tool validated, policy null), the Lens v1 utilities (EV least-dishonest
in aggregate — [[w42-lens-v1-utility-head-to-head]], which needs reading *against* this
page: EV wins on aggregate while the per-decision candlewax states are exactly where it
hides risk), [[rank-vs-price]] (route by consumer: play tolerates collapse, bids do
not), and [[jud]] (carry the full distribution to every decision, collapse only at
decision time — where the 2026-07-06 `judplay` mean-collapse showed the collapse choice
is itself where strength lives). **Any proposal of the form "distill X" must first name
the consumer and the licensed collapse.**

## Links

[[burl]] [[tool-orchestration]] [[zeb]] [[forge]] [[eq-browser-visualizers]]
[[reasoning-coherence-verification]] [[experiments/iter5-e2-candlewax-null]]
[[candlewax-spike]] [[jud]] [[rank-vs-price]] [[w42-lens-v1-utility-head-to-head]]
