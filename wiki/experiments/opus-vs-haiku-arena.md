---
title: "Opus vs Haiku: Head-to-Head Seed 900010"
kind: experiment
first_seen: 2026-04-19
last_updated: 2026-04-19
status: active
---

## Summary

[[burl-selfplay-arena]] head-to-head on seed 900010. Same deal (bad for team 0), same declaration. Opus salvages 7 points where Haiku was shut out. Opus shows better decision quality AND better tool-use economy at 6.5× the cost.

([burl/selfplay/arena.py @ 39aafaf](../sources/39aafaf.md))

## Results

| Run | Final | Bidder | Cost | Wall |
|---|---|---|---|---|
| Haiku | 0/42 | 0 | $0.84 | 7m22s |
| Opus | 7/35 | 7 | $5.58 | 8m50s |

Opening lead diverges: Haiku led 3(2|0) generic low trump; Opus led 2(1|1) double-ones after 7 `eq_outcome_distribution` evaluations.

## Tool-use efficiency delta (same 28 decisions each)

| Tool | Haiku | Opus |
|---|---|---|
| trump_declared | 24× | 1× |
| is_legal | 92× | 27× |
| eq_outcome_distribution | ~1.5/turn | ~1.5/turn |
| conditional_outcome | 0 | 0 |

Haiku re-queries system-prompt facts on every turn (`trump_declared` 24×); Opus trusts it after reading once. Both models call `eq_outcome_distribution` at similar rates when the primer nudges distribution probing. Neither model reaches for `conditional_outcome` — the 4th session observation of this pattern across 56 total decisions.

## Significance

Confirms Opus's decision quality advantage on bad deals. The `conditional_outcome=0` finding is the session's cleanest single datum: across 145+ decisions and every model tested (base Gemma, Haiku, Opus), no model reaches for the counterfactual tool zero-shot. STaR must synthesize demos to enable this behavior. See [[conditional-outcome-structural-nonuse]].

## Related pages

[[burl-selfplay-arena]] · [[haiku-4-5]] · [[burl]] · [[conditional-outcome-structural-nonuse]] · [[experiments/burl-move4-native-spike]] · [[sources/39aafaf]]
