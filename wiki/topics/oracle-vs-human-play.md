---
title: Oracle vs human play (load-bearing epistemic frame)
kind: topic
first_seen: 2026-01-06
last_updated: 2026-07-13
status: complete
---

## The frame

The [[forge-analysis]] workstream studies a **perfect-information minimax solver**:
all four players see all hands, all play optimally. This is not Texas 42 as humans
play it. The findings describe the *theoretical structure of the game tree under
omniscient optimal play*, not how people actually navigate hidden information.

The report's opening section
(`forge/analysis/report/00_executive_summary.md` lines 4–18) makes this caveat load-
bearing — every numbered finding downstream inherits it.

## What oracle findings DO say

- How minimax-optimal play behaves under full information.
- Theoretical structure of the Texas 42 game tree (basin counts, level-set topology, symmetry orbits).
- Statistical patterns in perfect-play outcome distributions (E[V], σ(V), correlations).

## What oracle findings DO NOT say

- How humans navigate hidden information.
- Strategies under real-world uncertainty.
- Actual human gameplay dynamics or outcomes.

## Why this matters for backlinks

When a [[forge-analysis]] finding cites that "n_doubles is the strongest E[V]
predictor (r=0.40)," the rigorous reading is: *under optimal play with all hands
visible, having more doubles raises your team's eventual point differential.* The
naive reading — "play more doubles" — assumes that result transfers from oracle
play to human play. That transfer is **untested**.

Any wiki page that wants to use a [[forge-analysis]] result to inform a Burl/LEM/Gus
training decision should link this page and acknowledge the gap.

## Marginalized data is a partial bridge

`data/shards-marginalized/` partly addresses the gap: P0's hand is fixed across 3
opponent configs, so the spread of E[V] and σ(V) over opponent realizations measures
**outcome uncertainty from imperfect knowledge of opponent hands**. The full game
tree is still solved with omniscient play; only the prior over opponent hands is
non-degenerate. Findings derived from marginalized data are closer to "what a human
would face" than findings from the standard oracle, but still not equivalent.

## Practical recommendations are hypotheses

The report's "implications for bidding" sections are explicitly framed as
hypotheses extrapolated from oracle analysis, not validated human-play claims. Any
project that adopts one (e.g. [[burl]] training Burl to over-weight doubles in
bidding) should treat the prior as informative-but-untested and run its own ablation.
