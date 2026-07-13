---
title: The analysis epic — forge/analysis's Jan 6-8 launch
kind: trail
first_seen: 2026-01-06
last_updated: 2026-01-08
status: complete
---

## What it is

The three-day launch of `forge/analysis/` (modules 01-26) that became [[forge-analysis]], the
workstream that today reads the oracle's game-tree output with classical statistics, ML
explainability, and dimensionality reduction. This page covers the launch itself — the
volume, the negative results, and the self-correction — as the closing act of
[[breakthrough-and-oracle]]. See [[forge-analysis]] for the workstream's current, mature
state (21 notebook themes, headline findings table, tooling).

## The volume

Began 2026-01-06 (`5ffdf58`, count-domino basin analysis, R²=0.76 for a linear
capture-probability model with learned coefficients matching true point values) and exploded
on Jan 7 into **131 epic-scoped commits** touching `forge/analysis/` — not the 286 a naive
`git log` count for that day would suggest, since the large majority of that day's total repo
commits are `bd sync:` auto-commits, not epic work. A SQLite/DuckDB substrate (SeedDB)
replaced ad-hoc per-script analysis, and the module count reached 26 by the close.

## The findings

- **Hidden information matters most mid-hand.** Mean V-spread across opponent hand
  configurations = 34.8 points; only 11% of hands are "stable" (spread < 10). Best-move
  consistency across opponent configs: 100% in the endgame, dropping through the mid-game —
  the aggregate figure (**54.5% overall best-move consistency**) reproduces cleanly from the
  current on-disk tables; a finer endgame/mid-game/early-game tri-split reported in-session
  does not reproduce exactly from the current tables and should be treated as a rounded,
  session-reported figure rather than a precise re-derivable number.
- **Six traditional 42 folk-heuristics, tested against the oracle, all refuted or not
  confirmed** (module 26a-26f, six commits, 2026-01-07): threshold cliffs, active voiding,
  naked lows, coverage protection, directional voids, coverage-vs-trump — each commit states
  its own negative result, one flagged "SURPRISING NEGATIVE RESULT."
- **Three-axis feature decomposition failed**: R²=22.8% at best (MLP-combined), both its Void
  Sufficiency and Linear Decomposition conjectures marked REFUTED (bead t42-bp9q, 2025-12-28,
  strictly upstream of the epic but part of the same feature-basis-search thread — see
  [[suit-algebra]]).
- **Count-centric abstraction is a dead end**, named plainly: "count centric was just not a
  good abstraction. dead end. didn't correlate with good play at all." (conv 33bdd626,
  2026-01-08).

## The self-caught overclaim

The moment that most embodies the project's ethos: a "19% skill, 81% luck" framing of a
variance decomposition was struck **the same epic, same day** it was noticed to be false
(`d29a317`, 2026-01-07): "both components are determined by the random deal — the oracle
plays perfectly with no human decisions measured." Negative results were recorded, not
buried — the project polices its own claims.

## The particle-filter side-quest

Mid-epic (2026-01-07, conv c1917412), Jason connected an unrelated side project's technique
to the hidden-information problem: "when messing around with go fish you used particles.
would that help us with our hidden information struggles." The question triggered a single
conversation but never became a named artifact — no bead, script, or class called "particle
filter" exists from this window. What it redirected into instead: module 25l
(`b698aa4`, same day), a 28×28 domino likelihood-ratio matrix extracted from 1M+ oracle
observations ("specific pairs have LR up to 8.7"), explicitly logged as scaffolding for a
future opponent model, not the model itself. Jason refused the shortcut on principle:
"tempting but those are heuristics. always the siren song of heuristics in this game but it
defies them. but we have learned something."

## The close: the epistemic audit (Jan 8, 2026)

The epic closed with a mechanical ~24-commit pass across every numbered report, adding
epistemic-status headers that distinguish "oracle (perfect-info) findings" from "human
gameplay advice," and renaming "Practical Implications" to "Implications for Human Play
(Hypotheses)" — capped by `3a6d2ec` rewriting the executive summary "per epic close
condition." The same day, a self-aware flag about the analyst's own state: "uh oh I'm dumb
this morning. maybe I slept badly... it's a loss of like 95%-92% but I can feel it." (conv
33bdd626) — worth preserving because it names a real methodological hazard for anything else
recorded that session.

A separate, minor target-drift note from the same window: bead `gh6r` asked to update
`forge/analysis/CLAUDE.md`; the findings instead landed in
`.claude/skills/texas-42-analytics/SKILL.md`. `CLAUDE.md` was never touched.

## Honest terminal status

**BUILT, and still mature today** — see [[forge-analysis]] for the workstream's current
21-notebook-theme shape and headline-findings table. The Jan 6-8 launch is the epic's
founding week, not its whole life; the workstream continued well past this era.

## What it ruled in/out about the wall

The analysis epic characterized the oracle beautifully and learned almost nothing about
*consuming* it. 131 commits, and the durable outputs were: folk wisdom is wrong, count-centric
is dead, the three-axis decomposition failed, hidden info matters most mid-hand. All true, all
worth knowing — but every one is a statement about what doesn't work or about the shape of
the problem, not a mechanism for turning a computed value into a plan. It confirmed the same
diagnosis [[strategy-fusion]] made from the theory side (naive perfect-information averaging
overstates achievable value) from the empirical side (best-move consistency collapses outside
the endgame) — independent support for the same wall, arrived at by a different route. See
[[breakthrough-and-oracle]] for the full era argument.

## Links

[[forge-analysis]] · [[the-oracle]] · [[strategy-fusion]] · [[breakthrough-and-oracle]] ·
[[oracle-vs-human-play]] · [[era2-breakthrough-oracle|conversation digest]]
