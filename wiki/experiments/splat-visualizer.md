---
title: The splat visualizer
kind: experiment
first_seen: 2026-10-11
last_updated: 2026-10-11
status: complete
---

## Summary

A browser page that draws a pour's splats: at every planned play of a deal,
each legal move's distribution of final bidder-team margins over the
mover's cell, with the plan's move marked, pmake at the contract as a
shaded tail, the means alone in a strip above (what a scalar sees), and the
deal's real ending as one line through everything. Below, the chosen move's
splat at every play of the hand as a ridgeline, so the real world can be
watched moving through the forecasts. It replaces, for this era, the
oracle-era [[eq-browser-visualizers]] (`eq_pdf_discs.html`): those PDFs were
clairvoyant per-world values averaged; these are [[scenario-player]]'s
honest plan over a cell. See [[candlewax]] for why the shape, not the mean,
is the object.

## How to run

```bash
# a pour (seconds); or use the checked-in sample and skip both steps
python -u -m scenario.atlas pour --deals 5 --seed 424242 --out scratch/viz/pour_viz_5 \
  --spec k0,n512,F=scratch/field_net/v3/margin_d250k_h128x64.npz
python -u -m scenario.viz.export scratch/viz/pour_viz_5 --out scenario/viz/data/pour_viz_5.json

python -m http.server 8000     # from the repo root
# http://localhost:8000/scenario/viz/                       the sample
# http://localhost:8000/scenario/viz/?data=/scratch/x.json  any other export
```

Keys: ←/→ a play, `[`/`]` the previous/next planned play, ↑/↓ or 1–5 a
deal, Home/End; click a ridge to jump. The URL hash (`#deal=1&play=10`)
holds the place. One HTML file, no dependencies, no build; the exporter is
`scenario/viz/export.py`, tested in `scenario/tests/test_viz_export.py`.

## What it shows, and the choices

- **Margins are even.** All 42 points are banked, so margin m means the
  bidders banked (42+m)/2: 43 reachable values, not 85. The exporter bins by
  bidder points (and refuses an odd margin); each bar is one exact score.
- **One axis, two colours.** The x axis is always the bidder team's margin,
  so the real-ending line stands still while you step. Bars are coloured by
  the side the ending favours (made / set at margin ≥ 2·30 − 42 = 18); the
  mover's side is bright, so a defender's splat glows on the left. The
  mover's number is its own tail (pmake for the bidding side, set for the
  defense).
- **The page sees all four hands; the mover did not.** The other seats are
  drawn behind glass, each hidden tile with a bar: the share of the mover's
  worlds that put it where it really is.
- **2D, not 3D.** The old discs were striking but needed orbiting and hid
  each other; the ridgeline keeps the stacked-along-the-hand look with the
  same x axis as the splats above, deterministic and readable at a glance.
- **Data:** the sample (5 deals, 88 planned positions, 52 forced) is 79 KB.
  The pour plans under `make` (q is pmake); past a decided contract the
  margins are the tape's, not the plan's (issue #107). The pour's seat 0
  always bids 30 and declares its best trump; the page says so.

## Verified

The exporter recomputes each move's q from its splat (`in_plan_units`) and
replays every deal to its recorded ending, refusing otherwise. Read off the
page and checked against `atlas.makes` / `in_plan_units` / the weighted mean
by hand: deal 1 play 10 (seat 1 defending; 6-5 plan set 59% μ +13.9, 6-3
58% +13.9, 6-2 55% +14.9: three humps at −6, +6, +16 and the deal really
ended +38), deal 3 play 9 (five moves, means spanning 2.9 points, set
71–78%), deal 1 play 2.

## Links

[[scenario-player]] | [[candlewax]] | [[eq-browser-visualizers]] | [[zeb]]
