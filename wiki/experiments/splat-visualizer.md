---
title: The splat visualizer
kind: experiment
first_seen: 2026-10-11
last_updated: 2026-10-13
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

## 2026-10-12: what must be true, zoom

**The panel.** Under the splats, [[scenario-player]]'s what-must-be-true
(`holding_predictors`) for any region of any move's splat: the made or set
side, a hump (valley split, each side ≥ 10% of the mass), a tapped bar, or a
dragged range. Holdings are ranked by |P(R ∧ h) − P(R)·P(h)| and read in the
mover's chair (partner, left = plays after it, right = before), with the real
holder marked: deal 3 play 9, *the 1-0 is set in 134 of 249 worlds where right
holds the 5-5; always (263 of 263) where partner does*; its +28..+42 hump
needs right's 5-5 (68 of 249, never otherwise). Clicking a holding conditions
the position on it (up to three, ANDed): each bar splits into the worlds where
it holds (bright, below) and the rest (faint, above), and a table gives every
move's mean and pmake/set if so / otherwise / whole cell, starring the best by
the plan's utility. The region, conditions and zoom live in the URL hash
(`#deal=3&play=9&sel=1-0:28:42&if=0:5-5&zoom=2`).

**Data: per world, not precomputed.** Per position `seats` (the three hidden
from the mover), `hidden` (their tiles), `holders` (one digit per world per
hidden tile: the index into `seats`); per move `outcomes` (one char per world,
`chr(48 + bidder points)`); `weights` only when unequal, with a document-level
`uniform` flag (true for this pour: every world weighs 1). The sample is
687 KB (234 KB gzipped), so the page computes everything from the worlds;
tests check the round trip and that the page's arithmetic equals
`holding_predictors` on made / set / margin ≤ 0 regions at every position.

**Zoom and touch.** ‹ › « » buttons step plays and planned plays; − / + zoom
the splats and ridges 1×–4× inside their own horizontal scrollers (kept in
step, one axis), never the page; pinch-zoom is not blocked. At 400 px the page
has no horizontal scroll. A custom two-finger pinch on the chart was left out:
it needs `touch-action: none`, which would block the browser's own pinch.

**Checked by hand** against `holding_predictors` and direct weighted means:
deal 1 play 10 (6-5, set side, 303 of 512: partner's 6-4 184/203, left's 6-4
60/179, left's 6-1 142/195 — the page's top three, in order); deal 3 play 9
(1-0 set side as above; under "right holds the 5-5", 249 worlds, the five
means +19.8 / +20.9 / +19.1 / +19.7 / +17.3 (6-2, 5-4, 3-3, 3-2, 1-0) and set 40.6–53.8%, otherwise
−18.7 … −23.3 and set 100%).

## 2026-10-13: Book against Zeb

A position may carry `book` notes (`scenario.metasplat sample`, the
[[metasplat-2026-10-13]] sample): a book rule that fires there, its allowed
tiles, and a contrast tile (the book's best where the plan leaves the book, the
best tile the book rules out where it keeps to it). The page then tags the
book's tiles and the contrast beside the dominoes, outlines the contrast's
splat in violet over the plan's on the same scale with its mean dashed, and
says under the header the shape of the difference (one of nine, fixed before
counting), Δp and Δμ with their noise bars, and the reason report's best
separating fact read on the pour's own worlds. The deal chips name each
sample's shape and jump to its annotated play.

```bash
python -m http.server 8000     # from the repo root
# http://localhost:8000/scenario/viz/?data=data/metasplat_sample.json
```

The sample (32 positions, 335 KB) is checked in; the test reads every note off
its own moves. Checked in a browser at 1400 px and 400 px (no horizontal
scroll).

## Links

[[scenario-player]] | [[candlewax]] | [[eq-browser-visualizers]] | [[zeb]] | [[metasplat-2026-10-13]]
