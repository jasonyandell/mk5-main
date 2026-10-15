---
title: The field that loses the count — the book's play against opponents who forget which tiles have gone (#172, 2026-10-15)
kind: experiment
first_seen: 2026-10-15
last_updated: 2026-10-15
status: active
---

## Summary

Issue #172, the second measurement of assumption A3 ([[book-assumptions]]). #151 seated a field
that forgets voids. The forgetting cost that field 0.32 marks/game, and the book player and Zeb
collected it equally ([[fallible-field-2026-10-15]]). The book's fallible opponent loses something
else, the *count*, and that is what its ploys lean on: the straight-off 84 that hopes for a
misdiscard, and the low-looking master late in a hand (S-0088-a, S-0079-a, S-0169-b). Here the field
loses the count.

*Results follow once every registered batch has run.*

## Why a view, not a fiber

A lost count is not a fiber of the planner. Every world reads the one true record (`world[seat] &
~rec.played`), so a world that holds a played tile drops it. A separate record per world would undo
the shared record the [[strategy-fusion-fence]] rests on. The reasoning is in #172. So the
forgetting is put where a person's is, in the seat's view of the game (`arena/count_lapse.py`).

- **The view.** The true state with a played tile t and a live tile u exchanged everywhere: hands,
  history and points. A relabeled deal is a deal, so every hand keeps its size, the points total 42,
  and every void the view shows is real. In it, t is still out and u has gone. The inner policy
  (deployed Zeb) decides there, planning over worlds that all share that one view. The game, its
  record and every other seat stay true.
- **Sticky.** A tile another seat played in a completed trick is forgotten with probability p, by a
  coin keyed on the deal, the seat and the tile. Once lost, it stays lost for the hand. This is how
  people lose the count; #151's lapse re-remembered at every decision.
- **The stand-in must keep the game legal.** u is the first tile, in a keyed order, that leaves a
  legal history with the same trick leaders; if none does, t is remembered. In random play about
  60% of forgotten tiles find a stand-in: 70% of those that lost their trick, but only 30–45% of
  those that won it. A played master is therefore lost at about 0.4p. That is the price of a view
  that could have happened.

## Design (pre-registered before any measured batch; see `scenario/count_lapse.py`)

- **Players.** Team B: deployed Zeb 0.3.0 deciding at the view of a seat that loses the count with
  p ∈ {0, 0.05, 0.1, 0.2} (at p = 0, deployed Zeb byte for byte). By the sixth trick the grid has
  forgotten about 0.75, 1.5 and 3 of the 15 tiles a seat saw others play. Team A: **B**, the book
  player (block c), or **Z**, plain deployed Zeb. Both teams bid by the heuristic rule at caution 0.
- **Arms.** Marks: B0, B5, Z5, B10, Z10, B20, Z20. The book's game: PB0, PB20, PZ20.
- **Primary.** G(p) = B_p − Z_p and ΔG(p) = G(p) − G(0). **The book predicts ΔG rises with p.**
  The read is ΔG(0.2). p* is the least p whose G(p) interval covers or lies above 0, or "beyond
  the grid".
- **Secondaries.** Z_p, the field's own cost; ΔG(0.05) and ΔG(0.1) as the trend; the book's game,
  (PB20 − PZ20) − (PB0 − ½) and PB20 − PZ20 on games won; the field's tiles forgotten and stood in
  for per decision.
- **Size.** 576 game pairs per marks arm (seeds 3101–3148 × 12), 192 per points arm (3201–3216 ×
  12); 384 batches through `scripts/cpu-slot`, two at a time. MDE on ΔG about 0.16 at #151's SD.
  Timing pilot seed 3090 excluded (19–21 s per 4-pair batch; at p = 0.2, 1.7 tiles forgotten per
  decision, 1.0 of them stood in for).
- **Estimator, batch-drop rule and fence.** As [[book-small-n-2026-10-14]].

## Links

[[book-authority]] · [[book-assumptions]] · [[fallible-field-2026-10-15]] ·
[[book-head-to-head-2026-10-13]] · [[strategy-fusion-fence]]
