---
title: The fallible field — the book's play against opponents who lose track of the voids (#151, 2026-10-15)
kind: experiment
first_seen: 2026-10-15
last_updated: 2026-10-15
status: active
---

## Summary

Issue #151, the [[book-authority]]'s first measurement of assumption A3 ([[book-assumptions]]):
opponents are fallible. Several of the book's lessons are written against a defender who sometimes
loses track, among them the straight-off 84 that hopes for a misdiscard and the low-looking master
late in a hand (S-0088-a, S-0079-a, S-0169-b). Every arena opponent has been Zeb, whose worlds
respect every void ever shown. Here the field forgets, and the question is the rate at which the
book's play stops costing against it: the crossover p*.

*Results follow once every registered batch has run.*

## The model, and what it is not

`Config.void_miss` = p (spec `V=<p>`, merged at 005f06ad). At each decision, each hidden seat's
**whole** void record is forgotten with probability p, by a coin keyed on that decision's record.
The worlds are drawn from what is remembered, and the record the plan advances through stays true.

- **A lapse per decision, not a count lost for the hand.** A seat forgotten at one decision may be
  remembered at the next. The sticky form, closer to "lost track", is #168. Because a lapse forgets
  less than a lost count, the grid runs to p = 0.4, past the issue's 0.2.
- **Voids only.** Forgetting which tiles have been played has no consistent form in this planner
  and is not modeled. The book's fallible opponent misses more than voids, so this is A3 restored in
  one dimension.
- At p = 0 the field is deployed Zeb byte for byte. Python and Rust agree at every p.

## Design (pre-registered before any measured batch; see `scenario/fallible.py`)

- **Players.** Team B, the field: deployed Zeb 0.3.0 with `void_miss` = p ∈ {0, 0.1, 0.2, 0.4}.
  Team A: **B**, the book player (the head-to-head's block c, crisis-read in its stack), or **Z**,
  plain deployed Zeb. Both teams bid by the heuristic rule at caution 0.
- **Arms.** Marks: B0, B10, Z10, B20, Z20, B40, Z40 (Z0 is Zeb against itself, 0 exactly). The
  book's game (250 points): PB0, PB40, PZ40.
- **Primary.** G(p) = B_p − Z_p, the book player's gap to plain Zeb against the same field, and
  ΔG(p) = G(p) − G(0). **The book predicts ΔG rises with p.** The read is ΔG(0.4). p* is the least
  p whose G(p) interval covers or lies above 0, or "beyond the grid".
- **Secondaries.** Each arm's margin; Z_p, the field's own cost of forgetting; ΔG(0.1) and ΔG(0.2)
  as the trend; the book's game, PB40 − PZ40 against PB0, on games won.
- **Size.** 576 game pairs per marks arm (seeds 2901–2948 × 12), 192 per points arm (3001–3016 ×
  12); 384 batches through `scripts/cpu-slot`, two at a time. MDE on ΔG about 0.19–0.27
  marks/game at #163's SDs. Timing pilot seed 2890 excluded (17–21 s per 4-pair batch).
- **Estimator, batch-drop rule and fence.** As [[book-small-n-2026-10-14]]. The fallible field is
  an information-state planner over a looser fiber ([[strategy-fusion-fence]]).

## Links

[[book-authority]] · [[book-assumptions]] · [[overbidders-2026-10-14]] ·
[[deviant-field-2026-10-14]] · [[book-head-to-head-2026-10-13]] · [[strategy-fusion-fence]]
