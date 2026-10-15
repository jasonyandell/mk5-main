---
title: The fallible field — the book's play against opponents who lose track of the voids (#151, 2026-10-15)
kind: experiment
first_seen: 2026-10-15
last_updated: 2026-10-15
status: complete
---

## Summary

Issue #151, the [[book-authority]]'s first measurement of assumption A3 ([[book-assumptions]]):
opponents are fallible. Several of the book's lessons are written against a defender who sometimes
loses track, among them the straight-off 84 that hopes for a misdiscard and the low-looking master
late in a hand (S-0088-a, S-0079-a, S-0169-b). Every arena opponent has been Zeb, whose worlds
respect every void ever shown. Here the field forgets, and the question is the rate at which the
book's play stops costing against it: the crossover p*.

## Results (complete, 2026-10-15; 384 of 384 registered batches ran, none dropped)

**One line.** Against a field that forgets voids, the book player gains exactly as much as plain
Zeb does, so its gap to Zeb does not move. ΔG(0.4) = −0.03 [−0.14, +0.08] marks/game, a null at an
MDE of 0.16. The gap G(p) stays near a mark a game at every p in the grid, so p* is **beyond the
grid**. The forgetting itself is real and costly to the field: deployed Zeb takes +0.32 [0.23,
0.41] marks/game from it at p = 0.4, and the book player takes about as much (B0 −1.04 to B40
−0.75).

Data: [texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10)
`book/fallible/` (per batch `meta.json` and `games.jsonl`, `summary.json`, the queue log, `CARD.md`).

### The arms (team A's margin against the field; 1,152 games per marks arm, 384 per points arm)

| p | B_p, the book player | Z_p, plain Zeb (the field's cost of forgetting) | G(p) = B_p − Z_p |
|---|---|---|---|
| 0 | −1.04 [−1.16, −0.91] | 0 exactly (Zeb against itself; not run) | −1.04 [−1.16, −0.91] |
| 0.1 | −0.91 [−1.03, −0.78] | +0.06 [+0.01, +0.11] | −0.97 [−1.10, −0.83] |
| 0.2 | −0.83 [−0.95, −0.70] | +0.12 [+0.05, +0.19] | −0.94 [−1.08, −0.80] |
| 0.4 | −0.75 [−0.87, −0.62] | +0.32 [+0.23, +0.41] | −1.07 [−1.21, −0.92] |

The book's game, share of games won against the field: PB0 39.1% [35.4, 42.7], PB40 43.2% [39.8,
46.6], PZ40 52.9% [50.0, 55.7].

### The registered contrasts (paired by game pair; MDE at 80% power)

| contrast | marks/game [95%] | pairs | SD/pair | MDE |
|---|---|---|---|---|
| **primary ΔG(0.4) = G(0.4) − G(0)** | **−0.03 [−0.14, +0.08]** | 576 | 1.37 | 0.16 |
| ΔG(0.1) | +0.07 [−0.00, +0.14] | 576 | 0.86 | 0.10 |
| ΔG(0.2) | +0.09 [−0.00, +0.19] | 576 | 1.13 | 0.13 |
| the field's cost: Z10 | +0.06 [+0.01, +0.11] | 576 | 0.65 | 0.08 |
| the field's cost: Z20 | +0.12 [+0.05, +0.19] | 576 | 0.84 | 0.10 |
| the field's cost: Z40 | +0.32 [+0.23, +0.41] | 576 | 1.04 | 0.12 |
| to 250: PB40 − PZ40, share of games won | −9.6 pts [−14.1, −5.2] | 192 | 0.31 | 6.2 pts |
| to 250: (PB40 − PZ40) − (PB0 − 50%) | +1.3 pts [−2.6, +5.2] | 192 | 0.27 | 5.5 pts |

The last contrast was registered as (PB40 − PZ40) − PB0 and computed that way, which leaves out
PB0's 50% chance baseline and reads −48.7 points. Adding the half back is a constant shift that
moves the mean and both ends alike; the summary now carries the corrected line. The design did not
change.

### Reading, under [[book-authority]]

- **Assumptions held and violated.** A3 is restored in one dimension. The field forgets a seat's
  whole void record, with probability p, afresh at each decision. It never forgets which tiles have
  been played, never misdiscards by inattention, and a seat it forgets at one trick it may remember
  at the next. The book's fallible opponent loses track of the count, and that is the form its ploys
  lean on: the straight-off 84 that hopes for a misdiscard and the low-looking master
  (S-0088-a, S-0079-a, S-0169-b). That form is not in this field. A1, A2, A4 and A6 are violated as
  in every arena block. The marks arms violate A5, and the points arms restore it.
- **What our model can see.** Forgetting voids costs the field, and the cost grows with p: 0.06,
  0.12 and 0.32 marks/game. Both team A players collect it, and they collect it equally. The planner
  takes it as readily as the book's rules do, so it does not show up as a gain for the book. ΔG(0.1)
  and ΔG(0.2) lean the book's way (+0.07, +0.09, each touching 0), and ΔG(0.4) does not. The trend is
  not monotone, and none of it clears its MDE.
- **The book's game.** To 250 the book player trails plain Zeb by 9.6 points of share against the
  forgetting field, the same distance it trails at p = 0 (+1.3 [−2.6, +5.2] on the corrected
  contrast).
- **The measurement that restores more of A3.** A field that loses the *count*, forgetting played
  tiles, so a ploy that hopes for a misdiscard has something to hope for. That needs a planner form
  this tracker does not have (#172). Also the sticky form of this one, where a seat forgotten stays
  forgotten for the hand (#168). Until one of them is measured, the book's ploys are priced only
  against opponents who never forget the count, which is their worst case.

### Receipts

Pre-registered at 4e53c9d0 (`scenario/fallible.py`). 384 batches through `scripts/cpu-slot`, two at a
time, 30–70 s a batch when a slot was free. While other desks held the machine, some batches waited
up to 17 minutes for a slot, and every one completed. Every batch directory holds its 24 games; none
dropped. Timing pilot seed 2890 excluded. Run 11:43–14:37 on 2026-10-15.

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
