---
title: The field that loses the count — the book's play against opponents who forget which tiles have gone (#172, 2026-10-15)
kind: experiment
first_seen: 2026-10-15
last_updated: 2026-10-22
status: complete
---

## Summary

Issue #172, the second measurement of assumption A3 ([[book-assumptions]]). #151 seated a field
that forgets voids. The forgetting cost that field 0.32 marks/game, and the book player and Zeb
collected it equally ([[fallible-field-2026-10-15]]). The book's fallible opponent loses something
else, the *count*, and that is what its ploys lean on: the straight-off 84 that hopes for a
misdiscard, and the low-looking master late in a hand (S-0088-a, S-0079-a, S-0169-b). Here the field
loses the count.

## Results (complete, 2026-10-16; 384 of 384 registered batches ran, none dropped)

**One line.** Against a field that loses the count, the book player gains as much as plain Zeb
does, and no more. ΔG(0.2) = +0.07 [−0.08, +0.22] marks/game, a null at an MDE of 0.21, so its gap
to Zeb does not close and p* is **beyond the grid**. Losing the count costs the field far more than
forgetting voids did: plain Zeb takes +0.71 [0.60, 0.82] marks/game from it at p = 0.2. #151's
void lapse cost +0.32 at p = 0.4. The book player takes about as much (B0 −1.07 to B20 −0.29).

Data: [texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10)
`book/count-lapse/` (per batch `meta.json` and `games.jsonl`, `summary.json`, the queue log, `CARD.md`).

### The arms (team A's margin against the field; 1,152 games per marks arm, 384 per points arm)

| p | tiles forgotten / stood in for, per field decision | B_p, the book player | Z_p, plain Zeb (the field's cost) | G(p) = B_p − Z_p |
|---|---|---|---|---|
| 0 | 0 / 0 | −1.07 [−1.19, −0.94] | 0 exactly (Zeb against itself; not run) | −1.07 [−1.19, −0.94] |
| 0.05 | 0.45 / 0.28 | −0.83 [−0.96, −0.71] | +0.15 [+0.07, +0.22] | −0.98 [−1.12, −0.84] |
| 0.1 | 0.90 / 0.54 | −0.64 [−0.77, −0.52] | +0.39 [+0.31, +0.48] | −1.04 [−1.18, −0.89] |
| 0.2 | 1.81 / 1.02 | −0.29 [−0.42, −0.17] | +0.71 [+0.60, +0.82] | −1.00 [−1.16, −0.84] |

The book's game, share of games won against the field: PB0 38.0% [34.1, 41.9], PB20 45.1% [41.1,
49.0], PZ20 57.6% [54.4, 60.7].

**Estimator (#191, 2026-10-17).** The Book arena's estimator is calibrated: twenty fixed-pair A/A replicates scatter as their intervals say (R_F 1.06), and two seeds' policies do not differ measurably ([[estimator-calibration-2026-10-16]]). The intervals here stand as reported.

### The registered contrasts (paired by game pair; MDE at 80% power)

| contrast | marks/game [95%] | pairs | SD/pair | MDE |
|---|---|---|---|---|
| **primary ΔG(0.2) = G(0.2) − G(0)** | **+0.07 [−0.08, +0.22]** | 576 | 1.82 | 0.21 |
| ΔG(0.05) | +0.09 [−0.02, +0.19] | 576 | 1.27 | 0.15 |
| ΔG(0.1) | +0.03 [−0.09, +0.16] | 576 | 1.53 | 0.18 |
| the field's cost: Z5 | +0.15 [+0.07, +0.22] | 576 | 0.93 | 0.11 |
| the field's cost: Z10 | +0.39 [+0.31, +0.48] | 576 | 1.05 | 0.12 |
| the field's cost: Z20 | +0.71 [+0.60, +0.82] | 576 | 1.30 | 0.15 |
| to 250: PB20 − PZ20, share of games won | −12.5 pts [−17.4, −7.6] | 192 | 0.35 | 7.2 pts |
| to 250: (PB20 − PZ20) − (PB0 − 50%) | −0.5 pts [−6.0, +4.9] | 192 | 0.38 | 7.7 pts |

The SD per pair came out above #151's (1.82 against 1.37 on the primary), so the registered MDE of
about 0.16 became 0.21. The 250 contrast carries PB0's 50% baseline as registered here.

### Reading, under [[book-authority]]

- **Assumptions held and violated.** A3 is restored in a second dimension: the field loses the
  count, sticky for the hand, at a rate of 0.45–1.8 tiles per decision. The stand-in rule keeps the
  view a legal history, so a forgotten trick-winner is lost at about 0.4p and a played master less
  often than a played loser. The field still never misdiscards by inattention, never forgets its
  own hand, and loses no voids (#168). The book's straight-off 84 hopes for a misdiscard, and a
  misdiscard is a play this field makes only when its forgotten count makes the discard look right.
  A1, A2, A4 and A6 are violated as in every arena block. The marks arms violate A5, and the points
  arms restore it.
- **What our model can see.** A field that loses the count bleeds marks steeply: 0.15, 0.39 and
  0.71 marks/game at p = 0.05, 0.1 and 0.2. That is more than twice what #151's void lapse cost at
  p = 0.4 (0.32), though the two p's count different things. Both team A players collect it, and they collect it equally (B gains +0.78, Z
  +0.71). The planner, which never models the field's lapse, takes the gifts as readily as the
  book's rules do. ΔG leans the book's way at every p (+0.09, +0.03, +0.07), but none of it clears
  its MDE and the trend is not monotone.
- **The book's game.** To 250 the book player trails plain Zeb by 12.5 points of share against
  this field. That is the distance it trails at p = 0 (−0.5 [−6.0, +4.9] on the contrast).
- **The measurement that restores more of A3.** The book's ploys are aimed: a lure is played *at*
  a seat that might lose track, and it is worth something only when the player chooses it for that
  reason. Neither team A player here models the field's lapse. So the next measurement is a
  player that does, with the lapse as a parameter of the model of the other seat ([[lem-the-teacher]]'s
  population, #182). Also the sticky void lapse (#168), and the two lapses together.

### Receipts

Pre-registered at 02f3b304 (`scenario/count_lapse.py`, `arena/count_lapse.py`). 384 batches
through `scripts/cpu-slot`, two at a time, 30–70 s a batch when a slot was free. While other desks
held the machine, some batches waited up to 58 minutes for a slot, and every one completed. Every batch
directory holds its 24 games; none dropped. Timing pilot seed 3090 excluded. Run 2026-10-15 14:53
to 2026-10-16 02:33.

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

## Stack version (#245)

These numbers were played on the book stack as it stood before #245 (2026-10-22), which
changed what the bidder counts as an off's guard (`bare-off`, `bidders-plan`; the double or
the top tile left in the suit, W42 p. 77). They reproduce at commit `64ecc6cf`. Re-runs on
the repaired stack are tracked on #245.
