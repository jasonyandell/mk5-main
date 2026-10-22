---
title: The book's own overbidders — its bidding against the two kinds it names, at power (#163, 2026-10-14)
kind: experiment
first_seen: 2026-10-14
last_updated: 2026-10-22
status: complete
---

## Summary

Issue #163, the [[book-authority]]'s next measurement of assumption A6 ([[book-assumptions]]).
[[deviant-field-2026-10-14]] seated the book's bidding against a field that overbids by a fixed
step, and its cost shrank as a null at MDE 0.52. That field overbid by rote. The book names two
overbidders of its own (catalog rows, in our words): one raises a weak hand over an opponent's bid
because they doubt they could set it (S-0155-a, the set-bidding fallacy); the other habitually bids
35 holding offs in the ten-count suits (S-0070-a). The book's answer to both is restraint: pass
and set him (S-0080-c).

## Results (complete, 2026-10-15; 416 of 416 registered batches ran, none dropped)

**One line.** Against both of the overbidders the book names, its bidding loses less ground than
against the style-free field, as the book predicts: X_S = +0.15 [+0.02, +0.28] and X_T = +0.23
[+0.03, +0.41] marks/game. It still trails the control in every field: by 0.51 [0.33, 0.69]
against the style-free field, 0.36 [0.16, 0.57] against the set-bidder and 0.29 [0.12, 0.45]
against the ten-count 35. The ten-count field sets itself, as the book says it will. The control
beats it by 1.30 [1.13, 1.45] marks/game. The set-bidding field costs itself nothing we can
measure.

Data: [texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10)
`book/overbidders/` (per batch `meta.json` and `games.jsonl`, `summary.json`, the queue log, `CARD.md`).

### The arms (team A's margin against the field; 1,152 games per marks arm, 384 per points arm)

| arm | team A | field | marks/game [95%] (points arms: share of games won) | A's contracts / hand | made A / B | mean bid A / B |
|---|---|---|---|---|---|---|
| HS | heuristic | set-bid | −0.09 [−0.18, +0.01] | 39% | 58.4% / 56.1% | 30.7 / 30.8 |
| RS | `risk-budget` | set-bid | −0.45 [−0.63, −0.27] | 47% | 52.8% / 56.9% | 40.9 / 31.5 |
| BS | the book player | set-bid | −1.25 [−1.38, −1.12] | 39% | 53.3% / 61.5% | 30.7 / 30.8 |
| HT | heuristic | ten-count 35 | +1.30 [+1.13, +1.45] | 17% | 47.2% / 42.3% | 35.9 / 35.0 |
| RT | `risk-budget` | ten-count 35 | +1.01 [+0.82, +1.20] | 24% | 47.5% / 43.1% | 47.6 / 35.0 |
| BT | the book player | ten-count 35 | +0.28 [+0.10, +0.45] | 17% | 42.2% / 46.9% | 35.9 / 35.0 |
| RH | `risk-budget` | style-free | −0.51 [−0.69, −0.33] | 56% | 53.5% / 60.4% | 39.6 / 30.9 |
| PHS, PRS | heuristic, `risk-budget`, to 250 | set-bid | 52.1% [49.0, 55.2], 45.1% [40.4, 49.7] | 39%, 46% | | |
| PHT, PRT | heuristic, `risk-budget`, to 250 | ten-count 35 | 77.9% [74.0, 81.8], 70.3% [65.9, 74.5] | 18%, 25% | | |
| PRH | `risk-budget`, to 250 | style-free | 38.8% [34.1, 43.5] | 56% | | |

A mean bid above 42 includes bids in marks (84 = two marks): when `risk-budget` outbids a 35 it
often goes to the marks.

### The registered contrasts (paired by game pair; MDE at 80% power)

| contrast | marks/game [95%] | pairs | SD/pair | MDE |
|---|---|---|---|---|
| **primary X_S = (RS − HS) − RH** | **+0.15 [+0.02, +0.28]** | 576 | 1.64 | 0.19 |
| **primary X_T = (RT − HT) − RH** | **+0.23 [+0.03, +0.41]** | 576 | 2.32 | 0.27 |
| C_S = RS − HS | −0.36 [−0.57, −0.16] | 576 | 2.50 | 0.29 |
| C_T = RT − HT | −0.29 [−0.45, −0.12] | 576 | 2.00 | 0.23 |
| RH | −0.51 [−0.69, −0.33] | 576 | 2.19 | 0.26 |
| the field: HS against 0 | −0.09 [−0.18, +0.01] | 576 | 1.20 | 0.14 |
| the field: HT against 0 | +1.30 [+1.13, +1.45] | 576 | 1.95 | 0.23 |
| the book player BS − HS | −1.16 [−1.30, −1.03] | 576 | 1.66 | 0.19 |
| the book player BT − HT | −1.02 [−1.17, −0.87] | 576 | 1.79 | 0.21 |

The SDs came in well under the 3.3 sized for, so the registered MDE of 0.39 became 0.19 and 0.27,
and both primaries clear 0.

S-0114-b, the book's game (192 game pairs per arm):

| field | PR − PH, share of games won | PR − PH, settled points/game | exploratory: X in the book's game, (PR − PH) − (PRH − 50%) |
|---|---|---|---|
| set-bid | −7.0 pts [−12.5, −1.6] | −28.3 [−41.9, −15.2] | +4.2 pts [−0.5, +8.9] |
| ten-count 35 | −7.6 pts [−12.0, −3.1] | −24.5 [−35.6, −13.6] | +3.6 pts [−1.8, +9.4] |

The last column was not registered. It applies the primary's form to games won: the same sign as
in marks, a null at an MDE of about 7–8 points.

### Reading, under [[book-authority]]

- **Assumptions held and violated.** A6 is restored for the two overbidders the book names, in
  our transcription of them (`populations/set-bid.silas`, `populations/ten-count-35.silas`, written
  from S-0155-a and S-0070-a and held to their readings in `silas/tests/bidding.rs`). Each field
  is deterministic and keeps one style for the whole match. A1 is violated. The book's answer,
  "pass and set them" (S-0080-c), assumes the reader knows who overbids, but `risk-budget` is the
  same rule in every field and reads no style. A2, A3 and A4 are violated as in every arena block.
  The marks arms violate A5, and the points arms restore it.
- **What our model can see.** The book's prediction holds in both fields. The mechanism is the
  book's restraint, applied by the field. The ten-count field takes 76–83% of the contracts, a
  mean bid of 35, and makes 42–43% of them. Against it `risk-budget` bids in 24% of hands, down
  from 56% against the style-free field. Its own contracts are where its cost lies (made 53.5% at
  a mean of 39.6 against the style-free field, where the field makes 60.4%). Fewer contracts leave
  less to lose, and the sets it collects are the control's too. Against the set-bidder the same
  holds in a milder form: 47% of hands, and the field makes 56.9%.
- **What the field does to itself.** The ten-count 35 sets itself: the control wins +1.30 [1.13,
  1.45] marks/game against it, and 77.9% [74.0, 81.8] of games to 250. The set-bidder takes 61% of
  the contracts at a mean bid of 30.8 and makes 56.1% of them. That costs it nothing measurable
  (the control is −0.09 [−0.18, +0.01] against it; to 250, 52.1% [49.0, 55.2]). As written, its
  raises are cheap: they rarely climb past 31. A set-bidder that raises further is the open half
  of S-0155-a.
- **The book's own game.** To 250, `risk-budget` trails the control by 7.0 and 7.6 points of
  share against the two fields. Against the style-free field it wins 38.8% [34.1, 43.5] of games.
  #152 put the same pairing, PR0 on seeds 1951–1958 (96 pairs), at 50.5% [43.8, 57.3]. The two
  arms have the same design and different seeds, and they are about 2.8 standard errors apart.
  Re-running PR0 seed 1951 at this study's commit (363b0c41) reproduced #152's games byte for byte,
  so the code did not move. Pooled over 288 pairs the arm is about 42.7%. #152's "level in the
  book's game at every k" is best read as a high draw at k = 0. In the book's game the cost of
  `risk-budget` does not vanish at this size.
- **The book player** costs about a mark a game against both fields (−1.16, −1.02), as against
  every field before ([[deviant-field-2026-10-14]]): the gap is in its play, not its bidding.
- **The measurement that restores more of A6.** A `risk-budget` that reads the field's style
  (A1 with A4: a regular who learns who overbids and passes to set them; #116, #130 slice 5), and a
  set-bidder whose raises climb.

### Receipts

Pre-registered at 363b0c41 (`scenario/overbidders.py`). 416 batches through `scripts/cpu-slot`, two
at a time, about 20–45 s a batch at load average 10–30. The queue was parked three times at the
Mayor's request so timed Demon confirmations could have a quiet machine. Each park came between
batches, with none in flight, and the queue resumed from where it stopped; every batch directory
holds its 24 games (points: 24), none dropped. Timing pilot seed 2590 excluded (the ten-count
field took 87% of contracts at 35 and made 39–43%). The PR0 re-run is in this page's reading; its
games are the published #152 batch.

## Design (pre-registered before any measured batch; see `scenario/overbidders.py`)

- **Players.** Every seat plays Zeb 0.3.0. The field (team B) bids by `populations/set-bid.silas`
  (**S**: bid-only-enough, except over an opponent's standing bid it raises up to six points past
  the hand's worth), `populations/ten-count-35.silas` (**T**: 35 on a suit of three with an off
  that exposes a ten-count tile), or the style-free heuristic (**H**). Both populations are held to
  their readings in `silas/tests/bidding.rs`. Team A bids by the heuristic (**H**, the control),
  the book's `risk-budget` (**R**), or plays as the book player (**B**, block c).
- **Arms.** Marks: HS, RS, BS, HT, RT, BT, RH. The book's game: PHS, PRS, PHT, PRT, PRH.
- **Primary.** X_f = (R_f − H_f) − RH for f ∈ {S, T}: the book's bidding against its overbidder,
  relative to the style-free field. **The book predicts X_f > 0.**
- **Secondaries.** C_f = R_f − H_f; the field against the control, H_f (the book predicts
  overbidders set themselves); the book player B_f − H_f; the book's game; mechanism.
- **Size.** 576 game pairs per marks arm (seeds 2601–2648 × 12), 192 per points arm (2701–2716 ×
  12): MDE 0.39 at SD 3.3. 416 batches through `scripts/cpu-slot`, two at a time.
- **Estimator, batch-drop rule and fence.** As [[book-small-n-2026-10-14]].

## Links

[[book-authority]] · [[book-assumptions]] · [[deviant-field-2026-10-14]] ·
[[book-head-to-head-2026-10-13]] · [[silas]] · [[strategy-fusion-fence]]

## Stack version (#245)

These numbers were played on the book stack as it stood before #245 (2026-10-22), which
changed what the bidder counts as an off's guard (`bare-off`, `bidders-plan`; the double or
the top tile left in the suit, W42 p. 77). They reproduce at commit `64ecc6cf`. Re-runs on
the repaired stack are tracked on #245.
