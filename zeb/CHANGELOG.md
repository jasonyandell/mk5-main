# Changelog

## 0.3.0 — 2026-10-14

Two new defaults, each measured on its own and then together:

- **Zeb reads the other seats' plays.** Under a pip trump, `play` and
  `analyze` draw their scenarios from a posterior that weighs each play by
  how likely the field net, softened, was to make it. A play of one of n
  legal tiles weighs ε/n + (1 − ε)/|A| if it is among A, the tiles the net
  scores within 2 points of its best, and ε/n otherwise, with ε = 0.28
  fitted to Zeb's own play. No scenario is ruled out. Alone, on fresh deals,
  it beat 0.2.0's plan by +0.37 marks a game, 95% interval [+0.24, +0.51].
  Doubles and no-trump are not read (the net does not play them), and a
  forecast is unchanged, because nothing has been played yet. Pass
  `reading: false` in a position request to plan over uniform scenarios as
  0.2.0 did.
- **Exact ties go to the margin.** Among the tiles tied exactly for the best
  chance to make the contract, Zeb plays the one with the best mean final
  margin under its plan, then the lowest id. It never gives up an estimated
  make. Alone it cost no marks (+0.01 [−0.04, +0.07] a game) and gained
  +4.55 points a hand. It applies under every declaration, so a forecast's
  chosen lead, and so its `mean` and `pmake`, can change.
- **Together**, in a pre-registered match at the default 160 worlds, 0.3.0's
  plan beat 0.2.0's by **+0.21 marks a game [+0.08, +0.34]** over 384
  paired games on duplicate deals; the A/A control (0.2.0 against itself at
  another seed) read +0.06 [−0.07, +0.18].
- **Cost.** Reading costs nothing at the opening lead and 1.3 to 2.5 times
  the unread median at tricks 1 to 3; from trick 4 on it is a few
  milliseconds either way. At the default 160 worlds a trick-2 or trick-3
  decision is about 50 ms median on an M5 Max (README, *Reading the plays*).
- Answers for a given seed change. Still bit-identical to the Python referee
  (`scenario/planner.py`), now on 38 positions, three of them with reading
  off.

## 0.2.0 — 2026-10-12

- The cell floor now applies only when the scenarios are a sample. Late in
  a hand, once every deal consistent with the play so far fits in `worlds`
  (with 512 from the third play of the sixth trick, sooner when a seat has
  shown a void), Zeb plans over all of them, so each of its cells is exact
  and the floor no longer hands the move down. Late-hand answers (`play`,
  `analyze`) can change; forecasts and early-hand answers do not. Measured
  in the arena it is the same player within noise; it is simply correct.
- Answers for a given seed change, because the scenarios no longer depend
  on the rung of the ladder that draws them; the distribution of answers
  does not.
- Still bit-identical to the Python referee (`scenario/planner.py`).

## 0.1.0 — 2026-10-11

The first release: Zeb, the scenario player, in WebAssembly.

- `forecast` gives one plan per declaration from the bidder's chair: the
  bidding side's expected margin and its make probability at every target
  from 30 to 42.
- `chooseBid` turns a forecast into a bid with the book's rule (the highest
  target some declaration makes with probability ≥ 0.8). It is a plain
  function.
- `play` returns a seat's move. Its make probability comes with every legal
  tile.
- `analyze` returns the same move with every legal tile's tail, for hints.
- `version` and `netId` are the receipts. Answers are deterministic in the
  request and its seed, and bit-identical to the Python referee
  (`scenario/planner.py`).
- Pip trumps are played against the embedded field net
  `margin_d250k_h128x64` (sha256 `6fa3723d…0275d5`). Doubles-trump and
  no-trump are played against a one-rung Walt.
