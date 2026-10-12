# Changelog

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
