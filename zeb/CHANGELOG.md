# Changelog

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
