# Changelog

## 0.4.1 - 2026-10-15

**The overcall, opt-in, tuned for opponents who fight back.** `chooseBid`
takes `overcall: true` (with `match`; it reads `match.holder` and
`match.passesAfter`). Two things change, both only when an opponent holds the
high bid:

- **The forecast is calibrated** (`cal`): each make probability goes through
  logit p~ = -0.0162 + 0.9237 logit p, so the hand-only rule and the mark
  option read an honest number over an opponent's bid.
- **A pass becomes an overcall** (`over`): when the hand-only rule would pass,
  `chooseBid` bids the one-mark call, `need` to 42, whose chance of our side
  taking the hand's mark is the largest (the lowest bid on a tie), if that
  beats what passing leaves. Every call from pass to 42 moves one mark, so
  this does not read the score: the larger chance is the better call at every
  score. The returned `pmake` is the calibrated one.

`overcall: true` also prices the mark bid and everything else with the
**contested field's constants**: the hand law, the auction odds and the
calibration refitted on a table that overcalls back (#170). `matchPrice`
takes `contested: true` for those odds. Without `overcall`, 0.4.0's frozen
constants and 0.4.0's answers: byte-identical.

- **Tuned for a table that contests openings, and it costs against one that
  does not.** The constants were fitted on bidders who overcall back (82.8%
  of opponent-held turns were overcalls). Against a table that contests,
  `over` and `cal` keep **+5.6 [+1.7, +9.4] points** of game share (against
  +14.3 [+11.5, +17.2] against a passive table), and the 0.4.0 constants lose
  to these, winning **43.5% [40.8, 46.2]** of games (chance is 50%). Against
  a passive table, one that mostly passes over an opening, the refit gives
  back **3.4 [0.8, 6.0] points** of game share relative to the 0.4.0
  constants (exploratory, 192 clusters), though it still beat the hand-only
  bidder by 17 points or more there. No single set of constants wins both
  tables; if the opponents you meet mostly pass, leave `overcall` off. Each
  interval is a 95% bootstrap over duplicate-deal clusters; see
  `wiki/experiments/match-aware-bidding-2026-10-14.md` (#170) in the
  repository.
- **The measured bidder leans on the seven pip trumps only**, and bid at 160
  worlds, with deployed Zeb playing both sides to 7 marks. Pass
  `legalDeclarations: [0, 1, 2, 3, 4, 5, 6]` to bid exactly as measured, and
  `worlds: 160` for the forecast. The default forecast also offers doubles and
  no-trump, which the overcall can then lean on. That is untested.
- A parity test in the repository holds `overcall: true` to the arena's
  `met,over,cal,contested` bidder (`arena/zeb_bid.py`, spec token
  `contested`): prices within 1e-12 and identical calls.
- Nothing else changes. The wasm module and every other function are
  0.4.0's.

## 0.4.0 — 2026-10-14

**Match-aware bidding, opt-in.** `chooseBid` takes a new `match` option: the
score, who holds the high bid, how many seats speak after you, and the
highest legal bid. With it, `chooseBid` is the hand-only rule plus one more
option, a mark bid (84, or one mark over an 84), at the declaration most
likely to take all 42 points. It bids the mark when the chance of winning the
game from there, by an exact match-equity table to `marksToWin` (7), beats
the hand-only call's. A tie keeps the hand-only call. There is no mark
option when `pointsToWin` is set or when `maxBid` is under 84. Behind a
partner's bid the hand-only call is a pass, but the mark bid is still priced.

- **Measured**, bidding at 160 worlds with both sides playing deployed Zeb
  and games to 7 marks: the match-aware bidder won **53.37% of games
  [52.47, 54.27]** against the hand-only bidder, over 4,096 duplicate
  orientations (a coin is 50%; the model's first-order prediction was
  52.94%). Most of the gain comes from hands where the hand-only bidder
  would bid 42 and neither side needs a single mark. See
  `wiki/experiments/match-aware-bidding-2026-10-14.md` in the repository.
- **The law behind it is fitted to the deployed field.** The hand law and
  the auction odds (how often a point bid stands, what passing leaves) were
  measured on Zeb playing Zeb and frozen. Against other fields the numbers
  will differ; follow-ups will refine them in 0.4.x.
- **The measured bidder leans on the seven pip trumps only.** Pass
  `legalDeclarations: [0, 1, 2, 3, 4, 5, 6]` to bid exactly as measured. The
  package's default forecast also offers doubles and no-trump, so its mark bid
  can lean on them too. That is untested, and probably a little bolder.
- `matchPrice({ ... })` exposes the two match equities and the call.
- **Without `match`, `chooseBid` is unchanged from 0.3.0**, and so is
  everything else. A parity test in the repository holds the JavaScript to
  the arena's Python (`arena/match_equity.py`, `arena/zeb_bid.py`) to 1e-12.

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
