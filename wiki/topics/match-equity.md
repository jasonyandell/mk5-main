---
title: Match equity — the value of a score in the game to seven marks
kind: topic
first_seen: 2026-10-14
last_updated: 2026-10-18
status: active
---

> **It’s not what you hold, it’s what you need.**
>
> — the shop, 2026-10-14

## What it is

The match equity V(a, b, d) is the probability that our side wins the game to seven marks
from a marks to b, with d = 1 when we shake first this hand (backgammon's match-equity table,
for 42). It is the unit a bid should be priced in: not P(make), not marks per hand.
`arena/match_equity.py` solves it exactly from one hand's law under a field: every hand moves
at least one mark, so V is a dynamic program over the 7 × 7 score grid with no iteration. Under
a coin per one-mark hand it is the oracle era's `race_wp` (`champion/utility.py`); Silas's
`WinValue::solve` is the same object for the points game and the marks game ([[silas]]).

## One-mark calls are score-blind

Every contract from 30 to 42 pays one mark, made or set. If every branch of a decision moves
exactly one mark, with the same continuation context (including next shaker) after each
outcome and a fixed future hand law,

    E[V] = V(a, b+1) + π · [V(a+1, b) − V(a, b+1)].

Here π is the chance our side takes the mark. For a monotone equity table the bracket is
nonnegative: maximizing π maximizes equity, with the same ordering when it is positive.
Thus the score does not change the best one-mark call **if the calls' mark-win probabilities
are themselves score-independent**. Score-sensitive opponents or different continuation
contexts can violate that premise. Different stakes (84 and up) change the equity comparison
even with fixed make probabilities. ("Within a hand, marks-optimal ≈ maximize
P(make), which is score-independent" was rung #27's play-side finding, [[champion-ladder]];
this is its bid-side form.)

## The 84 against the 42

Both need all 42 points; the 84 pays two. Assume the same make event and continuation
context for the comparison (a changed bid can otherwise change play). Bid 84 over 42 iff

    p · [V(a+2, b) − V(a+1, b)] > (1 − p) · [V(a, b+1) − V(a, b+2)].

The undivided inequality also handles zero increments and terminal boundaries.

Under the coin law the threshold p* is (rows: our marks; columns: theirs):

| us \ them | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| 0 | .50 | .45 | .40 | .33 | .25 | .14 | 0 |
| 1 | .55 | .50 | .44 | .38 | .29 | .17 | 0 |
| 2 | .60 | .56 | .50 | .43 | .33 | .20 | 0 |
| 3 | .67 | .63 | .57 | .50 | .40 | .25 | 0 |
| 4 | .75 | .71 | .67 | .60 | .50 | .33 | 0 |
| 5 | .86 | .83 | .80 | .75 | .67 | .50 | 0 |
| 6 | — | — | — | — | — | — | — |

At their 6 a set loses the game at 42 or 84 alike and a make at 84 is two marks: the 84 is
free (Jason, 2026-10-14). At our 6 the 42 dominates (the book's rule, Silas's `last-mark`
dominance). Even scores sit at one half, the marks-linear threshold. Against a lower point bid
the 84 must also beat that call's higher make chance.

## Where it is used

The hand law Zeb plays against itself is a coin to three places (the first shaker's side takes
50.3% of 4,251 hands), so the table above is also the measured one.

- [[match-aware-bidding-2026-10-14]] — Zeb's bidder with the 84 priced by match equity
  (`arena/zeb_bid.py`, `zeb:…,met`): 53.4% [52.5, 54.3] of games against the hand-only bidder,
  51.4% [50.6, 52.2] against the same 84 priced by expected marks. Every change is an added
  84; the opponents at 6 are about a third of the gain. Its follow-up (#160) applied the
  one-mark theorem to the pass over an opponent's bid: overcall whenever the call's π beats
  passing's 18%. The bidder with that repair and a calibrated forecast wins 71.0% [69.0, 73.0]
  of games against deployed Zeb and 63.4% [61.4, 65.4] against `met`. Against a table that
  contests back (#170), the repair keeps +5.6 [+1.7, +9.4] points of its +14. Its field model
  must be refit to that table: the frozen bidder wins 43.5% [40.8, 46.2] against the refit one,
  while the refit gives back about 3 points against a passive table.

## Links

[[rules-of-42]] · [[champion-ladder]] · [[silas]] · [[scenario-player]]
