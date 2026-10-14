---
title: The book player against Zeb — Winning 42's rules as a Silas stack, head to head (2026-10-13)
kind: experiment
first_seen: 2026-10-13
last_updated: 2026-10-13
status: active
---

## Question

[[book-study-2026-10-12]] priced the book's rules one decision at a time, in Zeb's own
worlds: a price is what Zeb's plan says the book's tile gives up. Two questions it could
not ask. A player that *follows* the book, seated at a real table against Zeb, scores how
many marks a game? And at the decisions where the book's tile is not Zeb's, what does the
book's tile actually gain or lose on the deal as dealt, and where? Jason: "we've not yet
had a contrast much finer than brute winrate." Compare and learn, not crown a winner.

## Pre-registration (written and committed before any pooled run)

Fixed before the first pooled batch; not moved afterwards. One 40-game batch at base seed
900 was run first, for timing and to check the decomposition's identity; it is excluded
from every number below.

**The book player** (`arena/book_play.py`): the 18 book-faithful policies of
`silas/examples/book/` ([[silas]] § The book, rule by rule), asked at every decision of its
seats; the first policy whose rule fires decides, in this order. Leading: the bidder
`first-lead` (trick one), `crisis`, `dead-double` (its cash), `bidders-plan`; his partner
`partner-leads`; a setter `trump-set` (its lead), `setter-leads`. Following: the
contract's exceptions (`defend-the-84`, `trump-set`), the count on the trick
(`partner-plays-the-top-trump`, `donate-when-safe`, `protect-the-count`, `pounce`), taking
the trick (`take-the-trick`, `helper-stretch`), the throws (`dead-double`,
`shed-for-a-void`), the follows with nothing at stake (`low-from-a-pair`,
`follow-with-the-off`, `follow-low`). The order is read off the book: the policies' own
headers name the overrides (the crisis stops the plan; `follow-low` defers to the count
rules; the dead double is thrown first); exceptions precede the general rule.

**Variants.** (a) where no rule fires, the lowest legal tile; a rule's set, its least
tile (the pure Silas stack). (b) where no rule fires, Zeb's tile; a rule's set, its
least tile. (c) as (b), but Zeb's best tile inside the rule's set (Zeb constrained by the
book). Zeb is deployed Zeb, `walt:k0,n512,F=margin_d250k_h128x64,f=16`, also asked at
every book decision for the log. Bidding: the arena's `heuristic`, which is Silas's
`bid-only-enough` call for call, for both teams; and (b) again with the book player
bidding `risk-budget` (block b-rb), with the bidding alone as its control (block rb:
`risk-budget` + Zeb's play). A marks bid is planned by Zeb as its contract's 42 points.

**The match.** The arena's paired design: games to 7 marks, each deal played in both
orientations (pooled halves), 36 games per batch, base seeds 1–11: 396 games, 198 deal
clusters per block. Control: Zeb against Zeb, one batch, must read exactly 0 (Zeb is
deterministic per coordinate). Headline: the book team's marks/game margin, mean over
clusters, 95% percentile bootstrap over clusters (10,000 resamples). Beside it: game
share, the share of hands the book team declares, each side's make rate as declarer, mean
bid, 84s.

**The decomposition** (blocks b, c, b-rb). At a book decision whose tile differs from
Zeb's (a *departure*), Zeb is rolled out from the position after the book's tile; with
R_0 the hand played by Zeb at all four seats from the deal (same auction), each
departure's delta is d_k = R_k − R_{k−1}, and Σ d_k is exactly the book team's hand minus
pure Zeb's on that hand (checked on every hand). Units: the book team's mark margin and
point margin per hand. Chance baseline at each departure: a uniformly random legal tile
other than Zeb's, Zeb continuing. Zeb's predicted price: q of the book's tile minus q of
Zeb's tile, in the book team's make probability.

**Per rule** (each rule and each policy; forced decisions excluded): decided per 100 book
decisions with a choice and per hand; fired at all; departures (share of decided);
predicted price; realized marks and points per departure with a 95% cluster bootstrap
interval; flips for/against; the random departure at the same positions and rule minus
random; contribution in marks per 100 hands. A rule **costs** where its marks interval
per departure is below 0, **gains** where above, else **null**; it **beats chance** where
rule − random is above 0.

**Situations** (all departures pooled, and each policy with 30 or more departures): trick
1, 2, 3, 4, 5–7; seat (bidder, his partner, a setter) × lead/follow; count at risk (the
count already on the trick plus the larger count of the two tiles) 0, 5, 10, 15+; the
contract (points the bidding side still needs: decided, 1–10, 11–20, 21+). A stratum
**concentrates** a rule's cost when its interval is below 0 and the rest of that rule's
departures have an interval containing 0 or above it.

**The order.** Co-firing counts: decisions where two or more policies fired with
different allowed sets, i.e. where the stated order decided.
