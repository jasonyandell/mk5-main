---
title: The book player against Zeb — Winning 42's rules as a Silas stack, head to head (2026-10-13)
kind: experiment
first_seen: 2026-10-13
last_updated: 2026-10-14
status: complete
---

> **Read under [[book-authority]] (2026-10-14).** The book is the authority and this arena is
> the suspect. Every number below is the book player against deployed Zeb in an arena where no
> seat models another across hands (A1), the book team sends its conventions to a partner who
> never reads them (A2), both opponents track every tile perfectly (A3), the auction is unread
> (A4), and the field has no bidding styles (A6) ([[book-assumptions]]). The numbers stand as
> measured. They are the scope of our model, not a verdict on the book. Of the published −1.88,
> 0.47 was the encoding ([[book-investigation-2026-10-14]] § After the repairs: −1.41 repaired).

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

## The match

Zeb against Zeb reads exactly 0 (36 games: the planner is a function of its coordinate, so
each deal plays the same in both orientations). Every block: 396 games, 198 deal clusters.

| book team | marks/game [95%] | game share | made as declarer, book / Zeb | book declares | book mean bid |
|---|---|---|---|---|---|
| (a) rules, else the lowest legal tile | **−2.98 [−3.20, −2.74]** | 13.9% | 41.2% / 70.2% | 50.0% | 30.6 |
| (b) rules, else Zeb's tile | **−1.88 [−2.09, −1.66]** | 25.0% | 50.0% / 67.5% | 50.1% | 30.6 |
| (c) as (b), Zeb's best inside a rule's set | **−1.41 [−1.60, −1.22]** | 29.8% | 52.6% / 65.5% | 50.0% | 30.6 |
| (b), bidding `risk-budget` | **−2.59 [−2.86, −2.31]** | 18.4% | 41.2% / 67.3% | 55.9% | 38.8 (173 bids of 84) |
| Zeb's play, bidding `risk-budget` (the bidding alone) | **−0.71 [−1.00, −0.41]** | 40.9% | 53.0% / 61.6% | 55.7% | 38.7 (177 bids of 84) |
| Zeb against Zeb (control) | 0 | 50% | 61.1% / 61.1% | 50% | 30.6 |

- **Every variant scores below Zeb in this arena, with intervals far from zero.** Where no rule fires, Zeb's tile
  instead of the lowest legal one is worth +1.10 [+0.89, +1.31] marks/game (paired by
  cluster); inside a rule's set, Zeb's best tile instead of the least one +0.47 [+0.30,
  +0.64]. The rest, −1.41 marks/game, is the rules' sets where they exclude Zeb's tile, part
  of it the encoding firing outside its rules' stated scope ([[book-investigation-2026-10-14]]).
- **The book's bidding and its play cost separately and add up:** −0.71 (bidding) and
  −1.88 (play) against −2.59 together (paired, b-rb − b −0.70 [−1.00, −0.41]).
  `risk-budget` declares a little more often, bids higher (an 84 in 8% of its contracts)
  and makes less (53% against Zeb's 61.6%).
- A rule decides 79% of the book player's decisions with a choice (8.7 such decisions a
  hand). Where one does, its tile is Zeb's about half the time: in block b, 14,370
  departures in 4,268 hands, 3.4 a hand.

## The departures, one at a time (block b)

The decomposition's identity held on every hand of every block (4,268 in b, 4,338 in c,
3,981 in b-rb): the per-departure deltas sum exactly to book minus pure Zeb, which is
**−0.173 [−0.194, −0.152] marks/hand** in b (−0.130 in c, −0.186 in b-rb), the game
number's −1.88 at 10.8 hands a game. 808 of 4,268 hands change outcome. Pure Zeb in the
book's chairs scores −0.001 [−0.011, +0.008] marks/hand against Zeb, as it must.

Pooled over the 14,370 departures: **−0.051 [−0.058, −0.045] marks per departure**, 526
flips for and 896 against. Zeb's plan forecast −0.077 (twice its price of −0.038 in the
book team's success probability: the make for the bidders, the set for the setters), so its
forecast runs 1.5 times pessimistic overall; a third of the excess is the winner's curse of
Zeb's own sample (re-planned on an independent sample the price is +0.0046 [+0.0039,
+0.0054] less negative, [[book-investigation-2026-10-14]]). A uniformly random other tile at
the same positions costs −0.087 [−0.097, −0.078], but at 44% of the positions the random
other tile is the book's own; where it is not, **the book's tile beats chance by +0.064
[+0.047, +0.080]** (+0.036 [+0.027, +0.045] counting those) **and loses to Zeb's by 0.051.**
In points the departures give up −0.16 [−0.27, −0.04]: pooled, the book does not trade
marks for points; it gains points once the contract is decided (+2.28 a departure) and in
the rules that take or give count (below).

Per policy (decides: per 100 book decisions with a choice, and per hand; marks per departure
with its 95% cluster interval; verdicts by the bands fixed above):

| policy | decides /100 | per hand | departs | Zeb's price | marks/departure [95%] | points | rule − random [95%] | marks/100 hands | verdict | vs chance |
|---|---|---|---|---|---|---|---|---|---|---|
| bidders-plan | 9.33 | 0.81 | 1,593 | −0.034 | −0.030 [−0.055, −0.005] | +0.37 | +0.041 [+0.012, +0.070] | −1.12 | costs | beats |
| shed-for-a-void | 8.43 | 0.73 | 1,491 | −0.019 | −0.023 [−0.042, −0.004] | +0.06 | +0.011 [−0.007, +0.028] | −0.80 | costs | null |
| dead-double | 7.91 | 0.69 | 1,484 | −0.032 | −0.044 [−0.075, −0.013] | −0.83 | +0.051 [+0.021, +0.083] | −1.55 | costs | beats |
| pounce | 7.87 | 0.69 | 1,590 | −0.034 | −0.011 [−0.039, +0.017] | +0.35 | +0.029 [+0.009, +0.050] | −0.42 | null | beats |
| take-the-trick | 7.30 | 0.64 | 1,240 | −0.033 | −0.092 [−0.122, −0.063] | **+1.88** | +0.079 [+0.049, +0.110] | −2.67 | costs | beats |
| setter-leads | 7.01 | 0.61 | 1,759 | −0.057 | −0.064 [−0.096, −0.031] | −0.38 | +0.011 [−0.019, +0.042] | −2.62 | costs | null |
| partner-leads | 6.08 | 0.53 | 1,019 | −0.035 | −0.043 [−0.083, −0.004] | −0.07 | +0.055 [+0.018, +0.092] | −1.03 | costs | beats |
| first-lead | 5.75 | 0.50 | 1,491 | −0.048 | −0.094 [−0.135, −0.051] | −2.05 | +0.063 [+0.023, +0.103] | −3.28 | costs | beats |
| low-from-a-pair | 5.59 | 0.49 | 738 | −0.044 | −0.073 [−0.115, −0.032] | −1.20 | two tiles | −1.27 | costs | — |
| donate-when-safe | 4.66 | 0.41 | 726 | −0.041 | −0.030 [−0.073, +0.011] | **+1.06** | +0.036 [−0.003, +0.074] | −0.52 | null | null |
| follow-low | 4.02 | 0.35 | 425 | −0.013 | −0.047 [−0.092, −0.005] | +0.23 | +0.000 [−0.019, +0.019] | −0.47 | costs | null |
| crisis | 1.40 | 0.12 | 363 | −0.080 | −0.121 [−0.190, −0.056] | −0.50 | +0.033 [−0.036, +0.101] | −1.03 | costs | null |
| protect-the-count | 1.40 | 0.12 | 120 | −0.040 | −0.067 [−0.173, +0.034] | −2.53 | +0.000 [−0.049, +0.049] | −0.19 | null | null |
| follow-with-the-off | 1.03 | 0.09 | 116 | −0.087 | −0.052 [−0.176, +0.068] | −1.47 | two tiles | −0.14 | null | — |
| partner-plays-the-top-trump | 0.95 | 0.08 | 144 | −0.047 | −0.056 [−0.137, +0.016] | −0.25 | +0.028 [0.000, +0.073] | −0.19 | null | null |
| trump-set | 0.20 | 0.02 | 49 | −0.019 | +0.000 [−0.113, +0.120] | −3.27 | two tiles | 0.00 | null | — |
| helper-stretch | 0.15 | 0.01 | 22 | −0.010 | −0.091 [−0.300, 0.000] | +4.55 | two tiles | −0.05 | null | — |
| defend-the-84 | 0 | 0 | — | | | | | | never fires: Zeb never bids 84 | |
| **all** | 79.1 | 6.9 | 14,370 | −0.038 | **−0.051 [−0.058, −0.045]** | −0.16 | **+0.036 [+0.027, +0.045]** | **−17.3** | costs | beats |

"Two tiles": with two legal tiles the random other tile is the book's, an identity, not a
finding. **In block b no policy and no rule gains** (no interval above zero); in block
b-rb one rule does, first-lead's *lead the top trumps* (+0.084 [+0.006, +0.166], 308
departures), one cell of about three hundred. Eight rules carry most of the cost (the
price is the book team's success probability, as above):

| rule | departs | marks/departure [95%] | Zeb's price | marks/100 hands | vs chance |
|---|---|---|---|---|---|
| setter-leads: lead the count that sets | 886 | −0.122 [−0.182, −0.062] | −0.091 | −2.53 | null |
| take-the-trick: void, trump it | 675 | −0.098 [−0.137, −0.060] (points +1.66) | −0.031 | −1.55 | beats |
| first-lead: double-up, lead low | 388 | −0.149 [−0.253, −0.048] | −0.069 | −1.36 | null |
| low-from-a-pair | 738 | −0.073 [−0.115, −0.032] | −0.044 | −1.27 | — |
| first-lead: lead a trump | 705 | −0.074 [−0.134, −0.014] | −0.041 | −1.22 | null |
| take-the-trick: take it last | 565 | −0.085 [−0.137, −0.034] (points +2.15) | −0.037 | −1.12 | beats |
| dead-double: the bidder cashes it | 299 | −0.147 [−0.241, −0.054] | −0.090 | −1.03 | beats |
| bidders-plan: draw the trumps | 588 | −0.071 [−0.126, −0.017] | −0.036 | −0.98 | null |

Three of the eight are mostly the encoding rather than the rule. *Lead a trump*: at 589 of its
705 departures the rule also allowed Zeb's tile and the least tile was played (−0.092); at the
other 116, +0.017 [−0.185, +0.222], and in block c the rule reads +0.119 [−0.053, +0.291].
*Low from a pair*: 210 of its departures decline a trick the higher tile takes, which no rule
of the stack covers (−0.219); the rest are −0.015 [−0.053, +0.024]. *The cash*: 241 of 299
come while the bidder holds the boss trump and his draw also fired. Without the departures
the census of [[book-investigation-2026-10-14]] labels as encoding artifacts, *the count that
sets* is −0.092 [−0.168, −0.018], and the two take-the-trick rules, *double-up, lead low* and
*draw the trumps* are unchanged.

Null at the table: the pounce's ten and five now (−0.009, −0.017 per departure), its guard
(no count on a lost trick, 0.000), the setter's double that calls for count (0.000), the
book's donation lead (+0.007), the safe trick's five (+0.008), the dump of a bare off that
calls no count (+0.032; +0.159 over chance), and "one point sets: lead a double" (+0.091
[−0.032, +0.231]). Block c reads the same in total (all departures −0.049 [−0.055, −0.042],
beating chance by +0.032); per policy three verdicts move there: donate-when-safe costs
(−0.068 [−0.111, −0.025]), follow-low and partner-leads are null (−0.009, 0.000). Every rule,
block and stratum is in `out/tables.md`.

## Where the departures cost, in this arena

Pooled over all departures of block b (c and b-rb read the same within their intervals):

| stratum | departures | marks/departure | points |
|---|---|---|---|
| trick 1 / 2 / 3 / 4 / 5–7 | 2,820 / 2,839 / 2,903 / 2,530 / 3,278 | −0.058 / −0.058 / −0.063 / −0.047 / −0.034 | −1.20 / −0.61 / −0.42 / +0.33 / +1.00 |
| bidder leading / following | 3,746 / 869 | −0.074 [−0.095, −0.053] / −0.092 [−0.129, −0.056] | −0.85 / +1.36 |
| partner leading / following | 1,019 / 1,546 | −0.043 [−0.083, −0.004] / −0.026 [−0.052, 0.000] | −0.07 / +0.54 |
| setter leading / following | 1,798 / 5,392 | −0.061 [−0.093, −0.029] / −0.035 [−0.047, −0.023] | −0.42 / −0.04 |
| count at risk 0 / 5 / 10 / 15+ | 7,791 / 3,329 / 2,701 / 549 | −0.045 / −0.055 / −0.070 / −0.033 [−0.080, +0.014] | −0.41 / +0.06 / +0.20 / +0.36 |
| the bidding side needs 21+ / 11–20 / 1–10 / decided | 7,107 / 3,272 / 1,452 / 2,539 | −0.067 / −0.065 / −0.037 / 0 | −0.96 / −0.40 / +0.09 / **+2.28** |

- **The cost is spread, not concentrated.** No pooled stratum meets the concentration
  criterion: every stratum with the contract open costs. It is largest in the bidder's chair
  (a departure of his costs two to three times his partner's) and early, and it runs while
  the contract is open (21+ and 11–20 points still needed: −0.067, −0.065), halves when the
  contract is nearly decided (1–10: −0.037), and turns into points once it is decided
  (+2.28 per departure, no marks): the book plays for count, Zeb for the contract.
- **Concentrated, by the criterion** (the stratum's interval below 0, the rest of that
  policy's departures not; a stratum that is the whole policy, such as first-lead at trick 1,
  is not a concentration; 26 of the 234 strata tested in block b meet it, with no correction
  for multiplicity): the bidder's plan while the bidder still needs 21 or more (−0.098) and
  with ten at risk (−0.066), and the crisis while he needs 21 or more (−0.201 per
  departure); the dead double when the bidder leads it (the cash, −0.147), at trick 3 and
  with 11–20 needed; the partner's leads at trick 4 and with 11–20 still needed (−0.118); the
  setter's leads with no count at risk (−0.100); follow-low in the partner's chair (−0.051),
  at trick 2 and with 21 or more needed; shed-for-a-void with 15 or more at risk (−0.170, 47
  departures) and with 11–20 needed; the safe trick's donation late (tricks 5–7) and when the
  bidder needs 10 or less (−0.145, 55 departures); the pounce late (−0.023).
- **Points against marks.** Taking the trick ("win any trick you can") gains +1.9 points per
  departure and loses 0.09 marks; the donation gains +1.1 points and is null in marks;
  first-lead loses both. Walt's +0.74 marks/game over E[Q] ([[scenario-player]]) was marks
  up, points down; the book is the same contrast from the other side.

## The order

Two or more policies fire at 16% of the decisions with a choice, with different sets at
14%, and the stated order picked a different least tile than the other policy's at 13%
(4,982 of 37,174 in b). The common pairs: dead-double with shed-for-a-void (1,787), pounce
with shed-for-a-void (1,165), crisis under first-lead at trick 1 (718), dead-double with
pounce (523) or first-lead (495). The order is a real choice at one decision in eight. Letting
every firing rule decide, with Zeb's best tile in the union of their sets, is worth +0.22
[+0.07, +0.36] marks a game over (c), and the book player still scores −1.2
([[book-investigation-2026-10-14]]).

## Corrections during the run

- **(c)'s tie-break had the setters' sign backwards.** Zeb's q is the bidding side's make
  probability; the first run of block c took its maximum for a setter too, the worst tile
  inside the set. Found by checking that in (c) Zeb's tile never departs when it is inside
  the set; fixed in `arena/book_play.py` with a test and block c re-run in full. The first
  run (−1.86 [−2.07, −1.65]) is published as `out/c-setter-sign-bug/` and is in no number
  above. After the fix the check holds: 0 such departures.
- **The decomposition's log key** first omitted the contract, which collides when the two
  orientations' auctions differ (b-rb only); it now carries declaration, bidder and bid, and
  every b-rb decomposition was redone under it.

## What this means

- **Asked to play its rules in our arena, the book player scores below Zeb** (A1–A4 and A6 violated; [[book-assumptions]]): 1.9 marks a game with
  Zeb's tile where no rule fires (b), 1.4 with Zeb's best tile inside a rule's set (c), 2.6
  with its own bidding too (b-rb). Its rules cost wherever they depart from Zeb, about 0.05
  marks a departure at 3.4 departures a hand. The pour priced each rule one decision at a
  time in Zeb's own worlds ([[book-study-2026-10-12]]: weak agreement on most, against none);
  at the table those departures add up to −17 marks per 100 hands, of which the rules' own
  disagreements with Zeb, without the encoding's artifacts, carry 56% (−0.042 [−0.051,
  −0.033] marks a departure, [[book-investigation-2026-10-14]]).
- **The book's rules carry real information even here.** Its tile beats a random departure at
  the same positions (+0.064 per departure where the random tile is not the book's own). Less
  of that information is visible to our arena than to the book's readers.
- **Zeb's forecast of a departure is honest in sign and, pooled, roughly in size** (realized
  −0.051 against −0.077 forecast, a third of the gap the winner's curse); per policy it is not
  (the pounce forecast −0.067, realized −0.011; take-the-trick −0.066 and −0.092).
- Nothing is adopted, and nothing here says the book is wrong for partners who both follow
  it: the book player's partner is the book player, and both opponents are Zebs, who read
  no book. The book's own game, readers of plays, its own deals and its scope are
  [[book-investigation-2026-10-14]].

## What our model leaves out

The book's lessons assume a table this arena does not seat ([[book-assumptions]]). Each item
below is a place the book's edge could live that these blocks price at zero or at its cost alone.
- **Opponents who learn you across hands** (A1). Both opponents are memoryless Zebs, so the
  value of being hard to read is zero by construction. The measurement is #149.
- **A partner who reads the conventions** (A2). The book player's partner is the book player,
  but its worlds stay uniform. Every convention's cost is charged here and none of its benefit.
  The measurement is #150.
- **Opponents who lose track** (A3). Zeb never forgets a tile, so the book's ploys meet their
  worst case. The measurement is #151.
- **The auction as a signal** (A4) and **a field with styles** (A6). Both teams bid `heuristic`
  at caution 0 and nobody reads a bid. The measurements are #117 and #152.
- **The book's game** (A5). Marks to 7 here; points to 250 is in [[book-investigation-2026-10-14]],
  where the book as Zeb's near-tie break wins 53.2% of games.
- **A bounded reader** (A7). The book was written for a player who cannot enumerate 512 worlds a
  decision. The measurement is #154.

## Data

[jasonyandell/texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10)
under `book/head-to-head/`, with `SHA256SUMS` and a card (`CARD.md`): per block and seed the
games, the book player's every decision (fired policies, allowed sets, Zeb's tile and q), the
decomposition per hand and the arena summaries; `out/results.json` and `out/tables.md`; the
scripts (`run.py`, `rollout.py`, `analyze.py`, `run_all.sh`) and logs. Code:
`arena/book_play.py` (the player and `SilasBidder`) and `scenario/core`'s
`SilasPolicy.bid`/`declare`.

## Links

[[book-study-2026-10-12]] · [[book-investigation-2026-10-14]] · [[silas]] · [[scenario-player]] · [[w42]] · [[arena]]
