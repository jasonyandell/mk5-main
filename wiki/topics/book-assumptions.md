---
title: The book's implicit assumptions — what Winning 42 assumes, what our arena omits, and the ledger to completion
kind: topic
first_seen: 2026-10-14
last_updated: 2026-10-14
status: active
---

## Summary

The [[book-authority]]'s core inventory. Every lesson of *Winning 42* ([[w42]]) is written for a
table: four people who know the game, play each other often, and play to 250 points. Our arena
([[scenario-player]], `arena/`) seats something narrower. Each assumption below lists the
lessons that depend on it (by the private coverage matrix's row ids; no book prose here), whether
the arena holds it, what has been measured nearby, the cheapest measurement that would hold it,
and the Book Authority's opinion. Then the ledger of what is left of the book, and the
recommended order.

The arena as of 2026-10-14, for reference: every seat is memoryless between hands; deployed Zeb
is deterministic per coordinate (which is why Zeb against Zeb reads exactly 0); its worlds are
uniform given the auction and every tile played, except where a reader is switched on; the book
team sends its conventions and reads as plain Zeb; both teams bid `heuristic` (caution 0) unless
a block says otherwise; every declaration is a pip trump; every hand is played out to its last
trick.

## The inventory

| # | the book assumes | lessons that depend on it (row ids) | the arena | measured nearby | cheapest measurement |
|---|---|---|---|---|---|
| A1 | **Opponents model you across hands**: they learn your style and exploit it, so varying your play and keeping a reputation have value | S-0080-b, S-0080-c, S-0081-a, S-0081-b, S-0089-e, S-0156-b, S-0070-a, S-0189-a, S-0037-a, S-0154-a; the bait lead of Hand 28 (S-0129-d) | **absent**: memoryless seats; deterministic Zeb; self-play pour | reads by the book (within hand, tile location only): null. Exploiter endgames: knowing Zeb exactly gains +0.004 contract probability per three-trick endgame. **#149** (a regular who learns bidding caution within a game, reads only): interaction +0.04 [−0.15, +0.22] marks/game, null ([[regulars-table-2026-10-14]]) | a responding regular (#116, #130 slice 5); play-ε learning |
| A2 | **Partners play one system and read each other by it** | S-0041-a, S-0041-b, S-0041-e, S-0047-d, S-0056-a, S-0057-a..d, S-0058-a, S-0060-c, S-0099-b, S-0100-b, S-0100-c, S-0148-h, S-0150-d..i, S-0151-g, S-0152-a, S-0180-c | **half**: the book team sends, never reads | convention overlay (gates, self-play partners): null; the handoff priced at −0.013 marks/hand with no reader of its kind | **#150**, book partners read each other |
| A3 | **Opponents are fallible**: they lose track, misdiscard, can be lured | S-0088-a, S-0083-b, S-0079-a, S-0079-b, S-0169-b, S-0080-a, S-0078-d | **absent**: Zeb never forgets a tile or a void | none | **#151**, the fallible-tracker field and its crossover p* |
| A4 | **The auction is a signal made by a known method**: a bid tells its level's shape | S-0021-a, S-0027-b, S-0027-d, S-0077-c, S-0077-d, S-0078-a, S-0078-b, S-0155-b, S-0165-a | **absent**: worlds uniform given the auction (#117) | pilot below: a deterministic bidder's opening carries 0.93 of 1.58 bits about its hand's strength bucket | #117 / #134; inside #149 |
| A5 | **The book's game and the match**: points to 250, a set rewarded, bidding that reads the score | S-0015-c, S-0017-c, S-0063-b, S-0114-a, S-0114-b, S-0047-a, S-0080-e, S-0110-a, S-0122-a, S-0129-b, S-0165-b, S-0170-a | **partial**: points to 250 exists (`points_to_win`); Zeb plans for the make, bids score-blind, never bids 84 (#115) | the book as Zeb's near-tie break wins 53.2% [51.1, 55.2] of games to 250, replicated | #115, #77 (marks-race utility), #130 slices 3–4 |
| A6 | **The field has bidding styles** (overbidders, wild bidders) | S-0080-c, S-0070-a, S-0155-a, S-0155-b, S-0114-b, S-0034-a, S-0184-a, S-0188-b, S-0190-a, S-0191-a | **absent**: both teams `heuristic`, caution 0 | `risk-budget` bidding −0.71 [−1.00, −0.41] marks/game against that style-free field | **#152**, a named deviant field |
| A7 | **The reader is a bounded human**: the rules are a fast policy, not a replacement for enumeration | the whole book; S-0076-a, S-0078-d | **violated by design**: compared at Zeb's full compute | the book as tie-break wins its game (A5) | **#154**, the book at small n |
| A8 | **A rule is applied with judgment inside its stated scope** | every rule file's header | **partial**: the Silas stack is literal | the census: 44% of the first head-to-head's loss was the encoding; repairs moved +0.47 [+0.33, +0.62] marks/game | #146 done; #153 |
| A9 | **The value of unpredictability is a response, not a fixed field** (an equilibrium claim) | S-0089-e, S-0081-a, S-0037-a, S-0154-a | **absent**: no mixed policies, no response map | none | #130 slice 5; #116; #149 is the fixed-learner first step |
| A10 | **The full declaration menu** (doubles as trumps, no trumps) | Ch. 9's 34 rows, Hands 23–25 | **absent**: pip trumps only | none | **#155** |
| A11 | **A lifetime at the table**: learning, enjoyment and conduct are part of the objective | S-0081-b, S-0080-e; Ch. 11's table-talk rows; S-0209-e | out of scope, correctly | none | none in marks |

### Opinions

- **A1 is the largest omission and the likeliest home of a measurable book edge.** The book's
  style chapter, its 84 chapter and its closing essay all assume a regular opponent who learns
  you. The arena prices every such lesson at exactly zero by construction, and the self-play
  pour contains no reader. Two nearby measurements bound the *trick-play* side as small. A
  reader who knew the book player's play rules exactly gained nothing within a hand. Knowing
  Zeb exactly is worth 0.4 percentage points of contract probability per endgame. The prediction is
  therefore that the cross-hand edge, when the arena can see it, shows in the **auction and the
  84** (what a bid or a straight off tells a learner who has seen the bidder before), not in
  trick play. That is #149's design. Jason's assumption is adopted: a tool that cannot reward
  modeling across hands is suspect for that reason, and Zeb's arena margin is presumed to
  contain the value of never being modeled until #149 says how much.
- **A2: the head-to-head charged the book every convention's cost and credited none of its
  benefit.** The book team sends (the donation, the top-trump lead, the partner's leads) to a
  partner whose worlds stay uniform. A convention is a joint deviation; pricing the sender alone
  measures half of it. The book study's overlay did pair sender and reader, on the gates rather
  than the book-faithful rules and with self-play partners, and read null. Expect a modest gain
  from #150, and the largest information flow in the 84 defence (Hand 35's discards), which
  waits on #128.
- **A3 is real and unpriced.** The straight-off 84 and the low-looking master are measured only
  against opponents who never forget, which is their worst case. The honest claim is a
  crossover: "the ploy pays against opponents who lose track more than p*". No human error rate
  is assumed, and p* is reported as the book's scope.
- **A4 is cheap, and A1 needs it.** The book reads a 30, a 31 and a 35 as shapes. Our worlds
  ignore the auction entirely. A learner that has seen a bidder's hands reads his next bid far
  more sharply (pilot below).
- **A5 is where restoring an assumption first paid.** Given its own game, the book's count play
  earned 53% of games as Zeb's tie-break. What remains is a utility that sees the score and an
  84 bid (#115).
- **A6:** the book's bidding has only been measured against the one field in which restraint
  cannot pay by setting an overbidder. Its −0.71 is the scope of that field.
- **A7, held firmly:** the book is a policy for a player who cannot enumerate. Its one measured
  win is in exactly that role, breaking ties for a planner. The prediction for #154 is that
  the book raises a bounded Zeb, by more as n falls.
- **A8:** every residual disagreement between an encoding and the book is presumed to be the
  encoding's until a reason report finds no condition in the header that would reconcile them.
  The three rows filed as "the book against its rule" are re-read under that presumption (#153).
- **A9:** the full statement of "keep them honest" needs mixed policies and a response map
  (#130 slice 5). #149 is the cheap step before that: a fixed learner, not an equilibrium.
- **A10 is plumbing,** but 34 rows can never fire until it is done.
- **A11 is outside a marks measurement.** It is recorded so that nobody charges the book a loss
  for advice whose objective is learning or enjoyment.

## The ONE next experiment: the regulars' table (#149)

**Run 2026-10-14 ([[regulars-table-2026-10-14]], pre-registered at `d432a216`).** Interaction
+0.04 [−0.15, +0.22] marks/game over 384 game pairs, A/A −0.02 [−0.20, +0.17]: a null at this
power (MDE about 0.27). The regular identifies a deterministic bidder (0.88 of its belief on
caution 0 by a game's end, 0.46 on a varied one), and that knowledge, read into the next hand's
tiles only, moves no marks measurably. In the charter's voice: our cross-hand reader of this
form cannot yet see the book's edge. It never responds, learns bidding only, and forgets at the
game's end. The sketch below was refined: the target plays Zeb 0.3.0, and Zv varies its bidding
only.

This is the experiment that most directly tests whether modeling the opponent across hands is
where the book's value lives. It is the smallest design that separates "Zeb is better at 42"
from "Zeb is never modeled".

**Pre-registration sketch** (to be committed in full before any measured batch):
- *Target team:* **Z**, deployed Zeb (`heuristic` bidding at caution 0, lowest-id ties), or
  **Zv**, varied Zeb: bidding caution drawn per hand from {−5, 0, +5}, since ceilings move in
  steps of 5. In play, Zv picks uniformly among tiles within 0.005 of the best q, the plan's
  own re-plan noise, so the variation is nearly free by Zeb's reckoning.
- *Opponents:* **R0**, the soft reader (uniform, ε = 0.28, [[reads-plays-soft-2026-10-14]])
  plus an auction likelihood whose bidder model is a flat prior reset every hand. **R** is the
  same reader with memory within a game: each opposing seat's caution and play ε are learned
  from the finished hands, every tile of which is public by then. R − R0 is exactly the
  cross-hand learning.
- *Primary:* I = (Zv − Z | R) − (Zv − Z | R0) in marks/game, 95% cluster bootstrap, paired
  deals. **The book predicts I > 0.**
- *Secondary:* S = (R − R0) against Z, the learning gain against a deterministic player
  (the book predicts the regulars gain). C = (Zv − Z | R0), the price of varying at a
  memoryless table (the book predicts it is small).
- *Both games* (7 marks; 250 points), an A/A arm, base seeds 1–22 (an interaction carries
  about twice an arm's interval; the minimum detectable effect is stated before running),
  every batch under `timeout 300`.
- *Reading:* I above 0 means the book's predictability lesson is visible in our arena, and
  Zeb's margin contained the value of not being modeled. I null means "our cross-hand reader
  of this form cannot yet see the book's edge". Its omissions are then named: it reads but does
  not exploit (no change to its own bidding against a learned overbidder, S-0080-c), and its
  memory lasts one game.
- *Fence:* every value is Zeb's plan over the reader's own cell. The learned model is a
  likelihood over information that is public when it is used ([[strategy-fusion-fence]]).

**Pilot (≤1 minute, no play; 2,000 simulated games of 11 hands, deals from the arena's
`deal_from_seed`; the script is posted on #149).** The bidder is the arena's `HeuristicBidder`,
which opens at 30 when its ceiling less its caution reaches 30. The learner keeps an interval
on the caution from one seat's revealed hands. An opening bounds it above by ceiling − 30;
a pass with three trumps bounds it below by ceiling − 29.

| bidder | learner's result within one game | opening's information about the hand's strength bucket |
|---|---|---|
| deterministic (caution 0) | identified to its behavioural class (an interval of 5) in **83.2%** of games, median hand 6 | **0.93 bits** of 1.58 |
| varied (caution U[−6, 6] per hand) | shown to fit no fixed caution in 65.0% of games, median hand 6 | **0.47 bits** |

The learner sees one seat's openings only; a full auction shows four actions a hand, so this
understates how fast it learns. The channel exists and opens within one game. Varying halves
what an opening tells. What that is worth in marks, net of what varying costs, is the
experiment.

## The completion ledger

From the private coverage matrix (570 merged rows; 2026-10-13, after batch 1; ids only):

| status | rows | what finishes it |
|---|---:|---|
| covered by a book rule, query, outcome or test file | 241 | measured at the table only for the 18 rules of the head-to-head; the rest are encoded, not seated |
| covered by an `odds/` file only | 24 | exact; done |
| covered by a gate only | 21 | the header audit (#153) |
| encodable now | 6 | #153 (batch 1 left 12 of its 86 with a stated reason; 6 remain encodable) |
| blocked by a gap | 149 | #130 is the first-cited gap of about 112: slice 3 (deal measure, play-out; 19 rows priced against a Silas field), slice 4 (the arena ledger; about 98 rows priced, including the bid book's 60 and cluster E's ten against a named field, #152), slice 5 (mixed policies; cluster E's four equilibrium rows, A9). Then #128 (policy likelihood, 19), #134 (the auction, 15), #135 (aggregate, 8), #132 (another seat's knowledge, 1), gap 9 (1) |
| n/a for an encoding | 129 | not n/a for a lesson: the Ch. 6 style rows, the Ch. 12 closing essay and the Ch. 14 profiles carry A1, A6 and A11, mapped above |

`silas/GAPS.md` § The book's silences records six choices the book does not make: which sure
trump leads, which setting lead, a count off on the setters' trick, the follower who is not last
and can take the trick, Hand 27's crisis, and the dead double against shed's lone tile. Each is
recorded in the GAPS file and none is guessed. Under the authority stance, each is resolved by
the book's own hands or left open, never by Zeb's preference.

## Recommended order

1. **#149, the regulars' table**: run; a null for a reading-only, fixed-style regular ([[regulars-table-2026-10-14]]). Its successor is a regular that responds (#116 / #130 slice 5).
2. **#150, book partners read each other**: cheap (it reuses `arena/book_read.py`), and it
   seats the benefit half of every convention the head-to-head charged.
3. **#153, the header audit and the last encodable rows**: cheap, and it finishes the encoding.
4. **#154, the book at small n**: cheap, and it tests A7 directly.
5. **#152, the named deviant field**: cheap arena-side, and it seats the book's bidding against
   its own field.
6. **#151, the fallible tracker**: medium, and it reports a crossover rather than a verdict.
7. **#130 slices 3–4**: the largest block of rows (about 117 priced).
8. **#117 / #134** (the auction read), then **#128** (policy likelihood in Silas).
9. **#155, the declarations**: plumbing that unlocks Ch. 9.
10. **#130 slice 5 / #116**: mixed policies and response maps, the full home of A1 and A9.

## Links

[[book-authority]] · [[w42]] · [[book-study-2026-10-12]] · [[book-head-to-head-2026-10-13]] ·
[[book-investigation-2026-10-14]] · [[book-reads-by-the-book-2026-10-14]] ·
[[reads-plays-soft-2026-10-14]] · [[regulars-table-2026-10-14]] · [[exploiter-endgames-2026-10-14]] · [[silas]] ·
[[scenario-player]] · [[strategy-fusion-fence]]
