---
title: The book's implicit assumptions — what Winning 42 assumes, what our arena omits, and the ledger to completion
kind: topic
first_seen: 2026-10-14
last_updated: 2026-10-22
status: active
---

> **Zeb isn’t better at 42; nobody at the table is punishing its predictability.** That is the reason Zeb “wins”.
>
> — the shop, 2026-10-14

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
team sends its conventions and reads as plain Zeb unless its partner read is on (`read=`, #150); both teams bid `heuristic` (caution 0) unless
a block says otherwise; every declaration is a pip trump; every hand is played out to its last
trick.

## The inventory

| # | the book assumes | lessons that depend on it (row ids) | the arena | measured nearby | cheapest measurement |
|---|---|---|---|---|---|
| A1 | **Opponents model you across hands**: they learn your style and exploit it, so varying your play and keeping a reputation have value | S-0080-b, S-0080-c, S-0081-a, S-0081-b, S-0089-e, S-0156-b, S-0070-a, S-0189-a, S-0037-a, S-0154-a; the bait lead of Hand 28 (S-0129-d) | **absent**: memoryless seats; deterministic Zeb; self-play pour | reads by the book (within hand, tile location only): null. Exploiter endgames: knowing Zeb exactly gains +0.004 contract probability per three-trick endgame. **#149** (a regular who learns bidding caution within a game, reads only): interaction +0.04 [−0.15, +0.22] marks/game, null ([[regulars-table-2026-10-14]]) | a responding regular (#116, #130 slice 5); play-ε learning |
| A2 | **Partners play one system and read each other by it** | S-0041-a, S-0041-b, S-0041-e, S-0047-d, S-0056-a, S-0057-a..d, S-0058-a, S-0060-c, S-0099-b, S-0100-b, S-0100-c, S-0148-h, S-0150-d..i, S-0151-g, S-0152-a, S-0180-c | **restored at the fallback and inside a rule's set** (`read=`, #150); a rule's guard still cannot read (gap 1) | convention overlay (gates, self-play partners): null. Book partners reading each other's plays and calls by the book: +0.10 [−0.06, +0.26] marks/game, null ([[book-partners-read-2026-10-14]]) | a guard that reads a partner posterior (#128; `silas/GAPS.md` 1) |
| A3 | **Opponents are fallible**: they lose track, misdiscard, can be lured | S-0088-a, S-0083-b, S-0079-a, S-0079-b, S-0169-b, S-0080-a, S-0078-d | **two dimensions measured**: a field that forgets a seat's voids with probability p per decision (#151); a field that loses the count, sticky, in a seat's view (#172) | forgetting voids costs the field 0.32 [0.23, 0.41] marks/game at p = 0.4 ([[fallible-field-2026-10-15]]); losing the count costs it 0.71 [0.60, 0.82] at p = 0.2 ([[count-lapse-2026-10-15]]). Both times the book player and Zeb collect it equally: ΔG(0.4) = −0.03 [−0.14, +0.08], ΔG(0.2) = +0.07 [−0.08, +0.22]; p* beyond the grid | a player that *aims* at the lapse, with the lapse as a parameter of its model of the other seat (#182, [[lem-the-teacher]]); #168, the sticky void lapse |
| A4 | **The auction is a signal made by a known method**: a bid tells its level's shape | S-0021-a, S-0027-b, S-0027-d, S-0077-c, S-0077-d, S-0078-a, S-0078-b, S-0155-b, S-0165-a | **one bidder measured** (#117): Zeb reading every opposing auction turn under the field's known heuristic rule; the levels carry only the lowest legal bid | pilot: a deterministic opening carries 0.93 of 1.58 bits about strength. The read by a known method: K = +0.17 [+0.06, +0.29] marks/game, but a re-seeded field reads +0.12 [+0.02, +0.23], so K − Za = +0.05 [−0.09, +0.19] ([[auction-read-2026-10-16]]); against a seed-varied field K = +0.12 [+0.00, +0.23] ([[seed-varied-2026-10-16]]); the estimator is calibrated and both A/A misses were chance ([[estimator-calibration-2026-10-16]]); pooled, exploratory, +0.14 [+0.06, +0.22] | the book's level semantics need a bidder whose levels mean something (#134) |
| A5 | **The book's game and the match**: points to 250, a set rewarded, bidding that reads the score | S-0015-c, S-0017-c, S-0063-b, S-0114-a, S-0114-b, S-0047-a, S-0080-e, S-0110-a, S-0122-a, S-0129-b, S-0165-b, S-0170-a | **partial**: points to 250 exists (`points_to_win`); Zeb plans for the make, bids score-blind, never bids 84 (#115) | the book as Zeb's near-tie break wins 53.2% [51.1, 55.2] of games to 250, replicated | #115, #77 (marks-race utility), #130 slices 3–4 |
| A6 | **The field has bidding styles** (overbidders, wild bidders) | S-0080-c, S-0070-a, S-0155-a, S-0155-b, S-0114-b, S-0034-a, S-0184-a, S-0188-b, S-0190-a, S-0191-a | **two dimensions measured**: a field that overbids by a fixed k (#152), and the book's two named overbidders (#163) | `risk-budget` bidding −0.51 [−0.69, −0.33] marks/game against the style-free field; against the set-bidder (S-0155-a) and the habitual ten-count 35 (S-0070-a) it loses less, X_S = +0.15 [+0.02, +0.28] and X_T = +0.23 [+0.03, +0.41], the book's prediction, while still trailing the control (−0.36, −0.29); the ten-count field sets itself (control +1.30 [1.13, 1.45]). #152's fixed-k field: shrink +0.35 [−0.02, +0.71] (MDE 0.52). To 250 the bidding trails the control by about 7 points of share in both named fields ([[overbidders-2026-10-14]], [[deviant-field-2026-10-14]]) | a `risk-budget` that reads the field's style (#116, #130 slice 5); a set-bidder whose raises climb |
| A7 | **The reader is a bounded human**: the rules are a fast policy, not a replacement for enumeration | the whole book; S-0076-a, S-0078-d | **measured to 16 worlds** (#154): the fallback still plans | the book as tie-break wins its game (A5); as Zeb's near-tie break it neither raises nor costs Zeb at 16, 64 or 160 worlds (D16 −0.01 [−0.18, +0.17], MDE 0.25); inside its sets the cost falls from 1.10 [0.93, 1.27] at 160 worlds to 0.21 [−0.02, 0.44] at 16, the predicted direction, never crossing 0 | done: **#161**, the reader with no plan: the rules are worth +2.47 [2.28, 2.65] marks/game to the reader, about 61% of a 16-world plan |
| A8 | **A rule is applied with judgment inside its stated scope** | every rule file's header | **partial**: the Silas stack is literal | the census: 44% of the first head-to-head's loss was the encoding; repairs moved +0.47 [+0.33, +0.62] marks/game | #146 done; #153 |
| A9 | **The value of unpredictability is a response, not a fixed field** (an equilibrium claim) | S-0089-e, S-0081-a, S-0037-a, S-0154-a | **absent**: no mixed policies, no response map | none | #130 slice 5; #116; #149 is the fixed-learner first step |
| A10 | **The full declaration menu** (doubles as trumps, no trumps) | Ch. 9's 34 rows, Hands 23–25 | **plumbed, opt-in** (#155): `MenuBidder` (`menu.silas` in Silas, equal on every hand of four doubles) declares doubles or no trump by the book's Ch. 9 conditions, Hands 23–25 as the book does; `risk-budget` declares them where they are offered. Deployed Zeb's net plays pip trumps only, so at a menu table every seat plays doubles and no-trump hands with the random field (`DTNT=random`), **provisional** | on Lem's field-v3 net ([[ch9-at-the-table-2026-10-17]]): where the menu changes the declaration, −0.010 [−0.028, +0.008] marks/hand on the first seeds and −0.032 [−0.045, −0.019] on fresh ones; no trump +0.024 [+0.002, +0.045], then +0.009 [−0.006, +0.025] on fresh seeds (not replicated); doubles trump −0.067 [−0.100, −0.034], then −0.098 [−0.121, −0.075], heaviest in hands holding one of the 6-6, 5-5 and 4-4, and the book's own 6-6-and-5-5 cut separates the hands as well (#201). **Our menu's ceilings omit terms the book counts first**: trump losers, a held count off, trick points (#230). `ch9_doubles_loss` encodes the book's surface count for doubles trump (matching its tallies on Hands 23–25); no player reads it yet. Provisional random field: −0.045 overall, doubles −0.106. **The book's line played out** ([[book-line-play-out-2026-10-22]], #230 step C): every registered primary missed, each partly our instrument. Hand 24's line makes 30 in 0.945 [0.938, 0.952] of uniform deals with a helping partner, its key move worth +0.27. Hand 25's named run is all but the only way to be set (post hoc, 0.015 other), yet it comes in 0.186 of deals against setters playing the book's own defence. Hand 23 makes 0.571, its trick-one budget exceeded mainly by the book's own pounce onto the 5-5. Our stack leaves the book's line on 23 and 24 (#245) | #245, the stack repairs; then step C again with a true no-help partner; what the book's Hand 23 assumes about the setters' trick-one count |
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
  than the book-faithful rules and with self-play partners, and read null. **#150 seated the
  receiving half** ([[book-partners-read-2026-10-14]]): each book partner reads the other's plays
  by the stack and its calls by `bid-only-enough`, and uses the posterior where the book is
  silent or inside the set it allows. +0.10 [−0.06, +0.26] marks/game against an A/A of −0.05:
  null, every registered sign positive. Our reader cannot yet see the edge, because the book's
  receivers are rules whose guards would need the belief (gap 1). The largest information flow,
  the 84 defence (Hand 35's discards), waits on #128.
- **A3, measured for voids and for the count; the aim is unpriced** ([[fallible-field-2026-10-15]],
  [[count-lapse-2026-10-15]]). A field that forgets voids pays 0.32 marks/game at p = 0.4. A field
  that loses the count, in a seat's view because a mis-counting opponent is not a fiber of this
  planner, pays 0.71 [0.60, 0.82] at p = 0.2. Both times the planner collects the cost as readily
  as the book's rules do, and the book player's gap to Zeb does not move (ΔG(0.2) = +0.07 [−0.08,
  +0.22]). p* is beyond both grids. The book's ploys are aimed: a lure is played at a seat that
  might lose track. Neither team A player models the field's lapse, so the aim itself is unpriced
  until a player carries the lapse in its model of the other seat (#182). The honest claim stays a
  crossover, with p* reported as the book's scope.
- **A4 is cheap, and A1 needs it.** The book reads a 30, a 31 and a 35 as shapes. Read by the
  field's known method, the auction is worth K = +0.17 [+0.06, +0.29] marks/game, the book's
  sign ([[auction-read-2026-10-16]]). Two A/A arms missed their intervals, and the
  calibration that followed shows both were chance: the estimator covers as it says
  ([[estimator-calibration-2026-10-16]]). The two K estimates agree, pooled +0.14 [+0.06, +0.22]
  (exploratory), the book's sign. A learner
  that has seen a bidder's hands reads his next bid far more sharply (pilot below).
- **A5 is where restoring an assumption first paid.** Given its own game, the book's count play
  earned 53% of games as Zeb's tie-break. What remains is a utility that sees the score and an
  84 bid (#115).
- **A6, measured for the book's own overbidders** ([[overbidders-2026-10-14]]): against the
  set-bidder and the habitual ten-count 35 the book names, its bidding loses less ground than against
  the style-free field (X_S = +0.15 [+0.02, +0.28], X_T = +0.23 [+0.03, +0.41] marks/game), the
  direction the book predicts, and the ten-count field sets itself as the book says. The bidding
  still trails the control in every field, because `risk-budget` reads no style: the book's "pass
  and set them" assumes the reader knows who overbids (A1). #152's fixed-step field leaned the
  same way as a null ([[deviant-field-2026-10-14]]); its "level to 250" at k = 0 did not replicate
  on new seeds (38.8% [34.1, 43.5] here against 50.5% [43.8, 57.3]; the code reproduces byte for byte).
- **A7, held, now measured to 16 worlds** ([[book-small-n-2026-10-14]]): the book is a policy for a
  player who cannot enumerate. Its one measured win is in exactly that role, breaking ties for a
  planner. #154 predicted the book raises a bounded Zeb by more as n falls; what it saw is that the
  near-tie break is free at every n (D16 −0.01 [−0.18, +0.17]) and that confining the plan to the
  book's sets costs 1.10 marks/game at 160 worlds and 0.21 [−0.02, 0.44] at 16, the predicted
  direction. The sets cost a bounded player on defense, not on its own contracts. Every #154
  fallback still planned over sampled worlds; #161 seated the reader the book writes for, who plans
  over none ([[no-plan-2026-10-14]]): the rules are worth +2.47 [2.28, 2.65] marks/game to the reader and
  recover about 61% of what a 16-world plan is worth. Inside a 16-world plan they are worth −0.10;
  the book is worth the most exactly where its reader lives.
- **A8:** every residual disagreement between an encoding and the book is presumed to be the
  encoding's until a reason report finds no condition in the header that would reconcile them.
  The three rows filed as "the book against its rule" are re-read under that presumption (#153).
- **A9:** the full statement of "keep them honest" needs mixed policies and a response map
  (#130 slice 5). #149 is the cheap step before that: a fixed learner, not an equilibrium.
- **A10 is plumbing,** and the plumbing is in (#155): the Ch. 9 rows can fire at a menu table. Every number
  from one is provisional until a net plays decls 7–9 (#177), because until then every seat plays those
  hands with the random field. The arena's count prices Hands 23 and 24 below the book's 31 and 30
  (the book prices them by their play lines), so the menu bidder passes Hand 23. That is the scope
  of our count, not a verdict on the hand.
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

From the private coverage matrix (570 merged rows; 2026-10-14, after batch 2, the header
audit of #153; ids only):

| status | rows | what finishes it |
|---|---:|---|
| covered by a book rule, query, outcome or test file | 261 | measured at the table only for the 18 rules of the head-to-head; the rest are encoded, not seated |
| covered by an `odds/` file only | 24 | exact; done |
| covered by a gate only | 0 | the header audit (#153, 2026-10-14): 20 listed in their rule's header, one no rule's |
| encodable now | 6 | all five with gap 7 on a part and one with no record of Hand 25; re-read 2026-10-14 against #144 and #148, unchanged |
| blocked by a gap | 149 | #130 is the first-cited gap of about 112: slice 3 (deal measure, play-out; 19 rows priced against a Silas field), slice 4 (the arena ledger; about 98 rows priced, including the bid book's 60 and cluster E's ten against a named field, #152), slice 5 (mixed policies; cluster E's four equilibrium rows, A9). Then #128 (policy likelihood, 19), #134 (the auction, 15), #135 (aggregate, 8), #132 (another seat's knowledge, 1), gap 9 (1) |
| n/a for an encoding | 130 | not n/a for a lesson: the Ch. 6 style rows, the Ch. 11 table-talk rows, the Ch. 12 closing essay and the Ch. 14 profiles now name their assumption (A1–A11) in the matrix |

`silas/GAPS.md` § The book's silences records six choices the book does not make: which sure
trump leads, which setting lead, a count off on the setters' trick, the follower who is not last
and can take the trick, Hand 27's crisis, and the dead double against shed's lone tile. Each is
recorded in the GAPS file and none is guessed. Under the authority stance, each is resolved by
the book's own hands or left open, never by Zeb's preference.

## Recommended order

1. **#149, the regulars' table**: run; a null for a reading-only, fixed-style regular ([[regulars-table-2026-10-14]]). Its successor is a regular that responds (#116 / #130 slice 5).
2. ~~#150, book partners read each other~~: done ([[book-partners-read-2026-10-14]]), null at
   ±0.16; the next step on A2 is a guard that reads (#128).
3. **#153, the header audit and the last encodable rows**: done 2026-10-14 (branch
   `book/header-audit`). Of the 21 gate-only rows, 19 were their rule's play and are listed;
   one was not (the setter's five at Hand 27, S-0126-a: `setter-leads.silas` had read p. 47's
   "low tile that calls for a five" as any tile calling any five, and led nothing; repaired
   from the book's own words); one is no rule's (the partner's throw with no count). Of the
   three "book against its rule" rows, one is reconciled by the book's own stated reason
   (Hand 29's spare ace, now `low-from-a-pair.silas`'s rule), one rests on a premise the book's
   own record contradicts (Hand 35 trick 4), and one stays open with a candidate condition
   and the measurement that would decide it (Hand 28 trick 2, a reason report from the right
   setter's cell). The six encodable rows wait on gap 7 and a record of Hand 25. What is left
   of the encoding is the 149 gapped rows.
4. **#154, the book at small n**: done 2026-10-14, a null at the tie-break and a signed filter trend; **#161** is its n = 0 arm.
5. **#152, the named deviant field**: done 2026-10-14, the cost shrinks the book's way as a null and
   vanishes in the book's game; **#163** is the book's kind of overbidder, powered.
6. **#151, the fallible tracker**: medium, and it reports a crossover rather than a verdict.
7. **#130 slices 3–4**: the largest block of rows (about 117 priced).
8. **#117 / #134** (the auction read), then **#128** (policy likelihood in Silas).
9. **#155, the declarations**: plumbing that unlocks Ch. 9.
10. **#130 slice 5 / #116**: mixed policies and response maps, the full home of A1 and A9.

## Links

[[book-authority]] · [[w42]] · [[book-study-2026-10-12]] · [[book-head-to-head-2026-10-13]] ·
[[book-investigation-2026-10-14]] · [[book-reads-by-the-book-2026-10-14]] ·
[[reads-plays-soft-2026-10-14]] · [[regulars-table-2026-10-14]] · [[book-partners-read-2026-10-14]] · [[exploiter-endgames-2026-10-14]] · [[silas]] ·
[[scenario-player]] · [[strategy-fusion-fence]]
