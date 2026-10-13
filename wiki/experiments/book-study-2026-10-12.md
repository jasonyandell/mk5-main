---
title: The book, read with the honest tools — Winning 42's rules priced on the pour and its conventions graded in the arena (2026-10-12)
kind: experiment
first_seen: 2026-10-12
last_updated: 2026-10-12
status: complete
---

## Summary

Jason, 2026-10-12: "my plan is not to ADOPT these conventions. it is to study them and,
for the ones that hold, try to find a mathematical reason for that and adopt THAT."
Roberson's *Winning 42* ([[w42]]) was read once end to end by one Fable session, which
wrote a ranked list of 26 candidates (7 conventions); Opus agents then priced every rule on
the 250,000-deal Zeb pour, checked the bid book against Zeb's forecast, recomputed the
book's odds by exact count, and graded the conventions in the arena with the overlay built
the same day ([[scenario-player]] § A convention overlay). The oracle-era verdicts
([[w42-book-validation-campaign]]) were not reused: they established no strategic claim
from the book (the "supported" rows are arithmetic, definitions, negative controls, a
tautology and one null).

**The picture in one line.** Of 21 gated rules, Zeb's plan sides with the book on 9, shows
no relation to it on 3, prices against it on 5 (robustly), and agrees weakly on 4; overall
Zeb's pick is the book's move at 52% of gated positions against a 37% random floor
(chance-corrected 0.23), and each strong player tested misses the book's move at about 45%
of them. The bid book's 25 of 30 is two hands better than "always bid"; the conventions
buy nothing measurable; three odds figures are corrected by exact count. No reason from the
book was extracted or adopted: Zeb today prices a move, it does not explain one (#124).

**What the measurements show**

- *Price is not agreement.* A rule's price is 0 whenever any book tile is among Zeb's exactly
  tied best, so it cannot tell "Zeb prefers the book's move" from "Zeb does not care". Over
  the face-off sample the 62% at price 0 splits into 43% Zeb strictly prefers a book tile,
  20% Zeb is indifferent between a book and a non-book tile (including every dead row), and
  38% Zeb strictly prefers a non-book tile (§ The face-off, per-gate table).
- *Zeb's plan sides with the book* (chance-corrected ≥ 0.44): the bidder's top-trump
  donation lead (C1a), the partner's count on a top-trump lead and sluff otherwise (C1b),
  no count on the bidder's double (C8), protect count (C13b), count-calling setter leads
  (C15), trump the bidder's double from the left (C22), the partner leads doubles (C23),
  and with four trumps holding the third, lead the double (C12A).
- *No relation to the book* (chance-corrected ≤ 0.07): the partner never leads trump (C9;
  random already scores 70% there), low from a pair (C13c, a two-tile choice at a coin
  flip), lead a singleton (C13d).
- *The gates are not the book's rules.* Written as [[silas]] policies and replayed on the
  book's own worked deals (`silas/examples/book/`, `silas/tests/book.rs`), at least 10 of
  the 21 Python gates part from the book on its own positions: the pounce (C7) also fires
  when the setter can follow (Hand 28); the donation lead (C1a) turns on what the partner
  can know; the trick-1 off lead (C10) is permitted, not prescribed (Hand 14 leads trump);
  four trumps missing the 2nd and 3rd (C12B) is played with three trumps in the book's
  examples; and C8, C11, C13b, C13d, C14, C15, C22 differ too. The handoff's book reading
  (C2) is false in the book's own Hand 30. Every per-rule verdict on this page prices the
  gate, not the book's rule, until the gate is re-checked (#123).
- *Zeb's plan prices against the book* (−0.05 pmake per decision or worse, surviving a
  four-times-stronger re-plan): biggest count first (C3), the pounce (C7), the trick-1 off
  lead (C10), four trumps missing the 2nd and 3rd, lead low (C12B), the six-five bait
  (C21). These are prices, not contradictions: no gate was checked against the book's own
  examples of the rule (#123) and none has a hand-checkable counterexample deal (#121).
- *Weak agreement* (0.19–0.30): the high-trump handoff (C2), double ahead of the off (C11,
  mostly indifference: 33 for / 39 indifferent / 28 against), void in follow position
  (C13a), crisis cash-the-double (C14).
- *Small prices are below Zeb's own noise.* Zeb's pick costs −0.005 to −0.016 pmake per
  decision when re-scored by a stronger copy of itself, so a mean price above about −0.015
  is unreadable per decision. Converted to marks per hand the page's own way (price × firing
  rate × 2), several such rules cost 0.02–0.07 marks/hand, larger than the +0.026 the
  posterior sampler shipped at; the conversion is calibrated at one point only (C2).
- *Bid-only-enough is arithmetic.* Under one-mark scoring every bid from 30 to 41 pays one
  mark, so a higher level buys nothing but the auction (32–34 points fall in 3% of made
  hands). Zeb's own `chooseBid` lands on 33 or 34 on 3 of the 30 book hands (#115).
- *"42 not 84 at one mark from the game"* is a correctness gate (the make condition is
  identical, the 84 only doubles the loss); Zeb's bidder never bids 84 today (#115).
- *Doubles-trump and no-trump as declarations:* the forecast ranks them where the book does
  on 1 hand of 3 — not a test.
- *The bid as a sentence* (31 means "one ten-count off") is a points artifact under marks:
  the sentence costs pmake(level) − pmake(30).
- *"Never fail to bid 84 with the double ahead of the off."* The book's showcase hand
  (p. 113: trumps 5-5 5-4 5-2 5-0 with the 6-5 out) forecasts p84 = 0.44 against the 0.73
  an 84 needs to beat a one-mark bid; counting alone caps it at 0.653, because an opponent
  holds the 6-5 with another five with probability 33/95. This is the study's one worked
  explanation: a counting fact with two drawn deals.
- *Three odds figures.* The 27 placements of three missing trumps are not equiprobable
  (a 1–1–1 split weighs 0.043 each, a 2–1 split 0.037, a 3–0 split 0.026): the book's 37%
  is 33/95 = 0.347 and its 52% is 47/95 = 0.495, a coin flip the bidder's way. The "80%"
  that an opponent cannot follow the first doubles lead is 23/30 = 0.767. "Partner holds two
  or more doubles 58% of the time" is the unconditional figure; given the bidder's own two
  doubles it is 0.557, given three 0.407. The appendix's three-trump and two-high-trump
  sections (pp. 183–184) were missing from the archive's OCR and were not checked.
- *The pounce* (play count before partner can) costs −0.076 pmake as a one-seat rule and
  more, not less, when the count is protected. Its "you'd have lost it anyway" is true
  descriptively (held-back count went to the bidder 62%) and selection-confounded.

**The conventions, graded.** Each was wired into both partners of the bidding team and
played against a plain line on the same 22,000 deals: sender only (S), reader only (R),
both (SR). A convention is confirmed iff SR − plain > 0 with the interval clear of zero
while S ≤ 0 and R ≤ 0. None was.

| convention | S | R | SR | SR − S | SR − R |
|---|---|---|---|---|---|
| C1 donation (count only on a guaranteed trick) | −0.0015 [−0.0029, +0.0000] | −0.0011 [−0.0028, +0.0006] | −0.0003 [−0.0023, +0.0017] | +0.0012 [−0.0003, +0.0027] | +0.0008 [−0.0008, +0.0024] |
| C2 the table's high-trump handoff | −0.0129 [−0.0163, −0.0095] | −0.0116 [−0.0149, −0.0083] | −0.0131 [−0.0170, −0.0091] | −0.0002 [−0.0022, +0.0018] | −0.0015 [−0.0049, +0.0020] |
| C2 handoff, the book's reading (a top trump was partner's only one) | same sender | −0.0087 [−0.0112, −0.0062] | −0.0313 [−0.0358, −0.0267] | | −0.0225 [−0.0264, −0.0187] |
| C3 biggest count first | priced only | −0.0016 [−0.0032, −0.0000] | −0.0022 [−0.0042, −0.0002] | | −0.0005 [−0.0022, +0.0011] |

marks/hand for the bidding team, 95% intervals, 22,000 paired deals per arm, bid 30, two
Zebs against two deterministic field nets.

- **The donation is null** in every arm. SR beats S and R by about +0.001 each, the
  convention's signature shape, at a size inside a tenth of a mark per hundred hands. The
  sender fired in 3,544 deals; the reader drew filtered worlds at 135,000 decisions.
- **The handoff is a priced signal that buys nothing here.** The sender costs −0.013
  marks/hand (the pour predicted −0.016: −0.039 pmake per firing × 19% of hands × 2). The
  reading on top of the sender recovers 0.000 ± 0.002. All of the cost sits where the bidder
  holds three trumps or fewer (−0.0144 [−0.0182, −0.0106], n = 19,706), exactly the stratum
  the entry predicted the information would pay in; at four trumps the gate rarely fires
  (flips 2/2).
- **The book's reading of the handoff costs 3 marks per hundred hands** and is the one
  mathematical lesson of the arena: *a reading must be the exact inverse of the sender's
  gate.* The sender hands off whenever he holds the top trump and another; the book's reader
  deletes every world where he holds another; so at every handoff from a two-trump partner
  the reader's filter keeps only false worlds (7,101 fallbacks of 198,025 decisions) and
  plans on them. Reading a signal nobody sends (every R arm) is the same error smaller:
  −0.009 to −0.012 marks/hand for the two handoff readings against a plain partner.
- **Biggest count first** is null-to-negative as a reading (−0.0016) and the same with its
  sender (−0.0022); the sender's own price on the pour is −0.064 per decision.

The ceiling was already known: on three-trick endgames conventions recover about 6% of
what partners would want to say ([[endgames-o2-vs-o1-2026-10-12]]). These verdicts are for
bid 30 against nets; the book's conventions are for partners who follow the book.

## Setup

**Wave 0, the reader.** One Fable 5.1 session, the whole OCR text in one context, two
hours, no code run. Output: 26 candidates in one shape (name, source page, type, the claim
in Walt units, the candidate reason, the grading experiment, comfort or strength, rank),
plus the sections *Already Zeb*, *Cannot hold* and *The table's convention* (Jason's
handoff, stated as a joint deviation). The OCR dropped every second screenshot of a
two-shot page, so worked hands and whole sections were never read (#123); pages 8, 36, 42,
57–58, 74, 85–86, 90 and 122 have no capture. The *Already Zeb* entries (pull trumps first
among them) were asserted by the reader and never checked.

**Wave 1, prices and counts.** Four Opus agents in worktrees, all under a minute per run:

- *Pour queries, bidder side and setter side.* Every rule's gate written as a predicate in
  the mover's chair (seat role, own hand, public record) over the 250,000-deal Zeb pour
  (4,343,426 positions, utility `make`, bid 30, seat 0 bids). The pour player is
  `k0,n512` over the 5k/50k field nets at code before the #110 and #112 fixes — not the
  deployed Zeb, with which it shares the exact-best set at 61% and the pick at 65% of live
  gated positions. Price = max pmake over the book's allowed tiles − max over all legal
  tiles (≤ 0; a setter's value is 1 − q).
- *The bid book.* 30 hands transcribed from the book's figures (two full deals partition all
  28 tiles, which checks the reads; three hands reconstructed from the play-out text); the
  forecast Zeb exposes (`forecast`, pmake at every target from the 30 plan, nine
  declarations) and `chooseBid` at cutoff 0.8; the 84 hands planned for the 42 contract,
  since the 42 column read off the 30 plan is low by up to 0.17.
- *The odds appendix.* Every placement of the marked tiles enumerated with exact
  multinomial weights; a 400,000-deal Monte Carlo cross-check agrees to three places.

**Wave 2, the arena.** `scratch/book/wave2/table2.py`: two Zebs
(`walt:k0,n512,F=margin_d250k_h128x64,f=16`, seats 0 and 2, seat 0 bids 30 and declares its
best trump) against two deterministic field nets (seats 1 and 3); 22,000 deals, seeds
1,000,000 up; the plain line is a function of the seed (deterministic planner, one
thread) so it is played once and every arm is paired against it deal for deal; a made 30
is +1 for the team, a set −1. Arms by spec letter `c=<name>:S|R|SR`. Eleven runs of about
15 minutes each (~2 h 50 m wall, 16 workers), no reader fallback in any SR arm except the
singleton reading's, which is the finding.

## Wave 1 numbers

Prices are pmake for the mover's team per gated decision, mean over gated positions;
*price 0* is the share where some book tile is among Zeb's exactly tied best (Zeb prefers
it or is indifferent); *Zeb* is how often the pour player's chosen tile is the book's. The
for / indifferent / against split and the chance-corrected agreement are in § The face-off.

| candidate | side | n | price | price 0 | Zeb | reading |
|---|---|---|---|---|---|---|
| C1a bidder leads the top trump for the donation | bidder | 10,298 | −0.009 | 86% | 58% | sides with the book; in 74% of gates the bidder does not hold the top trump |
| C1b partner sluffs non-count on a non-top trump lead | partner | 30,528 | −0.003 | 93% | 85% | sides with the book; random is 76% here |
| C1b partner gives the largest count on a top-trump lead | partner | 21,246 | −0.018 | 69% | 58% | sides with the book, a third against |
| C2 high-trump handoff | partner | 50,444 | −0.039 | 55% | 48% | weak; worst (−0.069) when the bidder's lead was itself the top trump |
| C3 biggest count first | partner | 25,985 | −0.064 | 49% | 37% | prices against; Zeb gives the smaller count 7,888 times, the largest 4,146 |
| C7 pounce | setter | 111,861 | −0.076 | 38% | 27% | prices against as a one-seat rule; −0.089 with count protected, −0.053 unprotected |
| C8 count on a double means void | setter | 54,466 | −0.011 | 81% | 75% | sides with the book; the reading is right 70% of the time in Zeb self-play (no partner sends it) |
| C9 the partner never leads trump | partner | 139,433 | −0.016 | 78% | 70% | no relation (random 70%); the partner leads trump 33% of the time he holds it |
| C10 lead the off first, trick 1, with one off and ≥ 4 trumps | bidder | 8,929 | −0.068 | 18% | 17% | prices against on trick 1; the direction holds (off-lead share rises 14% → 29% → 43% with 4, 5, 6 trumps) |
| C11 double ahead of the off, lead the double first | bidder | 24,526 | −0.015 | 71% | 45% | weak; mostly indifference |
| C12 four trumps, lead the double (holding the 3rd) | bidder | 4,113 | −0.014 | 55% | 54% | sides with the book's double over its low lead (−0.125); missing the 2nd and 3rd, the book's low lead prices against (−0.096) |
| C13 setter: void in follow position, protect count, low from a pair, lead a singleton | setter | 39,947–349,682 | −0.009 to −0.024 | 65–83% | 45–76% | protect count sides with the book; void in follow weak; low from a pair and lead a singleton no relation |
| C14 crisis: cash the double with most unplayed | bidder | 38,779 | −0.028 | 54% | 46% | weak; Zeb leans the book's way (the trump lead is −0.052) |
| C15 setter leads that call for count | setter | 318,488 | −0.014 | 79% | 50% | sides with the book, 40% indifferent; the one-point exception is real (quiet lead strictly best 38% there) |
| C21 the six-five bait lead | bidder | 1,010 | −0.053 | 17% | 11% | prices against |
| C22 trump the bidder's double from the left | setter | 45,115 | −0.009 | 83% | 66% | sides with the book (left seat 91% price 0; last seat 75%) |
| C23 the partner leads doubles | partner | 203,133 | −0.013 | 80% | 70% | sides with the book |

**The bid book.** Bid or pass agrees on 25 of 30 hands; the book bids 23 of the 30, so
"always bid" scores 23. Where both bid, the level agrees on 8 of 11, Zeb a notch higher on
one and lower on two. On the book's six 84 hands `chooseBid` never bids 84 (level agreement
0 of 6). The five bid-or-pass disagreements: the book passes a 0.84 hand on its
16-points-on-one-trick rule; it bids a reconstructed 0.80 hand on the cutoff; it calls two
0.67–0.72 hands easy (one is set in its own play-out); and its showcase no-doubles hand
forecasts 0.61 at 30. The at-risk ≤ 12 rule and its two-off paradox hold in direction (a
two-off hand at 0.91 beats one-off hands at 0.84 and 0.44). On the 84 hands an 84 beats a
one-mark bid iff p84 > (2·p30 + 1)/4: two lay-downs and one narrow yes, three prefer one
mark.

## What this means for Zeb

1. **Nothing was adopted, and no reason was found.** Zeb can price a move and show its
   per-world outcomes; it cannot yet say why (no plan trace, no reason vocabulary, no noise
   bar, no objective at ties). The reason report is #124; the book's own examples as
   positions, the test set for it, are #123.
2. **A signal's price is measured, and its information was worth zero here.** The handoff
   pays −0.013 marks/hand to tell the bidder where the top trump is not, and the bidder's
   plan, contingent over both placements already, cashes none of it. This is Jason's prior
   ("very few moves are free") with a number.
3. **The reading must be the sender's gate inverted.** A likelihood that is not the one the
   partner's policy induces costs more than the signal could pay (−0.031 for the book's
   singleton reading against the table's sender; −0.009 to −0.012 for any reading against a
   plain partner). This is the Bayes-consistency the posterior sampler (#111) was built on,
   seen from the convention side.
4. **The comfort claim is unmeasured.** "They can make better decisions" may mean a smaller
   cell at the same marks; the overlay can log the bidder's worlds remaining at trick 2 with
   and without the reading (#118).
5. **Ungraded conventions** need machinery the player lacks: the bid as a signal (C4) needs
   the auction read (#117); the 84-defense discard signal (C5) needs a setter team and the
   84 contract; the count-on-a-double void reading (C8) is a setter reading, and the arena
   seats only the bidding team's Zebs. Every rule was evaluated at bid 30 against self-play
   partners who never follow the book.

## The face-off (#119, 2026-10-12 night)

Is "Zeb's tile is the book's tile at 60–90% of gates" information about Zeb, or are the
gates easy? The same 21 gates, about 1,000 sampled positions each from the pour (20,977
rows over 20,866 positions, gates equally weighted), four more players asked for a move at
each (`scratch/faceoff/`, the research archive's `book/faceoff/`): random legal play; the
field net alone (trained on these same pour positions); jud, the oracle-distilled student;
and E[Q] n=10 (`lens:ev`, ten sampled worlds scored by the learned Q network, qMAE 0.94
points, averaged). Book agreement is "the pick is in the book's allowed set"; regret is
under the pour player's own q, so it measures closeness to that player, not strength.

| player | book agreement [95%] | chance-corrected | picks Zeb's tile | regret under Zeb's q |
|---|---|---|---|---|
| random | 36.7% [36.1, 37.4] | 0 | 27.5% | −0.059 |
| jud | 42.1% [41.5, 42.7] | 0.08 | 28.3% | −0.052 |
| Zeb, exact ties broken at random | 51.5% [51.0, 52.0] | 0.23 | 100% | 0 |
| E[Q] n=10 | 56.4% [55.8, 57.0] | 0.31 | 50.6% | −0.014 |
| field net alone | 57.8% [57.3, 58.4] | 0.33 | 54.4% | −0.010 |

A gated position has 4.55 legal moves and a book set of 1.46, so random agreement is 37%.
Against that floor every player's agreement is weak (chance-corrected 0.23–0.33), and at
about 45% of gated positions the strong players do not play the book's move.

Per gate, on live positions (`scratch/book-audit/kappa.py`; "for / indiff / against" is
Zeb's exact-best set: book tiles only / book and non-book tiles tied / non-book only):

| gate | legal | random | Zeb | chance-corr. | E[Q] | chance-corr. | for / indiff / against |
|---|---|---|---|---|---|---|---|
| C1a | 4.55 | 21% | 76% | +0.69 | 71% | +0.64 | 54 / 33 / 13 |
| C1b-top | 4.72 | 24% | 60% | +0.48 | 73% | +0.64 | 50 / 18 / 32 |
| C1b-nontop | 5.25 | 76% | 90% | +0.57 | 89% | +0.53 | 71 / 21 / 7 |
| C2 | 2.39 | 44% | 54% | +0.19 | 59% | +0.27 | 49 / 10 / 41 |
| C3 | 5.47 | 25% | 42% | +0.23 | 41% | +0.21 | 38 / 13 / 50 |
| C7 | 4.66 | 27% | 24% | −0.05 | 24% | −0.05 | 20 / 18 / 62 |
| C8 | 2.28 | 55% | 79% | +0.54 | 80% | +0.56 | 71 / 10 / 19 |
| C9 | 4.20 | 70% | 72% | +0.07 | 73% | +0.10 | 55 / 24 / 21 |
| C10 | 7.00 | 14% | 16% | +0.02 | 15% | +0.01 | 15 / 2 / 83 |
| C11 | 3.55 | 32% | 52% | +0.30 | 60% | +0.41 | 33 / 39 / 28 |
| C12A | 7.00 | 14% | 52% | +0.44 | 60% | +0.53 | 51 / 2 / 46 |
| C12B | 7.00 | 14% | 9% | −0.06 | 4% | −0.12 | 8 / 1 / 91 |
| C13a | 4.47 | 36% | 51% | +0.23 | 63% | +0.43 | 48 / 20 / 33 |
| C13b | 2.00 | 50% | 81% | +0.61 | 82% | +0.64 | 71 / 12 / 17 |
| C13c | 2.00 | 50% | 52% | +0.05 | 52% | +0.04 | 36 / 32 / 32 |
| C13d | 4.02 | 37% | 38% | +0.01 | 40% | +0.05 | 26 / 43 / 31 |
| C14 | 4.62 | 25% | 43% | +0.24 | 63% | +0.51 | 36 / 17 / 47 |
| C15 | 4.38 | 33% | 64% | +0.46 | 65% | +0.47 | 40 / 40 / 20 |
| C21 | 5.93 | 17% | 12% | −0.07 | 6% | −0.13 | 11 / 6 / 83 |
| C22 | 5.96 | 41% | 74% | +0.56 | 94% | +0.89 | 57 / 26 / 16 |
| C23 | 4.14 | 35% | 73% | +0.58 | 79% | +0.68 | 58 / 21 / 20 |

- **E[Q]'s pick lands on the book's move more often than Zeb's.** Same rows, same tie
  treatment: Zeb 52.5%, E[Q] 56.3%, random 36.2% (live and revived rows, n = 20,265); on
  live rows only, Zeb 52.0% and E[Q] 55.7%. Where E[Q]'s pick is outside Zeb's best set
  (41% of rows) the book holds E[Q]'s tile 30%, Zeb's 22%, both 15%, neither 33%. Neither
  player's agreement favors it as a reader of the book; regret separates the two strong
  players from jud and random only as closeness to the pour player.
- **Dead rows are not positions.** 17.6% of gated rows are dead (every tile ties). 57% of
  them are decided by the banked score — under the book's own Chapter 10 the hand is over
  and the position never occurs — and should be dropped; the rest are decided only in Zeb's
  512 worlds and are Zeb indifferent at `make` (the plan stops at `_settled`). Re-planning
  them at the nearest live bid revived 2,972, but 2,212 of those (74%) at a bid below 30,
  which is not a contract, and by the deployed Zeb rather than the pour player; that
  comparison is withdrawn.
- **Where the gap lives: Zeb's indifference.** `make` is 0/1 over 512 worlds, so exact ties
  are common; the book has a rule at those spots and E[Q], playing for points, has a
  preference. A tie is not a reason; a secondary objective at ties (make, then margin) is the
  principled candidate; measure first (#120).
- **The large prices survive a stronger plan; "Zeb's pick" does not.** 300 gated positions
  per rule re-planned at `k1,n2048,m8,f=16` (four times the worlds, a modeled partner; the
  referee, 3.1 h):

  | rule | book price, pour | book price, strong Zeb | negative price kept | strong tile = pour tile |
  |---|---|---|---|---|
  | C3 biggest count first | −0.052 | −0.051 ± 0.010 | 93% of the 142 negative | 68% |
  | C7 pounce | −0.080 | −0.079 ± 0.014 | 91% of 187 | 66% |
  | C10 trick-one off lead | −0.068 | −0.067 ± 0.006 | 97% of 254 | 29% |
  | C21 six-five bait | −0.059 | −0.052 ± 0.005 | 96% of 257 | 51% |

  The prices are stable; the pour player's pick matches the stronger plan's at 29–68% by
  rule. The pour player's own
  pick costs −0.005 to −0.016 under the stronger plan: that is the noise floor for a price.
  By the standard Jason set the same night, a price is not a case against the book: siding
  against a rule needs a hand-checkable counterexample deal (as the 84 hand got), and a
  gate checked against the book's own examples (#121, #123). C12B (−0.096) belongs on that
  list too.
- **"Strong but wrong" is not visible at these gates.** PIMC plays the counting rules about
  as the book does. Zeb's +0.74 marks/game over `lens:ev` was never made here: the project
  credits it to the `make` utility and the partner rung, and rung 0 alone is level with E[Q].
- **What agreement should mean.** Tile agreement answers "does the player land where the book
  points"; the book is about reasons. Explaining the book's move and agreeing with it are
  separate axes: at a deviation the question is whether Zeb's per-world outcomes state the
  book's reason, a counting fact the book missed, or a utility difference (marks vs points) —
  or whether the deviation belongs to the field model of the other seats, sampling noise, the
  bid-30 contract, the plan stopping at `_settled`, or the gate's translation of the book.
  A tie is none of these. The reason report is #124; the repaired book as a page is #121.

## Data

The measurements (scripts, pour queries, forecasts, per-deal arena output, the face-off's
per-position picks, the reports) are public in `jasonyandell/texas-42-zeb-research-2026-10`
under `book/wave1/`, `book/wave2/` and `book/faceoff/` ([publish-data-to-huggingface](../playbooks/publish-data-to-huggingface.md)).
The baselines, chance-corrected agreement, dead-row split and like-for-like comparison are
recomputed from `scratch/faceoff/out/` by `scratch/book-audit/stats.py`, `kappa.py`,
`dead_decided.py` and `live_at30.py`. The book's text, page images and transcribed hands
are not published.

## Links

[[w42]] · [[scenario-player]] · [[endgames-o2-vs-o1-2026-10-12]] ·
[[research-night-2026-10-12]] · [[the-wall]] · [[w42-book-validation-campaign]] ·
[[w42-book-claim-synthesis-and-ai-directions]] · [[w42-book-second-pass]] ·
[[publish-data-to-huggingface]]
