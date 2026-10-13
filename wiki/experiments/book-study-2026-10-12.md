---
title: The book, read with the honest tools — Winning 42's rules priced on the pour and its conventions graded in the arena (2026-10-12)
kind: experiment
first_seen: 2026-10-12
last_updated: 2026-10-13
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

**The picture in one line.** Of the book's 14 rules as the book means them, Zeb's plan
sides with 3, agrees weakly with 8, shows no relation to 3 and prices against none; Zeb's
pick is the book's move at 54% of the positions each rule governs against a 40% random floor
(chance-corrected 0.23), no higher than on the 21 gates first priced (52% against 37% on
the face-off's sample, 0.23), which over the whole pour were the book's rule for only 3
(§ The book's rules as the book means them). Each strong player tested misses the gates'
move at about 45% of gated positions. The bid book's 25 of 30 is two hands better than
"always bid"; the conventions buy nothing measurable; three odds figures are corrected by
exact count. No reason from the book was adopted: the reason report (#124) explains Zeb's
tile where it leaves the book's, and at the two rules first encoded against the book, what
it named was a condition the book states and the encoding lacked.

**What the measurements show**

- *Price is not agreement.* A rule's price is 0 whenever any book tile is among Zeb's exactly
  tied best, so it cannot tell "Zeb prefers the book's move" from "Zeb does not care". Over
  the face-off sample the 62% at price 0 splits into 43% Zeb strictly prefers a book tile,
  20% Zeb is indifferent between a book and a non-book tile (including every dead row), and
  38% Zeb strictly prefers a non-book tile (§ The face-off, per-gate table).
- *Zeb's plan sides with the book* (chance-corrected ≥ 0.44 over the whole pour) on three
  of its rules as the book means them: the partner's leads (+0.48, a double or a walker,
  then by the count each lead calls), protect the count (+0.45), and win low and keep the
  double (+0.48); and inside weaker rules on the bidder's top-trump leads, cashing winners
  with the off's double next-to-last, no count on a lost trick, the safe trick's five, and
  the setter's double lead when one point sets. As the gates were translated it sided with
  C1a, C1b, C8, C13b, C22, C23 and C12A.
- *No relation to the book* (chance-corrected ≤ 0.07): low from a pair (C13c, a two-tile
  choice at a coin flip, the same as the book's rule) and the trump set (+0.06), with the
  condition the book states; the dead double (+0.08) falls between this band and the weak
  one (≥ 0.11), near chance; as gates also the partner never
  leads trump (C9, which the book's partner-leads rule replaces) and lead a singleton (C13d).
- *The gates are not the book's rules.* Written as [[silas]] policies and replayed on the
  book's own worked deals (`silas/examples/book/`, `silas/tests/book.rs`), at least 10 of
  the 21 Python gates part from the book on its own positions: the pounce (C7) also fires
  when the setter can follow (Hand 28); the donation lead (C1a) turns on what the partner
  can know; the trick-1 off lead (C10) is permitted, not prescribed (Hand 14 leads trump);
  four trumps missing the 2nd and 3rd (C12B) is played with three trumps in the book's
  examples; and C8, C11, C13b, C13d, C14, C15, C22 differ too. The handoff's book reading
  (C2) is false in the book's own Hand 30. Over the whole pour the Silas encodings fire
  exactly where 17 of the 21 gates fire, with the same allowed tiles, so those verdicts
  price the gate, not the book's rule (#123); C22 is priced as the book's rule, C15 with
  its one-point exception, C14 without its "most unplayed" (§ The book's rules as Silas
  policies). Against the 14 book-faithful rules over the whole pour, only C2, C12A and C13c
  fire where their rule does with the same tiles; C1a's gate and the book's donation lead
  never fire together (§ The book's rules as the book means them).
- *Zeb's plan prices against the book* (−0.05 pmake per decision or worse): none of its
  rules as the book means them; inside them, the trump set's feed (−0.097, 76% against, even
  once the holder may yet trump), the pounce proper (a ten at once −0.060), the trick-1 off
  as a permission (−0.055), double-up lead low (−0.058), the six-five bait (−0.058), dumping
  a bare off (−0.064), and the two-trump straight-off lead (91% against). As gates,
  surviving a four-times-stronger re-plan: C3, C7, C10, C12B, C21. These are prices, not
  contradictions: none has a hand-checkable counterexample deal (#121).
- *Weak agreement* (0.11–0.37 over the whole pour): eight of the book's rules, among them
  the bidder's plan (+0.33), the crisis (+0.30), donate when safe (+0.25), the pounce with
  its guard (+0.22), the first lead and the setter's leads (+0.18 each), and the top-trump
  handoff (C2, +0.11: 46 for / 8 indifferent / 46 against, the book's worked hand and the
  gate alike). As gates also C11, C13a, C14 and C15 (+0.37; +0.46 on the face-off sample,
  which left out the dead rows).
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
- *The book's odds, exactly.* Every probability the book prints (32 claims) is a [[silas]]
  query computed exactly against the book's own model (each missing tile equally likely in
  each of the three other hands) and against the true deal (`silas/examples/book/odds/`).
  The equal-placement model is the recurring error: four trumps missing the 2nd-highest is
  33/95 = 0.347, not 37%; missing the 2nd and 3rd is 47/95 = 0.495, a coin flip the
  bidder's way, not 52%; three trumps, 388/855 = 0.454, not 47%; the doubles-lead "80%"
  is 23/30. The book's arithmetic also slips inside its own model: 243 placements, not 225;
  partner holds the 2nd-highest 27/81 = 1/3, not 21/81; partner holds three or more 51/243,
  not 45; six doubles is about 1/80 percent, not 1/50; two to four doubles is 0.577 (the
  printed "at least 36 percent" is the two-double row alone). The book prints no figure for
  partner's doubles; exactly, partner holds two or more 0.581 unconditionally, 0.557 given
  the bidder's own two, 0.407 given three. Whole-deal frequencies need a deal view Silas
  lacks (`silas/GAPS.md`) and are enumerated over all 1,184,040 hands instead.
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
| C15 setter leads that call for count | setter | 318,488 | −0.014 | 79% | 50% | agrees weakly (+0.37 over the whole pour); the one-point exception is real (quiet lead strictly best 38% there) |
| C21 the six-five bait lead | bidder | 1,010 | −0.053 | 17% | 11% | prices against |
| C22 trump the bidder's double from the left | setter | 45,115 | −0.009 | 83% | 66% | the gate sides with the book (left seat 91% price 0; last seat 75%); the book's rule fires at 5% of these and sides more (§ The book's rules as Silas policies) |
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

1. **Nothing was adopted.** Zeb prices a move and, through the reason report (#124: the
   plan's line, separating Silas facts, a noise bar), says why its tile beats another in
   its own worlds; it has no objective at ties (#120). The book's own examples as
   positions are #123.
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

## The book's rules as Silas policies (2026-10-13)

At positions where the book's rule, written as a [[silas]] policy, applies, how does Zeb's
plan value the tiles it allows? And how far is the policy's firing set from the gate's?

**Setup.** The 20 policies of `silas/examples/book/` (21 rules; C1b's two branches share
one file), compiled through `scenario_core`'s Silas bindings, asked at every one of the
4,343,426 planned positions of the 250,000-deal pour (the wave-1 population; the face-off's
20,977 rows are inside it with the same gate and book set). Each frame is built from the
record (declarer seat 0, bid 30, attributed plays); its `own-legal` set equals the pour's
legal tiles at every position. `SilasPolicy` binds no decision in Python, and a decision
returns one tile, not the allowed set; so each policy is compiled by `SilasPolicy` (which
proves it reads only the viewer) and its rules' guards, verbatim, are run with `SilasQuery`
in rule order, as `CompiledPolicy::decide` runs them: the first rule whose guard returns a
legal tile fires, and those legal tiles are its allowed set. Cost: 66 s per 50,000-deal
chunk, five chunks in parallel, about 5.5 CPU-minutes (`scratch/silas-reprice/`).

*Zeb* here is the pour player, `k0,n512`, utility `make`, bid 30, over the field net
`field_make_d5000_v2` (seeds 100,001 up, code `31ecd76d`) and `v2/margin_d50k_h128x64`
(seeds 200,001 up, code `b30603ad`), before the #110 and #112 fixes. Every q is a sampled
estimate over 512 worlds.

**Gate fidelity.** Per rule over all 4,343,426 positions: both fire, Silas only, gate only;
Jaccard; and where both fire, the share with the same allowed set.

| rule | both fire | Silas only | gate only | Jaccard | same allowed set |
|---|---|---|---|---|---|
| each of 17 rules (C1a, C1b ×2, C2, C3, C7, C8, C10, C11, C12A, C12B, C13a–d, C21, C23) | 1,010 to 349,682; 1,353,540 in all | 0 | 0 | 1.000 | 100% |
| C9 | 165,382 | 0 | 2,990 | 0.982 | 100% |
| C14 | 38,779 | 0 | 0 | 1.000 | 71% |
| C15 | 318,488 | 15,749 | 0 | 0.953 | 100% |
| C22 | 2,390 | 0 | 42,725 | 0.053 | 100% |

- The encodings were written to be their gates (Silas slice 3), and over the whole pour
  they are, at 17 of 21. This measures how well Silas encodes the gates, not how far the
  gates are from the book: "at least 10 of 21 gates misread the book" stays a finding on
  the book's own deals (`silas/tests/book.rs`).
- C9's 2,990 gate-only rows are the gate's ordering at its last tier with no trump held:
  the gate allows every legal tile there; the policy does not fire. C14's sets differ at
  11,257 positions, the policy's always a superset (every non-trump double; GAP aggregate).
  C15's Silas-only rows are the one-point exception, a third rule the gate lacked.
- C22 is the one encoding of the book's rule rather than the gate: a trump set, and from
  the last seat only on count already played. It fires at 5% of the gate's positions
  (1,837 left seat, 553 last seat).

**Prices on the Silas firing set.** pmake for the mover's team per decision. Rows decided
by the banked score (the bidder's side at 30, or the setters past 12) are dropped (393,736
rows over all rules; the hand is over); rows decided only in Zeb's worlds are kept as Zeb
indifferent, never revived. *Random* = book set ÷ legal; *Zeb* = Zeb's exact-best set
inside the book set, ties broken at random; chance-corrected = (Zeb − random) ÷ (1 −
random); *for / indiff / against* = Zeb's exact-best set only book tiles / both / only
non-book tiles, in %; price = max over the book set − max over legal. Brackets: 95%
cluster bootstrap over deals (1,000 resamples); they do not include the 512-world sampling
noise. † = above −0.015, unreadable per decision: the pour player's own pick costs −0.005
to −0.016 under a four-times-stronger plan (§ The face-off).

| rule | n | legal | book | random | Zeb | chance-corr. [95%] | for / indiff / against | price [95%] |
|---|---|---|---|---|---|---|---|---|
| C1a | 8,285 | 4.90 | 1.00 | 22% | 73% | +0.65 [+0.64, +0.66] | 66 / 18 / 17 | −0.011 [−0.012, −0.010]† |
| C1b-top | 19,609 | 4.99 | 1.09 | 25% | 60% | +0.46 [+0.46, +0.47] | 54 / 12 / 34 | −0.020 [−0.020, −0.019] |
| C1b-nontop | 27,684 | 5.53 | 4.24 | 74% | 86% | +0.47 [+0.46, +0.48] | 79 / 13 / 8 | −0.004 [−0.004, −0.003]† |
| C2 | 49,393 | 2.42 | 1.00 | 43% | 50% | +0.11 [+0.10, +0.12] | 46 / 8 / 46 | −0.040 [−0.041, −0.039] |
| C3 | 24,721 | 5.59 | 1.35 | 27% | 41% | +0.20 [+0.19, +0.21] | 37 / 10 / 53 | −0.068 [−0.069, −0.066] |
| C7 | 103,206 | 4.84 | 1.23 | 29% | 26% | −0.04 [−0.05, −0.04] | 21 / 12 / 68 | −0.082 [−0.083, −0.081] |
| C8 | 53,396 | 2.30 | 1.29 | 55% | 76% | +0.48 [+0.47, +0.48] | 72 / 9 / 20 | −0.011 [−0.011, −0.011]† |
| C9 | 138,221 | 4.36 | 2.80 | 63% | 65% | +0.04 [+0.03, +0.05] | 58 / 12 / 30 | −0.020 [−0.021, −0.020] |
| C10 | 8,929 | 7.00 | 1.00 | 14% | 17% | +0.03 [+0.02, +0.04] | 16 / 2 / 82 | −0.068 [−0.070, −0.067] |
| C11 | 21,136 | 3.75 | 1.08 | 31% | 50% | +0.28 [+0.27, +0.29] | 41 / 26 / 33 | −0.017 [−0.018, −0.017] |
| C12A | 4,113 | 7.00 | 1.00 | 14% | 53% | +0.46 [+0.44, +0.47] | 52 / 3 / 45 | −0.014 [−0.015, −0.014]† |
| C12B | 2,212 | 7.00 | 1.00 | 14% | 9% | −0.07 [−0.08, −0.05] | 8 / 1 / 91 | −0.096 [−0.101, −0.093] |
| C13a | 137,986 | 4.83 | 1.75 | 43% | 55% | +0.20 [+0.20, +0.21] | 40 / 22 / 38 | −0.015 [−0.015, −0.015]† |
| C13b | 38,988 | 2.00 | 1.00 | 50% | 78% | +0.56 [+0.55, +0.56] | 73 / 10 / 17 | −0.010 [−0.010, −0.009]† |
| C13c | 209,369 | 2.00 | 1.00 | 50% | 50% | +0.00 [−0.00, +0.00] | 41 / 18 / 41 | −0.020 [−0.020, −0.020] |
| C13d | 212,417 | 4.40 | 1.54 | 39% | 40% | +0.02 [+0.01, +0.02] | 32 / 19 / 50 | −0.040 [−0.040, −0.039] |
| C14 | 35,928 | 4.78 | 1.49 | 33% | 53% | +0.29 [+0.29, +0.30] | 48 / 10 / 42 | −0.027 [−0.027, −0.026] |
| C15 | 230,611 | 4.60 | 1.59 | 37% | 61% | +0.37 [+0.37, +0.37] | 53 / 17 / 30 | −0.019 [−0.019, −0.019] |
| C21 | 970 | 6.04 | 1.00 | 18% | 12% | −0.07 [−0.09, −0.04] | 11 / 3 / 86 | −0.055 [−0.058, −0.052] |
| C22 | 2,263 | 4.51 | 2.26 | 55% | 81% | +0.58 [+0.56, +0.61] | 52 / 43 / 5 | −0.002 [−0.003, −0.002]† |
| C23 | 171,155 | 4.35 | 1.50 | 37% | 72% | +0.55 [+0.55, +0.55] | 66 / 10 / 23 | −0.015 [−0.015, −0.015]† |

- **On the 17 identical rules the prices are the gates'.** The same metrics on the
  face-off's own rows reproduce its per-gate table; on the whole pour, with dead rows
  decided only in the model counted as indifferent, C15 falls from +0.46 to +0.37 and C2
  from +0.19 to +0.11 (both as gates too), and those two verdicts are corrected above.
- **C22, as the book's rule, sides with the book more than its gate:** +0.58 and 5%
  against, price −0.002, against the gate's +0.51, 17% against, −0.009. The gate's extra
  positions split by seat: from the left seat Zeb agrees (+0.70, 9% against, −0.003); from
  the last seat with no count on the trick it does not (+0.32, 26% against, −0.016),
  which is the condition the book adds.
- **C14 with every non-trump double allowed** reads +0.29, price −0.027, against the
  gate's +0.25, −0.030: Zeb's preference among doubles is not the gate's "most unplayed".
- **C15 with the one-point exception** reads as the gate does (+0.37 both; price −0.019
  against −0.020).
- The five rules priced against (C3, C7, C10, C12B, C21) and the three with no relation
  (C9, C13c, C13d) are unchanged; a Silas encoding that is its gate cannot move them.
  The book's own rules re-price them in the next section.

## The book's rules as the book means them (2026-10-13)

Where the book's rules, written as the book means them, apply, does Zeb's plan side with
them? And over the whole pour, how far were the gates from them?

**Setup.** The 14 book-faithful policies of `silas/examples/book/` ([[silas]] § The book,
rule by rule) and the 20 gate encodings of `examples/book/gates/`, each asked through
`SilasPolicy.allowed(frame)` (the fired rule and its allowed set; the fallback is no firing)
at all 4,343,426 planned positions of the pour; every frame's own-legal set equals the pour's
legal tiles. The gate encodings reproduce the previous section exactly (21 of 21 price rows;
no difference in fired rule or allowed set on its per-position output). Producing player,
metrics, drops and † as in the previous section: the pour player `k0,n512`, utility `make`,
bid 30, field nets `field_make_d5000_v2` (code `31ecd76d`) and `v2/margin_d50k_h128x64`
(`b30603ad`); every q sampled over 512 worlds; 525,116 rule-position rows decided by the
banked score dropped. Cost: 15 processes of 3 minutes (`scratch/book-reprice/`).

**Fidelity: how far the gates were from the book.** Each book rule (the rules in it that do
the gate's job) against the gate it replaces, over the whole pour. *Gate only* splits into
positions where the book rule fires another of its rules / none.

| book rule (its rules) | gate | both fire | book only | gate only (other / none) | Jaccard | same set |
|---|---|---|---|---|---|---|
| partner-plays-the-top-trump | C2 | 50,444 | 0 | 0 | 1.000 | 100% |
| first-lead (double-up, lead the double) | C12A | 4,113 | 0 | 0 | 1.000 | 100% |
| low-from-a-pair | C13c | 248,962 | 0 | 0 | 1.000 | 100% |
| partner-leads (a double or the top tile) | C23 | 203,133 | 27,141 | 0 | 0.882 | 82% |
| first-lead (one off may go first) | C10 | 7,001 | 0 | 1,928 (1,928 / 0) | 0.784 | 55% |
| crisis (the six-five bait) | C21 | 769 | 0 | 241 (0 / 241) | 0.761 | 100% |
| setter-leads | C15 | 332,928 | 103,824 | 1,309 (0 / 1,309) | 0.760 | 57% |
| setter-leads | C13d | 298,994 | 137,758 | 50,688 (0 / 50,688) | 0.613 | 13% |
| shed-for-a-void | C13a | 125,362 | 47,347 | 40,836 (0 / 40,836) | 0.587 | 66% |
| partner-leads | C9 | 165,382 | 130,765 | 0 | 0.558 | 53% |
| protect-the-count | C8 | 51,377 | 46,128 | 3,089 (0 / 3,089) | 0.511 | 100% |
| bidders-plan (the double ahead of the off; the off's double) | C11 | 10,065 | 1,203 | 14,461 (14,461 / 0) | 0.391 | 97% |
| protect-the-count | C13b | 37,469 | 60,036 | 2,478 (0 / 2,478) | 0.375 | 100% |
| pounce | C7 | 73,960 | 378,181 | 37,901 (0 / 37,901) | 0.151 | 74% |
| donate-when-safe (give a ten, a five) | C1b-top | 19,727 | 117,493 | 1,519 (0 / 1,519) | 0.142 | 100% |
| donate-when-safe (no count on a trick that can be lost) | C1b-nontop | 15,838 | 82,007 | 14,690 (3,317 / 11,373) | 0.141 | 100% |
| trump-set (its three ruffs) | C22 | 328 | 0 | 2,062 (0 / 2,062) | 0.137 | 100% |
| first-lead (double-up, lead low) | C12B | 2,212 | 20,898 | 0 | 0.096 | 0% |
| crisis (its three double leads) | C14 | 15,701 | 194,817 | 23,078 (64 / 23,014) | 0.067 | 59% |
| donate-when-safe (give a ten, a five) | C3 | 8,901 | 128,319 | 17,084 (16,633 / 451) | 0.058 | 100% |
| bidders-plan (the donation lead) | C1a | 0 | 23,087 | 10,298 (6,480 / 3,818) | 0.000 | — |

- Three gates were the book's rule over the whole pour (C2, C12A, C13c). The rest part from
  it, most of them widely: C1a's gate and the book's donation lead never fire together (the
  gate needs a trump out, the book none), and nine more have Jaccard below 0.4.
- The book's rules fire at 2,600,512 positions (60%), the gates at 1,461,098 (34%).

**Prices on each book rule's firing set.** Sorted by chance-corrected agreement; the last
column is the replaced gates' chance-corrected agreement and price from the previous
section.

| book rule | n | legal | book | random | Zeb | chance-corr. [95%] | for / indiff / against | price [95%] | gates replaced |
|---|---|---|---|---|---|---|---|---|---|
| partner-leads | 241,408 | 4.18 | 1.74 | 45% | 71% | +0.48 [+0.47, +0.48] | 65 / 11 / 25 | −0.016 [−0.016, −0.015] | C9 +0.04, −0.020; C23 +0.55, −0.015† |
| helper-stretch | 4,869 | 2.18 | 1.00 | 47% | 73% | +0.48 [+0.46, +0.50] | 63 / 19 / 18 | −0.017 [−0.019, −0.015] | new |
| protect-the-count | 95,394 | 2.37 | 1.32 | 54% | 75% | +0.45 [+0.44, +0.45] | 66 / 15 / 18 | −0.012 [−0.013, −0.012]† | C8 +0.48; C13b +0.56 |
| bidders-plan | 139,137 | 3.88 | 1.32 | 40% | 60% | +0.33 [+0.32, +0.33] | 38 / 36 / 26 | −0.016 [−0.017, −0.016] | C1a +0.65; C11 +0.28, −0.017 |
| crisis | 206,401 | 6.24 | 1.21 | 20% | 44% | +0.30 [+0.30, +0.31] | 43 / 4 / 53 | −0.021 [−0.021, −0.020] | C14 +0.29, −0.027; C21 −0.07, −0.055 |
| donate-when-safe | 213,454 | 3.75 | 1.82 | 48% | 61% | +0.25 [+0.25, +0.25] | 53 / 19 / 28 | −0.022 [−0.023, −0.022] | C1b +0.46, +0.47; C3 +0.20, −0.068 |
| pounce | 413,966 | 3.57 | 1.59 | 46% | 58% | +0.22 [+0.22, +0.23] | 51 / 15 / 34 | −0.026 [−0.026, −0.026] | C7 −0.04, −0.082 |
| shed-for-a-void | 147,078 | 4.86 | 1.43 | 34% | 47% | +0.20 [+0.20, +0.21] | 36 / 22 / 42 | −0.026 [−0.026, −0.026] | C13a +0.20, −0.015† |
| first-lead | 243,897 | 7.00 | 2.12 | 30% | 43% | +0.18 [+0.18, +0.19] | 42 / 2 / 56 | −0.024 [−0.024, −0.024] | C10 +0.03, −0.068; C12A +0.46; C12B −0.07, −0.096 |
| setter-leads | 286,093 | 4.47 | 1.92 | 44% | 54% | +0.18 [+0.18, +0.18] | 46 / 19 / 36 | −0.028 [−0.029, −0.028] | C15 +0.37, −0.019; C13d +0.02, −0.040 |
| partner-plays-the-top-trump | 49,393 | 2.42 | 1.00 | 43% | 50% | +0.11 [+0.10, +0.12] | 46 / 8 / 46 | −0.040 [−0.041, −0.039] | C2, the same |
| dead-double | 403,530 | 4.79 | 1.07 | 27% | 33% | +0.08 [+0.08, +0.08] | 22 / 24 / 54 | −0.024 [−0.024, −0.024] | new |
| trump-set | 5,718 | 3.78 | 1.11 | 33% | 37% | +0.06 [+0.04, +0.07] | 17 / 57 / 26 | −0.030 [−0.032, −0.027] | C22 +0.58, −0.002† |
| low-from-a-pair | 209,369 | 2.00 | 1.00 | 50% | 50% | +0.00 [−0.00, +0.00] | 41 / 18 / 41 | −0.020 [−0.020, −0.020] | C13c, the same |

The rules inside them that carry the reading (every rule's row is in `results.json`):

| rule | n | random | Zeb | chance-corr. | for / indiff / against | price |
|---|---|---|---|---|---|---|
| first-lead: lead the top trumps | 20,054 | 28% | 80% | +0.72 | 78 / 3 / 19 | −0.005† |
| first-lead: lead a trump | 124,983 | 32% | 58% | +0.39 | 57 / 2 / 40 | −0.011† |
| first-lead: two trumps, lead the straight off | 64,636 | 29% | 8% | −0.30 | 8 / 1 / 91 | −0.042 |
| first-lead: one off may go first | 7,001 | 31% | 42% | +0.16 | 41 / 1 / 58 | −0.055 |
| first-lead: double-up, lead low | 23,110 | 28% | 24% | −0.06 | 23 / 1 / 76 | −0.058 |
| bidders-plan: draw the trumps | 56,072 | 23% | 53% | +0.38 | 45 / 19 / 36 | −0.023 |
| bidders-plan: the donation lead | 18,495 | 32% | 33% | +0.02 | 8 / 69 / 24 | −0.012† |
| bidders-plan: dump a bare off (any count) | 2,741 | 43% | 36% | −0.12 | 33 / 8 / 59 | −0.064 |
| bidders-plan: the double ahead of the off | 3,090 | 28% | 27% | −0.01 | 8 / 56 / 36 | −0.020 |
| bidders-plan: cash the other winners | 16,335 | 45% | 85% | +0.72 | 76 / 17 / 7 | −0.003† |
| bidders-plan: the off's double next-to-last | 5,892 | 40% | 95% | +0.92 | 94 / 4 / 3 | −0.002† |
| crisis: the six-five bait | 734 | 18% | 9% | −0.11 | 8 / 1 / 91 | −0.058 |
| crisis: cash a double that is count | 63,472 | 17% | 46% | +0.36 | 45 / 4 / 51 | −0.029 |
| donate-when-safe: give a ten | 59,063 | 35% | 49% | +0.21 | 36 / 31 / 33 | −0.031 |
| donate-when-safe: give a five | 62,751 | 37% | 75% | +0.61 | 66 / 22 / 13 | −0.006† |
| donate-when-safe: no count on a trick that can be lost | 91,640 | 65% | 60% | −0.15 | 54 / 10 / 36 | −0.028 |
| partner-leads: a double or the top tile | 192,062 | 43% | 75% | +0.55 | 68 / 11 / 21 | −0.014† |
| pounce: no count on a lost trick | 173,055 | 62% | 88% | +0.68 | 79 / 16 / 5 | −0.003† |
| pounce: a ten now | 96,885 | 30% | 29% | −0.02 | 21 / 21 / 57 | −0.060 |
| pounce: a five now | 109,861 | 36% | 37% | +0.02 | 34 / 7 / 60 | −0.039 |
| setter-leads: one point sets, lead a double | 27,275 | 39% | 73% | +0.56 | 61 / 27 / 12 | −0.005† |
| setter-leads: the count that sets | 181,949 | 49% | 53% | +0.09 | 46 / 16 / 38 | −0.030 |
| setter-leads: a double that calls for count | 33,003 | 30% | 66% | +0.51 | 62 / 11 / 28 | −0.014† |
| setter-leads: a count tile | 38,407 | 41% | 39% | −0.03 | 22 / 35 / 43 | −0.045 |
| trump-set: feed the trump holder | 1,221 | 41% | 20% | −0.37 | 15 / 10 / 76 | −0.097 |
| trump-set: lead a high trump | 4,198 | 28% | 39% | +0.15 | 16 / 71 / 13 | −0.012† |
| dead-double: throw it | 260,356 | 31% | 33% | +0.04 | 18 / 34 / 48 | −0.020 |
| dead-double: the bidder cashes it | 143,174 | 19% | 31% | +0.15 | 29 / 6 / 66 | −0.032 |

**No firing in the pour.** Three rules need what bid 30 never gives, and carry no price:
low-from-a-pair's *keep the higher for the off* needs a bid of marks; setter-leads' *think
twice* needs a bid of 35 or more (at 30 its guard is the rule above it); setter-leads'
*count that calls for more* fires only once the bid is set (20,463 rows, all decided by the
banked score), since with any point banked the count that sets always fires first.

**Reading.**

- *Zeb sides with 3 of the 14:* the partner's leads (+0.48), protect the count (+0.45), and
  win low and keep the double (+0.48). It agrees weakly with 8, shows no relation to 2
  (low-from-a-pair 0.00, the C13c gate unchanged; the trump set +0.06), is near chance on
  the dead double (+0.08, between the bands), and prices against none. Each rule's firing set equally weighted, Zeb's pick is in
  the book set at 54% against a 40% random floor (chance-corrected 0.23); the 21 gates on
  the whole pour read 53% against 37% (0.25). Reading the book faithfully does not raise
  agreement.
- *The partner's leads* side with the book where C9 showed no relation: the book ranks the
  walker with the doubles and orders the rest by the count each lead calls.
- *The pounce* moves from against (C7, −0.04) to weak (+0.22) only through the book's guard
  (no count on a lost trick, +0.68). The pounce itself, a ten or a five played at once,
  still prices against (−0.060, −0.039).
- *The gates' other "against" rules stay against in the book's form:* the trick-one off as a
  permission (−0.055), double-up lead low with three trumps or four (−0.058), the six-five
  bait as a crisis lead (−0.058). The book's two-trump straight-off lead is new and against
  (91% against, −0.042). Biggest count first (C3) becomes the safe trick's ten (−0.031,
  +0.21) and five (+0.61). The book's donation lead (no trump out) is where Zeb is
  indifferent (69%); the gate's +0.65 was a different position.
- *The dead double and the trump set carry the conditions the book states.* The dead double
  is thrown on a trick the thrower cannot take, never a count double: every throw the book
  plays is one, and it throws no 5-5. Without the condition the throw fired where a trump
  would take the trick (−0.151, 76% against, 170,676 rows) and threw the 5-5, a ten
  (−0.094, 17,982 rows); with it, 260,356 rows at −0.020 (+0.04), and the rule moves from
  −0.062 to −0.024. The trump set's feed waits on a holder who may yet trump (still to
  play, not shown out of trump, a trump live outside the feeder's hand): without it, 52%
  of its firings came after every trump was played (−0.138). With it the feed still
  prices against (1,221 rows, −0.097, 76% against); the holder in fact holds a trump and
  is void in the double's suit at 256 of them (−0.069). The rule moves from −0.051 to
  −0.030.

**Worked reasons.** `scenario/reason.py` at three random positions per rule (s200001 up,
live, Zeb's exact-best set outside the book set, price below −0.05), Zeb's tile against the
book's best, under the pour player's spec at current code (`k0,n512`, `make`,
`margin_d50k_h128x64`, seed 0; the bar re-plans at seed 1 and one rung stronger). The
current plan picks the pour's tile at all six; every verdict is *reason*, and the book's
tile never succeeds where Zeb's fails.

| rule | position | book / Zeb | price (bar) | top separating fact: P(· \| Zeb's side) vs the rest, exact prior |
|---|---|---|---|---|
| feed the trump holder | s300001 deal 43342 ply 17; threes, every trump played; the bidder's 0-0 led | 5-5 / 2-2 | +0.555 (0.007) | the feeder's partner holds the 4-4: 0.55 vs 0.11, 5/14 |
| feed the trump holder | s200001 deal 9437 ply 21; blanks, every trump played; the bidder's 4-4 led | 5-5 / 6-3 | +0.250 (0) | the bidder's last tile is the 5-3: 1 vs 0, 1/4 |
| feed the trump holder | s300001 deal 36667 ply 21; deuces, every trump played; the bidder's 0-0 led | 6-4 / 6-5 | +0.133 (0.067) | the bidder is void in blanks: 0 vs 0.69, 3/5 |
| throw a dead double | s250001 deal 7003 ply 15; fours; a setter void in the deuce led, holding three trumps | 0-0 / 4-3 (ruff) | +0.398 (0.031) | an opponent holds the 4-4: 0.94 vs 0.45, 9/14 |
| throw a dead double | s350001 deal 33765 ply 18; fives; the bidder void in the six led, the 6-4 on the trick | 2-2 / 6-5 (ruff) | +0.180 (0.033) | the bidder's partner holds the 5-5: 1 vs 0.11, 4/15 |
| throw a dead double | s200001 deal 25254 ply 19; threes; a setter with no trump, the bidder's partner winning | 5-5 / 6-0 | +0.083 (0.017) | the bidder holds the 6-4: 1 vs 0.45, 1/2 |

In every feed position the set was over: the ten fed makes the bidder's 30 in 92–100% of
Zeb's worlds, and Zeb keeps it for a later trick the setters can win, on the holding the fact
names. Two dead-double positions are a ruff against a throw, and the third throws a ten onto
the bidder's side's trick. These reasons explain Zeb's tile; each names a condition the
encoding lacked, not a fault in the book's reasoning. The rules as fixed fire at none of
the six.

## Data

The measurements (scripts, pour queries, forecasts, per-deal arena output, the face-off's
per-position picks, the reports) are public in `jasonyandell/texas-42-zeb-research-2026-10`
under `book/wave1/`, `book/wave2/` and `book/faceoff/` ([publish-data-to-huggingface](../playbooks/publish-data-to-huggingface.md)).
The baselines, chance-corrected agreement, dead-row split and like-for-like comparison are
recomputed from `scratch/faceoff/out/` by `scratch/book-audit/stats.py`, `kappa.py`,
`dead_decided.py` and `live_at30.py`. The Silas re-price is recomputed from the pour by
`scratch/silas-reprice/walk.py` and `analyze.py` (`reconcile.py` checks it against the
face-off's table); the scripts, per-position output (fired rule and allowed set at all 4,343,426
positions) and `results.json` are in the same research repo under `book/silas-reprice/`. The
book-faithful re-price is recomputed by `scratch/book-reprice/walk.py`, `analyze.py` and
`tables.py` (`check_published.py` checks the gate encodings against `book/silas-reprice/`;
`diagnose.py` splits the dead double's throw and the trump set's feed; `reasons.py` writes
the worked reasons); the scripts, per-position output (every book rule's and gate's fired
rule and allowed set at all 4,343,426 positions), `results.json`, `diagnose.json` and the six
reason reports are in the same research repo under `book/book-reprice/`. The two rules'
rows with the book's conditions are recomputed by `scratch/book-reprice-fix/walk.py` (the
two rules as first encoded and as fixed), `analyze_fix.py` (which reproduces the first
encodings' published rows exactly), `diagnose_fix.py` and `union_fix.py`; scripts,
per-position output, `results.fix.json` and `diagnose.fix.json` are under
`book/book-reprice-fix/`. The book's text, page images and
transcribed hands are not published.

## Links

[[w42]] · [[scenario-player]] · [[endgames-o2-vs-o1-2026-10-12]] ·
[[research-night-2026-10-12]] · [[the-wall]] · [[w42-book-validation-campaign]] ·
[[w42-book-claim-synthesis-and-ai-directions]] · [[w42-book-second-pass]] ·
[[publish-data-to-huggingface]]
