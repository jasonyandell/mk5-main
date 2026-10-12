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
wrote a ranked list of 26 candidates (7 conventions); an army of Opus agents then priced
every rule on the 250,000-deal Zeb pour, checked the bid book against Zeb's forecast,
recomputed the book's odds by exact count, and graded the conventions in the arena with
the overlay built the same day ([[scenario-player]] § A convention overlay). The
oracle-era verdicts ([[w42-book-claim-synthesis-and-ai-directions]]) were discarded, not
reused: they were scored against perfect information ([[the-wall]]).

**The picture in one line.** The book is mostly exact counting over the hidden hands,
which Walt already does; its play rules are, almost all, what Zeb's plan does anyway;
its bid book agrees with Zeb's forecast on 25 of 30 hands; and its partnership
conventions, wired into both partners, buy nothing measurable. Nothing was adopted as a
rule. Three of the book's odds figures were off by about two points, in the direction an
expert's hand count would be, and are corrected below.

**What held, and the reason it holds**

- *Counting rules hold because the fiber is the object.* Pull trumps first, which trump to
  lead, the double ahead of the off, the setter's void creation in follow position, low
  from a pair, trump the bidder's double from the left seat: at 60–90% of gated positions
  the book's tile is exactly Zeb's best (price 0), and the rest is the fiber's tail the
  book cannot see. Nothing to adopt: the contingent plan over the cell computes them.
- *Bid-only-enough holds because one mark is one mark.* 32–34 points fall in 3% of made
  hands in every arena log, so pmake(32) − pmake(35) ≈ 0.03; under marks the level buys
  nothing but the auction.
- *"42 not 84 at one mark from the game" holds as a gate*: the make condition is identical,
  the 84 only doubles the loss. Zeb's bidder never bids 84 today (#115), so the gate is a
  correctness line for the day it does.
- *Doubles-trump and no-trump as declarations hold*: the forecast enumerates them, and ranks
  them where the book does on one hand of three.

**What did not hold, and why**

- *The bid as a sentence* (31 means "one ten-count off") is a points artifact: under marks
  every bid from 30 to 41 pays one mark, so the sentence costs pmake(level) − pmake(30).
- *"Never fail to bid 84 with the double ahead of the off."* The book's showcase hand
  (p. 113: trumps 5-5 5-4 5-2 5-0 with the 6-5 out) forecasts p84 = 0.44 against the 0.73
  an 84 needs to beat a one-mark bid; counting alone caps it at 0.653, because an opponent
  holds the 6-5 with another five with probability 33/95.
- *Three odds figures.* The 27 placements of three missing trumps are not equiprobable
  (a 1–1–1 split weighs 0.043 each, a 2–1 split 0.037, a 3–0 split 0.026): the book's 37%
  is 33/95 = 0.347 and its 52% is 47/95 = 0.495, a coin flip the bidder's way. The "80%"
  that an opponent cannot follow the first doubles lead is 23/30 = 0.767. "Partner holds two
  or more doubles 58% of the time" is the unconditional figure; given the bidder's own two
  doubles it is 0.557, given three 0.407.
- *Biggest count first* and *the six-five bait lead* contradict the pour (prices −0.064
  and −0.053 pmake per decision; Zeb plays the book's tile at 37% and 11% of gates).
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

marks/hand for the bidding team, 95% intervals, 22,000 paired deals per arm.

- **The donation is null** in every arm. SR beats S and R by about +0.001 each, the
  convention's signature shape, at a size inside a tenth of a mark per hundred hands. The
  sender fired in 3,544 deals; the reader drew filtered worlds at 135,000 decisions.
- **The handoff is a priced signal that buys nothing.** The sender costs −0.013 marks/hand
  (the pour predicted −0.016: −0.039 pmake per firing × 19% of hands × 2). The reading on
  top of the sender recovers 0.000 ± 0.002. All of the cost sits where the bidder holds three
  trumps or fewer (−0.0144 [−0.0182, −0.0106], n = 19,706), exactly the stratum the entry
  predicted the information would pay in; at four trumps the gate rarely fires (flips 2/2).
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
what partners would want to say ([[endgames-o2-vs-o1-2026-10-12]]). The book's two highest-
ranked conventions, graded with both halves, land inside that ceiling at zero.

## Setup

**Wave 0, the reader.** One Fable 5.1 session, the whole OCR text in one context, two
hours, no code run. Output: 26 candidates in one shape (name, source page, type, the claim
in Walt units, the candidate reason, the grading experiment, comfort or strength, rank),
plus the sections *Already Zeb*, *Cannot hold* and *The table's convention* (Jason's
handoff, stated as a joint deviation). Pages 8, 36, 42, 57–58, 74, 85–86, 90, 122, 131
and 181–186 are missing from the archive (page 42 is "stretching your doubles"; 185–186
are the second half of the odds appendix).

**Wave 1, prices and counts.** Four Opus agents in worktrees, all under a minute per run:

- *Pour queries, bidder side and setter side.* Every rule's gate written as a predicate in
  the mover's chair (seat role, own hand, public record) over the 250,000-deal Zeb pour
  (4,343,426 positions, utility `make`, bid 30, seat 0 bids). Price = max pmake over the
  book's allowed tiles − max over all legal tiles (≤ 0; a setter's value is 1 − q). Verdict
  scale: *already Zeb* when the book tile is best at ≥ 90% of gates, *cheap* when the mean
  price is ≥ −0.01, *expensive* below, *contradicts* when below −0.01 and best at fewer than
  half.
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
*ties* is the share where the book's tile is exactly best; *Zeb* is how often Zeb's chosen
tile is the book's.

| candidate | side | n | price | ties | Zeb | verdict |
|---|---|---|---|---|---|---|
| C1a bidder leads the top trump for the donation | bidder | 10,298 | −0.009 | 86% | 58% | cheap signal; in 74% of gates the bidder does not hold the top trump |
| C1b partner sluffs non-count on a non-top trump lead | partner | 30,528 | −0.003 | 93% | 85% | already Zeb |
| C1b partner gives the largest count on a top-trump lead | partner | 21,246 | −0.018 | 69% | 58% | expensive |
| C2 high-trump handoff | partner | 50,444 | −0.039 | 55% | 48% | expensive; worst (−0.069) when the bidder's lead was itself the top trump |
| C3 biggest count first | partner | 25,985 | −0.064 | 49% | 37% | contradicts; Zeb gives the smaller count 7,888 times, the largest 4,146 |
| C7 pounce | setter | 111,861 | −0.076 | 38% | 27% | contradicts as a one-seat rule; −0.089 with count protected, −0.053 unprotected |
| C8 count on a double means void | setter | 54,466 | −0.011 | 81% | 75% | already Zeb; the reading is right 70% of the time in Zeb play today |
| C9 the partner never leads trump | partner | 139,433 | −0.016 | 78% | 70% | expensive; the partner leads trump 33% of the time he holds it |
| C10 lead the off first, trick 1, with one off and ≥ 4 trumps | bidder | 8,929 | −0.068 | 18% | 17% | contradicts on trick 1; the direction holds (off-lead share rises 14% → 29% → 43% with 4, 5, 6 trumps) |
| C11 double ahead of the off, lead the double first | bidder | 24,526 | −0.015 | 71% | 45% | expensive, borderline |
| C12 four trumps, lead the double (holding the 3rd) | bidder | 4,113 | −0.014 | 55% | 54% | the book's double beats its low lead (−0.125); missing the 2nd and 3rd, the book's low lead contradicts (−0.096) |
| C13 setter: void in follow position, protect count, low from a pair | setter | 39,947–349,682 | −0.009 to −0.024 | 65–83% | 45–76% | already Zeb / cheap; leading a singleton −0.024, mild contradiction |
| C14 crisis: cash the double with most unplayed | bidder | 38,779 | −0.028 | 54% | 46% | expensive; Zeb leans the book's way (the trump lead is −0.052) |
| C15 setter leads that call for count | setter | 318,488 | −0.014 | 79% | 50% | already Zeb for doubles; the one-point exception is real (quiet lead strictly best 38% there) |
| C21 the six-five bait lead | bidder | 1,010 | −0.053 | 17% | 11% | contradicts |
| C22 trump the bidder's double from the left | setter | 45,115 | −0.009 | 83% | 66% | already Zeb (left seat 91% ties; last seat 75%) |
| C23 the partner leads doubles | partner | 203,133 | −0.013 | 80% | 70% | expensive, borderline |

**The bid book.** Bid or pass agrees on 25 of 30 hands; where both bid, the level agrees on
8 of 11, Zeb a notch higher on one and lower on two. The five disagreements: the book
passes a 0.84 hand on its 16-points-on-one-trick rule; it bids a reconstructed 0.80 hand
on the cutoff; it calls two 0.67–0.72 hands easy (one is set in its own play-out); and its
showcase no-doubles hand forecasts 0.61 at 30. The at-risk ≤ 12 rule and its two-off
paradox hold in direction (a two-off hand at 0.91 beats one-off hands at 0.84 and 0.44).
On the 84 hands an 84 beats a one-mark bid iff p84 > (2·p30 + 1)/4: two lay-downs and one
narrow yes, three prefer one mark. `chooseBid` lands on 33 or 34 on 3 of 30 hands, which the
book's bid-only-enough would never do (#115).

## What this means for Zeb

1. **Nothing to adopt as a rule.** Every play rule that holds is the plan over the fiber;
   every one that fails is a counting or utility fact the book rounded. The three
   corrected odds are the fiber's exact weights, which Zeb already draws from.
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
   seats only the bidding team's Zebs.

## Data

The measurements (scripts, pour queries, forecasts, per-deal arena output, the report) are
public in `jasonyandell/texas-42-zeb-research-2026-10` under `book/wave1/` and
`book/wave2/` ([publish-data-to-huggingface](../playbooks/publish-data-to-huggingface.md)).
The book's text, page images and transcribed hands are not published. The candidate list
and briefs quote the book and stay in `scratch/book/`.

## Links

[[w42]] · [[scenario-player]] · [[endgames-o2-vs-o1-2026-10-12]] ·
[[research-night-2026-10-12]] · [[the-wall]] · [[w42-book-claim-synthesis-and-ai-directions]]
· [[w42-book-second-pass]] · [[publish-data-to-huggingface]]
