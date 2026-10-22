---
title: The book's line played out — Hands 23 to 25 at the book's bids (#230 step C, 2026-10-22)
kind: experiment
first_seen: 2026-10-22
last_updated: 2026-10-22
status: active
---

## Summary

Registered, not yet run. Issue #230 step C. Chapter 9 works three hands (W42 pp. 76–82):

| hand | contract | the book's words for it (in our words) |
|---|---|---|
| 23 | doubles trump at 31 | a cinch; set only by one named, unlikely run of the cards |
| 24 | doubles trump at 30 | a cinch with no help from the partner; 35 with help |
| 25 | no trump at 30 | a cinch with no help; set only by one named run |

On Hands 23 and 24 the book's surface count would not bid: 22 and 23 at risk leave 20 and
19 (`arena.hand_metrics.ch9_doubles_loss`). The book then plays the line out in its head
and bids. Hand 25 is the other way round. The count of 13 is for the doubles-trump option,
which the book says comes in at 30 with partner help. The book gives no count for no trump
and chooses it as safer.

This experiment asks whether the book's second step, the line played out, holds over the
deals the declarer cannot tell apart. The book prices with uniform odds itself on these pages
("two-to-one", a percentage of the first trick), so uniform deals are its own frame.

Our pricing disagrees on Hand 23. The menu (`arena.hand_metrics.menu_choice`) prices it at 27,
so `MenuBidder` would pass a hand the book bids at 31. It prices Hands 24 and 25 at 42.

Step B was withdrawn on #230: a chair bidding by the surface count alone would pass these
hands. The book's method rests on the play-out, so the play-out is what we test.

## The instrument

**Deals.**
- The declarer's seven tiles are fixed: the book's own hands, from
  `silas/examples/book/hands.json`. Diagram facts are data; the book's prose stays in the private
  archive.
- The other 21 tiles are dealt uniformly, seven to each other seat, from
  `random.Random(230_300 + hand)`. Every arm of a hand uses the same deals.
- The declarer sits at seat 0 and leads trick one.

**The line (arms L, L0, B).** The declarer's leads follow the book's narrated line for that
hand, written in our words in the runner's `line_lead`. Each lead is the first tile of a fixed
order still held:
- **Hand 23:** the 1-1 first (a low double to draw the missing 5-5); then the 6-6, the other
  trumps, the 5-4 (the boss five behind the declarer's own 6-5), the 6-5 next-to-last, and the
  6-1 walking last.
- **Hand 24:** the 6-6, the 5-5, the 5-1 to draw the 6-5, the 6-4 (safe once the 6-5 is gone),
  the low trumps, and the 3-1 last.
- **Hand 25:** played as an 84. The 3-3, 1-1 and 4-4 first, the 4-4 third. Then, of the 5-5 and
  2-2, the one with fewer of its suit still out. Once the fives are gone the 5-0 goes. The offs go
  last, the 2-1 last of all.

The declarer's follows, and every other seat's plays, come from the arena's book stack:
`arena.book_play.BookPlay("lowest", "least")`, the eighteen book policies of
`silas/examples/book/`, first-firing-decides, the least allowed tile, the lowest legal tile
where none fires. No Zeb plays except in arm B.

**The partner without help (arm L0).** The book stresses "without partner help" on Hands 24
and 25. L0's partner plays the stack's tile unless it is a count tile and he holds a legal tile
without count; then he plays the least such tile.

**The read.**
- The bid is made when the declaring side's points reach it.
- Shares carry Wilson 95% intervals.
- Paired differences over the same deals carry the mean of the per-deal differences ± 1.96
  standard errors.

| arm | declarer leads | partner | setters | contract | deals per hand |
|---|---|---|---|---|---|
| L | the book's line | book stack | book stack | the book's | 4,000 |
| L0 | the book's line | without help | book stack | the book's | 4,000 |
| K (key move removed) | the line, its key move undone: Hand 23 opens with the 6-6; Hand 24 leads the 6-4 before the 5-1; Hand 25 leads the 5-5 and 2-2 first | book stack | book stack | the book's | 4,000 |
| B | the book's line | book stack | Zeb (`WaltPlay()`: level 1, random field) | the book's | 1,000 (the first of L's) |
| A | book stack | book stack | book stack | the book's | 4,000 |
| D | book stack | book stack | book stack | the book's rejected one: Hand 23 sixes at 31, Hand 24 aces at 30, Hand 25 doubles trump at 30 | 4,000 |

## Our encoding first (rule 6)

Before sealing we read the stack's declarer leads alone, with no points or outcomes, on eight
deals per hand from a seed the registration never uses.
- **Hand 25:** the stack plays the book's line: the 1-1, 3-3 and 4-4 (the 4-4 third), then the
  doubles, the offs last.
- **Hand 23:** it opens with the 0-0, not the book's 1-1. It then often leads the 6-1 right after
  the 6-6, where the book cashes the trumps and saves the 6-5 so the 6-1 walks last.
- **Hand 24:** it opens with the 5-5, not the 6-6, and leads the 3-1 before the 5-1, the off the
  book leads to draw the 6-5.

The `least` tie-break picks among sure trumps, and `bidders-plan`'s "dump a bare off while
holding a trump" fires on the 6-1 and the 3-1. So the general stack is not the book's line on
Hands 23 and 24. The line is therefore scripted per hand (L), and the stack is arm A, a measure
of our encoding: L − A, and the share of deals where A's leads depart from the line. Repairs to
`first-lead` and `bidders-plan` go on #230; they are not part of this test.

## Predictions

**Primary, one per hand.**
- **Hand 23 (L):** the share of deals set by anything other than the book's named run has a
  Wilson upper bound of at most 0.02. The named run, in our words: a setter wins trick one with
  the 5-5, leads a four next, and his partner wins that trick with the 6-4. The book says this is
  the only way. 0.02 is our tolerance for "only".
- **Hand 24 (L0, no help):** the make share's lower bound is at least 0.90. The book names no set
  scenario here; 0.90 is our reading of "cinch".
- **Hand 25 (L0, no help):** the share set by anything other than the named run has an upper bound
  of at most 0.02. The named run: a setter plays the 6-4 on an off the declarer leads at trick five
  or later, and the other setter wins that trick.

Sets on Hand 23 are sorted into the named run, count on the first lost trick (count besides the
5-5 on trick one), and other.

**Secondary.**
1. L's make share has a lower bound of at least 0.90 on every hand (our reading of "cinch", with
   a helping partner).
2. B, against Zeb's setters: lower bound at least 0.85.
3. L − K > 0 on every hand, interval above zero: the move the book's line turns on is what earns
   the make.
4. A − D > 0 on Hands 23 and 24, interval above zero (the book's doubles trump against the pip
   trump it turns down, both played by our stack). A − D ≥ 0 at the point on Hand 25, where the
   book says doubles trump also comes in with help.

   D is played by the stack, not by the book's lines for those contracts. The book sanctions
   sixes at 31 only with the 0-0 led before the off, so D's sixes and aces arms are scoped to our
   stack with a helping partner. The book claims doubles trump is safer; it predicts no size for
   the difference.

**Descriptive, no threshold.**
- L0's make share on Hand 23.
- The share reaching 35 with a helping partner on Hands 24 and 25.
- The encoding gap L − A, with A's departure share.

## What each outcome would mean

- **The primaries hold.** The book's play-out stands where its surface count would not bid. On
  Hand 23 it prices the hand at 31 where our menu says 27. The gap is our menu's static ceiling,
  and the next step is a ceiling read from the line (#230).
- **A primary misses.** The sets outside the named run are read deal by deal first:
  - is it a lead our script got wrong?
  - a follow the stack plays where the book would play otherwise?
  - a defence the book's analysis did not consider?

  Each is reported as the scope of our model: the stack's follows and defence, the setters'
  lowest-tile fallback, and deals with no auction inference ([[book-assumptions]]). We never
  write "the book is wrong" ([[book-authority]]).
- **B misses where L holds.** Our stack's setters are softer than Zeb's, and the result is scoped
  to them.
- **L − K is a tie.** On these deals, the key move earns nothing measurable against our setters.
  That is reported as a tie, not as agreement.

## The framing review (2026-10-22)

A fresh-context Opus reviewer read Chapter 9 whole from the private OCR: eleven files, p. 73 to
the start of Chapter 10, each to its last line. Two capture pages are absent from the OCR set
(p0103 and p0108), so the opening of Hand 23's sixes count and the start of the no-trump section
were read from the reviewed transcription's diagram facts only. The reviewer then read the first
draft. Changes made from its report:
- **Partner help:** a no-help arm (L0), which is now the primary on Hands 24 and 25.
- **"Only way":** the book's claim that one run is the only way to be set is the primary on Hands
  23 and 25. 0.90 for "cinch" stays as a secondary, labelled ours.
- **Hand 25's count:** the surface-count sentence is corrected.
- **Uniform deals:** stated as the book's own frame.
- **The line:** calibration on trick one gave way to the encoding check above. That check found
  the stack leaves the book's line on Hands 23 and 24, so the line is scripted.
- **Arm C:** the random declarer gave way to K, the line without its key move.
- **Arm D:** scoped to our stack with a helping partner.
- **Hand 23's named run:** tightened to the 5-5's holder, his lead of a four, and his partner's
  6-4.

Not changed:
- The no-trump defence uses the stack as is. `defend-the-84` reads only the defence of the last
  trick, and it applies to no trump as to an 84. The book's advice to set no trump like an 84
  otherwise falls to `setter-leads` and the follow rules. That is a scope note for misses on Hand
  25.

## Data

Per-deal rows (hand, arm, deal index, the four hands, the 28 plays, points, made), with
`run.py` and `read.py`, go to Hugging Face: `jasonyandell/texas-42-zeb-research-2026-10`
under `book/book-line-230c/`.
