---
title: The book's line played out — Hands 23 to 25 at the book's bids (#230 step C, 2026-10-22)
kind: experiment
first_seen: 2026-10-22
last_updated: 2026-10-22
status: complete
---

## Summary

**Result (2026-10-22): every registered primary missed, and calibration shows each miss measures
our instrument at least in part.** Read with that scope, three things stand:
- **Hand 24:** the book's line makes 30 in 94.5% [93.8, 95.2] of deals with a helping partner, and
  its key move, the 5-1 before the 6-4, is worth +0.27 of a make.
- **Hand 25:** read in the book's words (post hoc), its named run is all but the only way to be
  set. Other sets are 0.015 [0.012, 0.019] of deals. The run itself comes in 18.6% of deals
  against setters who play the book's own defence.
- **Hand 23:** makes 0.571. The book's 11-point budget for trick one is exceeded mainly by a sound
  pounce of count onto the 5-5's trick. The book names that risk itself on Hand 25.

Two instrument faults (a harmful no-help partner, a narrow classifier) and one recorded silence
(our declarer's ducks) are measured below. A suspected fourth fault turned out to be the book's
own pounce (corrected the same day). The encoding questions are #245. Data:
`book/book-line-230c/` on Hugging Face.

Registered and sealed at bc68aa03. Issue #230 step C. Chapter 9 works three hands (W42 pp. 76–82):

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

## Results (2026-10-22)

Every registered primary missed. Calibration after the read (rule 6) shows that each miss
measures our instrument at least in part, so none of them reads as a verdict on the book.
Read whole, though, the runs say something useful about all three hands.

The registered read (`read.txt` in the data):

| hand | primary | result | registered bar |
|---|---|---|---|
| 23 (L) | set outside the named run | 0.421 [0.406, 0.437] | upper ≤ 0.02: miss |
| 24 (L0) | make share, no help | 0.855 [0.844, 0.866] | lower ≥ 0.90: miss |
| 25 (L0) | set outside the named run | 0.278 [0.265, 0.292] | upper ≤ 0.02: miss |

| hand | S1 L make | S2 B Zeb setters | S3 L − K | S4 A − D | L − A (encoding gap) | reaches 35 (L) |
|---|---|---|---|---|---|---|
| 23 | 0.571 [0.555, 0.586] miss | 0.522 [0.491, 0.553] miss | −0.042 [−0.057, −0.026] miss | −0.314 [−0.332, −0.296] miss (sixes 0.708) | +0.177 [+0.163, +0.190] | — |
| 24 | **0.945 [0.938, 0.952] hit** | 0.757 [0.730, 0.783] miss | **+0.267 [+0.252, +0.282] hit** | **+0.172 [+0.156, +0.189] hit** (aces 0.738) | +0.035 [+0.026, +0.045] | 0.759 [0.746, 0.772] |
| 25 | 0.799 [0.786, 0.811] miss | 0.804 [0.778, 0.827] miss | −0.028 [−0.040, −0.016] miss | **+0.178 [+0.160, +0.196] hit** (doubles trump 0.649) | −0.028 [−0.042, −0.013] | 0.644 [0.629, 0.658] |

Calibration after the read (`diagnostics.txt`). Two instrument faults and one recorded silence,
each measured. A fourth suspected fault was the book's own rule; see the correction below:

1. **The no-help partner (L0) is not "no help".** Barred from playing count while he holds
   another tile, he keeps his count to the end and loses it to the setters. That happens in 417
   of Hand 24's 579 L0 sets and 980 of Hand 25's 1,511. Under L it happens in 29 of 219 and 0 of
   804. L0 measures a harmful partner, so the primaries on Hands 24 and 25 do not test the book's
   claim. Arm L, with a helping partner, is the clean read there.
2. **The Hand 25 classifier was narrower than the book's words.** It required the declarer to
   lead the trick the 6-4 falls on. The book says only that the 6-4 is played "on one of your
   offs" and the other setter holds the winner. Read as the book words it, **post hoc**, under L:
   - 744 of the 804 sets are the named run;
   - sets by anything else are 60 of 4,000 deals, **0.015 [0.012, 0.019]**, inside the 0.02
     tolerance. All 60 lose exactly 15 count.

   That read is post hoc and is labelled so wherever it is cited.
3. **Our declarer ducks on Hand 23, where the stack is silent.** In 429 of 4,000 deals, 0.107
   [0.098, 0.117], the declarer follows low under a setter's trump lead while holding a tile sure
   to win, and loses a second trick. That accounts for 412 of the 1,718 sets. Hands 24 and 25 have
   no such duck.

   No book rule fires there; the stack's lowest-legal fallback plays the low trump. "The follower
   who is not last and can take the trick" is a silence already recorded in `silas/GAPS.md` (the
   book's "win any trick you can" names no tile for him). The book's Hand 23 narrative does speak
   for this seat: after the planned loss the declarer can lose no other trick. Whether that
   sentence licenses a narrow rule is on #245.

**Correction (2026-10-22, same day).** The first version of this section listed a fourth
fault: setters throwing count "blind" on trick one, 268 sets, 0.067 [0.060, 0.075]. Reading the
deciding rule showed it is `pounce`'s `play-a-ten-now`, the book's own play (W42 p. 46): a setter
playing before his partner to a lead his partner may still beat plays count at once, a ten before
a five. It is the book faithfully encoded, not a fault. So all count on Hand 23's trick one,
469 + 268 sets, is the book's pounce.

What the instrument still shows, scoped to it:

- **Hand 24: the book's line holds, and its key move is the make.**
  - With a helping partner the line makes 30 in 94.5% of deals.
  - Without the 5-1 before the 6-4 it makes 67.8%, so drawing the 6-5 is worth 0.27 of a make.
  - Our doubles trump makes 0.172 more often than the aces the book turns down.
  - Against Zeb's setters the line makes 75.7%, short of the 0.85 bar. The cinch is scoped to
    our book-playing setters.
- **Hand 25: the book's structure holds; its likelihood is the open question.**
  - Read in the book's words (post hoc), the named run is all but the only way to be set: 60
    other sets in 4,000.
  - The run itself happens in 744 deals, 0.186 [0.174, 0.198]. The book calls its odds low.
  - Our setters play the book's own advice for setting a no-trump bid: save the pairs and the
    count for the last tricks, as against an 84 (`defend-the-84`, `setter-leads`). The
    named run is what that defence aims at.
  - So the gap is between two of the book's own pages: its read of how often setters hold the
    6-4 back, and its advice that they should. A setter who plays the book's defence reaches the
    named run far more often than "low odds" suggests. Which page the book means a bidder to
    plan against is a question about its readers (A-family assumptions, [[book-assumptions]]),
    not a ruling.
  - Removing the key move costs nothing here: L − K is −0.028. Leading the 5-5 and 2-2 first is
    a little better against these setters.
- **Hand 23: the book's budget is exceeded by count on the first lost trick.**
  - Trick one is where the hand is lost. The book budgets 11 for it: the 5-5 and its trick.
  - In 737 sets, 0.184, setters pounce count onto the trick the 5-5 takes, both plays of the
    book's own `pounce`:
    - 469, 0.117 [0.108, 0.128], by the setter after the 5-5, who has seen it win;
    - 268, 0.067 [0.060, 0.075], by the setter before it, who plays count at once on a lead his
      partner may still beat (W42 p. 46).
  - In 266 sets a setter wins trick one with a lower double and keeps the 5-5 back.
  - The book's named run sets only 33.
  - On Hand 25 the book itself names this risk: a void opponent may play count on a lost first
    trump trick, which it gives as about 80 percent likely. Its Hand 23 budget leaves that count
    out. What the book's 31 assumes about the setters there (that they don't see the pounce, or
    don't hold the count) is a reading for #230. Our setters, playing the book's rules, take it.
  - With the ducks removed the hand would still fall short of a cinch. Duck-free deals make
    0.634, an upper estimate, and selected.

**Our encoding.** The general stack leaves the book's line on all three hands. On Hand 23 that
costs 0.177. On Hand 24 it costs 0.035. On Hand 25 the stack does slightly better than the
script (−0.028), since its order of the first two doubles differs.

Of the departures, three are silences already recorded in `silas/GAPS.md`, where the arena's
tie-break decides:
- which sure trump leads (Hand 24 opens with the 5-5, not the 6-6);
- which low trump leads (Hand 23's 0-0 against the book's 1-1);
- the follower who is not last and can take the trick (the duck).

One was an encoding gap with the book's stated reason behind it: `bidders-plan` dumped the 6-1
early on Hand 23. The book saves the 6-5 for next-to-last, treating it as the 6-1's double
(p. 77, S-0105-d), but our `bare-off` counted only a literal double as an off's guard. Hand 24's
3-1 before the 5-1 depends on the world, not on the encoding: `bidders-plan` picks by the count
still out, by the book's rule.

**The repair (#245, PR #247, 2026-10-22).** An off's guard is now the double or the top tile
left in its suit (`stands-for-the-double`). On step C's same 4,000 deals, Hand 23's stack arm
makes **0.575 [0.560, 0.591]**, up from 0.394 [0.379, 0.409]: a paired **+0.181 [+0.168,
+0.195]**, with plays changed on 2,915 deals. The line − stack gap closes from +0.177 to
**−0.005 [−0.008, −0.001]**. The general stack now plays Hand 23 as well as the scripted line;
the rest of the shortfall is the book's pounce and the duck above. On Hands 24 and 25 no play
changed, and the line arms are unchanged. The duck stays a silence; whether Hand 23's "no
other trick" licenses a narrow rule is still open on #245.

## Data

Per-deal rows (hand, arm, deal index, the four hands, the 28 plays, points, made), with
`run.py` and `read.py`, go to Hugging Face: `jasonyandell/texas-42-zeb-research-2026-10`
under `book/book-line-230c/`.

## Stack version (#245)

These numbers were played on the book stack as it stood before #245 (2026-10-22), which
changed what the bidder counts as an off's guard (`bare-off`, `bidders-plan`; the double or
the top tile left in the suit, W42 p. 77). They reproduce at commit `64ecc6cf`. Re-runs on
the repaired stack are tracked on #245.
