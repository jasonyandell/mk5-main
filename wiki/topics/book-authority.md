---
title: The Book Authority — Winning 42 is the authority; our measurement is the suspect
kind: topic
first_seen: 2026-10-14
last_updated: 2026-10-14
status: active
---

## The charter

Jason, 2026-10-14, defining the role: "the thing responsible for driving the book to
completion. enforcing semantics like relative strength doesn't make the book less valuable.
the value is the lesson. if we can't measure the book as beneficial v Zeb, we haven't learned
enough yet. our measurements are considered the suspect, not the book. we must fully cover
even its implicit assumptions in detail and have an opinion that addresses the full scope as
best we can. eg the throw-em-off thing. the default assumption is a human opponent who is
actively modeling your behavior across hands. in our simulations Zeb wins because Zeb isn't
doing that, not because Zeb is better at 42. I believe and am willing to assume that modeling
other players makes a stronger player and the fact that our tool disagrees makes our tool
suspect. ... this needs reinforcement for a while until the pattern gets established and
fresh agents know to do this by simple example."

The stance, in three lines:

1. **Roberson's *Winning 42* ([[w42]]) is the authority.** Its reputation is earned at real
   tables; Zeb's ([[scenario-player]]) is earned in an arena that leaves out most of what those
   tables contain.
2. **A gap between the book and Zeb is a finding about our instruments:** "our tools cannot
   yet see the book's edge under assumptions X", where X names the omissions from
   [[book-assumptions]]. It is never a verdict on the book.
3. **The book is not done until every lesson, stated or implicit, has an encoding, an
   arena that holds its assumptions, and a measurement.** [[book-assumptions]] is the
   ledger; the next experiment is #149.

**What the stance does not change: the numbers.** Every number ships with its interval and its
chance baseline; a tie is not agreement; explaining a move is not agreeing with it; negative
results are reported in full and kept ([[book-study-2026-10-12]] set these rules and they
stand). The stance changes the headline and the burden of proof. A null or a loss is the
*scope of our model*, stated beside the assumptions it violates, with the measurement that
would restore them.

Supporting evidence for the working assumption that modeling other players makes a stronger
player: soft play-reading, a crude within-hand model of the other seats, made Zeb stronger by
+0.37 [+0.24, +0.51] marks/game against plain Zeb, replicated on fresh deals against an A/A of
−0.00 ([[reads-plays-soft-2026-10-14]]). That is the first modeling of other players our arena
has rewarded, and it models them only within one hand. The book's opponents model across
hands.

## Vocabulary

| avoid | use instead |
|---|---|
| "the book loses", "the book is wrong", "the book fails" | "against deployed Zeb in our arena (assumptions A1–A3 violated), the book player scores X [CI] against a baseline of Y" |
| "no rule gains", "nothing from the book was adopted" | "our arena shows no gain for any rule under these assumptions; adoption waits on the measurements that restore them" |
| "the opponent the book assumes does not rescue it" | "a reader of *this* form (within one hand, tile location only) cannot see an edge; the book's opponent also remembers and adapts" |
| "the book's information was worth zero" | "worth zero to a partner who does not read by the same convention / under a field with no styles" |
| "Zeb is better at 42" | "Zeb wins the arena's game, in which no seat models another across hands" |
| "the book against its rule" | "the encoding against the book: the rule's header omits a condition the play shows" |
| "the book is not noise" (as faint praise) | "the book's tile beats a random departure by +0.064 per departure, with less of its edge visible to us than to its readers" |

Use: *scope*, *under assumptions*, *our arena cannot yet see*, *the measurement that would*,
*the book's table*, *restored assumption*. Every book result names which inventory assumptions
([[book-assumptions]] A1–A11) its arm holds and which it violates.

## Worked examples (the same numbers, rewritten)

**1. The head-to-head headline ([[book-head-to-head-2026-10-13]]).**

- *Wrong:* "Asked to play its rules at a real table, the book loses to Zeb: 1.9 marks a game."
- *Right:* "Seated against deployed Zeb in an arena where no seat models another across hands
  (A1), the book's partners send its conventions but never read them (A2), and both opponents
  track every tile perfectly (A3), the book player (repaired encoding, Zeb's tile where no
  rule fires) scores −1.41 [−1.63, −1.18] marks/game against the Zeb-vs-Zeb baseline of
  exactly 0. That is the scope of our model. Of the published −1.88, 0.47 was our encoding
  ([[book-investigation-2026-10-14]]). The measurements that restore A1–A3 are #149, #150
  and #151."

**2. The book's own table ([[book-reads-by-the-book-2026-10-14]]).**

- *Wrong:* "The opponent the book assumes does not rescue it."
- *Right:* "A reader who uses the book only to place tiles, within one hand, changes nothing
  for the book player: −0.04 [−0.27, +0.19] marks/game, paired. The book's assumed opponent
  does more: it remembers the player across hands and changes its own play in response. This
  reader does neither. What the null rules out is narrow: within-hand tile location by the
  book's verdict sets is not where the book's edge lives. The same reader loses +0.50
  [+0.26, +0.74] to plain Zeb because its model is wrong. Modeling with the wrong model costs.
  It is not evidence that modeling is worthless."

**3. The book's game ([[book-investigation-2026-10-14]] § II).**

- *Wrong:* "This is the one win the investigation found."
- *Right:* "Where our measurement restored one of the book's assumptions, its own game (points
  to 250; A5), the book's edge became visible. As Zeb's near-tie break it wins 53.2% [51.1,
  55.2] of games to 250 against a 50% baseline, replicated on new deals. That is the pattern
  to expect: each assumption restored is a place to look, and the first one restored paid."

## How to report a book result

1. List the inventory assumptions ([[book-assumptions]]) the arm holds and those it violates.
2. Report the number with its interval and chance baseline, in full, whatever its sign.
3. Headline what our model can and cannot yet see. Never headline a verdict on the book.
4. Name the measurement that would restore the violated assumptions, and file it if no
   issue exists.
5. Where an encoding and the book disagree, the encoding is presumed wrong until a reason
   report ([[scenario-player]] § the reason report) shows the book's header states no
   condition that would reconcile them. The rule-fidelity census found 44% of the first
   head-to-head's loss was the encoding.

## Links

[[book-assumptions]] · [[w42]] · [[book-study-2026-10-12]] · [[book-head-to-head-2026-10-13]] ·
[[book-investigation-2026-10-14]] · [[book-reads-by-the-book-2026-10-14]] ·
[[reads-plays-soft-2026-10-14]] · [[scenario-player]] · [[silas]] · [[the-wall]]
