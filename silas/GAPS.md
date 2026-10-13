# What the book needs that Silas cannot yet say

The 21 rule gates of the book study ([book-study-2026-10-12](../wiki/experiments/book-study-2026-10-12.md))
are written as Silas policies in `examples/book/` and checked in `tests/book.rs`, most of
them on the book's own positions (Roberson, *Winning 42*). A rule that needs something
Silas lacks is written as the closest faithful encoding. Each missing piece is marked
`; GAP <name>:` at the spot where it would go, and the test shows what the encoding does
there instead. This file lists the gaps, ordered by how many of the book's rules each
one blocks.

"Viewer" means the fact is public, or is the viewer's own: a policy may read it. "World"
means it needs a hidden hand or a model of another seat's play.

| # | gap | kind | rules blocked | viewer / world | home |
|---|---|---|---|---|---|
| 1 | `declarer` | frame fact | all 21 gates | viewer | Silas |
| 2 | `none` | language construct | 12 gates | viewer (per pattern) | Silas |
| 3 | policy likelihood | belief operation | 5 readings | world | Silas |
| 4 | history | frame facts | 4 | viewer | Silas |
| 5 | cross-game policy comparison | new layer | 4 | world | Silas and Physics |
| 6 | `shown-void` | predicate | 3 | viewer | Silas |
| 7 | another seat's knowledge | modal construct | 3 | world | Silas |
| 8 | banked and bid | frame facts | 3 | viewer | Silas |
| 9 | the auction as information | frame facts and belief | 3 | world | Silas and Physics |
| 10 | aggregate | language construct | 2 | viewer | Silas |

## 1. `declarer`: who bid

**What.** A frame fact and a viewer predicate `(declarer c)`, the seat that won the
auction. `PolicyInput` carries `declaring_team`, but no predicate reads it, and
predicates see only the `Frame`. Every book rule is stated from a role: the bidder, his
partner, the setter on his left. Each encoding writes the bidder in as `S0` (the pour's
seat), which makes it a rule for one seating. A setter can be named relationally
(`(opponent me S0)`), but only once the bidder is named.

**Rules.** All 21 gates: C1a, C1b, C2, C3, C7, C8, C9, C10, C11, C12A/B, C13a–d, C14,
C15, C21, C22, C23 (pp. 24–54, 93–113, 183).

**Smallest addition.** Put the contract in the `Frame` (or give predicates the
`PolicyInput`), and add `declarer(c)` as a viewer predicate. With it, `(declarer b)
(partner b p)` replaces every literal seat.

## 2. `none`: a negated sub-pattern

**What.** Silas negates a single atom. The book keeps saying "no other": the only tile
of a suit, exactly two of a suit, no live tile beats the lead, no trump left outside my
hand, no second off, no count still out in that suit. Each is "there is no binding of
these fresh roles that satisfies this conjunction". The fix is a stratified form `(none
(roles ...) clause ...)` inside a pattern, whose roles are local and whose body is
checked once the outer roles are bound. It is decidable on the finite sorts, and
viewer-only whenever its body is.

**Rules.**
- C1b (p. 38): "the lead is the top trump". Written today as the else-branch of the
  not-top rule, which is negation by rule order. It holds only when the earlier rule
  would select.
- C9 (p. 40): the ordering by what a lead can draw.
- C10 (p. 24): exactly one off.
- C11 (p. 25): trumps drawn.
- C12A and C12B (p. 183): "the second-highest trump" and "my lowest trump". Written
  today as seven or fourteen literal patterns, one per pip declaration.
- C13a (p. 44): the only tile of a suit.
- C13b and C13c (p. 49): exactly two of a suit.
- C13d (the gate's reading of p. 44): the only tile of the suit led.
- C15 (p. 47): "not a low tile that calls for a five".
- The C2 reading (p. 25): "the highest trump the bidder does not hold".

Exact counts at small numbers ("exactly four trumps") can be had without it, by
pigeonhole against a full hand (C12). That works only on trick one.

**Smallest addition.** The `none` clause in `syntax.rs` and `eval.rs`. No new predicate,
no change to `Physics`.

## 3. Policy likelihood: reading a convention

**What.** A convention's receiver deletes the worlds in which the sender would not have
played what he played. Silas has `Belief::condition_likelihood` with a caller's closure.
What it lacks is the closure built from a `PolicyProgram`: for each world, replay the
record, and at each of the sender's past decisions ask whether his policy, at his
information state in that world, allows the tile he played. The sender's information
state includes his own hand, so this is world information. The arena's lesson is that
the likelihood must be the sender's gate exactly inverted.

**Rules.**
- C1, the second receiver (p. 25): no donation means no count.
- C2, the book's singleton reading (p. 25). `c02-reading-singleton.silas` gives 26/57 in
  Hand 30, where the reading is false.
- C3 (p. 106): a five given means no ten held.
- C8 (p. 54): a count follow means void.
- C21 (p. 107): the bait tells the partner where the 6-4 is.

**Smallest addition.** `Belief::condition_on_policy(&belief, seat, &CompiledPolicy,
history)`. It needs history (gap 4) and each seat's past frames, which a `World` plus the
record determine.

## 4. History: tricks before the current one

**What.** A policy's query sees the current trick's prefix, the played set and the voids.
It does not see who led an earlier trick, what each seat played to it, or which trick
this is. `PolicyInput.history` holds all of that, but predicates cannot read it. A
small set of viewer predicates would cover the book:
- `(trick-number n)`;
- `(played-in k c d)`, where seat `c` played `d` to trick `k`;
- `(led k c)`, where seat `c` led trick `k`.

**Rules.**
- C14 (p. 93): one trump trick so far, led by the bidder.
- C22 (p. 51): the trump-set trigger.
- C8 (p. 54) and C2 (p. 25): their readings, once the trick has closed.

Trick one alone is reachable today as `(quota S0 7)`.

**Smallest addition.** Give `PredicateContext::Viewer` the public record, or put the
history in the `Frame`. Add the three predicates above.

## 5. Cross-game policy comparison (Ch. 2, 7, 12)

**What.** Several of the book's rules are not about a play. They compare two whole
policies over many deals, under a scoring and a match state:
- bid only enough to win the auction, never 32 or 33 (pp. 21–22, 118–119);
- one mark from the game, bid 42, not 84 (p. 64);
- with a double protecting the off, an 84 bid is always right (p. 64);
- keep good setters honest with an occasional straight-off 84 (p. 65).

Silas policies choose plays inside one hand. They have no bid or declaration action, no
contract scoring, no match score, and no comparison of two policies' value over a
distribution of deals. Silas can compare two queries, not two policies. The last rule is
about opponents who adapt, so its "value" is over a population of opponents.

**Rules.** C16, C17, C18, C19, and the study's bid book (C24).

**Smallest addition.** Two pieces:
- In `Physics`: the contract and its scoring (marks, 84 = two marks, the game at seven),
  next to the trick tables.
- In Silas: a bid action in policies, the match state in the input, and an exact or
  sampled comparison of two policies' marks over a belief of deals, carrying its
  authority.

## 6. `shown-void`: a public void

**What.** `(shown-void c q)`, true when seat `c` has publicly failed to follow `q`. The
fact is in the viewer's `Kernel` (each hidden seat's voids). The only predicate that
reaches it, `void`, reads a world, so a policy may not use it. A shown void is not the
same as a void in every world: counting can make a seat void in every world without
showing it.

**Rules.**
- C1a (p. 25): the partner is out of trump. The encoding fires without this check and
  is wider than the gate.
- C14 (p. 93): exactly one opponent followed.
- C22 (p. 50): the trump set.

**Smallest addition.** One viewer predicate reading `kernel().hidden()`. This is the
cheapest gap on the list.

## 7. Another seat's knowledge

**What.** The book's conditions are often about what the partner can know, not what the
viewer knows:
- the bidder leads the top trump so that the partner knows the trick is safe (C1a,
  C1b);
- in Hand 33 (p. 111) the bidder leads his last trump although no trump is out, because
  the partner cannot know that.

The gate's "a trump still out" is the bidder's knowledge, and it excludes the book's
play (`tests/book.rs`, `c01a_does_not_fire_in_hand_33…`). Expressing this needs a
query evaluated at another seat's information state. That state includes the other
seat's hand, so it is world information: `(knows c query)`, true in a world when the
query holds in every world consistent with `c`'s view of it.

**Rules.** C1a, C1b and C3 (pp. 25, 37–38).

**Smallest addition.** A modal clause `(knows c ...)` evaluated by building seat `c`'s
`Kernel` from the world and the record. It needs history (gap 4).

## 8. Banked points and the bid

**What.** Viewer predicates `(banked t n)`, `(bid n)` and, more usefully, `(needs t n)`:
team `t` needs `n` more points to make or set. `PolicyInput` has `banked` and `bid`;
predicates cannot read them, and the language has no arithmetic to compare them.

**Rules.**
- C15 (p. 48): a lead that calls no count, only when one point sets. The book's own
  setter leads twice pick the count that sets over the gate's double (Hands 28 and 31,
  pp. 98, 108).
- C7 (p. 110): more urgent against 35 and 36.
- C17 (p. 64), at match level, overlapping with gap 5.

**Smallest addition.** The contract in the frame (as for gap 1), plus `needs` as a
derived viewer predicate. That avoids general arithmetic.

## 9. The auction as information (Ch. 2, 12)

**What.** The bid tells the table about the bidder's hand. In the book's examples, a 35
means two offs and a 36 means one off; a losing 31 means a double or two (pp. 53–54,
109). The partner's bid is the only information before play (p. 89). Silas has no
auction record in the frame and no model linking a bid to a hand. Reading a bid means
conditioning the belief on a bidding policy. That is gap 3 applied to bids, and it is
world information.

**Rules.** C4 and the bid inferences of pp. 53–54 and 109.

**Smallest addition.** The auction in the frame, a bid action (as for gap 5), and policy
likelihood (gap 3) over bids. The bid's meaning depends on the scoring, so the contract
belongs in `Physics`: under marks, levels from 30 to 41 pay the same.

## 10. Aggregate: comparing sizes

**What.** "The double whose suit has the most tiles still unplayed" compares the sizes
of two sets. "The lead whose called count is enough to set" compares a sum with a need.
Rule order and distinct roles give "at least n". `none` gives "exactly n". Neither gives
"more than another set". A clause such as `(card n (roles d) clause ...)` binding `n` to
a count, plus a comparison predicate on numbers, would do it.

**Rules.**
- C14 (p. 96).
- The book's C15 choice by what sets (pp. 98, 108; with gap 8).

**Smallest addition.** The `card` clause in the language, and `less(n, m)` on numbers.

## Where the encodings show the gate misreading the book

This is the first harvest of #123. Each item is checked in `tests/book.rs` on the
book's own deal.

- **C1a** fires wider than the gate. In Hand 28 (p. 98) the book leads the top trump
  with the partner not shown void. In Hand 33 (p. 111) the book leads its last trump
  with none out. The book's condition is the partner's knowledge (gap 7).
- **C2** is the table's convention. In the book's Hand 30 (p. 103) the partner plays the
  top trump holding two more, against the book's own p. 25 reading.
- **C7.** The book pounces while able to follow, too (Hand 28, p. 98). The gate requires
  a void.
- **C8 and C13b.** The book protects count on trump doubles (Hand 33, p. 111). The gates
  take only non-trump doubles.
- **C10.** p. 24 permits the off lead in that shape; it does not prescribe it. In Hand
  14 (p. 28), holding exactly that shape, the book leads trump.
- **C11.** The book leads the double ahead of the off with the last trump still in the
  partner's hand (Hand 30, p. 105). The gate requires trumps drawn.
- **C12B.** The book's examples hold three trumps (Hand 29, p. 101). The gate takes
  four only.
- **C13d.** No page states it. In Hand 28 (p. 98) the gate's tile is a singleton double,
  while the book leads the tile that calls the setting count.
- **C14.** The book's crisis comes after two trump tricks (Hand 26, p. 93), or is an off
  lead (Hand 31, p. 106). The gate's trigger is one trump trick.
- **C15.** Twice the book passes over the gate's count-calling double for the lead that
  calls the count that sets (Hands 28 and 31, pp. 98, 108).
- **C22.** The book trumps from the left seat; from the last seat it trumps only on
  count already played. The gate fires at both seats.
