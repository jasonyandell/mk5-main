# What the book needs that Silas cannot yet say

The 21 rule gates of the book study ([book-study-2026-10-12](../wiki/experiments/book-study-2026-10-12.md))
are written as Silas policies in `examples/book/` and checked in `tests/book.rs`, most of
them on the book's own positions (Roberson, *Winning 42*), each with the bidder in all four
chairs. A rule that needs something Silas lacks is written as the closest faithful
encoding. Each missing piece is marked `; GAP <name>:` at the spot where it would go, and
the test shows what the encoding does there instead. This file lists the open gaps,
ordered by how many of the book's rules each one blocks, and then the closed ones.

**Where the encodings stand.** 20 of the 21 encode their gate fully; one, C14, is partial
(gap 5, aggregate). Several gates in turn misread the book (the last section); the gaps
that would let an encoding follow the book past its gate are 1 to 4.

"Viewer" means the fact is public, or is the viewer's own: a policy may read it. "World"
means it needs a hidden hand or a model of another seat's play.

| # | gap | kind | rules blocked | viewer / world | home |
|---|---|---|---|---|---|
| 1 | policy likelihood (#128) | belief operation | 5 readings | world | Silas |
| 2 | cross-game policy comparison (#130) | new layer | 4 | world | Silas and Physics |
| 3 | another seat's knowledge (#132) | modal construct | 3 | world | Silas |
| 4 | the auction as information (#134) | frame facts and belief | 3 | world | Silas and Physics |
| 5 | aggregate (#135) | language construct | 2 (C14's gate; C15's book choice) | viewer | Silas |

## 1. Policy likelihood: reading a convention

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

**Smallest addition.** `Belief::condition_on_policy(&belief, seat, &CompiledPolicy)`.
The record is the frame, each seat's past frames follow from a `World` and the record,
and a policy is now a function of the frame (no modes or bindings, 2026-10-13), so the
likelihood of a play is determined by (policy, world, record): nothing else is missing.

## 2. Cross-game policy comparison (Ch. 2, 7, 12)

**What.** Several of the book's rules are not about a play. They compare two whole
policies over many deals, under a scoring and a match state:
- bid only enough to win the auction, never 32 or 33 (pp. 21–22, 118–119);
- one mark from the game, bid 42, not 84 (p. 64);
- with a double protecting the off, an 84 bid is always right (p. 64);
- keep good setters honest with an occasional straight-off 84 (p. 65).

Silas policies choose plays inside one hand. They have no bid or declaration action, no
match score, and no comparison of two policies' value over a distribution of deals. Silas
can compare two queries, not two policies. The last rule is about opponents who adapt, so
its "value" is over a population of opponents. The contract now carries its kind, stake
and declaration (`Bid::Points` or `Bid::Marks`, `threshold`, `stake`, `award`; 2026-10-13),
so one hand's mark transfer is defined.

**Rules.** C16, C17, C18, C19, and the study's bid book (C24).

**Smallest addition.** A bid action in policies, the match state (the game at seven),
and an exact or sampled comparison of two policies' marks over a belief of deals,
carrying its authority. The hand's award is `Contract::award`.

## 3. Another seat's knowledge

**What.** The book's conditions are often about what the partner can know, not what the
viewer knows:
- the bidder leads the top trump so that the partner knows the trick is safe (C1a,
  C1b);
- in Hand 33 (p. 111) the bidder leads his last trump although no trump is out, because
  the partner cannot know that.

The gate's "partner shown void, a trump still out" is the bidder's knowledge, and it
excludes the book's play twice (`tests/book.rs`, `c01a_is_the_gate_and_misses_the_book…`
and `c01a_does_not_fire_in_hand_33…`). Expressing this needs a query evaluated at another
seat's information state. That state includes the other seat's hand, so it is world
information: `(knows c query)`, true in a world when the query holds in every world
consistent with `c`'s view of it.

**Rules.** C1a, C1b and C3 (pp. 25, 37–38).

**Smallest addition.** A modal clause `(knows c ...)` evaluated by building seat `c`'s
frame from the world and the record (`Frame::new` with `c` as viewer and its dealt hand
read from the world and the record).

## 4. The auction as information (Ch. 2, 12)

**What.** The bid tells the table about the bidder's hand. In the book's examples, a 35
means two offs and a 36 means one off; a losing 31 means a double or two (pp. 53–54,
109). The partner's bid is the only information before play (p. 89). The frame now holds
the winning contract, but not the other seats' bids, and Silas has no model linking a bid
to a hand. Reading a bid means conditioning the belief on a bidding policy. That is gap 1
applied to bids, and it is world information.

**Rules.** C4 and the bid inferences of pp. 53–54 and 109.

**Smallest addition.** The whole auction in the record, a bid action (as for gap 2), and
policy likelihood (gap 1) over bids. The bid's meaning depends on the scoring, so the
contract's scoring belongs in `Physics`: under marks, levels from 30 to 41 pay the same.

## 5. Aggregate: comparing sizes

**What.** "The double whose suit has the most tiles still unplayed" compares the sizes
of two sets. "The lead whose called count is enough to set" compares a sum with a need.
Rule order and distinct roles give "at least n"; `none` gives "exactly n". Neither gives
"more than another set". A clause such as `(card n (roles d) clause ...)` binding `n` to
a count, plus a comparison predicate on numbers, would do it. Numbers are literal-only
today, so `to-set` can say "one point sets" but not "this lead's count sets".

**Rules.**
- C14 (p. 96): its gate's "most unplayed". The encoding allows every non-trump double.
- The book's C15 choice by what sets (pp. 98, 108): twice it leads past the gate's
  count-calling double to the tile that calls the count that sets.

**Smallest addition.** The `card` clause in the language, and `less(n, m)` on numbers.

## Closed

Closed in the engineering pass after the Lean-lens review (2026-10-13):

- **Modes and rigid bindings are gone.** A policy is a function of the frame; no book rule
  used either. This was a soundness defect, not a gap: the same frame could get two
  decisions.
- **The contract's kind and stake.** `(bid 84)` reads as the table says it, `(stake n)` is
  the marks at stake, and `to-make`/`to-set` read the threshold. It blocked no gate; it
  was a prerequisite of gaps 2 and 4.
- **The book's vocabulary.** `lib/book.silas` names the seats from the bidder, the trump
  (one predicate, `(trump q)`, under every declaration), the count, offs and walkers, and
  what a lead calls for; every rule is written with it, and what each allows is pinned
  unchanged (`tests/book_snapshot.rs`).

Closed in `0c5d7f66` (2026-10-13). The rules they blocked are re-encoded with them.

- **`declarer` (#126).** The contract is in the frame (`Contract { declarer, bid }`,
  inside the hand's `Record`), and `(declarer c)` reads it. Every book file now names the
  bidder as `(declarer b)` and his partner, setters and seats relationally; none names a
  chair, and `tests/book.rs` runs every check with the bidder in each of the four chairs.
  It blocked all 21.
- **`none` (#127).** `(none (roles ...) clause ...)`: stratified, three-valued, its roles
  distinct from one another but not from the roles around it (README, Queries). It gives
  "no other", "exactly n", "the top", "the second", "the lowest held", "trumps drawn" and
  "no count out in the suit". It blocked C1b, C2's reading, C9, C10, C11, C12A, C12B,
  C13a, C13b, C13c, C13d and C15: all twelve are now the gate, and C12A and C12B are one
  pattern each instead of seven and fourteen.
- **History (#129).** The record holds every play, attributed: `(trick-number n)`,
  `(led c d)`, `(played-to l c d)` (a trick is named by its lead, since numbers are
  literal-only) and `(trick-of d n)`. `PolicyInput` is gone; the frame is the one source.
  It blocked C14's trigger and C22's.
- **`shown-void` (#131).** `(shown-void c q)`: a hidden seat's void is the kernel's (the
  fact that forbids its worlds); the viewer's is read off the record. Tested to imply
  `void` in every world, and not conversely. It blocked C1a, C14 and C22.
- **Banked and bid (#133).** `(bid n)`, `(banked c n)`, `(to-make n)` and `(to-set n)`;
  the side is named by a chair, since teams are literal-only. It blocked C15's one-point
  exception, now a third rule. C7's "more urgent against 35 and 36" ranks the rule and
  changes no tile; C17 is a match-level rule (gap 2).

## Where the encodings show the gate misreading the book

This is the harvest of #123. Each item is checked in `tests/book.rs` on the book's own
deal.

- **C1a.** The encoding is now the gate, and the gate misses the book twice. In Hand 28
  (p. 98) the book leads the top trump with the partner not shown void; in Hand 33
  (p. 111) it leads its last trump with none out. The book's condition is the partner's
  knowledge (gap 3).
- **C2** is the table's convention. In the book's Hand 30 (p. 103) the partner plays the
  top trump holding two more, against the book's own p. 25 reading.
- **C7.** The book pounces while able to follow, too (Hand 28, p. 98). The gate requires
  a void.
- **C8 and C13b.** The book protects count on trump doubles (Hand 33, p. 111). The gates
  take only non-trump doubles.
- **C10.** p. 24 permits the off lead in that shape; it does not prescribe it. In Hand
  14 (p. 28), holding exactly that shape, the book leads trump.
- **C11.** The book leads the double ahead of the off with the last trump still in the
  partner's hand (Hand 30, p. 105). The gate, and now the encoding, require trumps drawn.
- **C12B.** The book's examples hold three trumps (Hand 29, p. 101). The gate takes
  four only.
- **C13d.** No page states it. In Hand 28 (p. 98) the gate's tile is the 1-1, a
  singleton, while the book leads the tile that calls the setting count.
- **C14.** The book's crisis comes after two trump tricks (Hand 26, p. 93), or is an off
  lead (Hand 31, p. 106). The gate's trigger is one trump trick.
- **C15.** Twice the book passes over the gate's count-calling double for the lead that
  calls the count that sets (Hands 28 and 31, pp. 98, 108).
- **C22.** The book trumps from the left seat; from the last seat it trumps only on
  count already played; and both only in a trump set. The gate fires at both seats on
  any trick. The encoding is the book's rule.

Where the gate and the book agree, the encodings now show it too: the C9 ordering picks
the book's lead in Hand 30 (p. 103, the 3-1, only a five out) and in Hand 31 (p. 107, the
3-1, no count out).

## Added by the book's odds (`examples/book/odds/`)

The odds the book prints are encoded one query per claim and checked exactly in
`tests/book_odds.rs`. Two kinds of claim need what Silas lacks; each such file is
marked `; GAP <name>:`, and the test computes the number in plain Rust instead.

| # | gap | kind | claims blocked | viewer / world | home |
|---|---|---|---|---|---|
| 6 | deal view | frame | 5 (Ch. 16 whole-hand statistics) | world | Silas |
| 7 | play-out over a belief | belief operation | 1 (Ch. 3, the 5%) | world | Silas |

### 6. Deal view: a probability over the hand the viewer is dealt

**What.** Chapter 16 states frequencies over a fresh hand: four of one suit, how
many suits are present, how many doubles, the commonest shape. A `Frame` fixes
the viewer's hand and draws only the three hidden hands, so no measure over the
viewer's own hand can be built. Each file's query is the event for one known
hand; the test enumerates all C(28,7) hands and checks the query hand by hand on
a sample.

**Claims.** c16-four-of-a-suit, c16-suits-represented, c16-doubles,
c16-two-to-four-doubles, c16-modal-hand.

**Smallest addition.** A deal kernel with four hidden seats and no viewer hand
(or the viewer's own hand hidden), so `Belief::uniform_resolving` can resolve
the tiles a whole-hand question reads.

### 7. Play-out over a belief

**What.** The Ch. 3 claim that the partner wins the deuce-trey on the trick
after the double, five times in a hundred, is the outcome of two tricks, so it
depends on how each hidden seat chooses among its legal tiles. Silas has
policies for the viewer and `step_belief` for observed plays, but no
probability of an outcome under policies for the hidden seats, played forward
world by world.

**Claims.** c03-double-ahead-later; also the straight-off 84's set rate
(Ch. 7), a play claim rather than a deal claim.

**Smallest addition.** A play-out over a belief: for each world, play every
seat's policy at its own information state up to a horizon, then evaluate a
query, carrying the belief's authority. It needs gap 3's per-seat frames.
