# What the book needs that Silas cannot yet say

The book's rules (Roberson, *Winning 42*) are written as Silas policies twice. The 21
rule gates of the book study ([book-study-2026-10-12](../wiki/experiments/book-study-2026-10-12.md))
are encoded in `examples/book/gates/` and checked in `tests/book.rs`: the record of what
that study priced. The rules as the book means them, read from the whole book and its
worked deals (the private coverage study's catalog), are the fourteen policies of
`examples/book/`, checked in `tests/book_rules.rs` on the book's own positions, each with
the bidder in all four chairs. A rule that needs something Silas lacks is written as the
closest faithful encoding. Each missing piece is marked `; GAP <name>:` at the spot where
it would go, and the test shows what the rule does there instead. This file lists the
open gaps, ordered by how many of the book's rules each one blocks, and then the closed
ones.

**Where the encodings stand.** 20 of the 21 gates are encoded fully; C14 is partial (gap
5). Of the fourteen book-faithful rules, nine need no gap; five carry one: the bidder's
plan (gap 3, the donation lead), the crisis (gaps 1 and 5), the partner's leads and the
dead double (gap 1), and the trump set (gap 5, its closest viewer fact). Each gap below
lists the rules it blocks; the gates' misreadings, and the rule that reads each, are the
last section.

"Viewer" means the fact is public, or is the viewer's own: a policy may read it. "World"
means it needs a hidden hand or a model of another seat's play.

| # | gap | kind | rules blocked | viewer / world | home |
|---|---|---|---|---|---|
| 1 | policy likelihood (#128) | belief operation | 5 readings; 3 places in the rules | world | Silas |
| 2 | cross-game policy comparison (#130) | new layer | 4 | world | Silas and Physics |
| 3 | another seat's knowledge (#132) | modal construct | 1 (the donation lead) | world | Silas |
| 4 | the auction as information (#134) | frame facts and belief | 3 | world | Silas and Physics |
| 5 | aggregate (#135) | language construct | 2 (the crisis's "most out"; the trump set's "as strong") | viewer | Silas |

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

In the book-faithful rules (each marked `; GAP likelihood`):
- `partner-leads.silas`: the one exception to "never lead trump" (p. 40) turns on how
  many trumps the bidder still holds, read from his bid and play; trump is never led.
- `crisis.silas` (the prelude's `trumps-he-cannot-draw`): in Hand 31 the bidder places
  both outstanding trumps with one setter by reading the other's count play (p. 106).
- `dead-double.silas`: in Hand 34 (p. 114) the partner keeps his dead 1-1 a while as a
  decoy for an opponent's reading; the rule throws it at once.

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

**Rules.** The donation lead (`bidders-plan.silas`, C1a as the book reads it; pp. 25, 29,
111). C1b and C3 are no longer blocked: their condition is the partner's own knowledge,
which is the viewer's when the partner is to play, and `donate-when-safe.silas` reads it
(a safe trick: the top unplayed lead, or the setters shown out of trump). The donation
lead's closest faithful reading is that the partner has not yet discarded on one of the
bidder's trump leads and no trump is out at all.

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
- C14 (p. 96) and `crisis.silas` (p. 93): "the double whose suit has the most tiles out".
  The rule allows every count-calling double.
- `trump-set.silas` (p. 52): "trumps as strong as the bidder's" compares two holdings; the
  rule reads it as holding the top trump still out.
- The 84 watch list's "keep the double with more tiles out" (pp. 66-81) is not encoded.

The book's C15 choice by what sets (pp. 98, 108) is no longer blocked for one called
tile: the trick's point, the lead's own count and one called five or ten make six sums,
and `to-set` against each is a disjunction of literals (the prelude's `lead-sets`). A sum
over several called tiles still needs `card`.

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

## Where the gates misread the book, and the rule that reads it

This is the harvest of #123, and of the coverage study's section 4. Each item is checked
in `tests/book_rules.rs` on the book's own deal (or on a position built from what the book
states, marked "ours"): the gate fails the book's play and the rule passes it.

- **C1a** (`bidders-plan.silas`). The book leads the top trump for the partner's count
  with the partner not shown void (Hands 12, 15, 23, 26, 28, 30) and with no trump out
  (Hands 15, 23, 33): its condition is what the partner can know (gap 3).
- **C1b, C3** (`donate-when-safe.silas`). The book gives count on any trick provably the
  bidder's from the partner's chair (Hand 30's 2-2 once the setters show out of trump); a
  low trump lead is safe once both setters have shown out; the last setter gives count to
  his partner's trick (Hand 29).
- **C2** (`partner-plays-the-top-trump.silas`). The book contradicts itself (p. 25's
  reading against Hand 30's play); the rule encodes the worked hand.
- **C7** (`pounce.silas`). The book's pounce needs a trick the setters may take (no count
  on the top trump or a walker), happens while able to follow (Hands 28, 30), plays the
  double first when count may still fall, and has a companion for the seat after the
  partner: no count on a trick already lost (Hand 34).
- **C8, C13b** (`protect-the-count.silas`). The book protects on the top trump (Hand 33),
  with any number of followers, and feeds count in a trump set (`trump-set.silas`).
- **C9, C23** (`partner-leads.silas`). The book leads the top tile of a suit with the
  doubles, and a six before a four or a five; trump never (its exception is gap 1).
- **C10, C12a, C12b** (`first-lead.silas`). The off on trick one is a permission (Hands
  12, 14 and 32 lead trump; in an 84, Hands 17 and 18, the off goes last); the low-trump
  lead holds with three trumps (Hands 16, 29), avoids the trump count tile, and allows the
  top with a trump count tile out; Hand 23 leads the 1-1, not the 0-0.
- **C11** (`bidders-plan.silas`). The book uses another suit's double for the off's count,
  strips and saves the off's double for next-to-last when it cannot win the lead back
  (Hands 16, 17, 18, 19, 25, 26), dumps a bare off before the double ahead of another
  (Hand 28 trick 3), and leads the double with the last trump in the partner's hand (Hand
  30).
- **C13a** (`shed-for-a-void.silas`). The book keeps a lone double (Hand 33) and the last
  guard of a count tile, sheds on the bidder's double too, and keeps the high tiles under
  doubles trump; against 84 it throws a dead double instead (Hand 35).
- **C13c** (`low-from-a-pair.silas`). Against 84 the book plays low from a pair with count.
- **C13d** (no rule). No page states it; its one instance is Hand 28 trick 4, where the
  book leads the count that sets (`setter-leads.silas`).
- **C14, C21** (`crisis.silas`). The book's crisis is a trump the bidder cannot draw
  (after two trump tricks in Hand 26; Hand 31's bait); it cashes a double that is count
  first, avoids the suit the trump holder followed, and never a double that calls no
  count.
- **C15** (`setter-leads.silas`). The book leads the count that sets past a double (Hands
  28, 31), a double when one point sets, a count tile that calls for more once its side
  has a trick (Hand 28 trick 5), and thinks twice about a low four, five or six against
  35.
- **C22** (`trump-set.silas`). The trump set needs the bidder's partner void too.

Double-dummy disagrees with the book at several of these (Hand 28 tricks 3 and 4, the
opening leads of Hands 34 and 35: `examples/book/leads.md`); the rules follow the book.

### The book against itself

Where two passages, or a rule and a worked hand, disagree, the rule encodes the worked
hand and its header records both:

- keeping the last trump: Hand 13 keeps it (S-0044-b), Hand 33 leads it for the donation
  (S-0144-k); `bidders-plan.silas` encodes Hand 33;
- the 6-5 off first: p. 132 (S-0167-b) against Hand 31's trump lead (S-0138-b);
  `first-lead.silas` encodes Hand 31;
- the partner's top trump: p. 25's reading (S-0041-a) against Hand 30's play (S-0135-c);
  `partner-plays-the-top-trump.silas` encodes Hand 30.

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
