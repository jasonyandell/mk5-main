# What the book needs that Silas cannot yet say

The book's rules (Roberson, *Winning 42*) are written as Silas policies twice. The 21
rule gates of the book study ([book-study-2026-10-12](../wiki/experiments/book-study-2026-10-12.md))
are encoded in `examples/book/gates/` and checked in `tests/book.rs`: the record of what
that study priced. The rules as the book means them, read from the whole book and its
worked deals (the private coverage study's catalog), are the eighteen policies of
`examples/book/`, checked in `tests/book_rules.rs` on the book's own positions, each with
the bidder in all four chairs; the frame facts it narrates (what an off risks, what an
84's defender watches) are queries, checked in `tests/book_queries.rs`. A rule that needs something Silas lacks is written as the
closest faithful encoding. Each missing piece is marked `; GAP <name>:` at the spot where
it would go, and the test shows what the rule does there instead. This file lists the
open gaps, ordered by how many of the book's rules each one blocks, and then the closed
ones.

**Where the encodings stand.** 20 of the 21 gates are encoded fully; C14 is partial, its
"most tiles out" left out as the study priced it (the gates are a record; the crisis rule
states it with the aggregate, Closed below). Of the eighteen book-faithful rules, twelve need
no gap; six carry one: the bidder's plan (gap 3, the donation lead), the crisis, the
partner's leads and the dead double (gap 1), the trump set (its "trumps as strong as
the bidder's" reads the bidder's hidden trumps: a belief, gaps 1 and 4, so the policy keeps
its closest viewer fact), and the defence of the 84 (gap 1; its "more tiles out" is
statable with the aggregate, Closed below, and not yet written). Each gap below lists the
rules it blocks; the gates' misreadings, and the rule that reads each, are the last section.

"Viewer" means the fact is public, or is the viewer's own: a policy may read it. "World"
means it needs a hidden hand or a model of another seat's play.

| # | gap | kind | rules blocked | viewer / world | home |
|---|---|---|---|---|---|
| 1 | policy likelihood (#128), narrowed again: the readings are built; a rule acting on one is not | belief in a guard | 4 places in the rules | world | Silas |
| 2 | cross-game policy comparison (#130), narrowed | new layer | 3 (C17 in part) | world | Silas and the arena |
| 3 | another seat's knowledge (#132) | modal construct | 1 (the donation lead) | world | Silas |
| 4 | the auction as information (#134), narrowed again | belief | 1 (pp. 53–54's offs) | world | Silas |

## 1. Policy likelihood: reading a convention, and acting on the reading

**What.** A convention's receiver deletes the worlds in which the sender would not have
played what he played: for each world, at each of the sender's past decisions, ask
whether his policy, at his information state in that world, allows what he did. The
sender's information state includes his own hand, so this is world information. The
arena's lesson is that the likelihood must be the sender's gate exactly inverted.

**Built: the readings (2026-10-13; README, The record as evidence).** `Reading` reads
named seats' calls (`Reading::calls`, the auction, built first) and their past plays
(`Reading::plays`, `with_plays`), each under a compiled policy, and `Belief::given`
weighs the worlds by them, exactly, `exact_given` one model per seat and phase. The
sender's gate is inverted exactly because it is the same compiled policy. It factorizes
because a seat's frame at each of its past decisions is its dealt hand (its hand in the
world and the tiles it played) and the public record, so a world's weight is a product
of per-seat weights, each a function of that seat's hand alone; `tests/play_reading.rs`
holds the counting to brute force. A play policy allows a set; the reading takes the
**consistent** likelihood by default (a play weighs one when allowed, zero otherwise: an
event, so the posterior is a count and a policy that allows everything is no evidence)
and offers the **uniform** one (`1/|allowed|`: the seat picks at random within the set,
an extra hypothesis) under its own label. A play the fallback decides is silence. The
readings moved (`tests/book_reading.rs`, `examples/book/reading/`):

- C2, the book's singleton reading (p. 25): 26/57 uniformly in Hand 30; read under the
  worked hand's sender, still 26/57; under p. 25's sender, 1.
- C3 (p. 106): a five given means no ten held: Hand 31's 5-0 read under
  `donate-when-safe` puts the 5-5 with a setter, 5/8 to 1 (S-0138-e's second reading).
- C8's reading (p. 54: a count follow means no choice) on a trump lead: Hand 31's 3-2 read
  under `protect-the-count` puts both trumps out with the left setter, 5/22 to 1 (S-0138-e's
  first reading), and gives the left setter his certainty at trick six (S-0140-e), 2/15 to
  1. C8's own position (p. 54, a double) is statable the same way and not yet written.
- C21 (p. 107): the bait read under `crisis` is evidence, not knowledge: 5/16 to 91/167
  (1549/1777 under the uniform likelihood). The book's certainty needs a bidder who never
  leads the 6-5 without the 6-4, which no rule it states says.
- C1, the second receiver (p. 25: no donation means no count), is statable the same way
  (`donate-when-safe` read on the partner) and not yet written.

**What remains: a rule acting on a reading.** A policy's guard reads its frame, never a
belief, so a rule cannot say "if the reading puts both trumps with one setter". The book's
rules that act on a reading are blocked there (each marked `; GAP likelihood`):

- `defend-the-84.silas`: the 84 defender throws first the low tile that tells his partner
  which double to drop (p. 72), and the bidder's partner keeps low tiles as decoys (p. 66,
  Hand 34; S-0147-d); both read another seat's discards as his policy. The rule allows
  every tile that wins nothing.
- `partner-leads.silas`: the one exception to "never lead trump" (p. 40) turns on how
  many trumps the bidder still holds, read from his bid and play; trump is never led.
- `crisis.silas` (the prelude's `trumps-he-cannot-draw`): in Hand 31 the bidder places
  both outstanding trumps with one setter by reading the other's count play (p. 106). The
  reading is now a number (5/22 to 1, above); the rule that acts on it is not.
- `dead-double.silas`: in Hand 34 (p. 114) the partner keeps his dead 1-1 a while as a
  decoy for an opponent's reading; the rule throws it at once.

**Built: the guard that reads (#128, 2026-10-14; README, The record as evidence).**
`ReadingPredicate` is a viewer atom, `(name)`, registered by the host: it holds when a
named reading (chairs counted from the bidder, their play policies, a likelihood) puts an
event at a probability of at least a literal bound, evaluated by `Reading::evidence` and
`Belief::given_for` at the frame, the viewer's own chair dropped, undefined on an
impossible record, remembered by record. `tests/guard_reading.rs` holds a rule that leads
trump into the left setter to firing exactly where Hand 31's reading is certain.

**Built: a reading in a policy file (#162, 2026-10-14; README, The record as evidence).**
`(reading name (read chair policy) ... clause ...)` in a policy declares the guard; the
reader is found by name (`compile_reading`; the `silas` command looks beside the file).
`examples/book/crisis-read.silas` is the crisis as Hand 31 reads it: its two-trump trigger
asks that one setter certainly holds both trumps out, read from the setters' plays under
`protect-the-count`. It leads the bait at Hand 31 trick 2 as `crisis.silas` does, and waits
where `crisis.silas` fires on two trumps out that no play places (`tests/reading_text.rs`).

**What remains.**
- **The arena.** `scenario_core.SilasPolicy` compiles each file alone against the standard
  registry, so `crisis-read` is not yet in the book player's stack; that binding is gated
  (`scenario/`) and is the second half of #162.
- **Hand 27's crisis** (S-0127-a) reads the partner's follow to the bidder's trump lead to
  place the undrawn trump with a setter, before the partner has shown out; `crisis-read`
  keeps `crisis.silas`'s shown-out condition and does not write that reading yet.
- **The other three rules, which are not a guard's reading.** `dead-double`'s decoy (Hand
  34) and `defend-the-84`'s decoys keep a tile for an opponent's reading of the viewer's
  own plays: that is the reading from another seat, gap 3 (#132), not a reading of others.
  `defend-the-84`'s signal (the low tile that tells the partner which double to drop) is a
  sender and receiver of one convention; the receiver reads the partner under the sender's
  own policy, so the file would read itself, which is refused. `partner-leads`'s exception
  reads the bidder's trumps from his bid as well as his play: a reading of calls needs the
  match state and the declarations offered, which a play frame does not hold.
- Cost: a reading per new record (40 ms on average at the book's positions, 1.3 s at worst).

## 2. Cross-game policy comparison (Ch. 2, 7, 12)

**What.** Several of the book's rules are not about a play. They compare two whole
policies over many deals, under a scoring and a match state:
- bid only enough to win the auction, never 32 or 33 (pp. 21–22, 118–119);
- one mark from the game, bid 42, not 84 (p. 64);
- with a double protecting the off, an 84 bid is always right (p. 64);
- keep good setters honest with an occasional straight-off 84 (p. 65).

The design is `docs/cross-game.md`: four objects (the match frame, a bid action, a
comparison of policies, an equilibrium claim) built in five slices.

**Narrowed (slice 1, 2026-10-13).** The match frame is in (README, The match):
`Scoring` (points to 250, marks to seven), `MatchState` (first to the target; the
deciding hand's bidder wins a tie at 250), `Outcome`, `Contract::settle` (marks: the
award; points: the book's rule, S-0017-c, with S-0082-a's "they get 84" for a set marks
bid as the other `SetReading`), `Utility::{Marks, Points, MatchWin}`, a match-win value
whose properties Silas checks (one on won states, zero on lost ones, in between never
rising with the other side's score nor falling with its own), the score-grid programme
that builds one from a hand-outcome model (`exact_given(model)`), and the `(dominates
...)` form, decided exactly by enumerating the hand's 43 outcomes from every state of
its frame, under both readings wherever points are scored (`examples/book/cross-game/`,
`tests/cross_game.rs`). What that moves, by the spec's clusters:

- **F, the scoring frame: closed** (S-0015-c, S-0017-c+S-0132-a, S-0020-a,
  S-0063-a+S-0154-b, S-0063-b, S-0072-b, S-0073-a, S-0113-a+S-0209-b, S-0114-a,
  S-0156-a, S-0166-a, S-0166-b, S-0168-b, S-0199-b, S-0208-b, S-0208-c, S-0208-e,
  S-0210-e). The definitions are the types above, and the contract's stakes (an 84 two
  marks, 126 three). Its two one-line theorems are tests: a set scores at least 43 for
  every points bid, the floor reached (S-0063-b), and a set outscores any make. Under
  S-0082-a's reading a set 42 pays exactly 42, one under the floor: the readings part
  there and nowhere else. Two of the eighteen keep a number for later: S-0073-a's "trump
  sets score 60 to 70" is a histogram over deals (slice 3), and the auction's own rules
  (the bidding limits, the forced bid, the raise steps after 84) are definitions whose
  legality is slice 2's `AuctionConfig`.
- **B, the three dominance rows: stated and decided.** S-0170-a's second half,
  "one mark from the game, 42 not 84", is `last-mark.silas`; S-0089-b, the book's own
  "within 42 points of 250", is `last-mark-points.silas`; S-0082-a (an 84 is two marks)
  is `settle`. Both hold, exactly; two marks out the 84 is not dominated
  (`two-marks-out.silas`). The policy form ("bid 42") adds a field hypothesis, that the
  field's overcalls after either bid settle in the same order; S-0165-b warns it can
  fail. It waits on the auction (slice 2) and its measurement (slice 4). The other ten
  B rows (far behind, gamble; ahead, protect) are expectations under a match-win value
  from a field's outcome table: the programme is here, the table is slices 3 and 4.
- **A, the contract half: stated.** Holding the contract, a lower point bid dominates a
  higher one under both scorings (`bid-only-enough.silas`, `bid-only-enough-points.silas`;
  by the hand's marks or points too). The nine rows (S-0028-d, S-0033-b, S-0034-a,
  S-0036-e, S-0053-a, S-0128-c, S-0137-e, S-0152-c+S-0153-b, S-0153-a) move to a file
  plus a field hypothesis: that the field rarely overcalls a 31 and not a 33 is a
  property of its bidding (slices 2 and 4).
- **C, the settlements.** The lay-downs' marks (S-0082-b; S-0188-a, three marks) and
  the 84 made (S-0149-a) are `settle`'s; the rest of C is a belief at the auction
  under a field (slices 3 and 4).
- **G, re-tagged: never this gap's** (S-0085-b, S-0087-b, S-0090-a+S-0096-a,
  S-0092-a+S-0097-a, S-0092-b, S-0095-a, S-0098-a, S-0098-b, S-0099-a, S-0099-b,
  S-0100-a, S-0100-b, S-0100-c, S-0101-a). Rules for playing and defending an 84 (save
  the doubles, hold pairs, the discard order) cited #130 only because the contract once
  flattened an 84 to 42; `(bid 84)` and `(stake 2)` are guards now (Closed, below). They
  are play rules, statable now except where another gap holds them: policy likelihood
  (#128) on four, another seat's knowledge (#132) on one, and a play-out over a belief
  (gap 7) for any claim about such a rule's set rate; the three the aggregate (#135) held
  are statable since it was built. The row-by-row split is the coverage archive's, which
  the spec summarizes and does not list.

**Narrowed again (slice 2, 2026-10-13).** The bid action is in (README, The auction;
`docs/cross-game.md` § 4, Slice 2 as built): the auction in the record with the
contract derived from it, `AuctionConfig` (the marks cap, the forced bid), the auction
frame and its predicates, `(bidding ...)` and `(declaring ...)` sections, and
`examples/book/bidding/`. What that moves:

- **A, bid only enough: statable.** `bid-only-enough.silas` is the rule as the arena
  plays it (`HeuristicBidder`, reproduced exactly: call for call on the arena's auctions,
  and on every hand), `never-32-or-33.silas` its "never 32 or 33". The claim that the
  extra points buy no auction is a property of the field's bidding: measured in slice 4.
- **D, the bid book: statable as a policy, two rows priced.** `risk-budget.silas` is
  the book's count as we read it, its table of calls on the study's 30 hands pinned
  (the book's call on 18; bid or pass on 22, against always bidding's 23). Over all
  1,184,040 hands its opening histogram is exact: S-0053-d's "about half are not
  biddable" holds (54.7% pass), and S-0153-a's reasoned bids 30, 31, 35 and 36 are
  61.5% of its bids. Both are prices of our reading, not of the book's judgment, which
  no one formula reproduces (its printed risk counts: 8 of 12). The bids' verdicts
  (whether a call makes) wait on slices 3 and 4.

**Rules still blocked.** C17 (the policy form, "bid 42 not 84", needs the field's
overcalls after either bid: slice 4; the dominance is stated), C18 (a belief at the
auction under a field: `P_F(make all 42 | shape) > 1/2`), C19 (an equilibrium: the field
adapts to the policy, so no fixed-field comparison can state it; slice 5), and the
verdicts on the bid book's calls (slices 3 and 4). C16's auction half is statable.

**Smallest addition.** A comparison of two hand policies over a measure of deals
against a named field: exact given the field over a listed set, a fiber or all hands,
sampled otherwise, with the arena's paired ledgers imported as outcomes that Silas
settles itself (slices 3 and 4).

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
109). The partner's bid is the only information before play (p. 89).

**Narrowed (slice 2 of gap 2, 2026-10-13).** The structure is in. The whole auction is
in the record, every call attributed, and the contract is derived from it; a play rule
reads it as public fact: `(shaker c)`, `(bid-of c n)`, `(passed c)`, `(no-bid)`,
`(high-bid n)`, `(high-bidder c)`. A bid is an action of a policy, a function of the
auction frame, so a bidding policy is the object a likelihood would invert. So "the
bidder bid 35" and "the partner passed" are now clauses a rule can state.

**Narrowed again (the bid as evidence, 2026-10-13; README, The record as evidence).** A
seat that has heard the calls conditions its belief on each read seat's calls (and the
declarer's declaration) being what that seat's bidding policy makes from its hand:
`Reading::calls`, `Belief::given`, and `silas query ... --auction ... --read ...`.
Exact by counting, `exact_given` the read policies, never sampled; held to brute force on
every world of small frames (`tests/reading.rs`). The match state the bids were made at
and the declarations offered are the reading's arguments, and the auction must be the
table's (`--auction`), never a contract-only hand's convention. What moved:

- **C4, the bid as a signal (p. 89): statable as a query.** In Hand 29 the partner reads
  the bidder's 30 in fours under `bid-only-enough`: the 4-4 from 1/3 to 6,003/8,669, two
  doubles or more from 313/456 to 6,814/8,669 (`examples/book/bidding/bidder-holds-two-doubles.silas`,
  `tests/book_auction.rs`).
- **C12B's condition** (S-0232-c, "only the auction could move it"): 194/285 from the
  bidder's hand; the three passes after his 30 move it to 69.4% read as
  `bid-only-enough`, 67.0% as `risk-budget`.
- **p. 109's "a losing 31 means a double or two"** reads the doubles alone: statable
  the same way, not yet written.

**What remains.**
- pp. 53–54's "a 35 means two offs, a 36 one off": a question about a whole hidden hand
  resolves most of the pool, and a belief's points are placements of what it resolves,
  so it is refused above the cap. It needs a seat's hand law (the read seat's agreeing
  hands, each weighted by its completions, which the counting already has) as a belief
  of its own.
- A policy acting on a reading: a rule's guard reads its frame, never a belief, so a
  play rule may state the bid ("partner bid 30") but not its reading; the book's rules
  that act on one are statable as queries, as the repaired rules are.
- A seat mid-auction reading the calls before its own: a `Frame` exists only once the
  auction is won, so the reading is at the play.
- The match state in the play frame, for a play rule that reads the score.

**Rules.** pp. 53–54's offs.

**Smallest addition.** A seat's hand law as a belief over that seat's hands (the counts
are there), and the match state in the play frame.

## Closed

Closed by the second ergonomics pass (2026-10-14; README, Short forms, The auction and
Diagnostics), the friction met writing the bidders of issue 141:

- **A suit's order before the declaration.** `(ranks-above a b q)` is the suits' face order
  (the double, then the other pip down), a tile fact of both phases, held to the trick key of
  every straight declaration on each suit it leaves alone; `(top-of-suit d t n)` and
  `(holds-top t n)` say "the n highest tiles of suit t" with it. The optimists' "top two of a
  suit" no longer anchors to the 6-5 and `q6`.
- **The extremum.** `(least t q c ...)` and `(most t q c ...)`, short forms for "a candidate no
  rival beats on q"; a cascade of declaring rules, one per length, is one rule.
- **A population of contracts from Silas files alone.** `silas coverage --bidders POLICY
  --play POLICY`: the random deals' contracts are a bidding policy's auctions and the play
  model is a stated policy. The study that needed Rust (issue 141's) is
  `examples/populations/` and one command.

**Prelude: what is still a family of literals.** Every bound of the bidding prelude takes a
`(number n)` (`risk-at-most t 14` needs no definition of its own), the frame's own numbers
are functions (#144), and the call is a number (closed below, #148). The book prelude's
`four-trumps-or-more` and `exactly-*-trumps` predate the aggregate and are scopes of distinct roles; restating them as a
`card` reads alike but changes the pinned cores of every rule that calls them, so they wait
for a deliberate re-pin with a proof of equal meaning, as the bidding prelude had.

Closed by the computed call (#148, 2026-10-14; README, The auction): a bidding rule's call
may be a number of its guard's answer, `(rule priced (bid (- 42 (risk t))) (least t (risk t)
(suit-at-least t 3)) (<= (risk t) 12))`. At each answer the number is a bid where it is one
value the table calls a legal bid; the rule calls the least such, and where none is (no
bid, not legal, no value) it passes to the next rule, as a literal level does: it never
clamps. Two additions to the numbers went with it: the difference `(- a b ...)`, an
interval over the integers (a part of a difference is never counted only to a cap), and a
definition whose body is one number, called where a number goes (`(risk t)`, so the risk is
spelled once in the prelude where it was spelled twice; `less-at-risk` is gone, `(< (risk
u) (risk t))` says it). What it moved:
- `risk-budget`'s thirteen levels are one rule; its calls at every least raise and its
  declarations won at 30, 42 and 84 were equal to the literal levels' on all 1,184,040
  hands (one digest, run once; the suite keeps every 8th hand and the all-hands pin of the
  opening, over 31 and over 36, `tests/bidding.rs`), and the old text is held to the old
  pinned core;
- its pin moved (`tests/core_pins.rs`), the prelude's `risk` is pinned new, `less-at-risk`
  is removed; `risk-at-most` and every other pin, `marginal-30` among them, are unchanged.
Not moved: `never-32-or-33` bids 34 over a 32 or 33 and the least raise else, a call that
is a function of the least raise but not one number of it without a maximum (`(max 34
(least-raise))`); its two rules are honest as they stand. `marginal-30` could say `(<= 13
(risk t) 14)` for its two `risk-at-most` clauses, a re-pin for wording alone.

Closed by functions as numbers (#144, 2026-10-14; README, Semantics and Short forms): a
predicate functional in its last argument, a number (`to-set`, `to-make`, `bid`, `stake`,
`trick-number`, `trick-of`, `banked`, `quota`, `count`, `bid-of`, `high-bid`, `least-raise`,
`score`, `short`), read without it, `(to-set)` or `(count d)`, is its number in a
comparison; where it has none (undefined, or holding for no `n`) the comparison is
undefined, so negation stays exact. A core form, since a definition's `(number n)` and a
sum cannot be spelled as literals where they are read; a built-in's relation is defined by
its number, so the two agree by construction. Comparisons chain as in Lisp, `(<= 1
(to-set) 6)`. What it moved:
- `bid-only-enough`'s three rules of least-raise ranges are one, "the least raise and the
  count exposed come to 42 at most"; `never-32-or-33`'s four are two;
- the book prelude's `to-set-within-6` to `-21` and the six `*-sets-with-*` sums are gone:
  `lead-sets` is one sum, `(<= 1 (to-set) (+ (count d) (count called) 1))`; `every-trick`
  is `(<= 42 (bid))`; `bid-35-or-more` is `(<= 35 (bid))`, written where it was read.
Every call of the two raising policies at every least raise (30 to 42, 84, 126, 168, and
none left) is pinned equal to the literal ranges' on all 1,184,040 hands, and the prelude's
definitions to their literals in all three truth values on every frame of 170 seeded hands
at every bid (`tests/bidding.rs`, `tests/functions.rs`).

Closed by the plays as evidence (#128 for the readings; 2026-10-13; gap 1 above): a belief
weighed by named seats' past plays, each under its play policy, exactly, under the
consistent likelihood (an event) or the uniform one (`1/|allowed|`), with the calls read
from the same hand. Four of the five readings gap 1 listed are numbers now (C2, C3, C8's
on a trump lead, C21); C1 is statable.

Closed by the bid as evidence (#134 in part, #128 for the auction; 2026-10-13; gaps 1
and 4 above): conditioning a belief on the calls of named seats, each under its bidding
policy, exactly.

Closed by the aggregate (#135, 2026-10-13; README, Semantics and Short forms):

- **Comparing sizes and sums.** `(card (roles ...) clause ...)` counts the bindings that
  satisfy a body, `(points (roles d) clause ...)` sums their count, `(+ ...)` adds, and
  `(<= a b)` compares, with `<`, `>=`, `>`, `=` and `/=` its short forms and a definition's
  `(number n)` its bound. Three-valued by intervals: decided wherever the body is, so
  always over the viewer's hand, and in each world over a hidden one. Numbers stay
  literal-only; nothing binds one. What it moved:
  - the crisis's "the double whose suit has the most tiles out" (C14's book reading, p. 96;
    `crisis.silas`, p. 93) is the prelude's `(more-out t s)`, and the crisis's last two rules
    keep only that double (`tests/book_rules.rs`, Hand 26); the C14 gate keeps its gap, as
    priced;
  - the bid rules (`lib/bidding.silas`): each "at most n points" is one `points` bound, a
    suit's length one `card`, the arena's tie-break one comparison, and the sixteen
    declaring rules of `bid-only-enough` are two. Every call and declaration of the three
    policies is pinned equal to the old encodings' on all 1,184,040 hands; `risk-budget`'s
    call costs 20 µs, not 150.
- **Statable, not yet written:** the 84 watch list's "keep the double with more tiles out"
  (pp. 66-81: `more-out`; `defend-the-84.silas` encodes doubles before pairs and allows either
  suit between two doubles or two pairs), and C15's choice by what sets
  over several called tiles (`lead-sets` sums one called tile, `(count called)`, since #144; a
  `points` over the called count tiles states the general case).
- **Not the aggregate's:** the trump set's "trumps as strong as the bidder's" reads the
  bidder's hidden trumps, which no policy may; as a query over the setter's belief it is
  statable now.

Closed by the cross-game spec's slice 2 (2026-10-13; gaps 2 and 4 above):

- **The auction in the record.** Calls, the auction's rules (`AuctionConfig`: the marks
  cap, the forced bid), legality held to the arena and the engine on every legal prefix,
  the passed-out hand, and the contract derived from the auction, never given.
- **The bid action.** `(bidding ...)` and `(declaring ...)` sections over an auction
  frame (the hand, the auction so far, the match state, the declarations offered),
  `CompiledPolicy::{bid, declare}`, and the auction's vocabulary (`lib/bidding.silas`).

Closed by the cross-game spec's slice 1 (2026-10-13; gap 2 above):

- **The scoring frame.** Points to 250 and marks to seven, the hand's settlement under
  either, the match state, and the match-win value with its checked properties. It
  closes the scoring definitions (cluster F) and makes a hand's marks or points a number
  Silas computes, never a ledger's.
- **Field-free comparison of contracts.** `(dominates ...)`, exact over the hand's 43
  outcomes: the last mark (S-0170-a, S-0089-b) and the contract half of "bid only
  enough".

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

- **pounce and bidders-plan, read again (2026-10-13).** The pounce's trick "the partner
  may still take" counted a ruff, so the rule played the 5-5 on the bidder's partner's 0-0
  in Hands 30 and 31, where the book's setter throws a plain tile; the book's premise is a
  lead that can lose its suit (S-0079-b: a tile the top of its suit is as good as a double).
  The prelude's `partner-may-take-it` now counts only a tile that follows. The bidder's plan
  dumped the bare off that calls for least count without counting the off's own: in Hand
  24 it led the 6-4, itself a ten, where the book leads the 5-1 (S-0107-d). Both rules
  were priced on the pour before these repairs.

### The book against itself

Where two passages, or a rule and a worked hand, disagree, the rule encodes the worked
hand and its header records both:

- keeping the last trump: Hand 13 keeps it (S-0044-b), Hand 33 leads it for the donation
  (S-0144-k); `bidders-plan.silas` encodes Hand 33;
- the 6-5 off first: p. 132 (S-0167-b) against Hand 31's trump lead (S-0138-b);
  `first-lead.silas` encodes Hand 31;
- the partner's top trump: p. 25's reading (S-0041-a) against Hand 30's play (S-0135-c);
  `partner-plays-the-top-trump.silas` encodes Hand 30.

These rules encode the book's stated rule over a worked play it gives no reason for:

- shedding for a void: p. 48 calls the void the first guideline, and Hands 32 and 33 shed
  so; in Hand 28 trick 2 (S-0130-c) and Hand 33 trick 5 the right setter throws a tile
  that voids nothing with a lone one in hand. `shed-for-a-void.silas` voids.
- the pair against the 84: in Hand 35 trick 4 (S-0151-b) the left setter breaks his pair
  of fives, which still catches the 5-2 or 5-1, and keeps a trey whose last target is gone.
  `defend-the-84.silas` throws the trey.
- doubles before pairs (p. 72) against Hand 34's afterword (S-0149-b), which would let the
  left setter keep two fours over his 3-3 with one trey out to two fours, the case on the
  narrated order (the archive's adjudication B1). `defend-the-84.silas` keeps the double,
  as the hand does.
- low from a pair (p. 49): in Hand 29 trick 3 (S-0133-j) the book says either ace will do
  and plays the higher, the 6-1, as he has another six; `low-from-a-pair.silas` allows
  only the 3-1 there.

### The book's silences (the encoding's repairs, #146)

The rule-fidelity census of the head-to-head ([book-investigation-2026-10-14](../wiki/experiments/book-investigation-2026-10-14.md))
found rules firing outside their headers' scope or against the stated order; those were
repaired, each pinned by a `repair_` test in `tests/book_rules.rs`. Where the census asked for
a choice the book does not make, the encoding makes none, and the arena's tie-break (least id,
or Zeb's best tile inside the set) still decides:

- **Which sure trump leads** (`first-lead.silas`, `lead-the-top-trumps`). The book says any of
  the top run; its hands lead the double, but Hand 28 leads the 6-0 holding the 0-0 as bait
  (S-0129-d, a reading of the setters: gap 1). No rank is encoded.
- **Which setting lead** (`setter-leads.silas`, `lead-the-count-that-sets`). The book's two
  hands choose between a lead that sets and one that does not; among leads that all set it
  states no order (Hand 28 trick 4 picks by "the odds", not by reach). No order is encoded.
- **A count off on the setters' trick** (`follow-with-the-off.silas`). The book plays the off
  the first time its suit is led and does not say whether an off that is itself count goes
  on a trick the setters are taking. The rule still plays it.
- **The follower who is not last and can take the trick** (`low-from-a-pair.silas`,
  `take-the-trick.silas`). The book's pair rule does not treat a higher tile that takes the
  bidder's side's trick (the catalog's note on S-0071-c); its "win any trick you can" (S-0056-b)
  is a general precept with no tile named for a follower not last. Low-from-a-pair is now silent
  there; no rule takes the trick.
- **Hand 27's crisis** (`crisis.silas`). The crisis now needs the partner shown out of trump.
  Hand 27 places the undrawn trump with a setter by reading the partner's follow to the
  bidder's trump lead (gap 1); without that reading the rule waits. Whether a crisis outlives
  a decided contract the book does not say; the rule does not ask.
- **The dead double against shed's lone tile** (`dead-double.silas`, `shed-for-a-void.silas`).
  The headers conflict (the dead double "thrown first"; the void "the first guideline") and the
  book's throws do not settle it; the stack's order is kept (dead double first).

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
(Ch. 7), a play claim rather than a deal claim; whether the first way to set Hands 17
and 18 always sets (c07-three-out-an-opponent); and how often Hand 34's two ways set
(c12-hand34-route-two), which are holdings the defender must keep to the end.

**Smallest addition.** A play-out over a belief: for each world, play every
seat's policy at its own information state up to a horizon, then evaluate a
query, carrying the belief's authority. It needs gap 3's per-seat frames.

## Added by the solved outcomes (`README.md`, Solved outcomes)

The book's claims of certainty need the outcome of the rest of the hand, not a fact about
the present. The coverage study named this "game value under best defence" and gave it no
number (S-0054-a + S-0210-a, S-0026-c). It is closed for the claims the book makes about
one player and narrowed to the two gaps below. The four outcome predicates state the
rest exactly: `(lay-down)`, `(certain-make)`, `(certain-set)` and the double-dummy bound
`(clairvoyant-dd-make)` (behind the `clairvoyant` feature: STRATEGY-FUSION FENCE). They are tested against brute force in `tests/outcome.rs`.

- **Stated now.** The tournament rule (S-0210-a) and the claim of the rest (S-0054-a) are
  `lay-down.silas`. The 84 that lays down (S-0082-b, S-0093-b, S-0094-a: Hands 21 and 22,
  certain at every seating) is there too, as is all seven of a suit (S-0188-a's lay-down;
  its three marks are `settle`'s, gap 2's slice 1). The book's test for a lay-down (S-0093-a) is
  `lay-down-shape.silas`: sufficient on every hand of the shape tried, and not necessary.
  Hand 32's claim (S-0143-c) is `claim-the-rest.silas`, made with the bidder already set.
  The hand that stops once made or set (S-0113-b) is the settled case, answered without
  search. Hand 26's "cannot be set once trumps are cleared" (S-0123-b) is
  `cannot-be-set.silas`: false as the bidder's own guarantee, exactly, though
  double-dummy play makes in all 756,756 worlds. Both are about one position, after
  trick 2 with all seven trumps played, 17 banked, the bidder to lead 3-3 1-1 5-5 2-2
  5-2, the other 15 tiles dealt five each in any way. The private adjudication's
  setting deal (A13) is one of those worlds, not an earlier fork: it differs in the
  partner's play (he keeps the 3-2 to the last trick and the bidder scores 31;
  double-dummy play has him shed it and the bidder scores 36).

| # | gap | kind | claims blocked | viewer / world | home |
|---|---|---|---|---|---|
| 8 | the side's guarantee | outcome | 1 (S-0123-b, as the book means it); large positions | world | Silas |
| 9 | outcomes other than the contract | outcome | 1 (S-0026-c, six tricks) | viewer | Silas |

### 8. The side's guarantee

**What.** `(certain-make)` and `(certain-set)` are what the viewer forces on his own. His
partner's plays range over every legal play, so a claim stands on the claimer's own
tiles. What the side can force with a partner who plays to the same goal from his own
information is different: two players with different information against a third
(decentralized). Silas does not compute it.

That side guarantee lies between the viewer's guarantee and the double-dummy bound, so it
is decided wherever the two meet. A lay-down or a forced outcome proves it; a world where
double-dummy play fails refutes it. Hand 26 is where they part. At its position after
trick 2 (all seven trumps played, 17 banked, the bidder to lead; 756,756 worlds),
double-dummy play makes in every world and the bidder alone cannot force the make: in
one world a partner who keeps the 3-2 to the last trick sets him, though he sees all four
hands. Whether the partnership can is open. The coverage study's reading is that the set
needs the partner to keep a count tile to the end.

The same section covers cost. The viewer's own guarantee is exact but exponential. From
the opening lead it is decided only by a lay-down or by a double-dummy counterexample among
the 10,000 worlds scanned. Otherwise the search runs out of budget and is refused.

**Smallest addition.** A search over the pair's strategies, each seat's choice a function
of its own frame (gap 3's per-seat frames), bounded as the viewer's is. Before that, a
second refuter: per-world minimax with the viewer alone against the other three. It is
sound for the viewer's guarantee and tighter than double dummy.

### 9. Outcomes other than the contract

**What.** The outcome predicates read the contract's threshold. "Five fours with the 4-4
and 6-4 guarantee six tricks" (S-0026-c, Hand 3) is a certainty about tricks, and the
book's 84 claims are certainties about every trick only because the bid is.

**Smallest addition.** A goal of `n` tricks (or `n` points) for the viewer's side, as a
literal argument: `(certain-tricks n)`. The search is unchanged; only the settled test
reads the goal.
