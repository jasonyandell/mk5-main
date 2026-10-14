---
title: Silas — a typed relational language over the worlds a seat cannot tell apart
kind: entity
first_seen: 2026-10-12
last_updated: 2026-10-13
status: active
---

## What it is

`silas/` is a standalone Rust crate: a small typed relational language for Texas 42. A
**query** binds roles (chairs, dominoes, contexts) and returns the answer tuples that
hold in each hidden world consistent with what a seat knows; a **pattern** is one
conjunctive case of a query, and cases combine by union. A **belief** is an exact
weighted set of those worlds; a query summarized over it gives exact event probabilities
and per-answer masses, and conditioning on a query gives the posterior. A **policy** is
a function of the frame, the viewer's information alone: a rule that reads a hidden hand
does not compile, and the same frame always gets the same decision. Comparing two
queries returns the first concrete world where they differ. Grammar, semantics,
predicate table and examples: `silas/README.md`, `silas/examples/`.

A **frame** is one record: the contract (declarer, bid, declaration), the viewer, its
dealt hand and the plays; everything else (seats of the plays, leader, trick, voids,
score, the viewer's kernel) is derived by one replay. Silas owns the game's nouns, the
language and every derived rule. A rules engine supplies only one declaration's four
tables through `Declaration` (led context, follow sets, trick key, count); Silas reads
them into a `Physics` value whose identity is a content hash of the tables, and computes
legality, trick winner and points, the trump and the fiber of hidden worlds itself, so
`Exact` rests on Silas's code alone. `silas::reference::Straight42` is the spec-faithful
reference (nine straight declarations); off-suit plays tie at the bottom of the trick
key, as texas-42's Lean `Declaration.key` defines.

## On Zeb's rules

`scenario/core` (feature `silas`, off by default) carries `ZebPhysics`, a table provider:
one of Zeb's ten declarations (`Rules`) as a Silas `Declaration`. The numbering is shared
and mapped explicitly: tile id = domino index (triangular), Zeb's led suit k = context
`q{k}` with 7 = `q*`, seat s = `S{s}`. Names are `zeb/P0`..`zeb/P6`, `zeb/DT`, `zeb/DS`
(doubles as their own suit), `zeb/NT`; on the nine shared declarations Zeb's tables are
the reference's, so the two physics have one identity. `scenario/core/tests/silas_physics.rs`
holds Zeb's own `legal`, `winner_offset`, `trick_points` and `Fiber` (the code its planner
runs) to Silas's derivations, and Silas's to `Straight42` (tables exhaustively; legal
plays, whole random hands and fibers seeded), and holds `DS` to its facts (no trump);
no disagreement was found. Zeb's default build is byte-identical with the feature off.
The `scenario_core` wheel carries the Silas classes (`SilasQuery`, `SilasPolicy`,
`SilasBelief`, `SilasSummary`, `SilasAuthority`, `silas_compare`); `scenario/silas.py`
builds a `SilasFrame` from Zeb's `Situation` (the bidder, the bid and the plays) and
checks Silas's reading of that record against Zeb's `Record`. Float weights enter only through
`SilasBelief.from_worlds`, labeled `sampled(n, seed)` or `exact_given(model_id)`; summaries
come back as `fractions.Fraction` (`scenario/tests/test_silas.py`: the 84 hand at 33/95,
`exact`, through Python).

## Numbers

Silas never computes in floating point. A float enters only through
`Weight::from_f32_exact` / `from_f64_exact`, which return the dyadic rational the float
denotes, bit for bit; every weight after that is an exact rational. Every belief and
summary carries an **authority**, a provenance lattice: the set of pinned models whose
weights entered it and the set of samples it was drawn from. Neither is `exact`; the
meet of two authorities is the union of each (total, never upgrades; a model's identity
survives a sample, which it once did not). Comparisons are three-valued (better, worse,
indistinguishable) against an explicit band, return the meet, and an exact pair's band
must be zero.

`Belief::uniform_resolving` answers questions about a few named tiles at the root
without enumerating the hand: one point per placement of the named tiles, weighted by
its exact number of completions; any read of an unnamed tile is refused. The book's 84
hand (p. 113: the bidder holds 5-5 5-4 5-2 5-0 4-4 4-2 2-2, fives trump) gives "an
opponent holds the 6-5 with another five" at exactly 33/95, `Exact`, over a root of
399,072,960 worlds ([[book-study-2026-10-12]]).

## The book, rule by rule

The book's rules are Silas policies twice over (2026-10-13). The 21 rule gates of
[[book-study-2026-10-12]] are encoded in `silas/examples/book/gates/`, tested in
`silas/tests/book.rs` and pinned by `silas/tests/book_snapshot.rs` (every point of the eight
narrated deals and of 1,600 random deals): they are the record of what that study priced,
and the published re-price reads them. On the book's own positions at least 10 of those
gates part from the book (#123); a formal reading of the whole book (a private catalog of
every rule and worked play, by id) lists every such mis-encoding.

The rules **as the book means them** are eighteen policies in `silas/examples/book/`,
named by what they say, each header giving its catalog ids, hands and pages, the rule in
our words, how it differs from the gate it replaces, its gaps, and where the book
contradicts itself. `silas/tests/book_rules.rs` asks each at the book's own positions
(`hands.json`, all four seatings): the book's tile is allowed where it plays the rule, the
rule is silent where it declines, and every catalogued mis-encoding has a test, named by
its id, that the gate fails and the rule passes.

| rule | replaces | what it says, in short | gaps |
|---|---|---|---|
| `first-lead` | C10, C12a, C12b | trump first (any of the top sequence); two trumps: the straight off; double-up odds; one off is a permission, never in an 84 | — |
| `bidders-plan` | C1a, C11 | draw with the top trump; the donation lead; bare offs at once while a trump can win back; else winners first, the off's double next-to-last | knowledge (#132) |
| `crisis` | C14, C21 | a trump the bidder cannot draw: the 6-5 bait, a count double, away from the trump holder's suit, its suit the most out | likelihood |
| `donate-when-safe` | C1b, C3 | count, biggest first, on a trick provably the side's; the last seat's donation; none on a trick that can be lost | — |
| `partner-leads` | C9, C23 | a double or a walker; then no count, a five, a six; never trump | likelihood |
| `partner-plays-the-top-trump` | C2 | the worked hand's play (the book's p. 25 reading disagrees) | — |
| `pounce` | C7 | count only on a trick the setters may take (a tile that follows can beat it; a ruff does not count), while able to follow; the double first; no count on a lost trick | — |
| `protect-the-count` | C8, C13b | no count on the bidder's master lead, the top trump too; not in a trump set | — |
| `shed-for-a-void` | C13a | a lone tile, never a double, a winner or a count tile's last guard; on any throw with no count to give, a tile he need not keep, then the guard; points bids only | — |
| `low-from-a-pair` | C13c | as the gate, and the 84 form with count | — |
| `setter-leads` | C15, C13d | a double when one point sets; the count that sets; count calling for more; think twice at 35 | — |
| `trump-set` | C22 | the bidder's partner void too; the big doubles; feed a holder who may yet trump; lead high | a belief (closest fact) |
| `dead-double` | new | throw a double no live tile can lead to, on a trick you cannot take, never the 5-5; the bidder cashes one while a setter holds trump | likelihood (a decoy) |
| `helper-stretch` | new | win low with the second tile and keep the double | — |
| `follow-low` | new | following trump with no way to take it and no count among the trumps, the lowest; the partner likewise in any suit | — |
| `take-the-trick` | new | take the other side's trick with the lowest tile that does: last, following or trumping; void, the lowest trump that beats it; the trump-set holder last trumps only count | — |
| `follow-with-the-off` | new | the bidder following a suit he cannot win plays the off his double does not protect | — |
| `defend-the-84` | new | against an 84 (or no trump, keeping count): keep doubles with tiles out and pairs with targets; throw what wins nothing, then a pair's guard, then a pair tile, a double last | likelihood, aggregate |

The prelude (`silas/lib/book.silas`) grew the concepts these rules share: the safe trick,
the trick the partner may still take, no setter can hold a trump, the dead double, a count
tile's last guard, the off a double protects and the bare off, a bid of marks, the lead
whose count sets (six literal sums against `to-set`, which closes the aggregate gap for one
called tile), the bidder's crisis, and the 84 defender's watch list, pair targets and the
tiles that guard a pair. Three passages contradict a worked hand (keeping the last trump,
the 6-5 off first, the partner's top trump); each rule encodes the worked hand and records
both. The gaps left, per rule, are in `silas/GAPS.md`: another seat's knowledge now blocks
only the donation lead, since the partner's own knowledge is the viewer's. Beside the
policies, queries state the frame facts the book narrates, checked on its hands
(`silas/tests/book_queries.rs`): what an off can lose and which side its double protects,
what an 84's defender watches for, when a hand is settled. All eighteen are priced under
Zeb on the pour ([[book-study-2026-10-12]]); the defence of the 84 never fires at bid 30.

## The book, second batch (2026-10-13)

Of the 86 rows of the private catalog that Silas could already say and no file had written
down, the batch encoded 71: what an off risks and which side its double protects (7), the
odds and narrated certainties (11), the bidder's plan and claims (8), follow low (8), take
the trick (4), follow with the off (2), the setter's throwaways (11), the defence of the 84
(11), plays the earlier rules already read (5), and the laws of play (4). Three more are
recorded where the book breaks its own stated rule, and twelve were left, each with the
reason (a policy likelihood for a reading, a play-out for a set rate, a frequency over a
fresh deal, or a narrated play the book gives no condition for). Each was read in the
reviewed transcription, written with its anchor and the book's condition in our words, and
asked at the book's own positions: every new rule fires on the book's hands and allows the
book's tile there, at all four seatings, except where the book breaks its own stated rule,
which each header records.

Two earlier rules were read again against the book's hands and repaired: the pounce
counted the partner's possible ruff (Hands 30 and 31 throw a plain tile on the bidder's
partner's double; S-0079-b), and the bidder's plan dumped an off without counting the off's
own count (Hand 24 leads the 5-1, not the 6-4). New odds: three trumps out all in one hand
3/38, an opponent 1/19; Hand 33's "you know" is 17/22; Hand 34's second way to set the 84
is a holding in 563/855 of the bidder's worlds (either way 92101/116280, the defender still
to guess right); the "only way" claims of Hands 16 and 34 are exact (1/15 and 2/15), the
set falling in exactly those deals under double-dummy play. Every new file was checked for
copied prose (7-word runs against the transcription) before commit.

## The engineering pass (2026-10-13)

A peer review through a Lean lens (`scratch/silas-lean/REVIEW.md`, unpublished) found the
query core sound and three structural defects, none yet a wrong number: a policy carried
modes across calls, so one frame could get two decisions; `Exact` rested on a physics's
own fiber methods and a string identity; and a frame had two constructors with two
meanings of `played` and `shown-void`. The pass fixed all three, gave the contract its
kind, stake and declaration (`(bid 84)`, `(stake 2)`), made authority a lattice, made
`trick-number` undefined once the hand is over, and wrote the denotational semantics into
the README (Kleene atoms, `some`/`none`, injective assignment with Bell-number
completeness, existential projection, the admissibility of `uniform_resolving`). For
working with it: `(define ...)` and `(include book)`, the `silas` command (`eval` with
witnesses and the allowed set, `query`, `fmt`, `check`), errors with line, column and
caret, `CompiledPolicy::allowed` (and `SilasPolicy.allowed` in Python).

## Short forms and diagnostics (2026-10-13)

Silas stays an s-expression language with a small exact core; the short forms are
tree-to-tree rewrites read away before the core reader, like `define`. Roles take their
sorts from the slots they fill, and a query may omit `(roles ...)` (its roles are the free
names in order of first use), `(out)` and a lone `(pattern ...)`; `and`, `or` and `not` of
any clause rewrite by Kleene's De Morgan laws (`(or a b)` is `(none (roles) (not a) (not
b))`); a rule says `(rule r (play x) clause ...)`; the fallback may go unsaid. Every book
rule, the prelude and the odds were rewritten (1,380 lines out, 927 in); the 2026-10-12
gates keep their explicit forms. `tests/core_pins.rs` pins each file's core from before
the rewrite, and every pin held. texas-42's Scheme/Fix had no shorthand to bring over
(every copy of its reader takes only the full form). For rules that silently never fire:
warnings read off the core (`check`; an unbound role under a negation, distinct roles
equated, more roles than values, a clause with its negation, an answer never asked to be
playable) and the funnel (`silas coverage`, and `eval`'s "stops at"): each pattern's
clauses in source order, counting where it stops over the book's lines and random deals.
The crate's files raise one warning (a redundant clause in `crisis`); coverage shows the
trump-set ruffs never fire in random play (3,000 deals, 84,205 frames: 29 reach the ruff,
one holds a trump) nor in the book's hands (no narrated position applies them), while
the feed fires 3 times and the lead 41: rare, not mis-guarded (#141).

## Speed (2026-10-13)

Silas counts a fiber without enumerating it. The c12 odds query (352/969 over 399,072,960
worlds) takes 2.2 ms, against 450 ms for an exact int32 brute force on MPS and 210 ms on 18
CPU threads, so the GPU is not a lever for Silas. What the time had gone to was CPU defects:
the count table rebuilt in full on every call, now built over reachable states (per call
6.6 → 0.6 µs); allocation in clause evaluation (a re-price call 12 → 4.5 µs); and the debug
test profile, now opt-level 1 (the slowest binaries killed at 300 s → 34 s for the suite).
BigRational is under 1% of every profile. The one route that could make a GPU worth revisiting
is compiling guards to bitmask operations, a language-design question.

Collapsing equivalent hidden tiles is not a speed lever: worlds never compress ([[endgame-equivalence-census]]).

## The double-dummy reference

`silas::Solver` (2026-10-13) is an exact perfect-information solver over any `Physics`:
integer fail-soft alpha-beta with a transposition table at each trick's start, keyed on
the exact state, holding proven bounds. It values every legal move at a fully specified
position under two objectives, the declaring side's points margin and make-or-set
against the bid (a null-window search). It **sees all four hands**: a reference, never a
player ([[the-wall]]). It explains a book move only where the book argues from a known
position; elsewhere it bounds what a real seat can achieve. A full deal solves in a
median 1.7 ms (worst 29 ms over 900 random deals). It agrees with an unpruned
exhaustive search on sampled endings, with an unpruned memoized search of whole game
trees (Hand 28 in the tests; 40 random deals and Hands 34 and 35 in a one-off check),
and under seat rotation. The forge oracle could not referee it on this machine: on MPS
(torch 2.11, macOS 26.5) `torch.sort` and `torch.unique` on int64 lose the high bits of
forge's packed states, so forge's enumeration collapses after two plies.

`grade_line` grades a line play by play; `silas/examples/book/solve_grade.rs` grades the
narrated lines of Hands 28–35 in under a second. Of 124 narrated plays with a choice, 21
fall short of the best double-dummy margin and 4 change make-or-set: the bidder's 3-1
lead to trick three of Hand 28 (any of 5-0, 6-1, 6-6 keeps all 42; the 3-1 sets him),
the bidder's opening 5-5 in Hand 34 and the left opponent's 4-3 that hands the 84 back
at trick five, and the bidder's opening 4-4 in Hand 35. Each of those is a bidder or
defender choosing without seeing the hands, so a short grade is a question for the
book's reasoning, not a verdict on it. Hand 33's narrated line breaks follow-suit at
trick six and is graded to there.

## Solved outcomes

The book's claims of certainty (the tournament lay-down rule, "cannot be set") are about
what one player can guarantee without seeing the hidden hands. They are not about
best play by everyone in one world. Four predicates (2026-10-13, `silas/README.md`
§ Solved outcomes) state them:

- `(lay-down)`: one fixed order of the viewer's tiles wins every trick left, in every world
  and against every play of the other three seats.
- `(certain-make)` and `(certain-set)`: the viewer, choosing from his frame alone, forces his
  side's goal against every play of the other three, his partner's included.
- `(dd-make)`: the per-world double-dummy outcome, an upper bound on any such guarantee.

The first three answer exactly or are refused. A lay-down proves the goal; one
double-dummy counterexample among 10,000 worlds refutes it. Otherwise an and-or search
over the viewer's frames decides it, since every play is public and his information
state is the frame. `tests/outcome.rs` holds them to brute force.

Results:

- Hands 21 and 22 lay down and are certain at every seating. The four-fives 84 of Hand 34
  is refuted at once.
- The book's lay-down shape is sufficient, not necessary.
- Hand 32's claim is of tricks, made already set.
- Hand 26 at the position after trick 2 (all seven trumps played, 17 banked, the bidder to
  lead 3-3 1-1 5-5 2-2 5-2, the other 15 tiles dealt five each in any of 756,756 ways):
  double dummy makes the 35 in every world, but the bidder alone cannot force the make.
  The private adjudication's setting deal (A13) is one of those worlds, not an earlier
  fork: the partner keeps the 3-2 to the last trick and the bidder, seeing all four hands,
  scores 31; double-dummy play has him shed it for 36. The book's "cannot be set" is the
  bound's (#143).

What the partnership can force together with a partner of his own information is open
(`silas/GAPS.md`, 8).

## The match and field-free comparison

The book's cross-game rules (bid only enough; one mark from the game, bid 42 not 84;
the 84 hand; keep the setters honest) compare contracts and policies under a scoring and
a match score. `silas/docs/cross-game.md` sorts the 140 rows of the private coverage
study that cite #130 into seven clusters and plans five slices. Slice 1 (2026-10-13,
`silas/README.md` § The match) is the frame and the comparisons that need no deal:

- **The frame.** Points to 250 (the book's game) and marks to seven (the tournament's),
  the hand's `Outcome`, and `Contract::settle`: under marks the award, under points the
  book's rule (made, each side scores what it took; set, the setters score the bid plus
  their points). The book's "they get 84" for a set 84 is the other `SetReading`; the
  two part only on a set marks bid under points, where a set 42 pays exactly 42, one
  under the floor of 43 every set points bid clears. A `MatchState` keeps its winner,
  since a tie at 250 goes to the deciding hand's bidder.
- **The value.** A match-win value is one on won states, zero on lost ones, and between
  never rises with the other side's score nor falls with its own; Silas checks a table
  and refuses one without those properties. The score-grid programme builds one from a
  hand-outcome model, `exact_given` that model.
- **`dominates`.** Two contracts compared over the hand's 43 outcomes from every state of
  a frame, under both readings, by marks, points or the match-win order (at least as good
  under every value). Exact, with a counterexample when it fails. Needing the last mark,
  42 dominates 84 under marks and, within 42 of 250, under points (S-0170-a, S-0089-b);
  two marks out it does not; a lower point bid dominates a higher one under both
  scorings (the contract half of "bid only enough"). In linear marks the 84 is a belief,
  better exactly when the make is likelier than not.

That closes the scoring definitions (cluster F), states the three dominance rows of
cluster B and the contract half of cluster A, and re-tags the fourteen 84 play rules
(cluster G) to the gaps that hold them.

Slice 2 (2026-10-13, `silas/README.md` § The auction; its choices are
`silas/docs/cross-game.md` § 4, Slice 2 as built) is the bid action:

- **The auction in the record.** Each seat's call, under an `AuctionConfig` (the marks
  cap, the forced bid), legal by construction; the contract is derived from the
  auction, never given. Legality is held to the arena's and the TypeScript engine's on
  all 834 legal prefixes of an auction (five marks included since #142, which found
  the engine refusing it). A hand known only by its contract gets the
  shortest auction its declarer wins.
- **Bidding and declaring in policies.** `(bidding ...)` and `(declaring ...)` rules
  read an auction frame (the hand, the auction so far, the match state, the
  declarations offered); predicates carry a phase, so a bid rule cannot read the play.
- **The bid book as policies** (`silas/examples/book/bidding/`). `bid-only-enough` is
  the arena's `HeuristicBidder`, exactly: call for call on the arena's own auctions and
  on all 1,184,040 hands, bid and declaration. `risk-budget` is the book's count of
  what is at risk as one reading; over all hands its opening call is exact: 54.7% pass
  (the book's "about half"), and 61.5% of its bids are the book's reasoned 30, 31, 35
  and 36. On the study's 30 bid-book hands it makes the book's call on 18 and agrees
  on bid or pass on 22, below always bidding's 23 ([[book-study-2026-10-12]]: Zeb's
  forecast, 25).
- **Cost.** Literal-only numbers made a bid level a rule and a points bound an
  enumeration: the reading cost 150 µs a hand on one core. The aggregate (below) took it
  to 20 µs.

What remains, a deal measure under a named field and the arena's paired ledgers, is
`silas/GAPS.md` gap 2; reading a bid as evidence about the hand (#134) is built (§ The
bid as evidence).

## The aggregate (2026-10-13)

Silas compares sizes and sums (gap 5, #135; `silas/README.md` § Semantics): `(card (roles
...) clause ...)` counts the bindings that satisfy a body, `(points (roles d) clause ...)`
sums their count points, `(+ ...)` adds, and the core's one comparison is `(<= a b)`, with
`<`, `>=`, `>`, `=` and `/=` as short forms. Numbers stay literal-only: nothing binds one,
and a definition's `(number n)` parameter takes the bound. The three-valued semantics is
intervals: a binding whose body is undefined may or may not count, so an aggregate is `[s,
s + u]`, and a comparison is true or false only when every completion agrees. That makes
`some` and `none` the thresholds `card ≥ 1` and `card = 0` in all three values, and makes
an aggregate decided wherever its body is: always over the viewer's own hand, and in each
world over a hidden one, where a belief gives its law. An aggregate is counted once per
decision, even across a policy section's rules, since its value is a function of its text
and the roles it names.

The bid rules were rewritten with it (`suit-at-least`, `exposes-at-most`, `risk-at-most`;
`bid-only-enough` declares by two rules instead of sixteen), and every call and
declaration of the three bidding policies is pinned equal to the old encodings' on all
1,184,040 hands. One core decides the opening in 7.5 µs (13), `risk-budget` in 20 µs (150)
and the arena declaration in 92 µs (210). The crisis rule now keeps only the double whose
suit has the most tiles out (the prelude's `more-out`). Still statable but unwritten: the
84 watch list's double with more out, and C15's sum over several called tiles.

## The bid as evidence (2026-10-13)

A seat that has heard the calls reads them (#134; #128 for the auction; `silas/README.md`
§ The auction as evidence): given a bidding policy for each read seat, the belief keeps
exactly the worlds in which every read seat's policy, from its own hand, makes its call
and, for the declarer, its declaration. A policy is deterministic, so each world weighs
zero or one, and whether it is kept depends seat by seat on that seat's hand alone: Silas
runs each read seat's policy once over every hand it could hold (at most 116,280) and
counts the worlds whose read hands all agree, by placement of the tiles the query reads,
never sampling. The number is `exact_given` the policies (one model per seat: its name and
a digest of its identity, the match state, the declarations offered), since another
policy is another number. It is held to brute force over every world of small frames.

From the opening, three seats read under `bid-only-enough` take about a second on 18
cores. C12B's condition, 194/285 from the bidder's hand, moves to 69.4% when the three
passes after his 30 are read as `bid-only-enough`'s and to 67.0% as `risk-budget`'s: the
auction moves it a point either way. In Hand 29 the partner reading the bidder's 30 in
fours puts the 4-4 in his hand at 6,003/8,669 against 1/3. `silas query ... --auction
... --read ...` asks it at the command line. What it cannot yet ask: a question about a
whole hidden hand ("a 35 means two offs", pp. 53–54), which needs a seat's hand law; a
reading inside a policy's guard; a seat mid-auction; and the plays as evidence, which
factorize the same way (a seat's play frame is its hand and the record) and wait only on
each past play's check (`silas/GAPS.md` 1 and 4).

## Where it sits

- **Origin.** Moved from texas-42 `1c15cc51` (CC0), where it was written as the
  query layer of that repo's walt crate; renamed and cut loose from that repo's rules,
  kernel and solver. texas-42's solver-backed teaching layer stayed behind; its answer
  key here is Zeb ([[scenario-player]]).
- **Zeb's reasons.** The reason report (#124, [[scenario-player]] § The reason report)
  names why Zeb prefers one tile in Silas: a declared space of queries (holdings, voids,
  the open trick's seats still to play, count at risk, the prelude's nouns as holdings,
  and every two of them joined by `and` or `or`, each fact in its own `some` scope),
  each evaluated on Zeb's sampled worlds, with a threshold and a chance level set before
  looking; the prelude's frame-only concepts are constant over one decision's worlds and
  enter as their world forms; the noise bar is Silas's three-valued compare
  on an `Estimate` from `SilasBelief.condition_likelihood` (a binding of Silas's own
  conditioning, added 2026-10-13).
- **Next.** The book's rules as the book means them are written (§ The book, rule by
  rule) and priced under Zeb on the whole pour ([[book-study-2026-10-12]] § The book's
  rules as the book means them: Zeb sides with 3 of 18 and prices against none; the dead
  double's throw and the trump set's feed carry the conditions the book states).
  Policy likelihood for the play (#128; the auction's is built, § The bid as evidence)
  and another seat's knowledge (#132) are next ([[w42]]). The cross-game
  slices go on from the auction to comparison over deals (§ The match and field-free
  comparison).
- **Lean.** The intended long-run spec for Silas and Zeb both, starting from texas-42's
  kernel-checked `Texas42` library; Rust is the first implementation it will be held to.
  What the book asks that Silas cannot yet express (cross-game policy comparison, the
  auction) is tracked as issues and becomes that agenda. The first cross-game theorem
  for it is the last mark's dominance, stated in `silas/docs/cross-game.md` § 3 and
  decided in Silas by enumeration.

## Links

[[scenario-player]] · [[w42]] · [[book-study-2026-10-12]] · [[the-wall]]
