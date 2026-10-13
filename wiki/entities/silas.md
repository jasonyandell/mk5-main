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

The rules **as the book means them** are fourteen policies in `silas/examples/book/`,
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
| `crisis` | C14, C21 | a trump the bidder cannot draw: the 6-5 bait, a count double, away from the trump holder's suit | likelihood, aggregate |
| `donate-when-safe` | C1b, C3 | count, biggest first, on a trick provably the side's; the last seat's donation; none on a trick that can be lost | — |
| `partner-leads` | C9, C23 | a double or a walker; then no count, a five, a six; never trump | likelihood |
| `partner-plays-the-top-trump` | C2 | the worked hand's play (the book's p. 25 reading disagrees) | — |
| `pounce` | C7 | count only on a trick the setters may take, while able to follow; the double first; no count on a lost trick | — |
| `protect-the-count` | C8, C13b | no count on the bidder's master lead, the top trump too; not in a trump set | — |
| `shed-for-a-void` | C13a | a lone tile, never a double, a winner or a count tile's last guard; points bids only | — |
| `low-from-a-pair` | C13c | as the gate, and the 84 form with count | — |
| `setter-leads` | C15, C13d | a double when one point sets; the count that sets; count calling for more; think twice at 35 | — |
| `trump-set` | C22 | the bidder's partner void too; the big doubles; feed a holder who may yet trump; lead high | aggregate (closest fact) |
| `dead-double` | new | throw a double no live tile can lead to, on a trick you cannot take, never the 5-5; the bidder cashes one while a setter holds trump | likelihood (a decoy) |
| `helper-stretch` | new | win low with the second tile and keep the double | — |

The prelude (`silas/lib/book.silas`) grew the concepts these rules share: the safe trick,
the trick the partner may still take, no setter can hold a trump, the dead double, a count
tile's last guard, the off a double protects and the bare off, a bid of marks, the lead
whose count sets (six literal sums against `to-set`, which closes the aggregate gap for one
called tile), and the bidder's crisis. Three passages contradict a worked hand (keeping the
last trump, the 6-5 off first, the partner's top trump); each rule encodes the worked hand
and records both. The gaps left, per rule, are in `silas/GAPS.md`: another seat's knowledge
now blocks only the donation lead, since the partner's own knowledge is the viewer's.

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
- Once Hand 26's trumps are cleared, double dummy makes in all 756,756 worlds, but the
  bidder alone cannot force the make. The book's "cannot be set" is the bound's.

What the partnership can force together with a partner of his own information is open
(`silas/GAPS.md`, 8).

## Where it sits

- **Origin.** Moved from texas-42 `1c15cc51` (CC0), where it was written as the
  query layer of that repo's walt crate; renamed and cut loose from that repo's rules,
  kernel and solver. texas-42's solver-backed teaching layer stayed behind; its answer
  key here is Zeb ([[scenario-player]]).
- **Zeb's reasons.** The reason report (#124, [[scenario-player]] § The reason report)
  names why Zeb prefers one tile in Silas: a declared space of holding and void queries,
  each evaluated on Zeb's sampled worlds; the noise bar is Silas's three-valued compare
  on an `Estimate` from `SilasBelief.condition_likelihood` (a binding of Silas's own
  conditioning, added 2026-10-13).
- **Next.** The book's rules as the book means them are written (§ The book, rule by
  rule) and priced under Zeb on the whole pour ([[book-study-2026-10-12]] § The book's
  rules as the book means them: Zeb sides with 3 of 14 and prices against none, once the
  dead double's throw and the trump set's feed carry the conditions the book states).
  Policy likelihood (#128) and another seat's knowledge (#132) are next ([[w42]]).
- **Lean.** The intended long-run spec for Silas and Zeb both, starting from texas-42's
  kernel-checked `Texas42` library; Rust is the first implementation it will be held to.
  What the book asks that Silas cannot yet express (cross-game policy comparison, the
  auction) is tracked as issues and becomes that agenda.

## Links

[[scenario-player]] · [[w42]] · [[book-study-2026-10-12]] · [[the-wall]]
