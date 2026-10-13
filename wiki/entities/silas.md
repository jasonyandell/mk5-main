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
an executable rule over the viewer's information alone: a rule that reads a hidden hand
does not compile. Comparing two queries returns the first concrete world where they
differ. Grammar, predicate table and examples: `silas/README.md`, `silas/examples/`.

Silas owns the game's nouns (dominoes, seats, teams, led contexts) and no rules. A
`Physics` (`silas/src/physics.rs`) supplies one declaration's four tables (led context,
follow sets, trick key, count) and the fiber of hidden worlds (count and enumerate);
legal plays, trick winner and points, beats and threat sets derive from the tables.
`silas::reference::Straight42` is the spec-faithful test physics (nine straight
declarations); off-suit plays tie at the bottom of the trick key, as texas-42's Lean
`Declaration.key` defines.

## On Zeb's rules

`scenario/core` (feature `silas`, off by default) carries `ZebPhysics`: one of Zeb's ten
declarations (`Rules`) and Zeb's `Fiber` behind the `Physics` trait, with `legal_plays`,
`trick_winner` and `trick_points` overridden by Zeb's own `legal`, `winner_offset` and
`trick_points`. The numbering is shared and mapped explicitly: tile id = domino index
(triangular), Zeb's led suit k = context `q{k}` with 7 = `q*`, seat s = `S{s}`. Identities are
`zeb/P0`..`zeb/P6`, `zeb/DT`, `zeb/DS` (doubles as their own suit), `zeb/NT`.
`scenario/core/tests/silas_physics.rs` holds it to `Straight42` on the nine shared
declarations (every table entry and trick-key order exhaustively; legal plays, whole random
hands and fibers seeded) and holds `DS` to its facts; no disagreement was found. To key
legality by Zeb's code, `legal_plays` takes the led tile rather than its context (2026-10-13).
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
summary carries an **authority**: `Exact` (enumerated or combinatorial support, no
model), `ExactGiven(model)` (weights from a pinned model; the model is an assumption,
the arithmetic exact), or `Sampled(n, seed)`; combining labels yields the weakest, and
nothing derived from a model or a sample is `Exact`. Comparisons on model or sampled
values are three-valued (better, worse, indistinguishable) against an explicit band; an
exact value's band must be zero.

`Belief::uniform_resolving` answers questions about a few named tiles at the root
without enumerating the hand: one point per placement of the named tiles, weighted by
its exact number of completions; any read of an unnamed tile is refused. The book's 84
hand (p. 113: the bidder holds 5-5 5-4 5-2 5-0 4-4 4-2 2-2, fives trump) gives "an
opponent holds the 6-5 with another five" at exactly 33/95, `Exact`, over a root of
399,072,960 worlds ([[book-study-2026-10-12]]).

## The book, rule by rule

The 21 rule gates of [[book-study-2026-10-12]] are Silas policies in `silas/examples/book/`,
tested in `silas/tests/book.rs` mostly on the book's own full deals (Hands 28–35),
replayed with every play checked legal, and every check run with the bidder in each of
the four chairs. On those positions at least 10 of the 21 Python gates part from the
book, so the per-rule prices on the study page are the gates' prices, not the book's
(#123).

Slice 3 (2026-10-13, `0c5d7f66`) closed the five viewer-side gaps the book hit hardest:
the frame now carries the hand's public record (contract and attributed plays, the one
source of the leader, trick, played set, shown voids and score; `PolicyInput` is gone),
read by `declarer`, `bid`, `banked`, `to-make`, `to-set`, `trick-number`, `led`,
`played-to`, `trick-of` and `shown-void`; and the language has `(none (roles …) …)`, a
stratified, three-valued negated sub-pattern (an undefined body never makes it true).
With them **20 of the 21 encodings are their gate exactly**; C14 still lacks the gate's
"most unplayed" (aggregate, #135). The open gaps in `silas/GAPS.md` are all world-side or
beyond one hand except that one: policy likelihood (#128), cross-game comparison (#130),
another seat's knowledge (#132), the auction (#134), aggregate (#135).

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
- **Next.** The book's rules past their gates (#123): policy
  likelihood (#128) and another seat's knowledge (#132) are what the book's own plays
  need next ([[w42]]).
- **Lean.** The intended long-run spec for Silas and Zeb both, starting from texas-42's
  kernel-checked `Texas42` library; Rust is the first implementation it will be held to.
  What the book asks that Silas cannot yet express (cross-game policy comparison, the
  auction) is tracked as issues and becomes that agenda.

## Links

[[scenario-player]] · [[w42]] · [[book-study-2026-10-12]] · [[the-wall]]
