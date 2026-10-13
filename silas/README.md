# Silas

Silas is a typed relational language over the worlds consistent with what one
seat of Texas 42 knows. A seat sees its own hand, the tiles played, and which
suits each other seat has failed to follow; every deal of the unseen tiles that
fits those facts is a *world*. Silas asks questions of those worlds:

- a **query** is a relation: the set of output tuples (dominoes, chairs,
  contexts) for which some pattern of typed atoms has a satisfying assignment;
- a **belief** is an exact finite measure over worlds; a query summarized over
  a belief gives exact probabilities, each carrying its **authority**;
- a **policy** is a function of what the viewer alone can see: an exact
  information-state table, relational rules guarded by queries, and a total
  fallback.

Silas owns the nouns of the game (dominoes, seats, teams, contexts), the
language, and every derived rule. A rules engine supplies only one
declaration's four tables, through the `Declaration` trait; Silas reads them
into a `Physics` and computes legality, the trick's winner and points, the
trump and the fiber of hidden worlds itself. `silas::reference::Straight42` is
the reference declarations.

```rust
use silas::{reference::Straight42, *};

let fives = "P5".parse::<Straight42>()?.physics();
let bidder: DominoSet = ["5-5", "5-4", "5-2", "5-0", "4-4", "4-2", "2-2"]
    .iter().map(|d| d.parse()).collect::<Result<_>>()?;
let contract = Contract::new(Seat::S0, Bid::number(84)?, fives)?;
let root = Frame::new(contract, Seat::S0, bidder, vec![])?;

let query = include_str!("examples/six-five-guarded.silas").parse::<Query>()?.compile(&Registry::standard())?;
let belief = Belief::uniform_for(root, &query, 1000, &mut Budget::new(10_000_000))?;
let p = query.summarize(&belief, &mut Budget::new(10_000_000))?.event_probability();
assert_eq!((p.value().to_string(), p.authority().is_exact()), ("33/95".to_string(), true));
```

That is the book's four-fives 84 (Roberson, *Winning 42*, p. 113): the chance
that an opponent holds the six-five together with another five is exactly
33/95 over the 399,072,960 deals the bidder cannot tell apart
(`tests/eighty_four.rs`). `uniform_for` resolved the three tiles the query
reads and nothing else: 27 points.

## Queries

```text
file     := form ... final
form     := (include library) | (define name (roles param ...) clause ...)
final    := query | policy                ; a dominance is its own file (The match)
query    := (query (roles role ...) (out name ...) pattern ...)
role     := (sort name)
sort     := chair | domino | context
param    := (sort name) | (number name) | (team name)
pattern  := (pattern clause ...)
clause   := (same name name ...) | atom | (not atom)
          | (some (roles role ...) clause ...)
          | (none (roles role ...) clause ...)
atom     := (predicate argument ...) | (definition argument ...)
argument := name | literal
```

Names begin with an ASCII letter and continue with letters, digits, `_` or
`-`. Literals are dominoes (`6-4`; either end first), chairs `S0`–`S3`,
contexts `q0`–`q6` and `q*` (the called suit), teams `T0` and `T1`, and
nonnegative integers. A literal spelling is never a role name. `;` starts a
comment. Text is bounded (1 MiB for a query, 4 MiB for a policy) and so is
nesting, and a query to 64 roles, its scopes' included. Unknown predicates,
wrong argument sorts, duplicate roles or outputs, and mixed-sort `same` groups
are refused at compilation, before any evaluation. Every error in program text
names its source, line and column, and shows the line with a caret under the
fault:

```text
rule.silas:3:5: unregistered predicate mystery
  |
3 |     (mystery c)))
  |     ^^^^^^^^^^^
```

`Query` and `Pattern` are plain data and can be built in Rust as well as
parsed; spans are locations, not content, and never affect equality.
`Query::compile(&Registry)` checks a query and binds its predicates;
`CompiledQuery::identity()` is its complete text identity, including every
predicate's version. `CompiledQuery::witnesses` gives each answer with one
satisfying binding of every role. `Program::parse` reads a text whose final
form is either.

## Semantics

This is the denotation the evaluator implements, and the statement a Lean
transcription would check it against.

- **Atoms are three-valued.** An atom denotes, at a frame `f`, a world `w` and
  an assignment, one of true, false or *undefined* (its precondition fails:
  no lead yet, the viewer not to play, the hand over). `not` swaps true and
  false and keeps undefined. A body's clauses combine by Kleene's conjunction:
  false if any clause is false, true if all are true, otherwise undefined.
- **Scopes are Kleene's quantifiers.** `(some (roles R) B)` is true when some
  assignment of `R` makes `B` true, false when every assignment makes `B`
  false, and undefined otherwise. `(none (roles R) B)` is its negation: true
  when every assignment is refuted, false when one satisfies, undefined
  otherwise. So an undefined atom inside `none` cannot make it true:
  `(none (roles (domino d)) (own-legal d) (double d))` is undefined when the
  viewer is not to play. `(not a)` is `(none (roles) a)`. A scope is
  stratified: its body is read once every enclosing role it names is bound,
  and it binds nothing outside itself.
- **Assignments are injective within a declaration.** The roles declared in
  one `(roles ...)` form, of one sort, denote distinct values unless `(same
  ...)` joins them (`same` closes transitively and joins only roles declared
  with it). A scope's roles are distinct from one another, not from the roles
  around them; a role is never distinct from a literal by declaration, so "no
  other" against a literal or an outer role is written out, as
  `(not (tile e d))`. Formally, a pattern's roles quotient into equality
  classes by `same`, and the pattern ranges over sort-respecting *injective*
  maps from classes to values.
- **Completeness is by set partition.** To cover every assignment of `n` roles
  of one sort, not only the injective ones, write one pattern per set
  partition of the roles, joining each block with `same`: those patterns
  cover every assignment exactly once. There are Bell(`n`) of them: 2 for two
  roles (the pattern without `same` and the one with it), 5 for three, 15 for
  four.
- **Answers are existential projections.** A pattern holds at an assignment
  when every clause is true; an undefined clause holds in no pattern. Roles
  not in `out` are existential: a query's answers at `(f, w)` are the output
  projections of the assignments at which some pattern holds, the union over
  patterns. Many witnesses give one tuple; overlapping or repeated patterns
  add no answers and no mass. `(out)` asks a yes-or-no question. A query with
  no patterns is false; an empty pattern is true.
- **A definition is its body.** `(define name (roles param ...) clause ...)`
  names a conjunction. A call substitutes its arguments for the parameters
  (parameters are names for the caller's arguments, not roles, and carry no
  distinctness of their own) and renames the body's scope roles apart from
  the names around the call. Called positively, the body's clauses join the
  caller's; negated, the body becomes a `none` (or a negated atom, when it is
  one atom). Definitions are expanded when text is read: a parsed query holds
  only primitives, prints in them, and its identity is theirs.
- **Viewer queries are measurable.** If every predicate a query names reads
  only the frame (`Viewer` access), its answers at `(f, w)` are the same for
  every world `w` of `f`. A policy's guards are such queries, so a policy is a
  function of the frame.
- **Resolving beliefs are exact on what they admit.** `uniform_resolving(f, T)`
  is the uniform measure on the fiber of `f` pushed to the placement of the
  tiles `T`: one point per class of worlds that agree on where `T`'s tiles are,
  weighted by the class's size. A read of `seat`'s holdings within `tiles` is
  *admissible* when it is constant on every class. The implementation admits
  a read unless some tile of `tiles` is unresolved, allowed to `seat` by its
  shown voids, and `seat` holds a tile; that predicate implies admissibility
  (it is conservative: a seat filled by its resolved tiles, or a tile forced
  by the other seats' voids, is refused too). Hence the summary of any query
  whose reads are all admitted over `uniform_resolving(f, T)` equals its
  summary over `uniform(f)`, exactly, and a refused read is an error, never an
  approximation.
- **Outcomes quantify over the fiber and the rest of the hand.** `(lay-down)`,
  `(certain-make)` and `(certain-set)` are Viewer atoms whose truth at `f` is a
  statement about every world of `f` and every play to the end of the hand:
  what the viewer can force, from his frame alone, against every play of the
  other three seats. `(dd-make)` is a World atom: the double-dummy outcome in
  one world, an upper bound on any such guarantee. Each answers exactly or is
  refused (Solved outcomes, below).

## Predicates

The core built-ins have horizon zero. Each carries the version in which its
meaning last changed: `silas-v0/core` for the first set, `silas-v1/core` for
the record's first predicates, `silas-v2/core` for those that changed or came
with the one-frame record and the full contract (marked v2). The four outcome
predicates, version `silas-v3/outcome`, look to the end of the hand (Solved
outcomes, below). *World* access
means the predicate reads a hidden world; *Viewer* means it reads only the
frame.

| Predicate | Access | Meaning |
|---|---|---|
| `live(d)` | Viewer | In the viewer's hand or the hidden pool. |
| `played(d)` | Viewer | v2: in the record: exactly the tiles not live. |
| `mine(d)` | Viewer | v2: in the viewer's remaining hand. |
| `holds(c,d)` | World | Seat `c` holds `d`. |
| `void(c,q)` | World | Seat `c` holds no tile that follows `q`. |
| `in(d,q)` | Viewer | `d` follows `q` under the physics. |
| `leads-context(d,q)` | Viewer | `d`, led, leads `q`. |
| `trump(q)` | Viewer | v2: `q` is the declaration's trump: its tiles follow no other context and beat every follower of any other context led. None under no trump. |
| `double(d)` | Viewer | `d` is a double. |
| `beats(a,b,q)` | Viewer | `a`'s trick key in `q` exceeds `b`'s. |
| `boss(d,q)` | Viewer | `d` is live, follows `q`, and no live tile beats it in `q`. |
| `master(d)` | Viewer | `d` is live and no live tile beats it when it is led. |
| `count(d,n)` | Viewer | `d` carries `n` count points. |
| `quota(c,n)` | Viewer | Seat `c` holds `n` tiles. |
| `tile(a,b)`, `chair(a,b)`, `context(a,b)` | Viewer | Equality; anchors a role to a literal. |
| `team(c,t)`, `partner(a,b)`, `opponent(a,b)`, `successor(a,b)` | Viewer | Partnerships and clockwise order. |
| `viewer(c)`, `leader(c)` | Viewer | The viewer; the leader of the open trick (or the next). |
| `next-actor(c)` | Viewer | `c` is to play; false for every seat once the hand is over. |
| `led-context(q)`, `current-winner(c)` | Viewer | The open trick; undefined before a lead. |
| `trick-play(c,d)` | Viewer | Seat `c` played `d` to the open trick. |
| `own-legal(d)` | Viewer | `d` is a legal play for the viewer; undefined unless the viewer is next. |
| `legal(c,d)`, `forced(c,d)` | World | `d` is a legal (the only legal) play for `c`; undefined unless `c` is next. |
| `declarer(c)` | Viewer | Seat `c` won the auction. |
| `bid(n)` | Viewer | v2: the bid as the table says it: 30 to 41 points, 42 per mark (84 for two). |
| `stake(n)` | Viewer | v2: the hand is played for `n` marks. |
| `banked(c,n)` | Viewer | Seat `c`'s side has taken `n` points in closed tricks. |
| `to-make(n)` | Viewer | The declarer's side needs `n` more points to reach its threshold; zero once made. |
| `to-set(n)` | Viewer | The other side needs `n` more points to set it; zero once set. |
| `trick-number(n)` | Viewer | v2: the trick being played or about to be led is the `n`-th, from one; undefined once all seven have closed. |
| `led(c,d)` | Viewer | Seat `c` led `d` to some trick of the record, the open one included. |
| `played-to(l,c,d)` | Viewer | Seat `c` played `d` to the trick led by `l` (`d` is `l` for its leader). |
| `trick-of(d,n)` | Viewer | `d` was played to the `n`-th trick. |
| `shown-void(c,q)` | Viewer | Seat `c` has publicly failed to follow `q`, read off the record. For a hidden seat it is the void that forbids its worlds. |
| `lay-down()` | Viewer | v3: some fixed sequence of the viewer's tiles wins every remaining trick in every world, against every play of the other three seats; undefined unless the viewer is to play. |
| `certain-make()`, `certain-set()` | Viewer | v3: the viewer, choosing from his frame alone, forces the make (the set) in every world against every play of the other three seats; undefined unless that is his side's goal. |
| `dd-make()` | World | v3: in this world the declaring side makes from here under double-dummy play. |

Numbers and teams are literal-only, so `banked` and `to-set` take a chair and
a trick is named by its lead: "one trump trick so far, led by the bidder" is
`(bidder b) (led b first) (trump-tile first)` with a `none` over any other
trump lead. A shown void implies a void in every world; the converse fails,
since counting can leave a seat void in every world without its having shown
it.

A new relation implements `Predicate` and is registered once under an
unreserved name with a `PredicateSpec`: semantic version, argument sorts,
horizon in plies, and access. It returns `Some(true)`, `Some(false)` or `None`
(undefined). The runtime checks argument sorts and withholds the world from
`Viewer` predicates. A `World` predicate reads the world only through
`Holdings`. It must be deterministic, keep to its declared horizon, and charge
its inner work to the `Budget`. The standard registry is the specification;
a custom predicate that declares `Viewer` access and captures a world in its
closure is outside it, and Rust cannot forbid one.

## The book prelude

`lib/book.silas` names the book's vocabulary; `(include book)` brings it in.
The seats, from the bidder: `(bidder c)`, `(bidders-partner c)`,
`(setter c)`, `(left-setter c)`, `(right-setter c)`, `(on-lead c)`,
`(bidders-lead)`, `(bidder-led d)`, `(bidder-winning)`. The count:
`(count-tile d)`, `(count-on-trick)`. The trump, said once for every
declaration: `(trump-tile d)`, `(trump-led)`, `(trump-out)`, `(may-yet-trump c)`,
`(shown-out-of-trump c)`, `(void-in-trump c)`, `(trump-set c)`,
`(top-trump d)`, `(next-trump above d)`, `(boss-trump d)`,
`(lowest-trump-held d)`, `(four-trumps-or-more)`, `(exactly-four-trumps)`,
`(exactly-three-trumps)`, `(exactly-two-trumps)`, `(sure-trump d)`,
`(trump-set-alone c)`. The viewer's play: `(follower d)`, `(discard d)`.
Leads: `(loses-in d q)`, `(can-lose d)`, `(off d)`, `(walker d)`,
`(beatable d)`, `(calls-for-count d)`, `(calls-for d n)`,
`(low-four-five-or-six d)`. The trick: `(last-to-play)`,
`(partner-winning)`, `(can-take-it)`, `(no-setter-trump)`, `(winner-cannot-be-beaten)`,
`(safe-trick)` (what the partner gives count to), `(partner-may-take-it)`
(what a pounce needs). Holdings: `(dead-double d)`, `(guards-count g)`. The
bidder's offs: `(draws-the-count-for dd o)`, `(strips-the-suit-of dd o)`,
`(protects dd o)`, `(bare-off o)`; his trouble: `(crisis)`. The bid:
`(every-trick)` (a bid of marks), `(bid-35-or-more)`, and `(lead-sets d)`,
whether a lead's own count, one called tile and the trick reach what the
setters need, as six sums against literal `to-set` values. Each is documented
where it is defined. Every rule in `examples/book/` is written with it.

## Beliefs and summaries

- `Belief::uniform(frame, cap)`: every world of the frame, weight one. The
  support is counted first and refused above the cap.
- `Belief::uniform_resolving(frame, tiles, cap)`: the same measure, resolved
  only on the given pool tiles (Semantics, above). This is how the 84 hand's
  399 million worlds become 27 points.
- `Belief::uniform_for(frame, &query, cap, &mut budget)`: `uniform_resolving`
  on exactly the tiles the query reads, found by resolving whatever a refused
  read needed until nothing is refused.
- `Belief::from_weights(frame, id, authority, entries, cap)`: a measure from
  the caller. Repeated worlds add their weights; zero weights drop out.
- `query.summarize(&belief, &mut budget)`: event probability, per-answer
  presence mass, the law of whole answer sets, and certainty facts
  (possible, certain, constant multiplicity, constant answers, one answer per
  world, one known identity). Presence probabilities need not sum to one;
  `summary.select(Selection)` turns answer sets into one answer per world under
  an explicit convention.
- `belief.condition(&query, id, &mut budget)` conditions on an answer
  existing; `belief.condition_likelihood(id, model, f)` multiplies by a
  likelihood in `[0, 1]` from a named model. Both return the posterior and
  the evidence's prior probability, and refuse an impossible event.
- `query.compare(&other, &belief, Comparison, &mut budget)` finds a world
  where two queries differ, in existence or in whole answers.

`step_frame`, `step_world`, `step_belief` and `step_belief_given` apply one
typed public play (`ObservedPlay`: actor, tile, and lead, follow or slough in a
named context) to a frame, a world and a belief. The stepped frame is the
frame of the record extended by the play. `compare_answers` sets transported
answers against fresh ones, and `anchor_back` takes a preimage for hindsight
analysis.

Every computation draws on an explicit `Budget`. Exhausting it, like every
other refusal, returns an error and no partial result.

## Policies

```text
policy   := (policy name (fallback lowest-legal) form ...)
form     := (rule name (select selector) (guard query))
          | (exact key (play domino))
selector := answer | tile domino
key      := (key (physics identity) (viewer chair) (dealt-bits n)
                 (declarer chair) (bid n) (history domino ...))
```

A policy is a function of the frame: the same frame always gets the same
decision, and nothing is carried from one decision to the next. A decision
tries, in order: the exact table, keyed by the frame's own source (the
physics's identity, the viewer, its dealt hand, the contract and every play);
the relational rules, in order; and the lowest legal tile. A rule's guard is a
query that may use only `Viewer` predicates. A rule *allows* the legal tiles
its guard answers (`answer`) or its tile when the guard holds and the tile is
legal (`tile`); the first rule that allows a tile decides, and the decision is
the least tile it allows (the definition: `min (answers ∩ legal)`).

`CompiledPolicy::allowed(frame)` returns the deciding layer and every tile it
allows; `decide` adds the decision and whether the contract is resolved;
`explain` adds the deciding rule's witnesses. A frame where the viewer is not
to play is refused. Rule names `exact` and `fallback` are reserved. The forms
`(initial ...)`, `(bind ...)` and a rule's `(in ...) (next ...)` are gone: a
policy has no modes and no bindings (2026-10-13).

## Frames and the record

A `Frame` is the hand's public record as one seat sees it: the `Contract`, the
viewer, the seven tiles it was dealt, and every play so far, as tiles, in
order. `Frame::new(contract, viewer, dealt, plays)` replays the record once:
the declarer leads the first trick and each trick's winner the next, so each
play's seat is read off the record; the leader, the open trick, the played
set, each seat's remaining size and shown voids, the banked points and the
viewer's `Kernel` are derived there, never supplied. `Frame::from_history`
builds the same frame from a scorekeeper's record (the viewer's remaining
hand and each play attributed to its seat) and checks the attributions.
`frame.after(tile)` is the frame one play later. The record is refused if it
repeats a tile, has the viewer play a tile it was not dealt or play
illegally, has another seat play one of the viewer's tiles, has a seat play a
tile of a context it has shown void in (leads included), or leaves no deal of
the hidden tiles.

A `Contract` is the declarer, the `Bid` and the declaration, whose rules are
the frame's `Physics`. A bid is `Points(n)` (30 to 41) or `Marks(m)`;
`Bid::number(84)` is two marks. The threshold (the points the declaring side
must take: `n`, or every point for marks), the stake (one mark, or `m`) and
the award are derived.

A `Kernel` is pure information: the viewer, its hand, the hidden pool, and
each hidden seat's capacity and shown voids. A frame derives one; a resolving
belief builds others for parts of the pool.

## The Physics boundary

```rust
pub trait Declaration {
    fn name(&self) -> String;
    fn led_context(&self, tile: Domino) -> Context;
    fn incidence(&self, context: Context) -> DominoSet;
    fn trick_key(&self, tile: Domino, led: Context) -> u8;
    fn count(&self, tile: Domino) -> u32;
}

impl Physics {
    pub fn new(declaration: &dyn Declaration) -> Result<Physics>;
    pub fn identity(&self) -> &str; // a content hash of the tables
    // The tables: led_context, incidence, trick_key, count. Derived by Silas:
    pub fn follows(&self, tile: Domino, context: Context) -> bool;
    pub fn beats(&self, led: Context, tile: Domino) -> DominoSet;
    pub fn threat(&self, tile: Domino) -> DominoSet;
    pub fn legal_plays(&self, hand: DominoSet, lead: Option<Domino>) -> DominoSet;
    pub fn trick_winner(&self, leader: Seat, plays: [Domino; 4]) -> Seat;
    pub fn trick_points(&self, plays: [Domino; 4]) -> u32;
    pub fn total_points(&self) -> u32;
    pub fn trump(&self) -> Option<Context>;
    pub fn count_worlds(&self, kernel: &Kernel) -> u128;
    pub fn worlds<'a>(&'a self, kernel: &'a Kernel) -> impl Iterator<Item = World> + 'a;
}
```

A rules engine supplies one declaration's four tables: the led context of
each tile, the tiles following each context, the trick key, and the count.
`Physics::new` reads them once (refusing a tile that does not follow the
context it leads) and keeps each context's keys as their order. Everything
else is Silas's code over those tables, and no engine can override it: an
`exact` number rests on Silas and the tables alone. The fiber of hidden worlds
is counted by a dynamic programme over the pool and enumerated by unranking
it. The identity is a 128-bit content hash of the tables, so equal identities
are equal tables, whatever engine supplied them: Zeb's tables and the
reference's are one physics. `Physics` is a value (its tables shared), cheap
to clone; `name()` is for people.

`Straight42` covers the nine straight declarations: a pip suit trump (`P0` to
`P6`), doubles trump (`DT`) and no trump (`NT`); `Straight42::physics()`
reads them. Doubles as their own unpowered suit (nello) is not a straight
declaration and is omitted. Its module documentation states the rules it
implements.

## The double-dummy reference

`Solver` is an exact perfect-information solver. **It sees all four hands.**
It is a reference, never a player: it explains a book move only where the
book's own argument is about a fully known position. Everywhere else its
value is an upper bound on what any real seat can achieve, and a move it calls
a mistake may be right for a seat that cannot tell the worlds apart (strategy
fusion: `wiki/topics/the-wall.md`).

An `OpenPosition` is fully specified: the contract (its declaration the
physics), the four hands, the current trick's leader and plays, and each
side's banked points (`OpenPosition::deal` gives a deal's opening lead; `play`
steps it). `solver.solve(&position, objective)` returns the value of every
legal move at the root (`RootValues`: each move's value, the best, and the
moves tied at best), and `solver.value` the position's own value. Two
objectives, both from the declaring side's point of view, maximized by it and
minimized by the defenders:

- `Objective::Margin`: the declaring side's final points less the
  defenders', banked points included;
- `Objective::Make`: one if the declaring side reaches the contract's
  threshold, zero if not, found by a null-window search that never computes
  the exact count.

The search is integer alpha-beta (fail-soft) over a `Physics`, whose four
tables it reads once. Its transposition table sits at each trick's start,
keyed on the exact state (the four hands, the leader and the declaring side),
and holds proven bounds on the declaring side's points still to come. Those
bounds are facts about the state, so one table serves both objectives and
every later position of the deal. Move ordering changes only the speed. On
this machine (Apple silicon, release build) a full deal from the opening lead
solves in a median 1.7 ms (90th percentile 5.9 ms, worst 29 ms over 900
random deals, the nine declarations in turn); the value of every opening move
takes a median 4.1 ms.

`grade_line(&deal, &line)` grades each play of a line from the deal's
opening lead: the seat, its legal plays, and under each objective the play's
value, the best value and the moves tied at best (`Grade`, `Judged`).
`examples/book/solve_grade.rs` grades every narrated play of the book's eight
full deals from `examples/book/hands.json`, in well under a second
(`cargo run --release --example solve_grade`).

`tests/solve.rs` holds it to an unpruned exhaustive search on sampled
positions with one to four tricks left (every root move, both objectives), to
an unpruned search of Hand 28's whole game tree memoized only on exact
trick-start states, to seat rotation, and to Hand 28's narrated line, which
it replays to the 25 and 17 the book awards.

## Solved outcomes

The book's tournament rule forfeits a bidder who lays his hand down when some
hidden layout and some defence could set him (Roberson, *Winning 42*, Ch. 14).
Its question is not what happens in one world under best play by everyone. It
is what one player can guarantee without seeing the hidden hands. Four
predicates answer it and bound it. Each looks to the end of the hand (horizon
28 plies), and each is exact when it answers and refused, never approximated,
when the work exceeds its caps.

Let `v` be the viewer, `f` his frame and `W(f)` its fiber. The *goal* of `v`'s
side is the make if `v` is on the declaring side, the set if not. The hand is
*settled* for the make once the declaring side has banked its threshold, and
for the set once the other side has banked the rest of the points plus one;
one or the other holds once every trick is played.

- **`(lay-down)`** (Viewer; defined when `v` is to play). True when some fixed
  sequence of `v`'s own tiles wins every remaining trick in every world of
  `W(f)` against every legal play of the other three seats, his partner's
  included. The sequence is his play to the open trick, if he is following,
  then one lead per trick. It is chosen once, from the frame, and reads nothing
  hidden, so it is a line `v` can play. "Every world, every play" is the rule's
  "any hidden layout and any defence", so for this one line the guarantee is
  exact.
- **`(certain-make)`** (Viewer; defined when `v` is on the declaring side). True
  when some strategy for `v` makes the contract in every world of `W(f)`
  against every legal play of the other three seats. A strategy is a choice of
  tile at each of `v`'s frames, so it is a function of what he sees and never
  of the world. **`(certain-set)`** is the same for a setter and the set.
- **`(dd-make)`** (World). True in a world when the declaring side makes from
  here under double-dummy play: both sides see all four hands and play their
  best (the `Solver`, `Objective::Make`).

**Whose play is optimized.** In `(certain-make)` and `(certain-set)` it is
`v`'s alone, over strategies that read only his frame. The setters' plays range
over every legal play, and so do his partner's: the guarantee stands on `v`'s
own choices, as a claim does. It is therefore a lower bound on what `v`'s side
could guarantee with a partner who plays to the same goal from his own
information. That team guarantee is not computed (`GAPS.md`, 8). In
`(dd-make)` every seat is optimized, and each world gets its own best line.

**The order.** For a viewer on the declaring side, at a frame where taking
every remaining trick reaches the threshold:

```text
(lay-down)  =>  (certain-make)  =>  his side can force the make  =>  (dd-make) in every world
```

Each implication is strict. The last is the double-dummy bound. In each world
the declaring side may play a different line, one that reads the hidden hands
(strategy fusion), so per-world minimax bounds every real guarantee from above
and is never below one. For a setter, `(certain-set)` implies `(dd-make)` is
false in every world.

**How `(lay-down)` is decided.** While every lead is `v`'s, a seat's replies
depend only on its own hand and `v`'s tiles, and the seats do not interact.
So the line fails if and only if some seat, holding some hand it holds in some
world, has a sequence of legal replies that leaves it a tile beating `v`'s in
some trick. Silas lists each hidden seat's possible hands: the fiber's
projection to that seat, a hand of the right size from the tiles its voids
allow, whose rest the other two seats can hold (Hall's condition for two). It
then carries, trick by trick, every hand each seat could still hold, and
searches the orders of `v`'s tiles depth first. The answer is exact, and the
work is charged to the budget.

**How `(certain-make)` and `(certain-set)` are decided.** In order:

1. A settled hand, or a goal out of reach even taking every remaining trick,
   answers at once.
2. If `v` is to play and `(lay-down)` holds, the answer is true: taking every
   remaining trick reaches any goal that is still in reach.
3. Silas looks through the worlds of `W(f)` for one where double-dummy play
   fails `v`'s goal. It reads at most `OUTCOME_SCAN_CAP` (10,000) of them, in
   a fixed order that would visit every world once (a stride coprime to the
   fiber's size). One such world is a proof that the answer is false: the
   defence that sets every declaring line there sets every strategy of `v`.
   Nothing is concluded from not finding one.
4. Otherwise Silas searches `v`'s information states, an and-or search over
   frames. At `v`'s turn some legal tile must win. At another seat's turn
   every tile that some world lets it play must lose; such a tile is one
   whose frame exists. Every play is public, so `v`'s information state after
   a history is exactly that history's frame, and the worlds still possible
   are its fiber. The search is therefore exact over `v`'s strategies. It
   memoizes on the record, stops at step 1 and 2's answers, and is refused
   when it exceeds the budget.

**Authority.** The three Viewer predicates read the frame and compute over its
whole fiber; their value is the same at every point of any belief. `(dd-make)`
reads every hidden tile. Over `uniform_resolving` it is refused unless every
pool tile is resolved, so `uniform_for` resolves the whole pool and the
belief's cap refuses a fiber beyond it. An answer over an enumerated fiber is
exact.

`examples/book/lay-down.silas` is the book's rule. `tests/outcome.rs` holds the
predicates to brute force on small positions and checks the double-dummy
bound against them. It also checks the book's lay-downs, Hands 21 and 22, and
hands that look as strong but can be set.

## The match

The book's rules compare whole contracts and policies under a scoring and a
match score, not plays in one hand. This is the frame they need
(`docs/cross-game.md`, slice 1); the bid action, the deal measure and the
comparison of policies over deals come later (`GAPS.md`, 2).

```rust
pub enum Scoring { Points { target: u32 }, Marks { target: u32 } }  // POINTS: 250; MARKS: 7
pub struct Outcome { declaring: u32, other: u32 }                   // the hand's 42 points, split
impl Contract {
    pub fn settle(&self, scoring: Scoring, outcome: Outcome) -> Result<Transfer>;
}
impl MatchState {
    pub fn apply(&self, contract: &Contract, outcome: Outcome) -> Result<MatchState>;
    pub fn won(&self) -> Option<Team>;
    pub fn short(&self, team: Team) -> u32;
    pub fn at_least(&self, other: &MatchState, team: Team) -> bool;
}
pub enum Utility { Marks, Points, MatchWin(WinValue) }
```

- **Settlement.** Under marks, `settle` is the contract's award: the stake to
  the side that gets it. Under points it is the book's rule (S-0017-c): made,
  the bidder's side scores the points it took (a marks bid, its value, 42 a
  mark) and the other side the points it took; set, the bidder's side scores
  nothing and the other side the bid's value plus the points it took. The book
  also says a set 84 gives the setters 84 (S-0082-a), without their points;
  `settle_reading` takes either `SetReading`, and the two differ only for a
  set marks bid under points. An outcome must split the hand's points.
- **The match.** A `MatchState` is a scoring, the score by team and the
  winner: the first side to reach the target, or, when both reach it on one
  hand (under points), the side that bid it. `apply` refuses a decided match.
- **Utility.** `Marks` and `Points` are the hand's net transfer to a side.
  `MatchWin(V)` is the side's chance of winning the match from the state the
  hand leaves. A `WinValue` is one on states the side has won, zero on states
  it has lost, and on the undecided states a table of exact rationals that
  Silas checks lies in [0, 1], never rises with the other side's score at a
  fixed own score, and never falls with the side's own at a fixed other. A
  table without those properties is refused; its authority names its model.
  `WinValue::solve` builds one from a hand-outcome model (the law of the
  contract reached and its outcome, from each undecided state) by a dynamic
  programme over the score grid, 7 by 7 under marks and 250 by 250 under
  points: every hand raises the total score, so the states are solved from
  the highest total down. Its authority is `exact_given(model)`.
  `Utility::expect` is the expectation over a hand's law, with the law's
  model met with the value's.
- **The match-win order.** `a.at_least(b, T)` holds when `a` is won by `T`,
  or `b` is lost by `T`, or neither is decided and `a` gives `T` no fewer
  points and the other side no more. It is exactly "`V(a) ≥ V(b)` for every
  `WinValue` `V` of `T`": the properties chain through the undecided state
  between `a` and `b`, and when the order fails a step value or a constant
  one half separates them.

```text
dominance := (dominates name (contracts contract contract) (frame (scoring kind n) place) (utility u))
contract  := (points n) | (marks m)
kind      := points | marks
place     := (short n) | (short-at-most n) | (score n m)       ; optional: every undecided state
u         := marks | points | match-win
```

A `dominates` file holds that one form. Both contracts are bid by S0 under
one declaration; the place is where the bidder's side stands, the other side
anywhere short of the target. The form **holds** when from every state of the
place, for every outcome of the hand (the 43 splits of 42 points) and, wherever
points are scored, under both readings of a set marks bid, the first contract
is at least as good for the bidder's side as the second: by its net marks or
points, or by the match-win order. The last needs no value, since the order
is every value at once. The answer is an enumeration over outcomes, with no
deal, belief or field, so it is `exact`; when the form fails, `decide` gives
the first counterexample. `examples/book/cross-game/` holds the book's: one
mark from the game, 42 dominates 84 (`last-mark`, S-0170-a; under points,
within 42 of 250, `last-mark-points`, S-0089-b); two marks out it does not
(`two-marks-out`); a lower point bid dominates a higher one under both
scorings (`bid-only-enough`, the contract half of cluster A); and in linear
marks the 84 is a belief statement, not a dominance (`eighty-four-in-marks`).
`tests/cross_game.rs` checks the settlement on the book's own vectors, the
award, the set's floor of 43, every example and the programme.

## Numbers

Weights are exact rationals (`Weight`, a nonnegative `BigRational`). Silas
performs no floating-point arithmetic, and the crate denies it in lint. A
float enters only through `Weight::from_f32_exact` or `Weight::from_f64_exact`,
which return the dyadic rational the float denotes, bit for bit, subnormals
included. They refuse NaN, the infinities and negative values. `-0.0` denotes
zero.

Every belief, summary, selection law and `Estimate` carries an `Authority`: the
pinned models whose weights entered the number, and the samples it was drawn
from. With neither it is `exact` (uniform or combinatorial support); a model
alone is exact arithmetic given that model; a sample means the support was
drawn, not enumerated. Its label is `exact`, or `exact_given(<model> + ...)`
and `sampled(<n>, <seed>)` joined by `and`.

Authorities form a lattice. `Authority::meet`, the authority of a number
computed from two inputs, is the union of their models and of their samples:
total, commutative, associative and idempotent, with `exact` its unit. It
never upgrades: `x.at_least(&x.meet(y))` for all `x`, `y`, and a meet is exact
only when both inputs are. Authority propagates through conditioning and
stepping. Only the uniform constructors produce `exact`; caller-supplied
weights must name their source. `compare(a, b, band)` returns `Better`,
`Worse` or `Indistinguishable` against an explicit, nonnegative band, with
the meet of the two authorities. The band must be zero unless a side rests on
a model or a sample: exact numbers are never blurred.

## The `silas` command

```text
silas eval  --hand ID [--trick N] [--seat ROLE] --policy FILE [--book FILE]
silas query FILE --hand ID [--trick N] [--seat ROLE] [--cap N] [--book FILE]
silas query DOMINANCE-FILE
silas fmt   [--check] FILE ...
silas check [DIR ...]
```

A position is a hand of `examples/book/hands.json`: a full deal at the point
of its narrated line (its tricks, then its tail, up to the first play the book
does not name) where `--seat` (`bidder`, `left`, `partner`, `right`) is to
play to trick `--trick` (default: the opening lead), seen from that seat;
or a hand the book gives alone, seen by its bidder before the first lead. The
bidder sits in S0. `eval` prints the position, the deciding layer, the fired
rule's witnesses and the allowed set:

```text
$ silas eval --hand hand-32 --trick 2 --seat bidder --policy examples/book/gates/c01a-top-trump-for-the-donation.silas
hand-32: trick 2, the bidder (S0) to play
  straight42/P0 (tables-92a0…); the bidder S0 bids 36: 36 points for 1 mark(s)
  record: S0:0-0 S1:2-0 S2:4-1 S3:6-2
  banked: T0 6, T1 0; hand: 1-0 2-2 3-0 3-1 5-0 6-0
  legal: 1-0 2-2 3-0 3-1 5-0 6-0
policy c01a-top-trump-for-the-donation
  fired: rule lead-the-top-trump
    6-0  <=  b=S0 p=S2 top=6-0
  allowed: 6-0
  plays: 6-0
```

`query` prints the exact answer over the worlds the seat cannot tell apart,
with its authority, resolving only the tiles the query reads:

```text
$ silas query examples/book/lay-down.silas --hand hand-21
hand-21: the bidder's hand before the opening lead
  straight42/P2 (tables-38dc…); the bidder S0 bids 84: 42 points for 2 mark(s)
  banked: T0 0, T1 0; hand: 1-1 2-0 2-2 3-3 5-2 5-5 6-2
query examples/book/lay-down.silas
  399072960 worlds, as 1 points resolving nothing
  P(answer) = 1 (exact)
```

On a `(dominates ...)` file, `query` needs no hand: it decides the form and
prints what it enumerated, the readings it checked and the verdict:

```text
$ silas query examples/book/cross-game/two-marks-out.silas
dominates two-marks-out: (marks 1) over (marks 2), bid by S0
  marks to 7, the bidder's side short 2: 7 states, 43 outcomes
  by the match-win order (at least as good under every match-win value)
  fails (exact): from 5-0, the hand ending 42 to 0: the first leaves 6-0 and the second 7-0 (won by T0)
```

`fmt` formats
canonically and idempotently: comments kept on their own lines, a form on one
line when it fits in 100 columns (policies, rules, queries, patterns,
definitions and dominances always take a line per item), at most one blank
line kept. Every `.silas` file in the crate is in that form
(`tests/language.rs`). `check` compiles every example and the prelude, each
policy, and each dominance's contracts, under all nine straight declarations.
The command is the default `cli` feature (it reads JSON); `cargo install
--path silas` installs it.

## Examples

`examples/` holds the queries the tests run: `partner-count.silas`,
`partner-has-count.silas`, `partner-master.silas`, `current-forced-play.silas`
and `six-five-guarded.silas`. `examples/book/` holds the book's rules as the
book means them, fourteen policies written with the prelude and named by what
they say (`first-lead`, `bidders-plan`, `crisis`, `donate-when-safe`,
`partner-leads`, `pounce`, `setter-leads`, ...), each header giving its
catalog ids, hands and pages and how it differs from the gate it replaces;
`tests/book_rules.rs` checks each on the book's own positions. Each names the
bidder through `declarer` and is tested at all four seatings.
`examples/book/gates/` holds the 2026-10-12 study's 21 rule gates, the record
of what that study priced (`tests/book.rs`); `tests/book_snapshot.rs` pins
what each gate allows on the book's deals and on 1,600 random ones.
`examples/book/odds/` holds the odds the book prints, one query per claim
(`tests/book_odds.rs`). The book's claims of certainty are queries on the
solved outcomes: `lay-down` (the tournament rule), `lay-down-shape` (the
book's test for one), `claim-the-rest` and `cannot-be-set`
(`tests/outcome.rs`). `examples/book/cross-game/` holds the book's
field-free comparisons of contracts as `dominates` forms (The match).
`GAPS.md` lists what the rules still cannot say.

## Provenance

Silas moved here from texas-42 at commit `1c15cc51` (CC0), where it was the
`walt::scheme` module, called Scheme/Fix. The move renamed it: `Fix` became
`Query`, a `Scheme` (one case) became `Pattern`, `CompiledFix` became
`CompiledQuery`, and the surface forms `(fix …)` and `(case …)` became
`(query …)` and `(pattern …)`. The rules it imported from texas-42's engine
were replaced by the `Physics` boundary and the `Straight42` reference. The
2026-10-13 engineering pass, after a peer review through a Lean lens, made a
policy a function of the frame, the frame one record with derived views, the
physics a value of tables whose fiber Silas owns, the contract carry its
kind, stake and declaration, and authority a provenance lattice.
