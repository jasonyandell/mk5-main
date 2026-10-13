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
final    := query | policy
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

## Predicates

All built-ins have horizon zero. Each carries the version in which its
meaning last changed: `silas-v0/core` for the first set, `silas-v1/core` for
the record's first predicates, `silas-v2/core` for those that changed or came
with the one-frame record and the full contract (marked v2). *World* access
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
declaration: `(trump-tile d)`, `(trump-led)`, `(trump-out)`,
`(shown-out-of-trump c)`, `(void-in-trump c)`, `(trump-set c)`,
`(top-trump d)`, `(next-trump above d)`, `(boss-trump d)`,
`(lowest-trump-held d)`, `(four-trumps-or-more)`, `(exactly-four-trumps)`.
The viewer's play: `(follower d)`, `(discard d)`. Leads: `(loses-in d q)`,
`(can-lose d)`, `(off d)`, `(walker d)`, `(beatable d)`,
`(calls-for-count d)`, `(calls-for d n)`. Each is documented where it is
defined. Every rule in `examples/book/` is written with it.

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
silas fmt   [--check] FILE ...
silas check [DIR ...]
```

A position is a hand of `examples/book/hands.json`: a full deal at the point
of its narrated line where `--seat` (`bidder`, `left`, `partner`, `right`) is
to play to trick `--trick` (default: the opening lead), seen from that seat;
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
with its authority, resolving only the tiles the query reads. `fmt` formats
canonically and idempotently: comments kept on their own lines, a form on one
line when it fits in 100 columns (policies, rules, queries, patterns and
definitions always take a line per item), at most one blank line kept. Every
`.silas` file in the crate is in that form (`tests/language.rs`). `check`
compiles every example and the prelude, each policy under all nine straight
declarations. The command is the default `cli` feature (it reads JSON);
`cargo install --path silas` installs it.

## Examples

`examples/` holds the queries the tests run: `partner-count.silas`,
`partner-has-count.silas`, `partner-master.silas`, `current-forced-play.silas`
and `six-five-guarded.silas`. `examples/book/` holds the book's rules as
policies written with the prelude, each naming the bidder through `declarer`
and tested at all four seatings (`tests/book.rs`); `tests/book_snapshot.rs`
pins what each allows on the book's deals and on 1,600 random ones.
`examples/book/odds/` holds the odds the book prints, one query per claim
(`tests/book_odds.rs`). `GAPS.md` lists what the rules still cannot say.

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
