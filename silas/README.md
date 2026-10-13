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
          | (<= number number)
atom     := (predicate argument ...) | (definition argument ...)
argument := name | literal
number   := n | (card (roles role ...) clause ...)
          | (points (roles (domino d)) clause ...) | (+ number ...)
```

Names begin with an ASCII letter and continue with letters, digits, `_` or
`-`. Literals are dominoes (`6-4`; either end first), chairs `S0`–`S3`,
contexts `q0`–`q6` and `q*` (the called suit), teams `T0` and `T1`, and
nonnegative integers. A literal spelling is never a role name. `;` starts a
comment. Text is bounded (1 MiB for a query, 4 MiB for a policy) and so is
nesting, and a query to 256 roles, its scopes' included. Unknown predicates,
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

### Short forms

The grammar above is the core. A text may also use these shorter forms, each
a rewrite into the core before the core reader sees it, so a short text and
its spelled-out form are one `Query`: equal as data, printed alike, one
identity. Nothing below adds meaning.

```text
query    := (query [(roles entry ...)] [(out name ...)] body)
entry    := (sort name) | name              ; a bare name: its sort from its uses
body     := pattern ... | clause ...        ; bare clauses: the query's one pattern
clause   := ... | (and clause ...) | (or clause ...) | (not clause)
          | (some (roles entry ...) clause ...) | (none (roles entry ...) clause ...)
          | (< a b) | (>= a b) | (> a b) | (= a b) | (/= a b)
```

- **Roles from their uses.** A bare name in any `(roles ...)` (a query's, a
  scope's, a definition's) takes the sort of the slot it fills: a
  predicate's parameter, a definition's parameter, or a `same` partner's.
  Without `(roles ...)` a query's roles are the free names its patterns use
  (not bound by a scope inside them), in order of first use, so the
  distinctness of roles declared together is exactly the explicit form's.
  A name used as two sorts, used where no slot sorts it, or standing where a
  number or a team goes is refused at the use.
- **Defaults.** Without `(out ...)` a query asks yes or no, `(out)`. A query of
  one pattern may give its clauses bare. A query with no body has no pattern
  and is false, as before.
- **Connectives,** by Kleene's laws, which hold in the three-valued semantics:
  `(and c ...)` is its clauses, spliced into the body around it. `(or c ...)`
  is `(none (roles) (not c) ...)`. `(not c)` negates any clause: `(not (not
  c))` is `c`; `(not (some R B))` is `(none R B)` and `(not (none R B))` is
  `(some R B)`; `(not (and c ...))` is `(none (roles) c ...)`; `(not (or c
  ...))` is `(not c) ...`. A negated atom or definition call is the core's
  own `(not ...)`. `(or)` is false and `(and)` true. `and` and `or` are
  reserved.
- **Comparisons,** over the integers, into the core's one `(<= a b)`: `(< a
  b)` is `(<= (+ a 1) b)`, `(>= a b)` is `(<= b a)`, `(> a b)` is `(<= (+ b
  1) a)`, `(= a b)` is `(<= a b) (<= b a)`, and `(/= a b)` is `(or (< a b)
  (> a b))`; a literal plus one is folded (`(> x 3)` is `(<= 4 x)`). `(not (<=
  a b))` is `(> a b)`, and each comparison's negation is its opposite, which
  the semantics below justifies. A definition's `(number n)` parameter may
  stand in a comparison and takes its call's literal; `(points (roles d)
  ...)`'s bare role is a domino.

So the book's "protect the count" exception, that the partner is not the
trump setter or has already played, reads three ways and is one clause:

```text
(or (not (trump-set pard)) (some (roles his) (trick-play pard his)))
(not (and (trump-set pard) (none (roles his) (trick-play pard his))))
(none (roles) (trump-set pard) (none (roles (domino his)) (trick-play pard his)))
```

`silas expand FILE` prints a file's core. `tests/surface.rs` holds every short
form to its explicit form, and `tests/core_pins.rs` pins the core of every
file and every prelude definition, taken before the short forms existed.

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
  only primitives, prints in them, and its identity is theirs. The short
  forms (Queries, above) are read away the same way and are not part of
  this semantics.
- **Aggregates are intervals.** `(card (roles R) B)` and `(points (roles
  d) B)` are stratified scopes, as `some` is: their roles distinct from one
  another, not from the roles around them, read once every enclosing role
  they name is bound. At an assignment each binding of `R` (each injective
  map from `R`'s classes to values) is *satisfied* when `B` is true,
  *refuted* when it is false, and *unsettled* when undefined. A binding
  weighs one under `card`, and under `points` the count of its domino `d`
  (the frame's count table; nonnegative). The aggregate denotes the interval
  `[s, s + u]`, `s` the weight of the satisfied bindings and `u` of the
  unsettled: every value that some completion of the undefined atoms could
  give. A literal `n` is `[n, n]`, and `(+ a ...)` adds intervals end to end.
  `(<= a b)` is true when `a`'s high end is at most `b`'s low end (every
  completion agrees), false when `a`'s low end exceeds `b`'s high end (no
  completion does), and undefined otherwise. So an aggregate is decided
  wherever its body is, and in particular over the viewer's own hand and the
  tiles' faces; over a hidden hand (`holds`) it is decided in each world, and
  a belief gives its law as for any `World` atom. Undefined is sound, never a
  guess: a comparison decided on intervals is decided alike in every
  completion, though one undefined on intervals may be decided in all of
  them (two aggregates over one undefined atom are bounded apart, as
  Kleene's `(or a (not a))` is undefined). The quantifiers are the two
  thresholds: `(some R B)` is `(<= 1 (card R B))` and `(none R B)` is `(<=
  (card R B) 0)`, in all three values. Negation is exact: not `a <= b` is `b
  + 1 <= a` on intervals. An aggregate's value at an assignment is a function
  of its text and the values of the enclosing roles it names, so an
  evaluation counts each once per binding of those, even across the rules of
  one policy section at one frame.
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
outcomes, below). The auction's, version `silas-v4/auction`, are in The
auction, below. *World* access means the predicate reads a hidden world;
*Viewer* means it reads only the frame.

Every predicate also has a *phase*: the play (a `Frame`, after the
declaration), the auction (an `AuctionFrame`, before it), or both. The
predicates below are of the play, except the tiles' and seats' own facts
(`double`, `tile`, `chair`, `context`, `team`, `partner`, `opponent`,
`successor`), `viewer`, `mine` (the dealt hand, at the auction) and `count`,
which are of both. A query naming a predicate of the other phase is refused at
a frame before any work, and a policy rule naming one at compilation.

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

Numbers and teams are literal-only (a number is never a role; the aggregates
compare sizes and sums without one), so `banked` and `to-set` take a chair and
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
(what a pounce needs: a tile that follows can still beat the trick's). Holdings:
`(dead-double d)`, `(guards-count g)`, `(more-out t s)` (suit `t` has more tiles out than
`s`, an aggregate). Defending the last trick, an 84 or no
trump: `(watched-by dd x)`, `(pair-target x q)`, `(keeps-a-pair t)`,
`(protects-a-pair t)`, `(wins-nothing t)`. The
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
- `Belief::given_auction(&evidence, tiles, cap)` and
  `Belief::given_auction_for(&evidence, &query, cap, &mut budget)`: the
  uniform measure conditioned on the auction, read as its bidders' policies
  make it (The auction as evidence, below), resolved as the uniform ones are.
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
policy   := (policy name [(fallback lowest-legal)] form ...)
form     := (rule name (select selector) (guard query))
          | (rule name (play action) body)     ; short form
          | (exact key (play domino))
          | (bidding brule ...) | (declaring drule ...)   ; The auction, below
selector := answer | tile domino
action   := name | domino
key      := (key (physics identity) (viewer chair) (dealt-bits n)
                 (auction (max-marks m) (forced yes|no) (shaker chair) (calls call ...))
                 (history domino ...))
```

A policy is a function of the frame: the same frame always gets the same
decision, and nothing is carried from one decision to the next. A decision
tries, in order: the exact table, keyed by the frame's own source (the
physics's identity, the viewer, its dealt hand, the auction and every play);
the relational rules, in order; and the lowest legal tile. A rule's guard is a
query that may use only `Viewer` predicates. A rule *allows* the legal tiles
its guard answers (`answer`) or its tile when the guard holds and the tile is
legal (`tile`); the first rule that allows a tile decides, and the decision is
the least tile it allows (the definition: `min (answers ∩ legal)`).

The fallback is the only one and may go unsaid. A rule's short form says
what it plays: `(rule r (play x) body)` is `(rule r (select answer) (guard
(query (out x) body)))`, and `(rule r (play 6-4) body)` is `(rule r (select
tile 6-4) (guard (query body)))`, where `body` is what follows `(out ...)` in
a query (its `(roles ...)` first, if given). So the book's protect-the-count
is

```text
(policy protect-the-count
  (rule follow-the-master-without-count (play plain)
    (viewer me)
    (setter me)
    (not (every-trick))
    (partner me pard)
    (bidder-led led)
    (led-context q)
    (not (loses-in led q))
    (or (not (trump-set pard)) (some (roles his) (trick-play pard his)))
    (follower plain)
    (count plain 0)
    (follower counted)
    (count-tile counted)))
```

`CompiledPolicy::allowed(frame)` returns the deciding layer and every tile it
allows; `decide` adds the decision and whether the contract is resolved;
`explain` adds the deciding rule's witnesses. A frame where the viewer is not
to play is refused. Rule names `exact` and `fallback` are reserved. The forms
`(initial ...)`, `(bind ...)` and a rule's `(in ...) (next ...)` are gone: a
policy has no modes and no bindings (2026-10-13).

## Frames and the record

A `Frame` is the hand's public record as one seat sees it: the `Auction`, the
declaration (a `Physics`), the viewer, the seven tiles it was dealt, and every
play so far, as tiles, in order. `Frame::new(auction, declaration, viewer,
dealt, plays)` derives the `Contract` from the auction (its last bidder at his
bid; refused while the auction is under way or passed out) and replays the
record once: the declarer leads the first trick and each trick's winner the next, so each
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
the frame's `Physics` (Lean's `Contract.ofBid`; `Contract::new` builds one for
the match's settlements, which need no auction). A bid is `Points(n)` (30 to
41) or `Marks(m)`; `Bid::number(84)` is two marks. The threshold (the points
the declaring side must take: `n`, or every point for marks), the stake (one
mark, or `m`) and the award are derived. A hand given with its contract alone
(the book's `hands.json`, Zeb's record through Python) is framed with
`Auction::won_by(declarer, bid)`, the shortest auction the declarer wins: he
speaks first and the others pass, the seats before him raising from 84 a
mark at a time for three marks or more. What the auction predicates read on
such a frame is that convention, not the table's auction.

A `Kernel` is pure information: the viewer, its hand, the hidden pool, and
each hidden seat's capacity and shown voids. A frame derives one; a resolving
belief builds others for parts of the pool.

## The auction

The hand's contract comes out of an auction (`docs/cross-game.md`, slice 2):
each seat calls once, from the shaker's left, a pass or a bid above the high
bid. A `Call` is `Pass` or `Bid(bid)`, ordered as the auction orders them.
`Auction::new(config, shaker, calls)` refuses a call that was illegal when
made, so an auction is legal by construction; `legal_calls`, `least_raise`,
`actor`, `high` and `result` read it, and `contract(declaration)` is its
last bidder at his bid.

Legality is texas-42's Lean `AuctionState.legalBid`, as `wiki/topics/rules-of-42.md`
§ Bidding states it: a point bid is 30 to 41 and above the high bid; before
any marks bid, one or two marks (42, 84) may be bid; after a marks bid only
one mark more. An `AuctionConfig` holds what varies: the marks cap (`STRAIGHT`
has none below the five that four calls can reach, Lean's `mark_le_ceiling`;
the TypeScript engine offers four) and the forced bid (the shaker may not pass
over three passes, the common variation; without it four passes are a
passed-out hand, no contract, and the next seat shakes). `tests/auction.rs`
holds the legal calls after every one of the 834 legal prefixes, at every
shaker, to the arena's (`tests/data/auction-legal.txt`, written from
`arena/auction.py` by `arena/test_silas_fixtures.py`); the engine is held to
the same file in `src/tests/rules/bidding/silas-auction-fixture.test.ts`.

An `AuctionFrame` is the auction as one seat sees it before the declaration:
the auction so far, the viewer, its seven tiles, the `MatchState`, and the
declarations the winner may choose from (they must agree on the count, the one
table read before a declaration). A bid or declaring rule reads it and
nothing else, so a bid policy is a function of the auction frame.

| Predicate | Phase | Meaning |
|---|---|---|
| `bears(d,q)` | both | `d` bears the pip of suit `q` (`q0` to `q6`; never `q*`). |
| `high-pip(d,q)` | both | `q` is `d`'s higher pip: the suit it leads when no trump calls it. |
| `shaker(c)` | both | `c` shook. |
| `bid-of(c,n)`, `passed(c)` | both | `c`'s call: the bid the table calls `n`, or a pass. |
| `no-bid()` | both | No seat has bid yet (false in the play). |
| `high-bid(n)`, `high-bidder(c)` | both | The high bid so far, and its bidder. |
| `legal-bid(n)`, `least-raise(n)` | auction | `n` is a legal bid, the least one; undefined unless the viewer is to call. |
| `scoring-marks()`, `scoring-points()` | auction | How the match is scored. |
| `score(c,n)`, `short(c,n)`, `short-at-most(c,n)` | auction | `c`'s side has `n`; needs `n` (at most `n`) to win. |
| `behind(c,n)` | auction | `c`'s side trails by at least `n` (one or more). |

The play frame reads the auction's record too (who bid what), but not yet
the match state: the score predicates are of the auction alone.

```text
section  := (bidding brule ...) | (declaring drule ...)
brule    := (rule name (select bid choice) (guard query))
          | (rule name (bid choice) body)          ; short form
choice   := n | pass | least-raise                 ; n: 30 to 41, or 42 per mark
drule    := (rule name (select declare kind) (guard query))
          | (rule name (declare kind) body)        ; short form; a name is answer
kind     := answer | q0 ... q6 | doubles | no-trump
```

`CompiledPolicy::bid(&AuctionFrame)` is the first bidding rule whose guard
holds and whose call is legal (`least-raise` is the least legal bid), else the
fallback: the least legal call, a pass when it is legal and 30 for a forced
shaker. `declare` is, of the declarations offered that the first declaring
rule to name one allows, the first offered (an `answer` rule names the pip
suits its guard answers, one context column), else the first offered. A
declaration is named by what its tables make trump (`Trump::of`: one pip's
tiles, the doubles, or nothing), so the order the declarations are offered
in is the tie-break. A bidding or declaring rule may name no `World`
predicate and none of the play; rule names are unique across the sections.
`explain_bid` and `explain_declare` add the deciding rule's witnesses; `silas
bid` prints them.

`lib/bidding.silas` (`(include bidding)`) is the auction's vocabulary: a
suit's length, its double, its offs, the count an off exposes, and the book's
count of what is at risk. Sizes and sums are aggregates (Semantics, above):
"at most 10 points exposed" is `(exposes-at-most t 10)`, one `(<= (points
...) 10)`, and a definition's `(number n)` takes the bound. A bid is still a
literal, so a bid level is a rule. `examples/book/bidding/` holds three
policies:

```text
(rule raise-to-32 (bid least-raise)
  (or (least-raise 30) (least-raise 31) (least-raise 32))
  (some (roles t) (suit-at-least t 3) (exposes-at-most t 10)))

(define exposes-at-most (roles (context t) (number n))
  (<= (points (roles c) (exposed c t)) n))
```

- `bid-only-enough` (S-0036-e) is the arena's `HeuristicBidder`: raise to the
  least legal bid while the hand is worth it, 42 less the count its offs
  expose under its best suit of three, and declare that suit. It reproduces
  the arena call for call and declaration for declaration on the arena's own
  auctions (400 deals at every shaker, raises and forced bids included,
  `tests/data/heuristic-bidder.txt`) and on all 1,184,040 hands
  (`tests/bidding.rs`). It opens on 403,197 of them (34.05%).
- `risk-budget` is the bid book's count as we read it (its header): each off
  risks its trick and the uncovered count of the suit it leads, the trump
  trick is at risk without the trump double, bid 42 less the risk to 12, try
  30 with two offs to 17, and 84 on the 84 shape. It is its reading on every
  hand. Its opening call over all 1,184,040 hands, exactly: a pass on 647,781
  (54.7%, S-0053-d's "about half"); 30 on 260,860, nearly half the bids; 30,
  31, 35 and 36 on 61.5% of them; 32 and 33 on 5.3%; 84 on 26,565. On the
  bid book's 30 hands it makes the book's call on 18 (always 30: 8) and agrees
  on bid or pass on 22 (always bid: 23; Zeb's forecast: 25); its count is the
  book's printed one on 8 of the 12 hands that print one.
- `never-32-or-33` (S-0152-c) raises past them to 34, or passes.

Cost. Before the aggregate each bound was a handful of `none` scopes over
sets of count tiles, each bid level re-derived the risk, and the declaration
compared suit lengths with scopes of up to seven roles. Rewritten with `card`
and `points` (the old and new calls and declarations are pinned equal on
every hand, `tests/bidding.rs`), one core decides `bid-only-enough`'s opening
in 7.5 µs a hand (13 before), `risk-budget`'s in 20 µs (150) and
`bid-only-enough`'s declaration in 92 µs (210): three policies' calls and
declarations on all hands take 31 s on 18 cores, 117 s before. Most of the
gain is counting each aggregate once per decision (the risk of a suit is read
by fifteen rules); what remains is the evaluation of the atoms in the
aggregates' bodies.

## The auction as evidence

A seat that has heard the calls knows more than its frame's voids: each
call is what its caller's bidding policy made from his hand. Given a policy
`π_P` for each seat `P` read, the belief keeps exactly the worlds `w` in
which, for every `P`,

```text
π_P.bid(auction before P's call, P, dealt_P(w), state, offered) = P's call
π_P.declare(the auction, P, dealt_P(w), state, offered) = the declaration   (P the declarer)
```

where `dealt_P(w)` is `P`'s hand in `w` and the tiles `P` has played. A
policy is a deterministic function of its auction frame, so each world's
likelihood is zero or one, and the posterior is the uniform measure on the
worlds kept: every kept world weighs one, the total is how many there are.

**How it counts.** Whether a world is kept depends, seat by seat, on that
seat's hand alone. `AuctionReading::evidence(&frame, &mut budget)` runs each
read seat's policy once on every hand it could hold (the fiber's projection
to that seat: at most C(21,7) = 116,280) and keeps the hands that agree.
`Belief::given_auction` then counts the worlds whose read hands all agree, by
placement of the tiles the query reads: with one seat read, for each of its
agreeing hands, the binomial number of ways the other two seats complete the
deal under each placement; with two or three, for each agreeing hand of the
seat with the fewest, every hand of the second within what is left, the
third taking the rest. Nothing is sampled. From the opening, with three seats
read, running the policies over all their hands takes 0.5 s on 18 cores for
`bid-only-enough` and 1.3 s for `risk-budget`, and counting the worlds their
agreeing hands (65,180 each under `bid-only-enough`) leave, under half a
second more (`tests/book_auction.rs`).

**Authority.** The arithmetic is exact, but the number is the deal's only
under the hypothesis that each read seat bids by its policy, so it is
`exact_given` one model per seat: its seat, its policy's name and a digest of
the policy's identity, the match state and the declarations offered (`S2
bids by bid-only-enough#cb7d…`). A different policy is a different model and
a different number. `Exact` stays the uniform deal's alone.

**What it reads.** The frame's auction, as the table's: a hand given with its
contract alone carries the shortest auction its declarer wins
(`Auction::won_by`), a convention, so the command line asks for the table's
auction (`--auction`). The play frame does not hold the match state, so the
reading takes the state the bids were made at, and the declarations offered,
as its own arguments. The viewer's own call is no evidence to him and is
refused. An auction no read hand agrees with is refused, as an impossible
condition is. A question whose answer reads most of a hidden hand ("he holds
two offs") resolves most of the pool, and the belief's points are placements
of what it resolves: such a question is refused above the cap, here as for
the uniform measure.

`tests/reading.rs` holds it to hand counts (a pass under "raise holding the
6-5" leaves 60 of 90 worlds; the 6-5 is then each other seat's in half), to a
policy that always passes (the belief unchanged), to one that bids exactly
holding the 6-6 (its holder certain: 38,760 × 3,432 worlds), and to brute
force over every world of a small frame under three auctions, the book's two
bidding policies and a toy one, and every set of read seats.

The book's readings (`tests/book_auction.rs`, `examples/book/bidding/bidder-holds-two-doubles.silas`):

- **C12B's condition** (the repaired low trump lead, S-0232-c): the bidder's
  hand alone puts it at 194/285; the three passes after his 30, read as
  `bid-only-enough` makes them, move it to 18,379,909/26,478,093 (69.4%);
  read as `risk-budget`, to 4,954,991/7,391,904 (67.0%). The auction moves it,
  a point either way.
- **The partner's view of a 30** (C4; p. 89): in Hand 29 (fours, 30), before
  the lead, the bidder holds the 4-4 in a third of the deals and in 6,003/8,669
  (69.2%) of those where `bid-only-enough` opens 30 and declares fours; two
  doubles or more, 313/456 (68.6%) against 6,814/8,669 (78.6%); with the
  setters' passes read too, 2,198,843/3,237,964 and 5,252,929/6,475,928.

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
(`docs/cross-game.md`, slice 1); the bid action is The auction (slice 2),
and the deal measure and the comparison of policies over deals come later
(`GAPS.md`, 2).

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
silas eval     POLICY --hand ID [--trick N] [--seat ROLE] [--book FILE]
silas bid      POLICY --hand ID [--seat ROLE] [--calls C,...] [--scoring S] [--book FILE]
silas query    QUERY --hand ID [--trick N] [--seat ROLE] [--cap N] [--book FILE]
               [--auction C,C,C,C [--shaker ROLE] --read ROLE=POLICY,... [--offered D,...] [--scoring S]]
silas query    DOMINANCE
silas coverage POLICY-OR-QUERY [--hand ID] [--random N] [--book FILE]
silas expand   FILE ...
silas fmt      [--check] FILE ...
silas check    [DIR-OR-FILE ...]
```

An option a command does not take is refused, with the options it does
take, and an unknown hand lists the hands. `silas help` prints the usage.

A position is a hand of `examples/book/hands.json`: a full deal at the point
of its narrated line (its tricks, then its tail, up to the first play the book
does not name) where `--seat` (`bidder`, `left`, `partner`, `right`) is to
play to trick `--trick` (default: the opening lead), seen from that seat;
or a hand the book gives alone, seen by its bidder before the first lead. The
bidder sits in S0. `eval` prints the position, where each rule before the
deciding one stops (the first clause, in source order, that fails at the
frame: Diagnostics, below), the deciding layer, the fired rule's witnesses and
the allowed set. Its policy is its file, or `--policy FILE`:

```text
$ silas eval examples/book/gates/c01a-top-trump-for-the-donation.silas --hand hand-32 --trick 2 --seat bidder
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

With `--auction` (the four calls from the shaker's left; `--shaker` is the
right by default, so the bidder calls first; it must be won by the bidder at
the hand's bid) and `--read` (the seats read and the bidding policy each bids
by), `query` reads the auction as evidence (The auction as evidence, above),
with `--offered` the declarations offered (by default the nine straight ones,
in `Straight42` order) and `--scoring` the match, at 0-0:

```text
$ silas query examples/book/bidding/bidder-holds-two-doubles.silas --hand hand-29 --seat partner \
    --auction 30,pass,pass,pass --read bidder=examples/book/bidding/bid-only-enough.silas \
    --offered P6,P5,P4,P3,P2,P1,P0
hand-29: trick 1, the partner (S2) to play
  ...
  auction: S3 shook: S0 30, S1 pass, S2 pass, S3 pass
  read S0: bid-only-enough calls 30 and declares straight42/P4 from 3034 of the 27132 hands it could hold
query examples/book/bidding/bidder-holds-two-doubles.silas
  46558512 worlds (5206344 kept), as 726 points resolving 0-0 2-2 3-3 4-4 5-5 6-6
  P(answer) = 1197/1517 (exact_given(S0 bids by bid-only-enough#cb7d652ef9593f67))
```

Unread, the same position gives 2949/4522 (`exact`).

On a `(dominates ...)` file, `query` needs no hand: it decides the form and
prints what it enumerated, the readings it checked and the verdict:

```text
$ silas query examples/book/cross-game/two-marks-out.silas
dominates two-marks-out: (marks 1) over (marks 2), bid by S0
  marks to 7, the bidder's side short 2: 7 states, 43 outcomes
  by the match-win order (at least as good under every match-win value)
  fails (exact): from 5-0, the hand ending 42 to 0: the first leaves 6-0 and the second 7-0 (won by T0)
```

`bid` puts a book hand's seat at its turn in the auction after `--calls`
(none by default) and prints the policy's call, the deciding rule with its
witnesses, and the declaration it would make if the others then passed.
`expand` prints a file's core: every definition expanded and every short
form spelled out. `fmt` formats canonically and idempotently: comments kept
on their own lines, a form on one line when it fits in 100 columns
(policies, their bidding and declaring sections, rules, queries, patterns,
definitions and dominances always take a line per item, with a leading
`(roles ...)`, `(out ...)` or `(play ...)`, or a rule's `(bid ...)` or `(declare
...)`, on the first), at most one blank line kept. Every `.silas` file in the crate is
in that form (`tests/language.rs`). `check` compiles every example and the
prelude (or the files and directories it is given), each policy, and each
dominance's contracts, under all nine straight declarations, and prints each
file's warnings. The command is the default `cli` feature (it reads JSON);
`cargo install --path silas` installs it.

## Diagnostics

A rule can be wrong in a way nothing refuses: it compiles, it reads well, and
it never fires. Two diagnostics look for that.

**Warnings** (`Query::lint`, `PolicyProgram::lint`, printed by `check`) are
read off the core, so a short text and its spelled-out form warn alike. Each
is located as an error is:

- a role declared and never used (it still takes a value distinct from its
  sort's other roles), or a scope's role bound and never used;
- more roles of one sort declared together than the sort has values (five
  chairs): the pattern never holds;
- `(tile a b)`, `(chair a b)` or `(context a b)` between two roles declared
  together and not joined by `same`: never true, or, negated, always true;
- a clause asked together with its own negation;
- an existential role used only under a negation: `(none (roles) (trick-play p
  his))` with `his` a role of the query asks for *some* tile `p` did not play,
  which almost any tile is; the binding belongs inside, `(none (roles his)
  ...)`;
- a rule with no pattern, and a rule whose answer it never asks to be the
  viewer's (`own-legal` or `mine`): answers the viewer cannot play are dropped
  without a word.

The crate's files raise one: `crisis.silas`'s `(not (tile his l))`, which
restates what distinctness already says (a seat's lead is not his follow).

**The funnel** (`Funnel`, `Coverage`, `why_not`) asks where a pattern stops.
Its steps are the pattern's clauses in source order, a definition call or a
short form one step; step `k` is the query of the first `k` steps, read under
the same semantics. Adding a clause only removes assignments, so the steps
that hold are a prefix, and the first that fails is where the pattern stops.
`eval` prints it for each rule that passed; `coverage` counts, over every
point of the book's narrated lines (and `--random N` seeded deals under the
nine declarations), how often each rule answered, allowed a legal tile and
decided, and how often each step held, flagging every zero with `!`. A rule
that answers but never decides is shadowed by an earlier one; one that
answers more often than it allows is answering tiles the viewer cannot play.

```text
$ silas coverage examples/book/protect-the-count.silas
coverage of policy protect-the-count over 205 frames of the book's 33 hands (...), and 0 random deals
  rule follow-the-master-without-count: decided 2, allowed 2, answered 2 of 205
         205/205  (viewer me)
          91/205  (setter me)
          64/205  (not (every-trick))
          ...
          28/205  (or (not (trump-set pard)) (some (roles his) (trick-play pard his)))
          10/205  (follower plain)
           8/205  (count plain 0)
           4/205  (follower counted)
           2/205  (count-tile counted)
  the fallback decided 203 of 205
```

The trump-set rules are rare, not mis-guarded (issue 141). Over the book's
205 frames none decides: it narrates three trump sets (Hands 32, 34, 35) and no
position where the rules apply, and the three frames that reach the setter's
ruff hold no trump. Over 3,000 random deals (84,205 frames) the three ruffs
never fire: 29 frames reach the ruff, one holds a trump, and it holds only one
(a rule without the second-trump clause would fire there, once); the feed fires
3 times and the lead 41. In the pour the ruffs fire at 328 of 4,343,426 positions.
Under a contract from a bidding policy (sampled, 3,000 contracts each, the
bidder drawing trump, defenders keeping theirs) a ruff rule answers in 0 games
under `bid-only-enough`, `risk-budget` and a marginal 30 (a hand `risk-budget`'s
count prices at 28 or 29: 13 or 14 at risk), and in 30 and 61 games under an
optimist who bids 30 on the top two tiles of two, or of three, suits whatever his
trumps. A sane bidder's setter is left a lone trump, which the ruffs' clause for a
second trump refuses; without it the games are 25, 13 and 18 (sane, sane, marginal)
against 146 and 250 (optimists).
`tests/diagnostics.rs` holds each warning, the funnel and the counts.

## Examples

`examples/` holds the queries the tests run: `partner-count.silas`,
`partner-has-count.silas`, `partner-master.silas`, `current-forced-play.silas`
and `six-five-guarded.silas`. `examples/book/` holds the book's rules as the
book means them, eighteen policies written with the prelude and named by what
they say (`first-lead`, `bidders-plan`, `crisis`, `donate-when-safe`,
`partner-leads`, `pounce`, `setter-leads`, `follow-low`, `take-the-trick`,
`defend-the-84`, ...), each header giving its catalog ids, hands and pages and
how it differs from the gate it replaces; `tests/book_rules.rs` checks each on
the book's own positions. Each names the bidder through `declarer` and is
tested at all four seatings. The facts of a frame the book narrates are
queries beside them, checked in `tests/book_queries.rs`: what an off can lose
(`off-risk`) and which side its double protects (`off-exposed`), what an 84's
defender watches for with a double (`watch-list`) or a pair (`pair-targets`),
and when a hand is over (`settled`).
`examples/book/gates/` holds the 2026-10-12 study's 21 rule gates, the record
of what that study priced (`tests/book.rs`); `tests/book_snapshot.rs` pins
what each gate allows on the book's deals and on 1,600 random ones. The
gates keep the explicit forms they were priced in; every other file is
written in the short forms (Queries, Short forms).
`examples/book/odds/` holds the odds the book prints, one query per claim
(`tests/book_odds.rs`). `examples/book/repaired/` holds the repaired book: a
rule with the condition Zeb's reasons show it omits, as a policy where the
bidder can read the condition from his own belief and as a query where he
cannot, each with a counterexample deal either way (`tests/book_repaired.rs`). The book's claims of certainty are queries on the
solved outcomes: `lay-down` (the tournament rule), `lay-down-shape` (the
book's test for one), `claim-the-rest` and `cannot-be-set`
(`tests/outcome.rs`). `examples/book/cross-game/` holds the book's
field-free comparisons of contracts as `dominates` forms (The match), and
`examples/book/bidding/` its bid rules as policies over the auction (The auction) and a bid read as evidence (The auction as evidence).
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
