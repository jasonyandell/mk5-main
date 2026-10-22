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
          | (define name (roles param ...) number)      ; a number of its own
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
          | (- number number ...)      ; the first less the rest, below zero too
          | (function argument ...)    ; a functional predicate, its number left out
          | (definition argument ...)  ; a definition of one number
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
          | (<= a b c ...) | (< a b c ...) | ...     ; a chain, but not /=
          | (least name number clause ...) | (most name number clause ...)
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
- **Chains,** as Lisp reads them: `(<= a b c)` is `(<= a b) (<= b c)`, and so
  for `<`, `>=`, `>` and `=`, each link read as above; `(<= 30 (least-raise)
  32)` is the book's "30 to 32". Negated, a chain is `(none (roles) link
  ...)`, as a negated `and` is. `/=` does not chain (neighbours apart is not
  all apart).
- **Extremes.** `(least t q c ...)` says `t` is a least candidate by the
  number `q`: it meets `c ...`, and no rival does with less. It is `c ...`
  and `(none (roles u) c' ... (< q' q))`, where `u` is `t` with the first
  suffix `-2`, `-3`, ... that no word of the form and no name around it
  takes, and `c'` and `q'` are `c` and `q` with `u` for `t`; `(most t q c
  ...)` is the same with `>`. Every candidate tied at the extreme answers
  (a declaring rule's tie-break is the order the declarations are offered
  in). Negated, an extreme is the negation of its conjunction, as an `and`
  is. `least` and `most` are reserved.

An optimist who declares the suit, of those he holds the top two of, with
the fewest tiles in hand (`examples/populations/optimist-2.silas`) needed a
rule per length before `least`; now it is one, and the two are pinned to
declare alike on every hand (`tests/bidding.rs`):

```text
(rule two-tiles (declare t) (holds-top t 2) (not (suit-at-least t 3)))
(rule three-tiles (declare t) (holds-top t 2) (not (suit-at-least t 4)))
(rule four-tiles (declare t) (holds-top t 2) (not (suit-at-least t 5)))
(rule more (declare t) (holds-top t 2))

(rule fewest-tiles (declare t) (least t (card (roles d) (of-suit d t)) (holds-top t 2)))
```

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
  one atom). A definition whose body is one number (a literal, a sum or a
  difference, an aggregate, a function, or another such definition's call)
  names that number instead, and its call stands where a number does, never
  as a clause: the prelude's `(risk t)` is the points at risk under `t`, and
  `risk-at-most` is `(<= (risk t) n)`. Definitions are expanded when text is
  read: a parsed query holds only primitives, prints in them, and its
  identity is theirs. The short forms (Queries, above) are read away the
  same way and are not part of this semantics.
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
  give. A literal `n` is `[n, n]`, `(+ a ...)` adds intervals end to end,
  and `(- a b)` is `[a_lo - b_hi, a_hi - b_lo]`: numbers are integers, and a
  difference may fall below zero (no literal does).
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
- **Functions are numbers.** A predicate *functional* in its last argument, a
  number, holds for at most one `n` at its other arguments, and where it is
  undefined it is undefined for every `n` (the table in Predicates marks
  them: `to-set`, `bid`, `least-raise`, `count`, ...). Given every argument
  but that one, `(p a ...)` is a number: `[n, n]` where `(p a ... n)` holds.
  Where `p` is undefined at `a ...` (the viewer not to call), or holds for no
  `n` (no bid yet, no raise left, a tile not played), there is no number, and
  a comparison that reads it, through any sum, is undefined: its
  precondition fails, as an atom's does. Negation stays exact (undefined is
  its own negation), so `(<= (to-set) 6)` is decided exactly where `to-set`
  has a value, and there it is `(or (to-set 0) ... (to-set 6))`. The interval
  could instead be every value `p` might take where it is undefined; that
  would decide only comparisons true of every such value (`(<= 30
  (least-raise))`), which no rule needs, at the price of a declared range
  for each function, and with no number at all it bounds nothing. A
  built-in function's relation is defined by its number (`Predicate::value`),
  so the two agree by construction. This is a core form, not a short form:
  a definition's `(number n)` is bound only when it is called, so `(<=
  (to-set) n)` cannot be spelled as literals where it is read, and a
  function inside a sum (`(+ (least-raise) (points ...))`) would spell as one
  disjunct a value; and a spelling as literals is false where the function
  has no number, so its negation would not be the opposite comparison.
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
- **A reading weighs worlds by the record.** A reading names hidden seats
  and, for each, a policy it acted by: its calls, its plays, or both. Seat
  `P`'s *dealt hand* in a world `w` is its hand in `w` and the tiles it has
  played; its frame at each of its past decisions (its auction frame, and its
  play frame before each of its plays) is that hand and the public record
  before the decision. `P`'s weight `L_P(w)` is the product over those
  decisions: a call (and, for the declarer, the declaration) weighs one when
  the policy makes it and zero otherwise; a play weighs, under the
  **consistent** likelihood, one when it is in the set `A` the policy allows
  and zero otherwise, and under the **uniform** likelihood `1/|A|` when it is
  in `A` and zero otherwise. Where neither a rule nor the exact table
  decides, the fallback stands for the policy's silence and `A` is every
  legal tile. The posterior is the uniform measure on the fiber weighted by
  `∏ L_P(w)` over the read seats; a seat not read weighs one, whatever it
  did. Since `L_P(w)` reads `w` only through `P`'s hand, the weight
  factorizes seat by seat.
- **The consistent likelihood is the default.** A play policy means the set
  it allows; its decision, the least tile of the set, is a tie-break by tile
  order and no part of what a rule says. The consistent likelihood conditions
  on an event, that every read play is one the policy allows, so the
  posterior is the uniform measure on the worlds kept, a count, as the
  auction's reading is. It assumes nothing about how a seat chooses within
  its set: a rule that allows more says less, and a policy that allows every
  legal tile (or is silent) leaves the belief unchanged. The uniform
  likelihood adds a hypothesis no book rule states, that the seat picks
  uniformly within the set, and with it the size of the set becomes
  evidence: a seat choosing at random among its legal tiles is not silent
  (it favours hands with fewer legal tiles), and widening one rule moves the
  weight of worlds it does not touch. Both are exact arithmetic; they are
  different models with different labels (The record as evidence, below).
- **Outcomes quantify over the fiber and the rest of the hand.** `(lay-down)`,
  `(certain-make)` and `(certain-set)` are Viewer atoms whose truth at `f` is a
  statement about every world of `f` and every play to the end of the hand:
  what the viewer can force, from his frame alone, against every play of the
  other three seats: information-state solves, one choice per frame. Behind
  the `clairvoyant` feature, `(clairvoyant-dd-make)` is a Clairvoyant atom: the
  double-dummy outcome in one world, an upper bound on any such guarantee
  (STRATEGY-FUSION FENCE). Each answers exactly or is
  refused (Solved outcomes, below).

## Predicates

The core built-ins have horizon zero. Each carries the version in which its
meaning last changed: `silas-v0/core` for the first set, `silas-v1/core` for
the record's first predicates, `silas-v2/core` for those that changed or came
with the one-frame record and the full contract (marked v2). The outcome
predicates (three, and a fourth under the `clairvoyant` feature), version `silas-v3/outcome`, look to the end of the hand (Solved
outcomes, below). The auction's, version `silas-v4/auction`, are in The
auction, below, with the suits' face order, `ranks-above`, version
`silas-v5/suit-order`. *World* access means the predicate reads a hidden world;
*Viewer* means it reads only the frame; *Clairvoyant* means it reads a hidden
world and plays it out with every seat seeing all four hands. A policy, and a
viewer evaluation, refuses World and Clairvoyant predicates; the refusal of a
Clairvoyant one names the STRATEGY-FUSION FENCE.

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
| `count(d,n)` | Viewer | `d` carries `n` count points. Functional: `(count d)`. |
| `quota(c,n)` | Viewer | Seat `c` holds `n` tiles. Functional. |
| `tile(a,b)`, `chair(a,b)`, `context(a,b)` | Viewer | Equality; anchors a role to a literal. |
| `team(c,t)`, `partner(a,b)`, `opponent(a,b)`, `successor(a,b)` | Viewer | Partnerships and clockwise order. |
| `viewer(c)`, `leader(c)` | Viewer | The viewer; the leader of the open trick (or the next). |
| `next-actor(c)` | Viewer | `c` is to play; false for every seat once the hand is over. |
| `led-context(q)`, `current-winner(c)` | Viewer | The open trick; undefined before a lead. |
| `trick-play(c,d)` | Viewer | Seat `c` played `d` to the open trick. |
| `own-legal(d)` | Viewer | `d` is a legal play for the viewer; undefined unless the viewer is next. |
| `legal(c,d)`, `forced(c,d)` | World | `d` is a legal (the only legal) play for `c`; undefined unless `c` is next. |
| `declarer(c)` | Viewer | Seat `c` won the auction. |
| `bid(n)` | Viewer | v2: the bid as the table says it: 30 to 41 points, 42 per mark (84 for two). Functional: `(bid)`. |
| `stake(n)` | Viewer | v2: the hand is played for `n` marks. Functional. |
| `banked(c,n)` | Viewer | Seat `c`'s side has taken `n` points in closed tricks. Functional. |
| `to-make(n)` | Viewer | The declarer's side needs `n` more points to reach its threshold; zero once made. Functional. |
| `to-set(n)` | Viewer | The other side needs `n` more points to set it; zero once set. Functional: `(to-set)`. |
| `trick-number(n)` | Viewer | v2: the trick being played or about to be led is the `n`-th, from one; undefined once all seven have closed. Functional. |
| `led(c,d)` | Viewer | Seat `c` led `d` to some trick of the record, the open one included. |
| `played-to(l,c,d)` | Viewer | Seat `c` played `d` to the trick led by `l` (`d` is `l` for its leader). |
| `trick-of(d,n)` | Viewer | `d` was played to the `n`-th trick. Functional; no number before `d` is played. |
| `shown-void(c,q)` | Viewer | Seat `c` has publicly failed to follow `q`, read off the record. For a hidden seat it is the void that forbids its worlds. |
| `lay-down()` | Viewer | v3: some fixed sequence of the viewer's tiles wins every remaining trick in every world, against every play of the other three seats; undefined unless the viewer is to play. |
| `certain-make()`, `certain-set()` | Viewer | v3: the viewer, choosing from his frame alone, forces the make (the set) in every world against every play of the other three seats; undefined unless that is his side's goal. |
| `clairvoyant-dd-make()` | Clairvoyant | v3, `clairvoyant` feature only: in this world the declaring side makes from here under double-dummy play. STRATEGY-FUSION FENCE: never in a policy (`wiki/topics/strategy-fusion-fence.md`). |

Numbers and teams are literal-only (a number is never a role; the aggregates
and the functions compare sizes, sums and the frame's numbers without one), so
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
`(every-trick)` (a bid of marks, `(<= 42 (bid))`), and `(lead-sets d)`,
whether a lead's own count, one called count tile and the trick reach what
the setters still need: `(<= 1 (to-set) (+ (count d) (count called) 1))`.
Each is documented where it is defined. Every rule in `examples/book/` is
written with it.

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
- `Belief::given(&evidence, tiles, cap)` and
  `Belief::given_for(&evidence, &query, cap, &mut budget)`: the uniform
  measure weighed by the record, the calls and plays of named seats read
  under their policies (The record as evidence, below), resolved as the
  uniform ones are.
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
          | (reading name (read chair policy) ... [(likelihood l)] [(at-least p)] body)
                                               ; The record as evidence, below
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
has none below the five that four calls can reach, Lean's `mark_le_ceiling`,
which the TypeScript engine offers too, #142) and the forced bid (the shaker may not pass
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
| `ranks-above(a,b,q)` | both | v5: `a` and `b` bear `q`'s pip and `a` ranks above `b` by their faces: the double first, then the other pip, highest first. |
| `shaker(c)` | both | `c` shook. |
| `bid-of(c,n)`, `passed(c)` | both | `c`'s call: the bid the table calls `n`, or a pass. `bid-of` is functional. |
| `no-bid()` | both | No seat has bid yet (false in the play). |
| `high-bid(n)`, `high-bidder(c)` | both | The high bid so far, and its bidder. `high-bid` is functional: `(high-bid)`. |
| `legal-bid(n)`, `least-raise(n)` | auction | `n` is a legal bid, the least one; undefined unless the viewer is to call. `least-raise` is functional: `(least-raise)`. |
| `scoring-marks()`, `scoring-points()` | auction | How the match is scored. |
| `score(c,n)`, `short(c,n)`, `short-at-most(c,n)` | auction | `c`'s side has `n`; needs `n` (at most `n`) to win. `score` and `short` are functional. |
| `behind(c,n)` | auction | `c`'s side trails by at least `n` (one or more). |

The play frame reads the auction's record too (who bid what), but not yet
the match state: the score predicates are of the auction alone.

Before a declaration no suit has a trick key, so a suit's order is its
faces': `ranks-above` is a tile fact, the same at every frame. It is not a
convention: under each of the nine straight declarations, on the tiles that
follow a pip suit the declaration leaves alone (and are not trump), it is
exactly the trick key's order in that suit (`tests/auction.rs`). So "the top
two of a suit" is said once for every suit, not anchored to the six-five:
`(top-of-suit d t n)` is `(bears d t)` with fewer than `n` tiles ranking
above `d`, a `card`, and `(holds-top t n)` is the viewer holding all `n` of
them.

```text
section  := (bidding brule ...) | (declaring drule ...)
brule    := (rule name (select bid choice) (guard query))
          | (rule name (bid choice) body)          ; short form
choice   := n | pass | least-raise                 ; n: 30 to 41, or 42 per mark
          | number                                 ; a computed call: (bid (- 42 (risk t)))
drule    := (rule name (select declare kind) (guard query))
          | (rule name (declare kind) body)        ; short form; a name is answer
kind     := answer | q0 ... q6 | doubles | no-trump
```

`CompiledPolicy::bid(&AuctionFrame)` is the first bidding rule whose guard
holds and whose call is legal (`least-raise` is the least legal bid), else the
fallback: the least legal call, a pass when it is legal and 30 for a forced
shaker.

A *computed call* is a number (Queries, `number`) in the call's place. Its
guard answers the roles the number names (the short form `(bid N)` makes
them its `out`, in order of first use; the explicit `(select bid N)` must
answer exactly those), so the number is a function of an answer. At each
answer it is an interval (Semantics, Aggregates); where that is one value
`n` and the table calls `n` a legal bid (`Bid::number`: 30 to 41, or 42 a
mark), the answer gives that bid. The rule allows the bids its answers give
and calls the least, as a play rule plays the least tile it allows; it is
the rule `(bid n)` with the clause `(= N n)` added to its guard, for each
legal `n`. A number that names no bid (29, 43, below zero), names one that
is not legal here (a bid at or under the high bid), or has no value (a
function with no number, an interval not decided) gives no call, and the
rule passes to the next as a literal level over a higher bid does. It never
clamps to the nearest legal bid: over a 40, a hand worth 38 passes, which is
what it would have said with a rule per level, and a policy stays total by
its later rules and the fallback. `(bid (least-raise))` decides as `(bid
least-raise)`. A computed call is a bid's only: a play or a declaration
names its answer. `explain_bid` keeps the witnesses of the answers that give
the call. `declare` is, of the declarations offered that the first declaring
rule to name one allows, the first offered (an `answer` rule names the pip
suits its guard answers, one context column), else the first offered. A
declaration is named by what its tables make trump (`Trump::of`: one pip's
tiles, the doubles, or nothing), so the order the declarations are offered
in is the tie-break. A bidding or declaring rule may name no `World`
predicate and none of the play; rule names are unique across the sections.
`explain_bid` and `explain_declare` add the deciding rule's witnesses; `silas
bid` prints them.

`lib/bidding.silas` (`(include bidding)`) is the auction's vocabulary: a
suit's length, its double, its top tiles, its offs, the count an off
exposes, and the book's count of what is at risk. Sizes and sums are
aggregates (Semantics, above): "at most 10 points exposed" is
`(exposes-at-most t 10)`, one `(<= (points ...) 10)`, and a definition's
`(number n)` takes the bound, so every bound of the prelude is one
definition with a number (`suit-at-least`, `offs-at-most`, `exposes-at-most`,
`risk-at-most`, `top-of-suit`, `holds-top`), never a family of literals,
and the risk itself is a number, `(risk t)`. The auction's own numbers are
functions (Semantics, Functions are numbers): "the least raise and the count
exposed come to 42 at most" is one comparison, so `bid-only-enough`'s three
rules of least-raise ranges are one. The call is a number too (a computed
call, above): "bid 42 less the risk under the least at risk" is one rule,
where it was a rule per level, thirteen. `examples/book/bidding/` holds three
policies:

```text
(rule raise (bid least-raise)
  (some (roles t)
    (suit-at-least t 3)
    (<= (+ (least-raise) (points (roles c) (exposed c t))) 42)))

(rule priced (bid (- 42 (risk t)))
  (least t (risk t) (suit-at-least t 3))
  (<= (risk t) 12))
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
  30 with two offs to 17, and 84 on the 84 shape: three bidding rules, the
  levels one computed call, whose every call at every least raise and every
  declaration won at 30, 42 and 84 was the thirteen literal levels' on all
  1,184,040 hands (one digest, run once; the suite keeps every 8th hand at
  every raise, and every hand at the opening and over 31 and 36,
  `tests/bidding.rs`). It is its reading on
  every hand. Its opening call over all 1,184,040 hands, exactly: a pass on 647,781
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
aggregates' bodies. The computed call reads every suit's risk whole (the
least has no literal to stop at), and two changes keep it cheaper than the
levels were: a pattern checks its atoms, then its comparisons, then its
scopes, each that reads only the frame before any that reads a world (a
conjunction's order is not its meaning), and an aggregate is
counted once per shape up to the names around it, so `(risk t)` in a rule,
in its rival under `least` and in its call is one count a suit.
`risk-budget`'s calls at all seventeen least raises and three declarations
take 1.5 ms a hand on one core; the thirteen levels take 2.1 ms on the same
evaluator, 2.9 ms before those two changes.

## The record as evidence

A seat that has seen the record knows more than its frame's voids: each call
is what its caller's bidding policy made from his hand, and each play what
its player's policy allowed from his. Given a policy for each seat read, the
belief weighs each world by how well the seat's record fits it (Semantics:
a reading weighs worlds by the record). A `Reading` names what is read:

```rust
let reading = Reading::calls(readers, state, offered)?          // the auction
    .with_plays(readers, Likelihood::Consistent)?;              // and the play
let reading = Reading::plays(readers, Likelihood::Uniform)?;    // the play alone
let evidence = reading.evidence(&frame, &mut budget)?;
let belief = Belief::given_for(&evidence, &query, cap, &mut budget)?;
```

**The calls.** The belief keeps exactly the worlds `w` in which, for every
seat `P` whose calls are read,

```text
π_P.bid(auction before P's call, P, dealt_P(w), state, offered) = P's call
π_P.declare(the auction, P, dealt_P(w), state, offered) = the declaration   (P the declarer)
```

A bidding policy is a deterministic function of its auction frame, so each
world weighs zero or one, and the posterior is the uniform measure on the
worlds kept: the total is how many there are. Its fallback, the least legal
call, is a call it makes: a pass is evidence.

**The plays.** For every seat `P` whose plays are read and each of its past
plays `d_i`, Silas builds `P`'s frame before the play (the auction, the
declaration, `P` as viewer, `dealt_P(w)` and the record before `d_i`) and asks
`P`'s policy what it allows there. Under the consistent likelihood the world
is kept when every `d_i` is allowed; under the uniform likelihood it weighs
`∏ 1/|A_i|`. A play the fallback decides is silence (every legal tile), so a
bidding policy read on the plays says nothing about them, and a play rule
speaks only where it fires: `protect-the-count`, read on a setter, reads only
his follows to the bidder's sure leads. A seat may be read on its calls and
its plays at once, under one policy or two; both read the one dealt hand.

**How it counts.** Whether a world is kept, and its weight, depends seat by
seat on that seat's hand alone (Semantics). `Reading::evidence(&frame, &mut
budget)` lists each read seat's possible hands (the fiber's projection to it:
a hand of its size from the tiles its voids allow, whose rest the other two
seats can hold, at most C(21,7) = 116,280), runs its policy on each (once for
the call, once per past play, stopping at the first refusal) and keeps the
agreeing hands with their weights. `Belief::given` then sums the weights of
the worlds whose read hands all agree, by placement of the tiles the query
reads: with one seat read, for each of its agreeing hands, its weight times
the binomial number of ways the other two seats complete the deal under each
placement; with two or three, for each agreeing hand of the seat with the
fewest, every hand of the second within what is left, the third taking the
rest, each weighed. Nothing is sampled. Under the uniform likelihood a seat's
weights are `1/d` with `d` a product of allowed sets' sizes, the `k`-th at
most `8 − k`, so every `d` divides 217,728,000; the sums are integers over
that scale until the last division, and exact. `tests/play_reading.rs` holds
the counting to brute force: every world of small frames, each read seat's
frames built from the world and the record, the policy asked at each, under
both likelihoods, every set of read seats, mixed policies, with and without
the calls. That the counted and the brute-force posteriors agree is the
factorization, tested.

**Authority.** The arithmetic is exact, but the number is the deal's only
under the hypothesis that each read seat acts by its policy, so it is
`exact_given` one model per seat and phase: `S2 bids by bid-only-enough#cb7d…`
(a digest of the policy's identity, the match state and the declarations
offered), `S3 plays within protect-the-count#438c…` under the consistent
likelihood and `S3 plays uniformly within protect-the-count#438c…` under the
uniform one (a digest of the policy's identity). A different policy, or the
other likelihood, is a different model and a different number. `Exact` stays
the uniform deal's alone.

**What it reads.** The frame's auction, as the table's: a hand given with its
contract alone carries the shortest auction its declarer wins
(`Auction::won_by`), a convention, so the command line reads calls only when
`--auction` names the table's. The play frame does not hold the match state,
so the reading of the calls takes the state the bids were made at, and the
declarations offered, as its own arguments. The viewer's own record is no
evidence to him and is refused. A record no read hand agrees with is refused,
as an impossible condition is. A question whose answer reads most of a hidden
hand ("he holds two offs") resolves most of the pool, and the belief's points
are placements of what it resolves: such a question is refused above the
cap, here as for the uniform measure, and asked of the seat's hand law
instead (below). A question over the trumps out reads
best with the trump bound first, `(trump q) (none (roles t) (in t q) ...)`,
so that it resolves the trumps alone.

**A seat's hand law.** A question of one read seat's whole hand ("the
bidder holds two offs", pp. 53–54) is asked of that seat's hand law, a belief
of its own:

```rust
let law = Belief::hand_law(&evidence, seat, cap)?;           // read
let open = Belief::uniform_hand_law(frame, seat, cap, &mut budget)?;   // unread
```

Its points are the seat's agreeing hands, each weighing its weight times the
ways the other two seats complete the deal: the sums the counting already
makes, keyed by the seat's hand instead of by a placement (with two or three
seats read, each world's weight is every read seat's, summed into the hand of
the one asked). At most C(21,7) = 116,280 points, whatever the question
reads of that hand. A query may read the seat's hand and the viewer's; a
read of either other hidden seat is refused, except that a tile the seat
holds is no one else's (`Belief::known` names the seat; `Holdings` refuses
the rest). The authority is the evidence's, `exact_given` the read policies,
and a question of the seat's hand alone has the same answer under the law as
under `Belief::given` (`tests/reading.rs` holds both to brute force on every
world of small frames, every set of read seats, under the three policies;
the uniform law to the uniform measure). The unread law is `exact`. On the
command line, `--law ROLE` asks it of a hidden seat, read or not:

```text
$ silas query examples/book/bidding/bidder-holds-two-offs.silas --hand hand-33 --seat left \
    --auction 35,pass,pass,pass --read bidder=examples/book/bidding/risk-budget.silas \
    --offered P6,P5,P4,P3,P2,P1,P0 --law bidder
hand-33: trick 1, the left (S1) to play
  ...
  read S0: risk-budget calls 35 and declares straight42/P4, and risk-budget allows its plays 4-4, from 1587 of the 38760 hands it could hold
query examples/book/bidding/bidder-holds-two-offs.silas
  133024320 worlds (5446584 kept), as 1587 points, the hand law of S0
  P(answer) = 1 (exact_given(S0 bids by risk-budget#7996dcc962c3ff7e + S0 plays within risk-budget#d6e3de7f59acb0b2))
```

Before the lead his 35 and fours make two offs 1,587/1,751 (77/646 unread);
the 4-4 he led takes away the one-off 35s, which lack the trump double
(`tests/book_auction.rs`). Without `--law` the question is refused above the
cap. Because a law leaves the other seats unresolved, a pattern checks every
clause that reads only the frame before any that reads a world, so that
`(right-setter s)` binds `s` before `s`'s hand is read.

**Cost.** The calls: from the opening, with three seats read, running the
policies over all their hands takes 0.5 s on 18 cores for `bid-only-enough`
and 1.3 s for `risk-budget`, and counting the worlds their agreeing hands
(65,180 each under `bid-only-enough`) leave, under half a second more
(`tests/book_auction.rs`). The plays: one policy evaluation per possible hand
per past play, each building the seat's frame from the record. The hands a
seat could hold shrink as it plays faster than its plays grow, so the cost is
at the first tricks. Over every point of the book's eight narrated deals, seen
from each seat (656 frames), with the three hidden seats read under each of
`protect-the-count`, `donate-when-safe`, `follow-low`, `bidders-plan` and
`crisis` and under both likelihoods (6,560 readings, the question asked
after), a reading took 40 ms on average on 18 cores: 0.18 s in trick one,
0.05 s in trick two, 0.014 s in trick three, under 2 ms from trick five; the
slowest, 1.3 s (`crisis`, uniform, in trick one).

`tests/reading.rs` holds the calls to hand counts (a pass under "raise
holding the 6-5" leaves 60 of 90 worlds; the 6-5 is then each other seat's in
half), to a policy that always passes (the belief unchanged), to one that bids
exactly holding the 6-6 (its holder certain: 38,760 × 3,432 worlds), and to
brute force over every world of a small frame under three auctions, the
book's two bidding policies and a toy one, and every set of read seats.
`tests/play_reading.rs` holds the plays to a policy that allows every play
(the belief unchanged under the consistent likelihood, every set of seats; a
random player, under the uniform one, is evidence), to one that plays the 6-6
whenever it can (not having led it, the seat does not hold it: 5/16 to 0),
to brute force, and the calls and plays to one hand.

**A guard that reads (#128).** A rule acts on a reading through a viewer
predicate the host names and registers: `ReadingPredicate::new(name,
readers, likelihood, event, bound)`, the readers named by chair from the
bidder (`ReadChair::RightSetter`, ...), so the one guard serves every
seating; the event a query with no outputs; the bound a probability in
(0, 1]. `(name)` holds at a frame when the plays of the readers, read under
the likelihood, put the event at a probability of at least the bound (1:
the reading makes it certain). The viewer's own chair is dropped, since his
record is no evidence to him; with no chair left the event is asked of the
uniform deal. A record no read hand agrees with is no reading: the atom is
undefined there, and neither it nor its negation holds. It is a function of
the frame, because the reading is, so a policy may use it; the reading's
models are in the predicate's version, so a policy that reads one policy is
not the policy that reads another. Its answer is remembered by the frame's
record (the policy key), so a guard asked once per binding, or a policy
priced over a pour, pays for each record once. The standard registry does
not know a reading; a policy that names one is compiled against the
registry that holds it.

```rust
let right = (ReadChair::RightSetter, protect_the_count);
let guard = ReadingPredicate::new("left-holds-the-trumps", vec![right],
    Likelihood::Consistent, left_setter_holds_the_trumps_out, one)?;
let mut registry = Registry::standard();
registry.register(guard)?;
let policy = PolicyProgram::parse(text, name)?.compile(&registry, &physics)?;
```

**A reading in a policy file (#162).** A policy may declare its readings
before its rules, and its rules name them like any atom:

```text
(policy crisis-read
  (reading one-setter-holds-the-trumps-out
    (read left-setter protect-the-count)
    (read right-setter protect-the-count)
    (trump q)
    (some (roles s) (setter s) (none (roles t) (in t q) (live t) (not (mine t)) (not (holds s t)))))
  (rule ... (one-setter-holds-the-trumps-out) ...))
```

A chair is `bidder`, `left-setter`, `partner` or `right-setter`; a reader is
named by its policy's name; the likelihood is `consistent` unless said, and
the bound 1 unless `(at-least p)` says otherwise (`p` is `n` or `n/d`). The
body is the event's, as a rule's short form is its guard's, and prints as
`(event (query ...))`, which reads back. Such a policy is compiled with
`PolicyProgram::compile_reading(registry, physics, resolve)`, where `resolve`
finds a reader by name; plain `compile` refuses it and says so, and so does a
reader that cannot be found or a policy that reads itself, however far round.
The `silas` command finds a reader as the policy file of that name beside the
file that reads it. `examples/book/crisis-read.silas` is the book's crisis as
Hand 31 reads it (`tests/reading_text.rs`).

`tests/guard_reading.rs` asks the book's two readings of Hand 31 as guards at
every seating (certain read, 5/22 and 2/15 unread), holds a rule that leads
trump into the left setter to firing exactly where the reading is certain,
and holds the viewer's chair, an impossible record, the identity and the
memory to their statements.

The book's readings of a bid (`tests/book_auction.rs`; `examples/book/bidding/bidder-holds-two-doubles.silas`,
`losing-31-holds-a-double.silas`, `bidder-holds-two-offs.silas`):

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
- **A losing 31** (W42-p054-b, p. 54: a double or two): the deal of Hand 33
  under an auction of ours (the left setter 30, the partner passes, the right
  setter 31, the bidder 35), seen by the left setter before the lead. The
  31, read as `bid-only-enough` makes it, lifts "a double" from 2,441/2,584
  (94.5%) to 24,033/24,989 (96.2%) and "two or more" from 313/456 (68.6%) to
  18,684/24,989 (74.8%); read as `risk-budget` (a risk of exactly 11), to
  certainty and 2,158/2,187 (98.7%).
- **A 35 means two offs** (W42-p054-a; the bidder's hand law): Hands 33 and 32
  are one deal; the bidder opens, the rest pass, and the left setter reads
  his call and declaration as `risk-budget` makes them, before the lead. The
  35 in fours puts two offs (a tile neither trump nor double) at
  1,587/1,751 (90.6%) and one off at 164/1,751, never none or three, from
  77/646 (11.9%) for two unread; with every call read, 1,897,452/2,112,947.
  The 36 in blanks puts one off at certainty. `bid-only-enough` raises to the
  least bid and never bids 35 or 36 in one round.

The book's readings of a play (`tests/book_reading.rs`, `examples/book/reading/`),
with the bidder in every chair, each the same under both likelihoods but the last:

- **The left setter's certainty** (S-0140-e): in Hand 31 (deuces, 30), on lead
  to trick six with the 1-1 and the 5-2, the left setter knows the bidder
  holds both trumps still out, so either lead loses both tricks. Over the 15
  worlds he cannot tell apart it is 2/15: the right setter never showed out of
  trump. Read under `protect-the-count`, the right setter's 3-2 on the
  bidder's 2-2 at trick one, a trump that is count, was his only trump: 1.
- **The bidder's two readings after trick one** (S-0138-e, the same hand): the
  same 3-2 puts both trumps out with the left setter, 5/22 to 1; the
  partner's 5-0 on the top trump, read under `donate-when-safe` (the biggest
  count first), puts the 5-5 with a setter, 5/8 to 1.
- **The partner's top trump** (C2; p. 25 against Hand 30): from the bidder's
  chair after the partner's 6-6 on his 6-5, the partner is out of trump in
  26/57 of the worlds. Read under the worked hand's sender
  (`partner-plays-the-top-trump`) the play says nothing about his other trumps:
  26/57 still. Read under p. 25's sender (the top trump only when it is his
  only one), 1. The two passages are two policies, and two numbers.
- **The bait** (C21; p. 107, the same hand): the partner, seeing the bidder
  lead the 6-5 at trick two, knows he holds the 6-4. Uniformly 5/16. Read
  under `crisis`, whose bait leads the 6-5 holding the 6-4, 91/167: where
  the rule is silent the bidder might have led anything. Under the uniform
  likelihood, 1549/1777: a silent lead is one tile in six, the bait one in
  one. The book's certainty needs a bidder who never leads the 6-5 without
  the 6-4, which no rule it states says.

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

## The clairvoyant (double-dummy) reference

**STRATEGY-FUSION FENCE.** `ClairvoyantSolver` is an exact perfect-information
(double-dummy) solver. **It sees all four hands.** Its value read as the value
of a move for a seat that cannot see them, world by world and then averaged,
is strategy fusion: strong but wrong. It is a reference, never a player: it
explains a book move only where the book's own argument is about a fully
known position. Everywhere else its value is an upper bound on what any real
seat can achieve, and a move it calls a mistake may be right for a seat that
cannot tell the worlds apart. The ruling, the inventory and the sanctioned
uses: `wiki/topics/strategy-fusion-fence.md`.

The fence has three layers in Silas. The module and every public item
(`ClairvoyantSolver`, `clairvoyant_grade_line`, `OpenPosition`, `Objective`,
`RootValues`, `Deal`, `Grade`, `Judged`) and the predicate
`(clairvoyant-dd-make)` exist only under the non-default `clairvoyant` cargo
feature; the crate's tests and its showcase enable it explicitly (a
dev-dependency of the crate on itself), and `scenario/core` does not. The
names say what they are at every call. A policy that names the predicate is
refused, and the refusal names the fence. What a player uses instead is an
information-state solve: `(lay-down)`, `(certain-make)`, `(certain-set)`
(Solved outcomes, below), or Zeb's plan.

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

`clairvoyant_grade_line(&deal, &line)` grades each play of a line from the deal's
opening lead: the seat, its legal plays, and under each objective the play's
value, the best value and the moves tied at best (`Grade`, `Judged`).
`examples/book/clairvoyant_grade.rs` grades every narrated play of the book's
eight full deals from `examples/book/hands.json`, in well under a second
(`cargo run --release --features clairvoyant --example clairvoyant_grade`).

`tests/clairvoyant.rs` holds it to an unpruned exhaustive search on sampled
positions with one to four tricks left (every root move, both objectives), to
an unpruned search of Hand 28's whole game tree memoized only on exact
trick-start states, to seat rotation, and to Hand 28's narrated line, which
it replays to the 25 and 17 the book awards.

## Solved outcomes

The book's tournament rule forfeits a bidder who lays his hand down when some
hidden layout and some defence could set him (Roberson, *Winning 42*, Ch. 14).
Its question is not what happens in one world under best play by everyone. It
is what one player can guarantee without seeing the hidden hands. Three
predicates answer it, and a fourth, behind the fence, bounds it. Each looks to the end of the hand (horizon
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
- **`(clairvoyant-dd-make)`** (Clairvoyant; the `clairvoyant` feature only).
  True in a world when the declaring side makes from here under double-dummy
  play: both sides see all four hands and play their best (the
  `ClairvoyantSolver`, `Objective::Make`). STRATEGY-FUSION FENCE: a bound and a
  showcase, never a policy's input.

**Whose play is optimized.** In `(certain-make)` and `(certain-set)` it is
`v`'s alone, over strategies that read only his frame. The setters' plays range
over every legal play, and so do his partner's: the guarantee stands on `v`'s
own choices, as a claim does. It is therefore a lower bound on what `v`'s side
could guarantee with a partner who plays to the same goal from his own
information. That team guarantee is not computed (`GAPS.md`, 8). In
`(clairvoyant-dd-make)` every seat is optimized, and each world gets its own best line.

**The order.** For a viewer on the declaring side, at a frame where taking
every remaining trick reaches the threshold:

```text
(lay-down)  =>  (certain-make)  =>  his side can force the make  =>  (clairvoyant-dd-make) in every world
```

Each implication is strict. The last is the double-dummy bound. In each world
the declaring side may play a different line, one that reads the hidden hands
(strategy fusion), so per-world minimax bounds every real guarantee from above
and is never below one. For a setter, `(certain-set)` implies `(clairvoyant-dd-make)` is
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
   Nothing is concluded from not finding one. (The fence's sanctioned use:
   the clairvoyant outcome as a bound that refutes, never as a value.)
4. Otherwise Silas searches `v`'s information states, an and-or search over
   frames. At `v`'s turn some legal tile must win. At another seat's turn
   every tile that some world lets it play must lose; such a tile is one
   whose frame exists. Every play is public, so `v`'s information state after
   a history is exactly that history's frame, and the worlds still possible
   are its fiber. The search is therefore exact over `v`'s strategies. It
   memoizes on the record, stops at step 1 and 2's answers, and is refused
   when it exceeds the budget.

**Authority.** The three Viewer predicates read the frame and compute over its
whole fiber; their value is the same at every point of any belief. `(clairvoyant-dd-make)`
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
and the deal measure and the comparison of policies over deals are The field
(slice 3).

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

## The field

A field is four players, one a seat; a comparison asks how much better one
player does than another in a chair of that field (`docs/cross-game.md`,
slice 3; `tests/field.rs`).

```rust
pub enum Player { Policy(Arc<CompiledPolicy>), Chance { seed: u64 } }
pub struct Field;                                   // Field::new([Player; 4]), id(): the players, seat by seat
pub struct Table { config: AuctionConfig, offered: Arc<[Physics]>, state: MatchState }
pub fn play_out(world: &World, shaker: Seat, field: &Field, table: &Table, budget: &mut Budget) -> Result<Settlement>;
pub enum Settlement { Played { auction, contract, outcome, plays }, PassedOut { auction } }
pub enum Deals { Listed(Vec<World>), Uniform(Sample) }
pub enum Seats { Rotate, Fixed(Seat) }
pub struct Trial { field, deals, seats, table, utility: Utility, confidence: BigRational }
impl Trial { pub fn compare(&self, a: &Player, b: &Player, budget: &mut Budget) -> Result<Paired>; }
```

- **A player** is a Silas policy, which bids, declares and plays by its
  sections (a section it lacks is its fallback: it passes, or bids 30 when
  forced, declares the first offered and plays its lowest legal tile), or the
  chance player: the arena's `RandomBidder` and lowest-legal play, drawn
  exactly. It bids with probability 3/10 (always when it may not pass),
  uniformly among the legal bids of at most one mark, declares uniformly
  among the pip suits offered, and plays its lowest legal tile; its draws are
  keyed on its seed and its auction frame, so it too is a function of what it
  sees.
- **A play-out** (`play_out`) is one deal (`World::deal`, seven tiles to
  each seat) from one shaker: every seat calls by its player at its own
  `AuctionFrame`, the winner declares, then all 28 tiles are played, each
  seat choosing by its player at its own `Frame`. The settlement is the
  auction, the contract, the `Outcome` (the two sides' points, 42 together)
  and the plays. Four passes without the forced bid are `PassedOut`: no
  contract; the next seat shakes, which is another deal, so the hand moves
  no score (nothing in marks or points; the match-win value of the state
  itself).
- **The deals.** `Listed` is those deals, each once: a claim about them
  alone. `Uniform(Sample { n, seed })` is `n` deals from a seeded splitmix64
  stream, a Fisher–Yates shuffle with every draw exactly uniform, so each of
  the 472,518,347,558,400 deals is equally likely. `Rotate` puts the player
  under test in every chair, `Fixed(seat)` in one, each at every shaker: 16
  or 4 seatings a deal.
- **The comparison.** `a` and `b` each take the chair, on the same deal, at
  the same shaker, against the same three other seats (the field's player
  in the chair sits out). A deal's difference in the utility of the chair's
  side is its mean over the seatings: one cluster. The answer (`Paired`) is
  the mean over deals (`delta`, an `Estimate`), each deal's difference, the
  flips (deals `a` did better, `b` did, tied) and an interval: over a listed
  set, the mean itself, exact given the field; over a sample, the empirical
  Bernstein bound (Maurer and Pontil 2009, Theorem 4, both sides) at the
  trial's confidence, from the sample variance and the width of the range a
  difference lies in (four times the most marks four calls reach; in points,
  four times a set marks bid at that most; 2 for a match-win value), its
  logarithm and root bounded above in exact rationals and its half-width
  rounded up to a millionth. The verdict is `Better` or `Worse` only when
  the interval clears zero. Every comparison runs `a` against the chance
  player too (`chance()`), from the same play-outs of `a`.
- **Authority.** A field is a model: every answer is `exact_given(field)`,
  the field named by its players (`NAME#DIGEST`, as a reading names a
  policy, or `chance#SEED`), met with the sample's `sampled(n, seed)` and the
  utility's value. None is `exact`. A match-win value is one side's, so a
  chair on the other side is refused, and a utility whose unit is not the
  table's scoring is refused, since the players bid for that game.

## The arena ledger

A field with a seat only the arena can play (Zeb, a net, an overbidding
heuristic) is compared by the arena and imported as data
(`docs/cross-game.md` § 2.5, slice 4; `tests/ledger.rs`).

```rust
pub struct Ledger { field: String, players: [String; 3], sample: Sample, seats: Seats, table: Table, rows: Vec<LedgerRow> }
pub struct LedgerRow { deal: usize, world: World, shaker: Seat, chair: Seat, arm: Arm, settlement: Settlement }
pub enum Arm { A, B, Chance }
impl Ledger { pub fn from_json(source: &str) -> Result<Ledger>; }   // the `ledger` feature (on with `cli`)
impl Paired { pub fn from_ledger(ledger: &Ledger, utility: &Utility, confidence: &BigRational) -> Result<Paired>; }
```

- **The format** (`silas-ledger-1`, written by `arena/ledger.py`) is one JSON
  object: the field's identity, the three players' names, the sample (`n`,
  `seed`), the seatings (`"rotate"` or a chair), the auction (`max_marks`,
  `forced`), the declarations offered (`P6` ... `P0`, `DT`, `NT`), the scoring
  (`kind`, `target`), the score, and one row per deal, seating and arm: the
  hands as dealt (tiles `h-l`), the shaker, the chair, the arm (`a`, `b`,
  `chance`), the calls, the contract (declarer, bid, declaration) and the
  outcome (each side's points), or neither for a hand passed out, and the 28
  plays as `[seat, tile]`. Every key is required and no other is read: a
  ledger that carries a utility (marks won, a transfer, a value) is refused.
- **Reading it.** `from_ledger` replays every row under Silas's rules: the
  auction must be the table's, the contract its auction's, the declaration
  offered, each play by the seat to play, a tile it holds and may play, and
  the ledger's outcome the replay's; each deal must hold every seating of
  every arm once, on one deal. It settles every hand by `Contract::settle`
  and values it by its own `Utility`, so the units are Silas's: the utility's
  unit must be the ledger's scoring (a marks ledger read as points is
  refused), since the players bid for that game. The answer is a `Paired`, as
  `Trial::compare`'s, with the chance arm from the ledger's own chance rows,
  and the authority `exact_given(field)` met with `sampled(n, seed, arena)`:
  the arena's deal stream, not Silas's (`Sample::arena`).
- **The differential check.** For a field of Silas policies the arena's
  ledger and Silas's `play_out` agree hand for hand, call for call and tile
  for tile (`data/ledger-*.json`, written by `arena/test_ledger.py`); each
  reads to the arena's own paired mean to the rational, and its flips. That
  is how a ledger from a field Silas cannot play earns trust in its plumbing.
  `silas compare --ledger FILE` prints the comparison.

## The decision quotient

A solved course of a viewer's decisions is reduced to the fewest states that
give the same answers to every block sequence (Jason's note *A minimal
comparison algebra for a declared 42 family*, 2026-10-21; the theorems are in
`scenario/lean/Stab/Residual.lean`; `src/quotient.rs`, `tests/quotient.rs`).

```rust
pub struct Decision<A, X> { menu: Vec<A>, best: Vec<A>, blocks: Vec<Block<A, X>> }
pub struct Block<A, X> { play: A, seen: X, next: usize }
pub enum Interface { Raw, Shape, Fork }
impl Course<A, X> { pub fn new(decisions: Vec<Decision<A, X>>) -> Result<Course<A, X>>;
                    pub fn quotient(&self, interface: Interface) -> Result<Quotient>; }
pub struct Quotient { class: Vec<usize>, states: usize, rounds: Vec<Vec<usize>> }
impl CourseFile { pub fn from_json(source: &str) -> Result<CourseFile>; pub fn report(&self) -> Result<Value>; }  // the `course` feature
```

- **The course.** A decision is its legal menu, its complete maximizing set
  and its blocks: an own play, what was seen until the next own decision, and
  that decision. A block the course lacks is REJECT; a decision with no blocks
  is the final tile, whose play is HALT. `new` refuses an unsorted menu, a
  maximizing set outside it, a play outside it, a repeated label, a dangling
  block and a cycle. Nothing here knows 42: tiles and observations are any
  ordered type.
- **The quotient** is the coarsest right congruence that keeps the output,
  computed twice, bottom-up by signatures and by Moore refinement from the
  output partition; `quotient` refuses if they differ. Round `k` separates
  exactly what block sequences of length at most `k` reveal, so the rounds are
  the block-depth ladder. With at most `n` own tiles it is exact after `n - 1`
  blocks, not one (`ladder_42`; `tests/quotient.rs` has the three-tile witness).
- **Interfaces.** `Raw` outputs the menu and maximizing set as tiles and labels
  blocks by play and observation. `Shape` keeps no tile identity: menu size,
  maximizing slots, and under each slot the multiset of successor classes.
  `Fork` keeps the set instead. Values are never outputs.
- **The file** (`silas-decision-course-v1`, written by `lem/quotient.py
  --export`) lists every decision with its group (the root), split, stage,
  ply, menu, maximizing set, `q` aligned with the menu, mass, settled flag and
  blocks; it carries no hidden hand. `from_json` audits it: the maximizing set
  is the argmax of `q`, the blocks under each play partition the mass, and
  Bellman holds exactly. `report` gives every count of the Python report.
  `silas quotient --course FILE` prints it; on the 256 roots of the Gus
  maskbreadth study it reproduces `lem/quotient.py`'s report in every count
  (`wiki/experiments/comparison-algebra-2026-10-21.md`).

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
silas query    QUERY --hand ID [--trick N] [--seat ROLE] [--cap N] [--book FILE] [--law ROLE]
               [--read ROLE=POLICY,... [--likelihood consistent|uniform]
                [--auction C,C,C,C [--shaker ROLE] [--offered D,...] [--scoring S]]]
silas query    DOMINANCE
silas compare  --ledger FILE [--utility marks|points] [--confidence P/Q]
silas quotient --course FILE
silas coverage POLICY-OR-QUERY [--hand ID] [--random N [--bidders POLICY [--offered D,...] [--scoring S]]
               [--play POLICY]] [--book FILE]
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

With `--read` (the seats read and the policy each acted by), `query` reads
the record as evidence (The record as evidence, above): every read seat's
past plays, under `--likelihood` (`consistent`, the default, or `uniform`):

```text
$ silas query examples/book/reading/bidder-holds-the-trumps-out.silas --hand hand-31 --trick 6 --seat left \
    --read right=examples/book/protect-the-count.silas
hand-31: trick 6, the left (S1) to play
  ...
  read S3: protect-the-count allows its plays 3-2 4-0 5-1 3-3 4-1, from 2 of the 9 hands it could hold
query examples/book/reading/bidder-holds-the-trumps-out.silas
  15 worlds (2 kept), as 1 points resolving 2-0 6-2
  P(answer) = 1 (exact_given(S3 plays within protect-the-count#438c5fb627cfa9a5))
```

Unread, 2/15 (`exact`); under `--likelihood uniform` the worlds weigh 1/60
together and the answer is 1, `exact_given(S3 plays uniformly within ...)`.
With `--auction` too (the four calls from the shaker's left; `--shaker` is
the right by default, so the bidder calls first; it must be won by the bidder
at the hand's bid) each read seat's calls are read as well, under the same
policy, with `--offered` the declarations offered (by default the nine
straight ones, in `Straight42` order) and `--scoring` the match, at 0-0. A
book hand's own auction is a convention, so without `--auction` no call is
read, and before any play there is nothing to read:

```text
$ silas query examples/book/bidding/bidder-holds-two-doubles.silas --hand hand-29 --seat partner \
    --auction 30,pass,pass,pass --read bidder=examples/book/bidding/bid-only-enough.silas \
    --offered P6,P5,P4,P3,P2,P1,P0
hand-29: trick 1, the partner (S2) to play
  ...
  auction: S3 shook: S0 30, S1 pass, S2 pass, S3 pass
  read S0: bid-only-enough calls 30 and declares straight42/P4, and bid-only-enough allows its plays 4-0, from 3034 of the 27132 hands it could hold
query examples/book/bidding/bidder-holds-two-doubles.silas
  46558512 worlds (5206344 kept), as 726 points resolving 0-0 2-2 3-3 4-4 5-5 6-6
  P(answer) = 1197/1517 (exact_given(S0 bids by bid-only-enough#cb7d652ef9593f67 + S0 plays within bid-only-enough#4563ff381ae7b32f))
```

Unread, the same position gives 2949/4522 (`exact`). The bidder's lead is
read too and says nothing: `bid-only-enough` has no play rules, and its
fallback is silence.

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
It stays: without it the rule means the same, but its core is not the one
pinned (`tests/core_pins.rs`).

**The funnel** (`Funnel`, `Coverage`, `why_not`) asks where a pattern stops.
Its steps are the pattern's clauses in source order, a definition call or a
short form one step; step `k` is the query of the first `k` steps, read under
the same semantics. Adding a clause only removes assignments, so the steps
that hold are a prefix, and the first that fails is where the pattern stops.
`eval` prints it for each rule that passed; `coverage` counts, over every
point of the book's narrated lines (and `--random N` seeded deals under the
nine declarations), how often each rule answered, allowed a legal tile and
decided, and how often each step held, flagging every zero with `!`, and in
how many of the random deals each rule answered. A rule that answers but
never decides is shadowed by an earlier one; one that answers more often
than it allows is answering tiles the viewer cannot play.

A rule can be rare because the hands it is for are rare, and which hands
come up depends on who bids. `--bidders POLICY` makes each random deal's
contract the auction a bidding policy makes from it: the shaker drawn, every
seat calling by the policy under `--scoring` (marks at 0-0 by default), the
winner declaring by it from `--offered`, and a deal passed out dealt again.
The book's hands are then left out, since their contracts are not the
policy's. `--play POLICY` is the play model, stated: every seat draws its
tile uniformly from what that policy allows, every legal tile where it is
silent (the set the uniform likelihood reads a play against,
`CompiledPolicy::permits`); without it each seat plays a uniformly random
legal tile. `examples/populations/` holds the bidders and the play model of
issue 141's study, which needed Rust before:

```text
$ silas coverage examples/book/trump-set.silas --random 3000 \
    --bidders examples/populations/optimist-2.silas --offered P6,P5,P4,P3,P2,P1,P0 \
    --play examples/populations/draw-trump.silas
coverage of policy trump-set over 3000 random deals
  contracts: optimist-2 makes every call (the shaker drawn, marks to 7) and declares from P6 P5 P4 P3 P2 P1 P0
    3000 contracts from 19134 deals (16134 passed out); bids 30 x3000
  play: every seat uniformly within what draw-trump allows (every legal tile where it is silent)
  rule trump-a-big-double: decided 14, allowed 14, answered 14 of 84000, in 13 of the 3000 random deals
  ...
```

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
Under a contract from a bidding policy (the command above, 3,000 contracts
each, offered P6 to P0, played within `draw-trump`), `trump-a-double` answers
in 0 deals under `bid-only-enough`, `risk-budget` and `marginal-30`, and in
19 under `optimist-2`; under the three that are not optimists the only ruff
that answers at all is `trump-in-on-count`, once, under the marginal 30. The first sampled study, in Rust
before the command, found the same (3,000 contracts each, the
bidder drawing trump, defenders keeping theirs): a ruff rule answers in 0 games
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
`examples/book/bidding/` its bid rules as policies over the auction (The auction) and a bid read as evidence, and `examples/book/reading/` the plays read as evidence (The record as evidence).
`examples/populations/` is not the book's: bidders and a play model that
make populations of contracts for `coverage` (Diagnostics), each held to
its reading on every hand (`tests/bidding.rs`).
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
