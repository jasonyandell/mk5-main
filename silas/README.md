# Silas

Silas is a typed relational language over the worlds consistent with what one
seat of Texas 42 knows. A seat sees its own hand, the tiles played, and which
suits each other seat has failed to follow; every deal of the unseen tiles that
fits those facts is a *world*. Silas asks questions of those worlds:

- a **query** is a relation: the set of output tuples (dominoes, chairs,
  contexts) for which some pattern of typed atoms has a satisfying assignment;
- a **belief** is an exact finite measure over worlds; a query summarized over
  a belief gives exact probabilities, each carrying its **authority**;
- a **policy** chooses plays from what the viewer alone can see: exact
  information-state rules, relational rules guarded by queries, and a total
  fallback.

Silas owns the nouns of the game (dominoes, seats, teams, contexts) and the
language. It owns no rules: every declaration-dependent or world-dependent
fact comes through the `Physics` trait. `silas::reference::Straight42` is the
reference physics and the test oracle.

```rust
use std::sync::Arc;
use silas::{reference::Straight42, *};

let fives: Arc<dyn Physics> = Arc::new("P5".parse::<Straight42>()?);
let bidder: DominoSet = ["5-5", "5-4", "5-2", "5-0", "4-4", "4-2", "2-2"]
    .iter().map(|d| d.parse()).collect::<Result<_>>()?;
let hidden = [Seat::S1, Seat::S2, Seat::S3].map(|seat| Hidden { seat, capacity: 7, voids: ContextSet::EMPTY });
let kernel = Kernel::new(Seat::S0, bidder, DominoSet::FULL.difference(bidder), hidden)?;
let root = Frame::new(fives, kernel, Seat::S0, vec![], DominoSet::EMPTY)?;

let query = include_str!("examples/six-five-guarded.silas").parse::<Query>()?.compile(&Registry::standard())?;
let belief = Belief::uniform_resolving(root, ["6-5", "5-1", "5-3"].iter().map(|d| d.parse()).collect::<Result<_>>()?, 1000)?;
let p = query.summarize(&belief, &mut Budget::new(10_000_000))?.event_probability();
assert_eq!((p.value().to_string(), p.authority()), ("33/95".to_string(), &Authority::Exact));
```

That is the book's four-fives 84 (Roberson, *Winning 42*, p. 113): the chance
that an opponent holds the six-five together with another five is exactly
33/95 over the 399,072,960 deals the bidder cannot tell apart
(`tests/eighty_four.rs`).

## Queries

```text
query    := (query (roles role ...) (out name ...) pattern ...)
role     := (sort name)
sort     := chair | domino | context
pattern  := (pattern clause ...)
clause   := (same name name ...) | atom | (not atom)
atom     := (predicate argument ...)
argument := name | literal
```

Names begin with an ASCII letter and continue with letters, digits, `_` or
`-`. Literals are dominoes (`6-4`; either end first), chairs `S0`–`S3`,
contexts `q0`–`q6` and `q*` (the called suit), teams `T0` and `T1`, and
nonnegative integers. A literal spelling is never a role name. `;` starts a
comment. Text is bounded to 1 MiB and 32 levels of nesting, and a query to 64
roles. Unknown predicates, wrong argument sorts, duplicate roles or outputs,
and mixed-sort `same` groups are refused at compilation, before any
evaluation.

The semantics:

- **Distinct roles of one sort denote distinct values** within a pattern,
  unless `(same ...)` identifies them; `same` groups close transitively. A
  pattern without `same` and one with it together cover every assignment.
- Roles range over whole sorts: 28 dominoes, four chairs, eight contexts. Add
  `(live d)` when a tile must be live.
- Roles not in `out` are existential: many witnesses give one tuple. Patterns
  combine by union; overlapping or repeated patterns add no answers and no
  mass. `(out)` asks a yes-or-no question. A query with no patterns is false;
  an empty pattern is true.
- `not` negates one atom over the finite world. An atom may be *undefined*
  when its precondition fails (no lead yet, say); then neither it nor its
  negation holds, so negation never manufactures a fact from missing context.

`Query` and `Pattern` are plain data and can be built in Rust as well as
parsed. `Query::compile(&Registry)` checks a query and binds its predicates;
`CompiledQuery::identity()` is its complete text identity, including every
predicate's version.

## Predicates

All built-ins have version `silas-v0/core` and horizon zero. *World* access
means the predicate reads a hidden world; *Viewer* means it reads only the
frame.

| Predicate | Access | Meaning |
|---|---|---|
| `live(d)`, `played(d)` | Viewer | In the viewer's hand or the hidden pool; explicitly supplied as played. Not live does not imply played. |
| `holds(c,d)` | World | Seat `c` holds `d`. |
| `void(c,q)` | World | Seat `c` holds no tile that follows `q`. |
| `in(d,q)` | Viewer | `d` follows `q` under the physics. |
| `leads-context(d,q)` | Viewer | `d`, led, leads `q`. |
| `double(d)` | Viewer | `d` is a double. |
| `beats(a,b,q)` | Viewer | `a`'s trick key in `q` exceeds `b`'s. |
| `boss(d,q)` | Viewer | `d` is live, follows `q`, and no live tile beats it in `q`. |
| `master(d)` | Viewer | `d` is live and no live tile beats it when it is led. |
| `count(d,n)` | Viewer | `d` carries `n` count points. |
| `quota(c,n)` | Viewer | Seat `c` holds `n` tiles. |
| `tile(a,b)`, `chair(a,b)`, `context(a,b)` | Viewer | Equality; anchors a role to a literal. |
| `team(c,t)`, `partner(a,b)`, `opponent(a,b)`, `successor(a,b)` | Viewer | Partnerships and clockwise order. |
| `viewer(c)`, `leader(c)`, `next-actor(c)` | Viewer | The viewer, the trick's leader, the seat to play (none when it holds nothing). |
| `led-context(q)`, `current-winner(c)` | Viewer | The current trick; undefined before a lead. |
| `trick-play(c,d)` | Viewer | Seat `c` played `d` to the current trick. |
| `own-legal(d)` | Viewer | `d` is a legal play for the viewer; undefined unless the viewer is next. |
| `legal(c,d)`, `forced(c,d)` | World | `d` is a legal (the only legal) play for `c`; undefined unless `c` is next. |

A new relation implements `Predicate` and is registered once under an
unreserved name with a `PredicateSpec`: semantic version, argument sorts,
horizon in plies, and access. It returns `Some(true)`, `Some(false)` or `None`
(undefined). The runtime checks argument sorts and withholds the world from
`Viewer` predicates. A `World` predicate reads the world only through
`Holdings`. It must be deterministic, keep to its declared horizon, and charge
its inner work to the `Budget`.

## Beliefs and summaries

- `Belief::uniform(frame, cap)`: every world of the frame, weight one. The
  support is counted first and refused above the cap.
- `Belief::uniform_resolving(frame, tiles, cap)`: the same measure, resolved
  only on the given pool tiles. Each point stands for every world that agrees
  with it on those tiles, weighted by their exact number. `Holdings` refuses
  any read that could tell those worlds apart, so an answer is either exact or
  refused. Never approximate. This is how the 84 hand's 399 million worlds
  become 27 points.
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
named context) to a frame, a world and a belief. A hidden play removes the
tile from the pool, shrinks its seat, and records a void on a slough. The
fourth play closes the trick. `compare_answers` sets transported answers
against fresh ones, and `anchor_back` takes a preimage for hindsight analysis.

Every computation draws on an explicit `Budget`. Exhausting it, like every
other refusal, returns an error and no partial result.

## Policies

```text
policy   := (policy name (initial mode) (fallback lowest-legal) form ...)
form     := (bind name query)
          | (rule name (in mode) (next mode) (select selector) (guard query))
          | (exact key (play domino))
selector := answer | tile domino | rigid name
key      := (key (physics id) (viewer chair) (hand-bits n) (pool-bits n)
                 (hidden (chair capacity (voids context ...)) ...)
                 (leader chair) (prefix domino ...) (played-bits n)
                 (history (chair domino) ...) (banked n n) (bid n)
                 (declaring-team team))
```

A decision tries, in order: the exact table, keyed by the complete public
information state under one physics; the relational rules of the current
mode, in order; and the lowest legal tile. A rule's guard is a query that may
use only `Viewer` predicates; `answer` takes the least legal domino it
returns, `tile` and `rigid` act when it holds. Bindings are evaluated once,
when a controller starts. Every action is checked for legality, and
`choose_traced` reports which layer chose. A refusal leaves the controller as
it was.

## The Physics boundary

```rust
pub trait Physics: Debug + Send + Sync {
    fn identity(&self) -> String;
    fn led_context(&self, tile: Domino) -> Context;
    fn incidence(&self, context: Context) -> DominoSet;
    fn trick_key(&self, tile: Domino, led: Context) -> u8;
    fn count(&self, tile: Domino) -> u32;
    fn count_worlds(&self, kernel: &Kernel) -> u128;
    fn worlds<'a>(&'a self, kernel: &'a Kernel) -> Box<dyn Iterator<Item = World> + 'a>;

    // Provided, derived from the above:
    fn follows(&self, tile: Domino, context: Context) -> bool;
    fn beats(&self, led: Context, tile: Domino) -> DominoSet;
    fn threat(&self, tile: Domino) -> DominoSet;
    fn legal_plays(&self, hand: DominoSet, lead: Option<Domino>) -> DominoSet;
    fn trick_winner(&self, leader: Seat, plays: [Domino; 4]) -> Seat;
    fn trick_points(&self, plays: [Domino; 4]) -> u32;
}
```

The required methods are one declaration's four rule tables (the led context
of each tile, the tiles following each context, the trick key, and the count)
and the fiber of hidden worlds of a `Kernel`. A `Kernel` is pure information:
the viewer, its hand, the hidden pool, and each hidden seat's capacity and
shown voids. The physics says what a void forbids. The provided methods derive
everything else; an implementation may override one only with an equal,
faster computation. `legal_plays` takes the tile that led the trick, not its
context: every trick is led by a tile, and a rules engine that keys its
legality by the led tile (Zeb's does) can override it with its own code. A
`Frame` carries its physics as `Arc<dyn Physics>`, so
queries, beliefs and policies are not generic. The rules sit behind one
pointer per frame, and a `Predicate` stays an object any extension can
implement.

`Straight42` covers the nine straight declarations: a pip suit trump (`P0` to
`P6`), doubles trump (`DT`) and no trump (`NT`). Doubles as their own unpowered
suit (nello) is not a straight declaration and is omitted. Its module
documentation states the rules it implements.

## Numbers

Weights are exact rationals (`Weight`, a nonnegative `BigRational`). Silas
performs no floating-point arithmetic, and the crate denies it in lint. A
float enters only through `Weight::from_f32_exact` or `Weight::from_f64_exact`,
which return the dyadic rational the float denotes, bit for bit, subnormals
included. They refuse NaN, the infinities and negative values. `-0.0` denotes
zero.

Every belief, summary, selection law and `Estimate` carries an `Authority`:

- `Exact`: uniform or combinatorial support, with no model;
- `ExactGiven { model }`: exact arithmetic on weights from a pinned model,
  whose string names its content hash, rung and seed;
- `Sampled { n, seed }`: worlds drawn, not enumerated.

Authority propagates through conditioning and stepping and never upgrades.
`Authority::meet` gives the weakest of two labels, merges two models into one
label, and refuses to merge two different samples. Only the uniform
constructors produce `Exact`; caller-supplied weights must name their source.
`compare(a, b, band)` returns `Better`, `Worse` or `Indistinguishable` against
an explicit, nonnegative band, which must be zero when both values are exact.

## Examples

`examples/` holds the queries the tests run: `partner-count.silas`,
`partner-has-count.silas`, `partner-master.silas`, `current-forced-play.silas`
and `six-five-guarded.silas`.

## Provenance

Silas moved here from texas-42 at commit `1c15cc51` (CC0), where it was the
`walt::scheme` module, called Scheme/Fix. The move renamed it: `Fix` became
`Query`, a `Scheme` (one case) became `Pattern`, `CompiledFix` became
`CompiledQuery`, and the surface forms `(fix …)` and `(case …)` became
`(query …)` and `(pattern …)`. The rules it imported from texas-42's engine
were replaced by the `Physics` trait and the `Straight42` reference.
