# Silas through a Lean lens — peer review

Reviewed 2026-10-13 against `silas/` at its 2026-10-13 00:44 state (`README.md`,
`GAPS.md`, `src/*.rs`, `tests/*.rs`, `examples/`), texas-42's kernel-checked
`lean/Texas42/{Basic,Trick,Deal,Play,Cells,Belief,Information,Strategic}.lean`
(trust rules: no `sorry`, no `native_decide`, TRUST-01), and the Walt mathematical
basis (`texas-42/walt/math/`, the §8 Bayes material that `Belief.lean` formalizes).
Read-only; nothing was run. Line numbers are from the files as read.

The one-paragraph verdict. The query core (eval.rs) is sound: the three-valued
semantics is strong Kleene with "undefined holds in no pattern" at the top, `none`
is Kleene ¬∃, the evaluation order cannot change a value, and `uniform_resolving`
weights exactly and refuses conservatively. What is not yet Lean-shaped sits in the
surrounding structure: a policy is not a function of the information state (the
controller carries mode and binding state across calls); the `Exact` authority
stamp rests on two trait methods (`count_worlds`, `worlds`) and a string identity
that nothing checks; and the frame has two constructors with two meanings for the
same predicates, so "one source of truth" does not hold at the type level. All
three are cheap to fix now and expensive to specify around later.

---

## 1. Soundness findings in Silas as built

Severity scale: **high** = a Lean statement of the intended semantics would be false
or unformulable; **medium** = the implementation is right but the specification it
would have to satisfy is awkward or duplicated; **low** = a local inconsistency.

### F1. A policy's decision is not a function of its information state (high)

`src/policy.rs:105-109` (`PolicyController { mode, rigid }`), `:236-249`
(`initialize` evaluates bindings at whatever frame it is handed), `:302-327` (the
mode advances on every rule that fires).

The README (line 207-213) says a decision is tried "in order … the relational rules
of the current mode"; the current mode is a property of the *controller*, i.e. of
the sequence of `choose` calls, not of the frame.

Counterexample. A policy with `(initial a)` and two rules, `r1 (in a) (next b)
(select tile 6-6)` and `r2 (in b) (next b) (select tile 0-0)`, both guarded by
`(query (roles) (out) (pattern))`. Call `choose` twice on the *same* frame (the
viewer holding both tiles, on lead): the first call returns 6-6 and moves the mode
to `b`; the second returns 0-0. Same `PolicyKey`, two actions. Likewise a controller
initialized at trick 3 is in mode `a` while one carried from trick 1 may be in `b`;
and `rigid` bindings are read at the initialization frame, so "the top trump at the
start" differs by when the controller was started.

Why this matters for Lean. The Lean object is `PolicyKernel : ViewerCtx → PubState →
Domino → ℚ` (`Belief.lean:156`) — a function of the viewer's private context and the
public record. GAPS.md gap 1 (policy likelihood) needs exactly that: "at each of the
sender's past decisions ask whether his policy allows the tile he played". With
hidden controller state the likelihood of a play is not determined by (policy,
world, record); you would have to specify a replay convention for the mode, and that
convention is itself a semantic choice nobody has written down (does the mode advance
on the rule that *would* have fired, even when the sender played something else?).

Fix (cheap now). Make `CompiledPolicy::decide(frame) -> Domino` a pure function.
Either drop modes — all 28 rules in `examples/book/` are `(in play) (next play)` and
no book file uses `bind`, so the encodings do not change — or define the mode as a
derived view: fold the policy over the viewer's own past plays in the record from
the hand's initial frame (`Frame::from_record(.., plays=[])`, reconstructible since
`hand0 = hand ∪ own plays`), advancing on the rule whose selection equals the
recorded play, and evaluate `bind` at that initial frame. The record now carries
everything modes were a workaround for (`led`, `played-to`, `trick-of`,
`trick-number`), so dropping them is the cleaner choice. `PolicyController` then
becomes, at most, a memo that is checked against the replay.

### F2. `Exact` is stamped on numbers whose correctness rests on an unverified trait method, and physics equality is a string (medium)

`src/physics.rs:38-41` (`count_worlds`, `worlds` are required trait methods),
`src/belief.rs:116` (`uniform_resolving` weights are `count_worlds(&rest)` and the
belief is `Authority::Exact`), `src/belief.rs:81-86` (`uniform` likewise),
`src/frame.rs:229-235` (`Frame: PartialEq` compares `physics.identity()`),
`src/physics.rs:17-18` ("Two physics with the same identity must be the same rules"
— a promise), `src/policy.rs:219` (`PolicyKey.physics` is that string).

The fiber of hidden worlds is not a rule table; it is a *derived* object of
(incidence, voids, capacities, pool). Lean defines it that way (`Cells.lean:482`,
`IsWorld v P A := … A s ⊆ allowed P s ∧ (A s).card = capacity P s ∧ disjoint ∧
⋃ A = pool`). Silas instead asks each `Physics` to supply both the count and the
enumeration, and then labels the result "no model contributed a weight". If a
physics returns a count that disagrees with its own `incidence` (an off-by-one in a
void, a cached table), `uniform_resolving` produces a wrong rational labelled
`Exact`, and `Frame::eq` will say two frames with that physics are equal to frames
with the reference physics because the identity strings match. The README's "may
override one only with an equal, faster computation" (line 261) is unverifiable from
inside Silas; today it is held only by `scenario/core/tests/silas_physics.rs`.

The question asked ("can anything reach `Exact` from a non-exact input?"): through
the measure algebra, no — `from_weights` refuses `Exact` (`belief.rs:40`),
`condition` preserves, `condition_likelihood` and `step_belief_given` meet with
`ExactGiven`, `Estimate::new` is crate-private, `meet` never upgrades. The only path
is a physics whose fiber is wrong, and that path is open by design.

Fix (cheap now, while only `Straight42` and `ZebPhysics` exist). Make the physics a
value: four tables (`led_context: [Context; 28]`, `incidence: [DominoSet; 8]`,
`key: [[u8; 8]; 28]`, `count: [u32; 28]`), built once from any rules engine;
`identity` becomes a content hash of the tables, so identity equality *is* table
equality. `count_worlds`, `worlds`, `legal_plays`, `trick_winner`, `trick_points`,
`beats`, `threat` become Silas functions of the tables (the `Fiber` DP in
`reference.rs:168-238` is already that). Zeb's engine then becomes an implementation
under differential test, which is the stated end state ("Rust and Python held to
Lean"), rather than a trusted override. If the override is kept for speed, require
the table form anyway and keep the override as an internal accelerator checked
against it.

### F3. Two frame constructors, two meanings for the same predicates (medium)

`src/frame.rs:242-260` (`Frame::new`: caller supplies `leader`, `prefix`, `played`,
and a `Kernel` with caller-supplied voids and capacities), `:271-331`
(`from_record`: everything derived), `:341-364` (`after` rebuilds from the record
and then *checks* the caller-computed kernel against it — the signature of dual
authority), `src/frame.rs:156-160` (`Record` caches `banked`, derivable from
`plays`), `src/frame.rs:189-200` vs `:303-309` (shown voids derived twice, by two
loops; agreement is tested at `tests/record.rs:245`, enforced nowhere).

Consequences. `(played d)` means "FULL \ live" on a record frame and "whatever the
caller said" on a `Frame::new` frame; `tests/query.rs:416-417` asserts a tile can be
neither live nor played. `(shown-void viewer q)` is undefined on one and defined on
the other; the hidden seats' `shown-void` is defined on both but from different
sources. `Frame::new` accepts unreachable residues (leader `S1` with a two-tile
prefix and untouched capacities). The README (line 226) says "the record is the one
source of it"; the type says there are two.

This is exactly texas-42's TYPE-02 rule ("derived views, never stored state … storing
both authorities is forbidden") and Lean's shape: one `(ViewerCtx, PubState)`
(`Cells.lean:165, 455`) with `hand`, `pool`, `allowed`, `capacity` as functions.

Fix. `Frame = { physics, viewer, hand0, contract, plays }` (plus the auction when
gap 4 lands). `kernel()`, `leader()`, `prefix()`, `played()`, `banked()`,
`shown_voids(seat)` are methods that replay. Delete `Frame::new`; the tests that
build synthetic residues (`tests/query.rs:231-252, 337-352`, `tests/common`) can
build records, and the belief-only uses that need just a `Kernel` already have
`Kernel`. `Frame::after` then needs no cross-check. Cost: medium; it touches every
test fixture once.

### F4. `trick-number` is false, not undefined, after the seventh trick (low)

`src/registry.rs:238`: `record.trick_number() == Some(number(n))` yields
`Some(false)` for every `n` when `trick_number()` is `None`; README line 139 says
"undefined once all seven have closed". Counterexample: a 28-play record;
`(not (trick-number 7))` holds, by the README it should not. One-line fix: return
`None` when the record is complete. Decide and document `next-actor` the same way
(it returns false-everywhere when no seat is to play; the README's "none when it
holds nothing" is consistent with that, but say "false", not "none").

### F5. Distinctness is implicit between roles and explicit between a role and a literal; the completeness claim is only true for two roles (medium, for the Lean statement)

`src/eval.rs:574` (any two outer classes of one sort are distinct), `:418` (a
`none`'s locals distinct from one another, not from the outer roles),
`README.md:69` ("A pattern without `same` and one with it together cover every
assignment").

Two observations. (a) For `n` roles of one sort the patterns needed to cover every
assignment are the set partitions of `n` — Bell(n): 5 for three roles, 15 for four.
The README's sentence is true for two. (b) The six-five example
(`examples/six-five-guarded.silas`) must write `(not (tile other 6-5))` because
`other` is a role and `6-5` a literal, while `(holds foe a) (holds foe b)` gets
`a ≠ b` for free. Existential roles are distinct from output roles too, so
`(query (roles (domino d) (domino e)) (out d) (pattern (holds S1 d) (holds S1 e)))`
answers `d` only when S1 holds some *other* tile — an inequality the text does not
show. Not a bug; a semantics that Lean will have to state as "there exists an
*injective* assignment of the pattern's equality classes" and that readers will
trip on.

Options. Keep it and state it (the Lean denotation is `∃ f : Classes ↪ Value`,
sort-respecting), documenting the Bell-number completeness; or move to standard
relational semantics (any assignment) with explicit `(not (tile a b))`, which is
the cleaner Lean object. The second costs a pass over 20 book files. I lean to the
first for now and the second before Lean: the implicit inequality is the kind of
thing that makes a formal spec read differently from the program.

### F6. `Authority::meet` forgets provenance; `compare` bands the wrong label (low)

`src/measure.rs:182`: `Sampled ⊓ ExactGiven{m} = Sampled` — the model name `m` is
dropped, so a number that depends on a pinned model and a sample no longer names the
model. `:183-184`: two different samples refuse to combine, so `meet` is partial.
`:227-229`: a nonzero band is refused only when *both* sides are `Exact`; an
`ExactGiven` number is exact arithmetic under an assumption, and a band on it is a
number the system does not have (model uncertainty), while a `Sampled` number's band
should follow from `n` and a stated confidence, not be chosen by the caller.

Fix. Authority as a provenance *product*: `{ given: BTreeSet<ModelId>, sample:
Option<(n, seed)> }` with `Exact = {∅, None}`; `meet` is componentwise union (two
samples: a set of samples, or refuse — but say which). `compare` accepts a band only
when a side is sampled, and returns the meet of the two authorities with the verdict.
In Lean this is a lattice and "never upgrades" is a one-line monotonicity lemma.

### F7. `uniform_resolving`: the weighting is exactly right; the refusal is sound and conservative (no defect; one spec note)

`src/belief.rs:94-129` and `src/frame.rs:446-460`.

Weighting. For a placement of the resolved tiles `T`, the point's weight is
`count_worlds(rest)` where `rest` has pool `P \ T`, capacities `c_j − |placed_j|`,
and the same voids; `allowed(rest)` is `(P \ T) \ incidence(voids)`, so the count is
exactly the number of full worlds agreeing with the placement. `place` respects
`allowed` and room; placements with no completion are dropped; `Kernel::new` inside
`point` cannot fail (capacity sums match by construction). Correct.

Refusal. `hand_within(seat, tiles)` refuses iff some tile of `tiles` is unresolved
and allowed for a hidden `seat` of nonzero capacity. Sound: an unresolved tile not
allowed for the seat is held by it in no world, so the representative's answer is
every world's answer. Conservative in two ways: it uses the seat's *total* capacity
rather than the capacity left after the resolved placement (a seat whose resolved
tiles fill it holds no unresolved tile, yet is refused), and a tile forced to one
seat by the others' voids is still refused. Both are refusals, never wrong answers.
`rebuild` keeps `unresolved` through `condition`/`step`; `step` reads legality
through the belief's `Holdings` before touching representatives; `Difference.world`
and `Belief::worlds()` expose representatives, which is documented. Airtight within
the crate; a caller holding a `World` from `worlds()` can of course read it.

Spec note for Lean. State the admissibility predicate abstractly — a read
`(seat, tiles)` is admitted iff `A ↦ A seat ∩ tiles` is constant on the class — and
prove the implemented predicate implies it. Then "a summary over
`uniform_resolving T` equals the summary over `uniform` whenever no read is refused"
is a corollary, and the conservative slack can be tightened later without touching
the theorem.

### F8. Three-valued core: sound, and simpler than the README makes it sound (no defect)

`src/eval.rs:288-304` (atom → `Option<bool>`, negation flips a defined value),
`:342-350` (a clause holds iff `Some(true)`), `:358-428` (`none`).

The semantics is strong Kleene. Within a `none`, a binding is refuted iff some
clause is `Some(false)`, satisfied iff all are `Some(true)`, else unsettled; `none`
is `Some(true)` iff all refuted, `Some(false)` iff some satisfied, else `None`. That
is Kleene ¬∃. At the top, a pattern holds iff every clause is `Some(true)`. Two
consequences worth writing down as lemmas: `(not a)` ≡ `(none (roles) a)`; and the
early exits in `witness` (`:406`, `:410-413`) cannot change a value, because an
atom's value at a partial binding equals its value at every extension — only the
budget charge depends on order. One thing the summary layer then cannot express:
`event_mass = 0` conflates "false in every world" with "undefined in every world"
(`tests/query.rs:255-264` relies on the conflation). Acceptable, but a `Certainty`
field `defined_everywhere` would let the Lean statement be two-valued at the top
without losing the distinction.

### F9. Policy measurability: holds, with one hole that is a trust boundary and one that is F1 (medium)

`src/policy.rs:208-215` (`viewer_only` refuses any `World` spec in guards and
bindings, including inside `none`, since `Compiler.used` is shared across nested
bodies), `src/eval.rs:298-303` (a `Viewer` predicate under a world reader receives
only the frame; a `World` predicate under a viewer reader is an error),
`src/registry.rs:54-57` ("it cannot audit what a closure captures").

Paths checked: guard and binding queries (viewer-only by spec); the exact table (a
`PolicyKey` of public data, validated at compile under the physics); the legal set
(the viewer's hand); `contract_resolved` (public); the physics object (a declaration
— but see F2: a physics could carry hidden state, and the string identity would not
show it). No path reads a `World`. The remaining hole is a custom predicate that
declares `Viewer` and captures a world in its closure; Rust cannot forbid it. For
Lean, the standard registry is the specification and custom predicates are outside
it; say so in the README. The other hole is F1: measurable in (information state,
controller history), not in information state.

### F10. `Contract` flattens marks to 42 and loses the stake; the auction is absent (low now, blocks gaps 2 and 4)

`src/frame.rs:141-148` ("A marks bid (84 and up) is the whole 42 here"),
`:205-211` (`to_set = 43 − bid − banked`, defined for `bid < 30` too, giving an
unreachable 43). Lean's `Contract` carries `kind : point | mark`, `amount`, and the
`declaration` (`Play.lean:41-46`), with `threshold`/`stake`/`award` derived, and
`Information.lean:47` proves the mechanical projection that drops the losing bids
is not injective. Silas's record is that projection. See §3.4.

### F11. Smaller notes

- `src/policy.rs:308-314`: `(select answer)` silently drops illegal answers before
  taking the minimum. Document it as the definition (`min (answers ∩ legal)`), or
  refuse a guard that returns an illegal tile — a guard that forgot `own-legal` is a
  bug, not a preference.
- `src/dynamics.rs:141-149`: `step_belief` keeps `Exact` under the likelihood-one
  model. This is justified for the uniform prior (legality-conditioning pushed
  through the play is a bijection onto the successor fiber — Lean's
  `remainder_injective`), and it is a theorem to state, not a comment.
- `src/frame.rs:156-158`: the record attributes each play to a seat and then
  validates the attribution (`:291`). Lean's record is `List Domino` with actors
  derived (`PubState.actor`). Storing the seat is harmless redundancy, but the
  canonical form for hashing and for `PolicyKey.history` should be the tile list.
- `src/registry.rs:222-228`: `shown-void` for a hidden seat reads the kernel, for
  the viewer the record. Under F3's fix both read the record.
- `README.md:69` and `:139`: the two sentences F4 and F5 correct.

---

## 2. The Lean shape of Silas

### 2.1 What is already there

| Silas | Lean (`Texas42`) | Status |
|---|---|---|
| `Pip`, `Domino{hi,lo}`, index | `Pip := Fin 7`; `Domino {high, low, le}`; `card_domino` | there |
| `Seat`, `Team`, `plus`, `partner` | `Seat := Fin 4`, `Seat.next`, `Seat.team`, `Seat.after` | there |
| `Context::{Natural p, Called}` | `Suit.{natural p, called}` | there |
| `Physics` four tables | `Declaration`; `ledSuit`, `effMem`/`effSuit`, `key` (Lex (tier, rank)), `countPoints` | there; Silas's `tier*16+rank` is the same order (`eq_of_key_eq` is the injectivity Silas's `beats` assumes) |
| `beats`, `threat`, `boss`, `master` | `Beats`, `beatsSet`, `threat`; `boss`/`master` are `threat ∩ live = ∅` | there |
| `trick_winner` (first of greatest keys) | `PlayState.trickWinner` (strict `<` keeps the earlier play; unique by PA-A11) | there |
| `legal_plays(hand, lead)` | `legalSet` with `legalSet_lead/follow/slough` | there |
| `Contract{declarer, bid}` | `Contract{bidder, kind, amount, declaration}`, `threshold`, `stake`, `award` | Lean is richer (F10) |
| `Record`, `from_record` | `PubState.replay K ds`: leader, trick, played-by-seat, voids, score | there; Silas must derive, not store (F3) |
| `Kernel{viewer, hand, pool, hidden{capacity, voids}}` | `ViewerCtx{viewer, hand0}` + `hand`, `pool`, `allowed`, `capacity` of a `PubState` | there, as functions |
| `World`, `Kernel::contains` | `IsWorld v P A` (viewer coordinate normalized to `∅`) | there; `World` ≅ `{A // IsWorld}` with the viewer's hand filled in |
| `count_worlds`, `worlds` | the fiber `Φ`; `losslessness` says it equals the image of rule-compatible deals; 399,072,960 and the 90-world witness are kernel-checked | there; Silas should *define* it this way (F2) |
| `Belief` (finite measure) | `FinPMF (Seat → Finset Domino)` after normalizing by `total_mass` | there |
| `condition`, `condition_likelihood` | `FinPMF.condition p L`; `condition_mul` is the chain rule | there |
| `step_belief`, `step_belief_given` | condition on legality, then `FinPMF.map` through the play; `physicalBelief = posterior.map remainder` | there in the deal-domain form |
| `step_frame` / `ObservedPlay.class` | `PubState.step`; `voidsAfter_lead/follow/slough` | there; `PlayClass` is derived, Silas validates it (`dynamics.rs:39-50`) |
| policy | `PolicyKernel : ViewerCtx → PubState → Domino → ℚ`; `likelihoodFrom`, `bayesWeight` | there; a Silas policy is the Dirac kernel — once F1 is fixed |
| `Strategic` | `BeliefProc`, `beliefVal_eq_exp_latentVal`, `bestResponse_eq` | there; the home of gap 2 |

### 2.2 What must be added (a `Texas42.Silas` namespace)

1. **Syntax and denotation.** `Sort`, `Value` (a sum), `Term`, `Atom`, `Pattern`,
   `NoneClause`, `Query` as inductive data; `Three := Option Bool`; an atom's
   denotation `⟦p⟧ : Frame → World → List Value → Three` given by one table per
   built-in (the "Predicates" table of the README becomes definitions with
   docstrings, which is PROOFS.md rule 4: definitions are the trust surface);
   Kleene `∧` over clauses; `none` as Kleene ¬∃ over injective local bindings;
   `⟦q⟧ : Frame → World → Finset Answer` as the projection of satisfying injective
   class assignments, union over patterns.
2. **Frame.** `Frame := (Declaration, ViewerCtx, Contract, List Domino)` and nothing
   else; every Silas accessor a definition over `PubState.replay`.
3. **Belief.** `Belief := FinPMF (fiber)`; `summarize` as expectations of indicators;
   `uniform`, `uniform_resolving T := (uniform Φ).map (A ↦ s ↦ A s ∩ T)`; the
   admissibility predicate of F7.
4. **Authority** as the provenance lattice of F6, attached to `Estimate`.
5. **Policy** as `Frame → Domino` (deterministic) and as the Dirac `PolicyKernel`;
   the layered decision (exact table, ordered rules, fallback) as a definition.
6. Later: `knows` (§3.3), `card` (§3.5), the auction record (§3.4), the match
   utility (§3.2).

### 2.3 The core theorems

- **Evaluator soundness.** `CompiledQuery.evaluate f w = ⟦q⟧ f w` for every compiled
  `q`, frame `f`, world `w ∈ Φ(f)`, and sufficient budget. The proof structure:
  atoms are order-independent (their value at a partial binding equals the value at
  every extension), so pruning (`unary`, `constant`, the `answers.contains`
  short-circuit at `eval.rs:551-553`) is sound. This is the theorem that makes the
  Rust evaluator an implementation rather than the definition.
- **Measurability as typing.** If every predicate of `q` has access `Viewer`, then
  `⟦q⟧ f w = ⟦q⟧ f w'` for all `w, w' ∈ Φ(f)`; hence a policy built from such
  guards is a function of `(ViewerCtx, PubState)`. Needs F1.
- **Summaries are expectations.** `event_probability = β.exp (fun w => if ⟦q⟧ f w ≠ ∅
  then 1 else 0)`; presence and set masses likewise; `select` is a kernel composed
  with the summary.
- **Conditioning is Bayes.** `condition q = FinPMF.condition β (indicator ∘ ⟦q⟧)`;
  `condition_likelihood` and `step_belief_given` are `FinPMF.condition` with the
  given `L`; the chain rule is `condition_mul`. Evidence probability is the
  normalizer `Z`.
- **Stepping commutes with the deal-domain posterior.**
  `step_belief_given (physicalBelief ds) d L = physicalBelief (ds ++ [d])` when `L`
  is the policy kernel of the actor — the square between Silas's remainder-domain
  stepping and Lean's deal-domain `posterior.map remainder`, proved from
  `likelihoodFrom_append` and `remainder_injective`. Its special case: the uniform
  fiber measure conditioned on legality and pushed through a play is the uniform
  measure on the successor fiber, which is why `step_belief` may keep `Exact`.
- **`uniform_resolving` equals `uniform` on admitted reads.** For every query whose
  reads are all admitted under `T`, `summarize q (uniform_resolving T) = summarize q
  uniform`. Follows from the admissibility predicate (F7) and evaluator soundness.
- **Authority never upgrades.** Monotonicity of every operation in the provenance
  lattice.
- **Record derivations agree with `PubState`.** Silas's `from_record` (voids,
  capacities, pool, banked, leader, prefix) equals `PubState.replay` and
  `ViewerCtx.{allowed, capacity, pool}`. With F3 this is by definition; today it is
  a theorem about two loops.
- **Three-valued coherence.** `(not a) = none (roles) a`; `none` is Kleene ¬∃;
  top-level `U ↦ F`.

### 2.4 Rust choices that would make the Lean statements awkward — change now

Collected from §1: F1 (policy state), F2 (fiber as a trait method; string identity),
F3 (two frames; cached `banked`; voids derived twice), F5 (implicit distinctness —
at least document the injective denotation), F6 (authority not a lattice), F10
(`Contract` without `kind`/`stake`/declaration; no auction). Also: record plays as
`Vec<Domino>` canonically (actors derived), and `Number`/`Team` sorts prepared to be
bound by functional atoms (`card`, `count`, `bid`) rather than enumerated — the
`Sort::values` design (`syntax.rs:25-32`) already returns an empty domain for them,
which is the right shape for "bound only by a functional clause".

---

## 3. The open gaps through a Lean lens

### 3.1 Policy likelihood (gap 1)

Object. The history likelihood of Lean's `Belief.lean:161`:
`L_π(ω) = ∏_{t : actor_t = c} π_c(a_t | ω.hands c at t, record_{<t})`, and the
posterior `β.condition L_π` — `bayesWeight` exactly. Nothing new is needed in Lean;
what is needed in Rust is a policy that *is* a kernel of `(hand, record)` (F1) and
`Belief::condition_on_policy(seat, &CompiledPolicy)` that, for each point, builds
the seat's frame at each of its past decisions (`Frame::from_record` with that seat
as viewer and its hand from the world — refused on a resolving belief unless the
seat's hand is resolved) and multiplies the kernel values.

Two decisions to make now. (a) A *convention* is set-valued ("the sender would have
played one of these"); the natural kernel is uniform on the allowed set, and the
posterior weights then depend on that within-set choice, not only on its support.
Say which kernel a convention denotes; the Dirac kernel of a deterministic policy is
the special case. The book's readings (C1, C2, C3, C8, C21) are gates, so Dirac
suffices for them; `(out d)` guards already return sets, so the set-valued reading
is one definition away. (b) Authority after conditioning on a policy is
`ExactGiven{policy identity}` — the table's convention is a model of the table, not
a rule of the game. `CompiledPolicy::identity()` is already the right string.

### 3.2 Cross-game policy comparison (gap 2)

The objects, from the inside out.

- **Hand outcome.** `award K P_D : Team × ℕ` (`Play.lean:71`) — the mark transfer
  of one contracted hand. Silas needs `Contract.kind`/`stake` to compute it (F10).
- **Match state and utility.** `m : Team → ℕ` (marks), the game at seven;
  `V : (Team → ℕ) → ℚ` the probability of winning the match from `m` — itself the
  value of a stochastic game, but for the book's claims a fixed `V` (even
  `V m = [m T ≥ 7]` at the last mark) suffices.
- **A hand policy** `σ_s : Info_s → Action` over the whole hand: bids, the
  declaration, and plays — one function, since the auction is information for
  play (§3.4).
- **A field** `F = (σ_s)_{s ≠ me}`, possibly stochastic (a FinPMF over pure fields).
- **The comparison.** For my policies `σ₁, σ₂`, a deal belief `D` (uniform over
  deals — 472,518,347,558,400 of them — so `Sampled`, or a pinned sample
  `ExactGiven`), a field `F`, and a match state `m`:
  `Δ(σ₁, σ₂; F, D, m) = E_{ω ~ D}[ V(m + award(σ₁, F, ω)) − V(m + award(σ₂, F, ω)) ]`,
  returned as an `Estimate` with the meet of the authorities. This is `beliefVal`
  with the segment reward zero and the terminal utility `V ∘ (m + award)`; the
  theorem `beliefVal_eq_exp_latentVal` already says the value is a function of
  `(record, belief)`, so the Rust comparison can be organized as an expectation over
  worlds of per-world rollouts, which is what the arena does.

The four claims, quantified.

- **"Bid only enough; never 32 or 33" (pp. 21-22, 118-119).** Two policies that
  differ only at the auction: `σ_min` bids the least winning amount, `σ_+k` bids
  `k` more. The claim is `Δ(σ_min, σ_+k; F, D, m) ≥ 0` for the book's field. Its
  content is a statement about the *field*: against opponents who bid what their
  hands are worth, the extra two points buy no auction the smaller bid would have
  lost (`P_F(outbid at 31 ∧ ¬outbid at 33) ≈ 0`) and cost `P(make 31) − P(make
  33)`. So the quantified statement has the field as a premise; without a field it
  is not a theorem, it is a definition of what "bid only enough" means. The Lean
  shape: a dominance *given* a bidding-field property, and the property stated as
  a predicate on `F`.
- **"One mark from the game, bid 42, not 84" (p. 64).** This one is field-free.
  Both contracts need all 42 points (`threshold 42 = threshold (mark 2)`); the
  stake differs. At `m` with the bidder's team one mark short, `V(m + award(42, P_D))
  ≥ V(m + award(84, P_D))` for every `P_D`: equal when made (both win), and
  `−1 ≥ −2` marks when set. A pointwise dominance over deals, hence over every
  belief and field — a two-line Lean theorem about `award` and a monotone `V`. Make
  it the first cross-game claim Lean checks.
- **"With a double protecting the off, an 84 bid is always right" (p. 64).** Not
  field-free: it says `E[marks | 84] > E[marks | 42]`, i.e. `2(2p − 1) > (2p − 1)`
  where `p = P(make | hand, F)` — true iff `p > 1/2` (and the match utility is
  linear in marks, which it is not near seven). So the quantified claim is
  `P_F(make all 42 | hand has a double-protected off) > 1/2` — a belief statement
  at the auction (`uniform` over the 399,072,960 deals, which `uniform_resolving`
  can hold) under a play model `F` (§3.6). Authority: `ExactGiven{F}` or
  `Sampled`.
- **"Keep the setters honest with an occasional straight-off 84" (p. 65).** The
  opponents adapt, so the field is a *function of my policy*: `F = BR(σ)` (the
  setters' best response to the frequency with which I bid 84 on such hands). The
  claim is about a *mixed* `σ_p` (84 with probability `p`): `∃ p > 0,
  Δ(σ_p, σ_0; BR(σ_p), D, m) > 0`, or in equilibrium language, `σ_0` is not a best
  response to `BR(σ_0)` because the setters can exploit a bidder who never bluffs.
  A one-sided `compare` cannot state this; it needs mixed policies (a `FinPMF` over
  pure policies) and a response map. Lean has `bestResponse_eq` for a *fixed*
  field; the adaptive version is a two-partnership zero-sum game, and the honest
  statement is a Stackelberg or equilibrium inequality. Build the fixed-field
  comparison first and name the adaptive claim as out of scope until then.

Rust implications for the design to converge on: `Contract` with kind/stake;
`Physics`-level `award`; a policy that covers bids, declaration and plays; a field
as a seat-indexed assignment of policies (optionally a sampled mixture); a
`compare_policies` whose result is an `Estimate`. The band in `compare` finally has
a job here, and it should be derived from `n` at a declared confidence (Hoeffding on
a bounded mark differential), not chosen ad hoc.

### 3.3 Another seat's knowledge (gap 3)

Object. The epistemic accessibility of seat `c` at a world: `ω ~_c ω'` iff
`ω.hands c = ω'.hands c` (same record), an equivalence relation, so `K_c` is an S5
modality whose cells are exactly the fibers `Φ(ViewerCtx c, PubState)` — Lean
already has those cells. `(knows c Q)` at `ω` is Kleene ∀ over `Φ_c(ω)`: `T` iff
`Q = T` everywhere, `F` iff `Q = F` somewhere, else `U`. Nested `knows` is free
(`(knows p (knows b Q))`), and "the partner can know the trick is safe" is
`(knows p (none (roles (domino t)) (live t) (not (own-legal t)) (beats t lead q*)))`
with `own-legal` read at `p`'s frame — which is the right subtlety: inside
`knows c`, the viewer predicates are `c`'s.

Rust. A `World` predicate with a sub-query argument; evaluation builds `c`'s kernel
from the world and the record (`Frame::from_record` with viewer `c`, hand
`ω.hands c`) and runs the sub-query over its fiber. Refusal on a resolving belief
unless `c`'s hand is resolved. Cost is a nested fiber enumeration per world; cap it
with the budget. The grammar should scope roles the way `none` does (bind nothing
outside).

### 3.4 The auction as information (gap 4)

Lean has the object and the warning. `DealLocalInfo` (`Information.lean:25`) is
(viewer, hand0, auction, declaration, plays); `mech_not_injective` proves the
mechanical projection `(ViewerCtx, PubState)` that keeps only the winning contract
loses information that every seat observed. Silas's `Contract{declarer, bid}` is
that projection. The fix is structural, not a predicate: the record carries
`auction : Vec<(Seat, Bid)>` with `Bid = Pass | Point(n) | Mark(m)` and the
declaration chosen; `Contract` is derived; `PolicyKey.history` includes bids;
predicates `(bid-of c b)`, `(passed c)`. Reading a bid is then gap 1 applied to a
bidding kernel — nothing new mathematically, which is the point of doing it this
way. The scoring (what a 31 is worth versus a 41 under marks) lives in `award`, so
the physics boundary grows by the auction's legality (`Auction.lean:legalBid`) and
`award`; both are already Lean definitions.

### 3.5 Aggregate (gap 5)

Object. `card` of a filter: `(card n (roles r ...) body)` binds `n` to
`#{β : body β = T}`, three-valued — defined iff no binding is unsettled (a `U`
could change the count), else `U`. Then `(less n m)` on numbers. This keeps `Number`
literal-only in the enumeration sense (`Sort::values` stays empty for it) and bound
only by functional clauses (`card`, `count`, `bid`, `to-set`, `banked`), which is
how Lean would type it: a number role is a `ℕ` determined by an equation, not a
quantifier over `ℕ`. Compile-time check: every number role must appear as the output
of exactly one functional clause before any `less` that reads it (a stratification
like `none`'s). "Most unplayed" is then `(card n (roles (domino e)) (live e) (in e
q)) (none (roles (context r)) (not (context r q)) (card m (roles (domino e)) (live e)
(in e r)) (less n m))`. The alternative — a direct `(more (roles ...) body₁ (roles
...) body₂)` — avoids number roles but is a one-off; `card` is the general object.

### 3.6 Solved outcome

The value of a *world* under a play model is a fact; the probability is a belief
statement (keep texas-42's "support ≠ belief"). Two models matter:

- **Perfect-information minimax** — Lean has `ConstellationSuffix.minimax`/`value`
  (exact, fuel-indexed, kernel-evaluable at small depth). As a Silas `World`
  predicate `(solved-makes)` / `(solved-margin n)` with `horizon_plies = 4k` it
  reads every hand (refuses on unresolved tiles) and carries no model, so `Exact`
  survives. It is the oracle, which is *algorithmia non grata* for play; as an
  analysis predicate for the book's "this hand makes against any defense" it is
  the right object, and its authority label is honest.
- **A field rollout** — `award(σ, F, ω)` for a Silas policy field: `ExactGiven{F}`.
  This is the per-world term of §3.2 and the `p` of "always right".

Either way the belief-level number is `β.exp` of the per-world value, which is the
summary layer already built. What is missing is a predicate whose horizon is not
zero — `PredicateSpec.horizon_plies` exists for exactly this, and the budget is the
guard.

---

## 4. Ranked recommendations

1. **Make a policy a pure function of its information state.** Drop controller
   modes (every book rule is `(in play) (next play)`; no `bind` is used) or derive
   the mode by replaying the record from the hand's first frame; evaluate `bind` at
   that frame. Unblocks gap 1 and the measurability theorem. Cost: low-medium
   (`policy.rs`, one test).
2. **One frame, derived views.** `Frame = (physics, viewer, hand0, contract, plays)`;
   kernel, leader, prefix, played, banked, voids as functions; delete `Frame::new`
   and the cached `banked`; one void derivation. Cost: medium (fixtures in four test
   files).
3. **Physics as tables; the fiber owned by Silas.** Identity = content hash of the
   four tables; `count_worlds`/`worlds`/`legal_plays`/`trick_winner`/`trick_points`
   become functions of the tables; Zeb's engine becomes an implementation under
   differential test. Cost: medium-low (`physics.rs`, `reference.rs`, ZebPhysics).
4. **`Contract` with kind, stake and declaration; the auction in the record;
   `award`.** The prerequisite for gaps 2 and 4, and the object Lean already has.
   Cost: low-medium.
5. **Authority as a provenance lattice; `compare` bands only sampled values, with
   the band derived from `n`; return the meet with the verdict.** Cost: low.
6. **Policy likelihood (gap 1) as `Belief::condition_on_policy`,** with the
   convention kernel decided (Dirac for a tile, uniform-on-set for a set-valued
   guard) and authority `ExactGiven{policy identity}`. Cost: medium, after 1 and 2.
7. **Write the denotational semantics down** (a short `SEMANTICS.md` or module doc):
   Kleene three-valued atoms, `none` as Kleene ¬∃, top-level `U ↦ F`, injective
   class assignment with the Bell-number completeness, existential projection, and
   the admissibility predicate of `uniform_resolving`. Fix `trick-number`. This is
   the document Lean transcribes. Cost: low.
8. **`knows` (gap 3)** as Kleene ∀ over the other seat's fiber, scoped like `none`,
   refused on unresolved hands. Cost: medium.
9. **`card` and `less` (gap 5)** with three-valued definedness and stratified number
   roles. Cost: medium.
10. **Cross-game comparison (gap 2) in two steps:** first the field-free dominance
    "42 not 84 at one mark from the game" as the first Lean-checkable cross-game
    claim (cost: low once 4 lands); then `compare_policies(σ₁, σ₂, field, deals, m)`
    returning an `Estimate` (cost: high), with the adaptive "keep the setters
    honest" named as an equilibrium claim and left out of scope until the
    fixed-field comparison exists.

The three most serious soundness findings, restated: **F1** (a policy's action
depends on controller call history, not on the information state — the
measurability theorem is false as built and policy likelihood is ill-defined);
**F2** (`Exact` rests on unverified `count_worlds`/`worlds` methods and a string
identity, so a wrong physics produces wrong numbers labelled "no model"); **F3**
(two frame constructors give `played`, `shown-void` and the record predicates two
meanings, and the kernel's voids and the record's banked points are stored beside
the plays they derive from). None of the three is a wrong number today; each is a
specification Lean could not state as the README states it.
