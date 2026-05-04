---
title: BookStrategyPlayer — multi-step strategy framework with recording
kind: entity
first_seen: local-2026-05-03
last_updated: local-2026-05-04
status: design (build pending)
---

## What it is

BookStrategyPlayer is a player architecture that plays via a library of named multi-step
**strategies** rather than a single per-decision utility. Each strategy is a small object
with an explicit lifecycle (recognition → commitment → execution → completion). Strategies
that don't apply at a given decision fall back to a baseline player ([[w42-lens-v1-utility-head-to-head|Lens]]
with the EV utility, the established strongest fixed-utility baseline).

The framework was motivated by [[w42-book-claim-synthesis-and-ai-directions]]'s "single-
decision blind spot" insight: most book claims are multi-step plans, but most W42 probes
test single-decision contrasts. A planner that can express plans is the natural next
abstraction level for testing the book.

## The Strategy protocol

Each strategy implements four lifecycle methods, all pure functions of `(game_state, plan_state)`:

```python
class Strategy(Protocol):
    name: str
    def applies_to(self, gs, ps) -> bool: ...
    def priority(self, gs, ps) -> float: ...
    def commit_recognition(self, gs, ps) -> dict: ...   # called once per plan instance
    def next_action(self, gs, ps) -> tuple[Action | None, dict]: ...
    def plan_done(self, gs, ps) -> bool: ...
    # Optional:
    def observe(self, opp_action, opp_seat, gs, ps) -> dict: ...   # opponent modeling hook
```

By convention: following through on an active plan returns priority ≥ 10.0; recognizing a
new opportunity returns 1.0–5.0. This monotonicity prevents plan-flipping mid-execution.

`PlanState` carries:

- `active_plans: dict[str, dict]` — namespaced per-strategy private state
- `facts: dict[str, Any]` — publishable cross-strategy facts (e.g., `void_suits`)

Strategies do not call each other's methods. They contribute facts (publishable) and
actions (arbitrated). Composition is the framework arbitrating among declared facts and
proposed actions, not strategies negotiating with each other.

## The player loop

Five phases per decision:

1. **Retire** completed/dead plans (each strategy's `plan_done` is consulted)
2. **Recognize** new applicable strategies; call `commit_recognition` for each fresh one
3. **Arbitrate** — highest-priority applicable strategy wins; fallback to Lens(ev) if none
4. **Apply state delta** — `next_action` returns `(action, plan_state_delta)`; the player applies the delta
5. **Bail or play** — `action=None` means strategy bailed at the last second; use fallback

`BookStrategyPlayer(strategies=[], fallback=Lens(ev))` is **identical** to Lens(ev). The
framework is a strict superset of the baseline. Any margin in head-to-head measurement
is exactly the strategy library's contribution.

## Five composition modes

The protocol's purity makes meta-strategies compose naturally:

1. **State-conditioned strategy selection** — `priority(gs, ps)` already takes game state.
   `TrumpPulling.priority` returns `5.0 - gs.trick_idx` (high early, fades); `ThrowawayLadder`
   does the inverse. The arbitration loop selects the right strategy per state. Free out of
   the box. Connects to [[w42|t42-nwuu]] (state-conditioned utility) — same idea, applied to
   strategies instead of utilities.

2. **Strategy chaining** — one strategy's plan publishes a `fact`; another strategy reads it
   via `applies_to`. Example: `SingletonLeadToVoid` publishes `facts.void_suits`;
   `CountSteeringIntoVoid` reads it. Context-aware strategies use the same mechanism:
   a parent strategy publishes an explicit context fact (`active_context`, `target_suit`,
   `plan_phase`, etc.) and another strategy becomes applicable only in that context.
   Requires formalizing publish/subscribe in `PlanState` (~30 LOC framework addition).

3. **Hierarchical strategies** — a Strategy's `next_action` can delegate to a sub-player
   with its own library. Finite acyclic nesting needs no protocol change and maps directly
   to the book's chapter structure — Ch 3 (bidder play), Ch 5 (setter defense), Ch 8
   (84-bid endgame) are each natural sub-libraries. Cyclic self-recursion is not part of
   the design; construction should reject cycles and records should carry a `strategy_path`
   for nested decisions.

4. **Opponent-aware (counter-strategies)** — add an `observe(opp_action, ...)` hook to the
   protocol. Strategies accumulate opponent statistics in shared facts. Counter-strategies
   trigger when opponent patterns match a precondition. (~50 LOC framework addition.)

5. **Cross-hand match memory** — current architecture is per-hand. Lifting state to
   match-level (e.g. "watch opponent for 5 hands then switch to counter") requires extending
   the player with a `match_state` arg passed through lifecycle methods. Bigger lift; the
   parallel-hand simulator currently treats hands as independent.

## The book IS a meta-strategy

The deepest implication of the architecture: **the book's chapter structure is exactly mode
1 + mode 3.** Chapters partition state space (Ch 3 = bidder offense, Ch 5 = setter defense,
etc.) and within each chapter live multiple specific tactics as sub-strategies. A faithful
book-encoded player is:

```python
TheBook = BookStrategyPlayer(
    strategies=[
        Chapter3_BidderPlay(),       # composite of trump-pulling, etc.
        Chapter4_PartnerSupport(),
        Chapter5_SetterDefense(),    # composite of void-creation, low-trump-trap, etc.
        Chapter8_AllTricks84(),      # composite of throwaway-ladder, etc.
        Chapter12_HighBidPlay(),     # composite of pounce-window, etc.
    ],
    fallback=Lens("ev"),
)
```

`Lens(ev) vs TheBook` is then a single head-to-head match measuring how much the *entire
book* is worth in points-per-hand — the question the [[w42-book-validation-campaign]] has
been trying to answer all along, just at the wrong abstraction level until now.

## Recording → training pipeline

Per-decision records get logged structurally:

```python
@dataclass
class DecisionRecord:
    hand_id, trick_idx, decision_idx
    game_state_tensor                   # serialized GameStateTensor
    plan_state_snapshot                 # active_plans + facts at decision time
    strategy_path                       # nested strategy path, empty for top-level fallback
    legal_actions
    applicable_strategy_names           # which fired
    strategy_priorities                 # all returned scores, not just winner
    chosen_strategy                     # None = fallback
    chosen_action
    fallback_action                     # what Lens(ev) WOULD have chosen — free counterfactual
    immediate_trick_outcome             # filled at trick-end
    hand_point_margin                   # filled at hand-end
    plan_completion_status              # per-strategy completed/disrupted/abandoned
```

The free counterfactual (`fallback_action`) is the key. At every decision we record both
what the strategy player did *and* what Lens(ev) would have done. The difference is the
strategy's contribution exactly. Re-running the hand with the counterfactual action gives
ground-truth per-decision causal effect (paired-seed style); off-policy estimation gives a
cheaper approximation once a value model is trained.

The same records are also the strategy-discovery surface. Decisions where no strategy
applies are not empty data; they are uncovered territory. The useful coverage buckets are:

- `uncovered` — no strategy applied and fallback played.
- `covered_same_as_fallback` — a strategy applied but matched Lens(ev).
- `covered_diff_positive` — a strategy diverged and improved outcome.
- `covered_diff_negative` — a strategy diverged and hurt outcome.
- `bailed` — a strategy recognized context but declined to play.
- `disrupted` — a committed plan failed to complete.

High-regret or high-tail-risk uncovered regions become candidates for new strategy
discovery. A candidate strategy is composed into the library, rerun in paired-seed
head-to-head, and kept only if coverage, completion, and point-margin diagnostics justify
it. This is future exploration, not a Phase 1 requirement beyond recording the fields
needed to identify the regions.

Three trainable models, in increasing ambition:

- **Model A — strategy selector.** Replaces hand-crafted `priority(gs, ps)` with a learned
  function. Input: encoded game_state + plan_state + applicability mask. Output: per-
  strategy logits. Loss: regression on per-strategy expected hand_point_margin (Q-learning
  over macro-actions). The framework is unchanged; only the arbitration step gets a learned
  head.

- **Model B — plan-success predictor.** P(plan completes successfully) and E[points if it
  completes] per (state, strategy). Useful as input to Model A or as an early-bail signal.

- **Model C — end-to-end policy distillation.** Predict actions directly, bypassing the
  library at inference. The strategies become *training scaffolding*. Discovery side-effect:
  if the trained policy consistently picks an action where no encoded strategy applied,
  that region is a candidate for a new strategy nobody wrote down. **Strategy discovery
  bootstraps from gameplay data.**

## Where the parked models suddenly have roles

This pipeline naturally re-activates the three parked-or-experimental models:

- **[[gus]]** (belief-state encoder) → input encoder for Model A. Gus turns partial-
  observability into a clean belief tensor; the strategy selector consumes it. This is what
  Gus was originally designed for — feeding decision-time models. The decision-time model
  isn't [[burl]], it's the strategy selector.

- **[[burl]]** (slow experimental LLM) → candidate Model A. Strategies have human-readable
  names; an LLM is unusually well-suited to "given this game situation, which named
  strategy should fire?" because it can reason about strategy rationales in natural
  language. The structured action space (15-30 named strategies, not raw 7-domino choice)
  is what makes Burl tractable — small action space, narrative reasoning, STaR-style chain-
  of-thought to justify the pick. **This is the use case Burl was built for and we just
  hadn't found it yet.**

- **[[zeb]]** (parked AlphaZero variant) → either training-data generator or a backup Model
  C target. Zeb's policy head trained on strategy-labeled data instead of raw action labels
  has a much smaller, structured output space.

## Why this is the cleanest path past EV the campaign has identified

[[w42-lens-v1-utility-head-to-head]] established that EV is the ceiling for fixed
*pointwise* utilities at one-step lookahead. The other paths to "beat EV":

| path | status |
|---|---|
| Multi-utility heads | **dead** — Lens v1 killed it |
| Soft-cliff utilities (disaster, etc.) | **dead** — disaster confirmed EV is the ceiling for fixed pointwise tweaks |
| Generic MCTS (rung-2) | expensive build; doesn't directly use book wisdom |
| Lookahead-Lens K=2 | impractical — branching factor ~3-4 over 8-12 plies (4 players × K=2-3) |
| **Strategy-selector trained from book-strategy gameplay** | **uses book wisdom as structured action space; trains on cheap self-play with free counterfactuals; deployed model has planning capability without paying MCTS branching cost** |

The strategy-selector path was not on the original list because it required the framework
to exist first. Now that we're scoping the framework, training the selector is the natural
next layer — and the training data accumulates as a pure side effect of running matchups.

## Phased build plan

**Phase 1 (now, ~3-4h build):**
- Strategy framework (~400 LOC)
- 1-3 starter strategies (~150 LOC each). `singleton_lead_to_void` is a valid first
  measurement target, but it should be treated as a measurement/negative-control candidate,
  not assumed to rehabilitate [[w42-bookval-v1-wave2-void-creation]]. The canonical
  follow-position void variant remains the cleaner book-positive candidate if the first
  strategy is meant to test the book's strongest version.
- DecisionRecord serialization to parquet (~50 LOC; logged but not yet trained on)
- Head-to-head measurement vs Lens(ev) (~50 LOC, reuses `w42/lens_v1/parallel_match.py`)
- Publish/subscribe `facts` extension to PlanState (~30 LOC; mode 2 enabler)

Phase 1 deliverables are useful even if Phase 2/3 never happen: a measurement instrument
+ training pipeline as free side effect + the option to spend on Phase 2/3 only after
Phase 1 confirms strategies are worth measuring.

**Phase 2 (after ~5-10 strategies, ~50K labeled decisions):**
- Train Model A (learned strategy selector) on recorded data
- A/B test learned-selector vs hand-crafted-priority head-to-head
- Decide whether to invest in Burl-as-selector (LLM path) or smaller NN

**Phase 3 (research frontier, gated on Phase 2):**
- Strategy discovery via clustering of fallback-invoked decisions where the learned
  selector deviates from hand-crafted priorities
- End-to-end policy distillation (Model C) if Phase 2 lifts are real

## Algebraic specification

The framework's structure is determined by nine algebras and ten laws. Stating them
explicitly removes most "design choice" degrees of freedom — most decisions become forced
moves once the algebras are fixed. Property-based tests for L1-L10 should ship with
Phase 1.

### The nine algebras

**A1. `Library` is a commutative idempotent monoid.**
```
Library    = Set Strategy
identity   = ∅
operation  = ∪    (set union)
```
Two strategies merged into one library forms a *set*, not a list. Arbitration uses `max`,
which is commutative and idempotent — so listing the same strategy twice is equivalent to
listing it once, and order doesn't matter. **Forced**: dedupe by `Strategy.name`.

**A2. `Arbitration` is a bounded join-semilattice.**
```
carrier   = (Strategy, priority) ∪ {⊥}
join (∨)  = argmax-priority with canonical strategy.name tie-break (with -∞ identity)
laws      = associative, commutative, idempotent
```
The framework picks the join over all applicable (strategy, priority) pairs. Bottom ⊥
exists (no strategies applicable → fall back). **Forced**: arbitration code is
`applicable.fold(max_priority_then_name, ⊥)` — three lines, no branching. Equal priorities
must not reintroduce library-order dependence.

**A3. `PlanState` evolves under the State monad (per hand).**
```
HandComputation a = State PlanState a
```
Each decision is a pure function `(GameState, PlanState) → (Action, PlanState')`. The
hand is a sequence threaded by `>>=`. **Forced**: PlanState is per-hand-immutable-but-
rebound, never globally mutable — which is what lets parallel-hand simulation run
without locks.

**A4. `Recording` is the Writer monad.**
```
type      = WriterT [DecisionRecord] Identity
operation = `tell` per decision; `runWriter` per hand
```
Records accumulate. They never read back — pure write. **Forced**: recording can never
affect player behavior; A/B testing recording-on vs recording-off is observably identical.

**A5. `Fallback` and `Library` together form the Reader monad.**
```
PlayerEnv = (Library, Fallback)
type      = Reader PlayerEnv
```
Library + fallback are immutable per-player. **Forced**: a player object is immutable
after construction.

**A6. `Strategy` is a Σ-algebra (a record of operations on a carrier).**
```
Strategy : (GameState × PlanState) → ⟨5 + 1 functions⟩
  applies_to          : (GS, PS) → 𝔹
  priority            : (GS, PS) → ℝ
  commit_recognition  : (GS, PS) → Δ PS
  next_action         : (GS, PS) → (Maybe Action × Δ PS)
  plan_done           : (GS, PS) → 𝔹
  observe (optional)  : (Action, Seat, GS, PS) → Δ PS
```
Universal-algebra style: a fixed signature of operations on a common carrier. **Forced**:
every strategy implements the same interface; framework code never inspects strategy
internals; refactoring one strategy can't break others.

**A7. `Δ PlanState` is a partial monoid where facts must be monoid-valued.**
```
Δ PlanState         = (Δ active_plans, Δ facts)
Δ active_plans      = Map String (Maybe Plan)        -- None = retire
Δ facts             = Map String MonoidValue         -- e.g. Set, Counter, Max
fact merge          = pointwise monoid-op
plan merge          = namespace-keyed overwrite (or delete on None)
```
The crucial constraint: **facts must be values in a commutative monoid** (sets union,
counters add, max-tracked values take max). This is what makes fact accumulation order-
independent across decisions and across parallel hands. **Forced**: facts API only
accepts values with declared monoid laws — implementation rejects naked scalars unless
wrapped in a monoid type.

**A8. Hierarchical composition is a functor `BookStrategyPlayer → Strategy`.**
```
wrap : BookStrategyPlayer → Strategy
wrap(BSP).next_action(gs, ps) = (BSP.choose_action(gs), Δ-from-sub-recording)
```
A sub-player wraps as a Strategy. The functor preserves structure: `wrap(BSP1 ∪ BSP2)` is
observably equivalent to coordinating wrap(BSP1) and wrap(BSP2) at the parent for finite
acyclic strategy graphs. **Forced**: chapters can be sub-libraries; nesting depth is
unbounded by the protocol but finite in any instantiated player; construction rejects
cycles rather than providing a recursive execution stack.

**A9. The full player is a composed monad stack.**
```
PlayerM = ReaderT (Library, Fallback) (StateT PlanState (WriterT [DecisionRecord] Identity))
```
Reader for env, State for plan-thread, Writer for recording, no IO (Identity). **Forced**:
the entire decision logic is a pure function from env+state to (action, state', records).
Testable without a simulator. Replayable from records (which is L10 below).

### The ten laws

These are invariants the algebras force. Each is testable by property-based tests
(Hypothesis or equivalent) and should ship with Phase 1.

**L1 — Empty library identity.**
```
BSP(∅, fb).choose_action(gs) ≡ fb.choose_action(gs)         ∀ gs
```
Adding strategies STRICTLY EXTENDS behavior; never alters the fallback baseline.

**L2 — Library order independence.**
```
BSP({s₁, s₂}, fb).choose_action(gs) ≡ BSP({s₂, s₁}, fb).choose_action(gs)
```
Falls out of A2 (max is commutative).

**L3 — Library idempotence.**
```
BSP({s, s}, fb) ≡ BSP({s}, fb)
```
Falls out of A1 (set semantics).

**L4 — Do not re-recognize active plans.**
```
If s.name ∈ ps.active_plans, commit_recognition(s, _, _) is not called.
```
Plans are committed once and persist until retired. A strategy may recognize again after
retirement only if the library intentionally permits a new plan instance. One-shot-per-hand
strategies require an explicit tombstone fact; they are not the framework default.

**L5 — Priority monotonicity for active plans.**
```
∀ s ∈ Library, ∀ ps : s.name ∈ ps.active_plans :
    s.priority(gs, ps) ≥ max{ s'.priority(gs, ps) | s'.name ∉ ps.active_plans }
```
Following through dominates starting fresh. Convention enforced as an assertion.

**L6 — Bail and Retire are orthogonal operations.**
```
Bail   (next_action returns None action and identity/private-preserving delta)
       : DOES NOT modify ps.active_plans[s.name]
Retire (plan_done returns True, OR delta.active_plans[s.name] = None)
       : DELETES ps.active_plans[s.name]
```
Two semantically distinct ways for a strategy to "give up" — one preserves the chance to
fire again, the other doesn't.

**L7 — Fact merge is associative.**
```
merge(merge(a, b), c) ≡ merge(a, merge(b, c))
```
Falls out of A7 (facts are monoid-valued).

**L8 — Fact merge is commutative.**
```
merge(a, b) ≡ merge(b, a)
```
Same source. Lets parallel hands accumulate facts without ordering concerns.

**L9 — Strategy namespace hermeticity.**
```
Strategy s reads/writes only ps.active_plans[s.name] (private) and ps.facts.* (shared).
∀ s, s' : s ≠ s' ⇒ s does not read/write ps.active_plans[s'.name]
```
Inter-strategy communication goes through `facts`, never through plan-state poking. This
is the law that makes adding a strategy a *pure addition* — it CANNOT break existing
strategies.

**L10 — Recording fidelity.**
```
∀ hand h : Records(h) + (forge oracle + seeds) ⊨ Replay(h)
```
Decision records contain enough state to deterministically replay the hand. Falls out
of A4 + the immutability of A3/A5. **Critical for training-data integrity**: the records
ARE the training data; they must be sufficient. Records include the player/library
identity, fallback identity, `strategy_path`, legal actions, chosen action, fallback
action, plan-state snapshot, and all RNG/forge world-sampling seeds needed for replay.

### The core operation

```
choose_action : (GameState, PlanState) → (Action, ΔPlanState, DecisionRecord)
choose_action(gs, ps) =
    let ps₁  =  retire_done(ps, library)                          -- A6 plan_done; A7 Δ_apply
        new  =  { s ∈ library | s.applies_to(gs, ps₁) ∧ s.name ∉ ps₁.active_plans }
        ps₂  =  ps₁ ⊕ ⊕{ s.commit_recognition(gs, ps₁) | s ∈ new } -- A7 Δ-fold
        app  =  { s ∈ library | s.applies_to(gs, ps₂) }
    in  case argmax-priority(app, gs, ps₂) of                     -- A2 ∨-fold
            ⊥          → (fb.choose_action(gs), ε, Record(fallback))
            Some(s)    → let (a, δ) = s.next_action(gs, ps₂)
                         in  case a of
                                Nothing  → (fb.choose_action(gs), δ, Record(bail, s))
                                Just(α)  → (α,                    δ,           Record(strategy, s, α))
```

Six lines of meaningful logic. Everything else is implementation noise.

### What the algebra forces in implementation

A handful of decisions that look like preferences are actually forced:

| design "choice" | actually forced by | what would break |
|---|---|---|
| Strategies stored as `dict[name → Strategy]`, not `list` | A1 + L3 | dedup gets messy with lists |
| Arbitration = `max(priority)` with canonical `strategy.name` tie-break, not weighted vote | A2 | weighted vote or order-dependent ties break commutativity (L2) |
| Facts must be monoid-valued | A7 + L7 + L8 | merge ordering becomes load-bearing |
| Per-hand PlanState, not global | A3 | parallel hands need locks |
| Recording is pure-write | A4 | recording could secretly affect play |
| Strategies hermetic in namespace | A9 + L9 | adding a strategy could break others |
| Bail preserves active plan unless explicit retire delta is emitted | L6 | a last-second fallback silently destroys plan state |
| Hierarchical sub-player wraps as Strategy in an acyclic graph | A8 | composite strategies need bespoke plumbing, or cycles need a stack/termination protocol |

### Maintenance benefits the algebra delivers

1. **Property-based tests fall out for free.** L1-L10 are property tests directly. Hypothesis
   (or equivalent) can fuzz strategies and game states and check the invariants
   automatically. A bug in a new strategy that violates L9 (writes to another strategy's
   namespace) gets caught before it ships.

2. **Refactoring is safe by construction.** If the algebras hold, replacing the arbitration
   implementation (e.g. swapping `max` for a learned head, per Model A in the training-
   pipeline section) preserves L1-L4 and L7-L10 automatically. Only L5 needs re-checking
   under a learned head.

3. **Future strategies have a precise contract.** When a future strategy author (human
   or agent) adds strategy #20, they don't need to read framework code. They need: the
   Strategy signature (A6), the namespace rules (L9), the bail-vs-retire distinction (L6).
   Three concepts. The rest of the system is downstream of those.

## Composition limits worth knowing

Three things the framework can't do, no matter how clever the strategies:

1. **One domino per decision.** If two strategies both want different dominoes, priority
   arbitration picks one and the other's plan goes infeasible. The framework can detect
   and log the conflict but not resolve it — game constraint, not framework limit.
2. **No automatic plan negotiation.** Two active plans can rest on contradictory
   predictions of opponent behavior; the framework doesn't reason about plan compatibility.
   Strategy-author responsibility.
3. **Partnership coordination via assumed-mirror only.** A strategy that requires partner
   to play the same library can be modeled by having the partner-seat run a copy of the
   same player. Asymmetric self-play; the simulator already supports this.

## Status

- **Designed:** 2026-05-03 in conversation with the orchestrator. Architecture, lifecycle
  protocol, composition modes, recording format, and phased build plan all specified.
- **Bead pending:** Phase 1 build (framework + first strategy + measurement harness +
  recording) to be filed.
- **Built:** not yet.
- **Strategies encoded:** none yet.

## Links

- [[w42-book-claim-synthesis-and-ai-directions]] — the methodology insight that motivated
  this design
- [[w42-book-validation-campaign]] — the campaign this serves
- [[w42-lens-v1-utility-head-to-head]] — the EV-as-ceiling result that constrained the
  design space
- [[w42-bookval-v3-utility-argmax-divergence]] — Wave 4.0 measurement that gated the
  whole architecture-decision branch
- [[gus]] — input encoder candidate
- [[burl]] — Model A (strategy selector) candidate; the use case Burl was built for
- [[zeb]] — training-data generator / Model C target
- [[forge]] — the simulator the player runs against
