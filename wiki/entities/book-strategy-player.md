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

Each strategy implements five lifecycle methods, all pure functions of
`(game_state, plan_state)`:

```python
class Strategy(Protocol):
    name: str
    version: str
    def recognizes(self, gs, ps) -> bool: ...
    def priority(self, gs, ps) -> float: ...
    def commit_recognition(self, gs, ps) -> dict: ...   # pure proposal, committed only if winner
    def next_action(self, gs, ps) -> tuple[Action | None, dict]: ...
    def plan_done(self, gs, ps) -> bool: ...
    # Optional:
    def observe(self, opp_action, opp_seat, gs, ps) -> dict: ...   # opponent modeling hook
```

Recognition and commitment are deliberately separate. `recognizes` means "this fresh
strategy can propose a plan instance now." `commit_recognition` builds that proposed plan
state, but the framework applies it only if the fresh strategy wins arbitration. Losing
fresh strategies do not become active plans. Active strategies are eligible for arbitration
because they already have private plan state, not because they newly recognize the current
state.

By convention: following through on an active, unblocked plan returns priority ≥ 10.0;
recognizing a new opportunity returns 1.0–5.0. This monotonicity prevents plan-flipping
mid-execution, but the law is scoped to strategies that are actually eligible to act.
Priorities must be finite real numbers; `NaN` is a contract violation because it breaks
ordering laws.

`PlanState` carries:

- `active_plans: dict[str, dict]` — namespaced per-strategy private state
- `facts: dict[str, FactValue]` — publishable cross-strategy facts (e.g.,
  `SetFact({"sixes"})`, `CounterFact({"seat_2:trump": 1})`, `MaxFact(17)`)

Strategies do not call each other's methods. They contribute facts (publishable) and
actions (arbitrated). Composition is the framework arbitrating among declared facts and
proposed actions, not strategies negotiating with each other.

## The player loop

Five phases per decision:

1. **Retire** completed/dead plans (each strategy's `plan_done` is consulted)
2. **Propose** candidates from active strategies plus fresh strategies whose `recognizes`
   precondition holds; fresh proposals compute `commit_recognition` speculatively
3. **Arbitrate** — highest-priority candidate wins; fallback to Lens(ev) if none
4. **Commit the winner** — apply only the winning fresh strategy's recognition delta
5. **Bail, validate, or play** — `next_action` returns `(action, plan_state_delta)`;
   `action=None` means bail to fallback, and illegal actions are recorded contract
   violations that fall back in batch runs

`BookStrategyPlayer(strategies=[], fallback=Lens(ev))` is **identical** to Lens(ev). The
framework is a strict superset of the baseline. Any margin in paired-seed head-to-head
measurement estimates the strategy library's hand-level contribution. Per-decision
`fallback_action` records are local counterfactual proposals, not exact causal effects
unless paired replay or a value model evaluates the divergent branch.

## Five composition modes

The protocol's purity makes meta-strategies compose naturally:

1. **State-conditioned strategy selection** — `priority(gs, ps)` already takes game state.
   `TrumpPulling.priority` returns `5.0 - gs.trick_idx` (high early, fades); `ThrowawayLadder`
   does the inverse. The arbitration loop selects the right strategy per state. Free out of
   the box. Connects to [[w42|t42-nwuu]] (state-conditioned utility) — same idea, applied to
   strategies instead of utilities.

2. **Strategy chaining** — one strategy's plan publishes a `fact`; another strategy reads it
   via `recognizes` or `priority`. Example: `SingletonLeadToVoid` publishes
   `facts.void_suits: SetFact({suit})`;
   `CountSteeringIntoVoid` reads it. Context-aware strategies use the same mechanism:
   a parent strategy publishes lawful context facts such as
   `active_contexts: SetFact({("Chapter5_SetterDefense", "void_creation")})`. Scalar
   phase, target, and local context belong in `active_plans[s.name]` unless wrapped in a
   lawful `FactValue` such as `UniqueFact`, `MaxFact`, or timestamped/canonically-tied
   `LastByDecision`.
   Requires formalizing publish/subscribe in `PlanState` (~30 LOC framework addition).

3. **Hierarchical strategies** — a Strategy's `next_action` can delegate to a sub-player
   with its own library. Finite acyclic nesting needs no protocol change and maps directly
   to the book's chapter structure — Ch 3 (bidder play), Ch 5 (setter defense), Ch 8
   (84-bid endgame) are each natural sub-libraries. A chapter wrapper exposes a lawful
   parent-level priority, namespaces internal state by `strategy_path`, and normally
   returns `None` when no inner strategy applies. It should not silently consume control by
   playing its own fallback unless explicitly configured as a terminal fallback. Cyclic
   self-recursion is not part of the design; construction should reject cycles and records
   should carry a `strategy_path` for nested decisions.

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
    serializer_schema_version
    framework_version
    library_fingerprint
    strategy_versions
    fallback_fingerprint
    player_config_hash
    forge_version
    rules_version
    hand_seed
    decision_rng_seed_or_stream_position
    initial_deal_id_or_seed
    seat_id
    partnership_config
    hand_id, trick_idx, decision_idx
    game_state_snapshot                 # lossless replay input
    game_state_tensor                   # model input; not assumed lossless
    plan_state_snapshot                 # active_plans + facts at decision time
    strategy_path                       # nested strategy path, empty for top-level fallback
    legal_actions
    active_strategy_names
    recognized_strategy_names           # fresh recognizers
    candidate_strategy_names            # active + fresh candidates
    strategy_priorities                 # finite scores for all candidates, not just winner
    chosen_strategy                     # None = fallback
    chosen_action
    fallback_action                     # what Lens(ev) proposes under pure dry-run
    strategy_status                     # fallback/strategy/bail/illegal_strategy_action/etc.
    immediate_trick_outcome             # filled at trick-end
    hand_point_margin                   # filled at hand-end
    plan_completion_status              # per-strategy completed/disrupted/abandoned
```

The local counterfactual proposal (`fallback_action`) is the key cheap signal. At every
decision the record stores both what the strategy player did and what Lens(ev) would have
proposed. That local difference is not automatically the strategy's causal contribution:
one different domino can change legal actions, trick winners, revealed information, later
strategy applicability, and hand margin. Exact causal effect requires paired-seed replay
from the decision or from hand start; off-policy estimation gives a cheaper approximation
once a value model is trained. Counterfactual fallback evaluation must be observationally
pure: it cannot consume the live RNG stream or mutate player, forge, match, or plan state.

The same records are also the strategy-discovery surface. Decisions where no strategy
applies are not empty data; they are uncovered territory. The useful coverage buckets are:

- `uncovered` — no strategy applied and fallback played.
- `covered_same_as_fallback` — a strategy applied but matched Lens(ev).
- `covered_diff_hand_won` — a strategy diverged and the hand-level margin was positive.
- `covered_diff_hand_lost` — a strategy diverged and the hand-level margin was negative.
- `covered_diff_replay_positive` — paired replay estimates the divergent strategy branch as positive.
- `covered_diff_replay_negative` — paired replay estimates the divergent strategy branch as negative.
- `bailed` — a strategy recognized context but declined to play.
- `disrupted` — a committed plan failed to complete.
- `illegal_strategy_action` — a strategy returned an action outside `legal_actions`; tests
  should fail hard, while large simulations may record the violation and use fallback.

High-regret or high-tail-risk uncovered regions become candidates for new strategy
discovery. A candidate strategy is composed into the library, rerun in paired-seed
head-to-head, and kept only if coverage, completion, and point-margin diagnostics justify
it. This is future exploration, not a Phase 1 requirement beyond recording the fields
needed to identify the regions.

Three trainable models, in increasing ambition:

- **Model A — strategy selector.** Replaces hand-crafted `priority(gs, ps)` with a learned
  function. Input: encoded game_state + plan_state + candidate mask + strategy metadata.
  Target: replay-estimated or value-model-estimated expected hand margin for each candidate
  strategy. The cheap initial version imitates hand-crafted priorities on successful or
  completed plans and downweights disrupted/negative plans; the strong version trains on
  paired-seed counterfactual replays. The framework is unchanged; only arbitration gets a
  learned head.

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
| **Strategy-selector trained from book-strategy gameplay** | **uses book wisdom as structured action space; trains on cheap self-play with local fallback counterfactuals and replay hooks; deployed model has planning capability without paying MCTS branching cost** |

The strategy-selector path was not on the original list because it required the framework
to exist first. Now that we're scoping the framework, training the selector is the natural
next layer — and the training data accumulates as a pure side effect of running matchups.

## Phased build plan

**Phase 1 (now, ~3-4h build):**
- Core framework contract: `PlanState`, `DeltaPlanState`, `FactValue`, `Strategy`,
  `Candidate`, `ArbitrationResult`, and `DecisionRecord`.
- Monoid fact wrappers: `SetFact`, `CounterFact`, `MaxFact`, `MinFact`, `UniqueFact`,
  `BoolOrFact`, and `BoolAndFact`; reject naked scalar facts.
- Library construction: reject duplicate strategy names, reject `NaN`/non-finite priorities
  during evaluation, canonical `strategy.name` tie-break, and acyclic strategy graph checks
  for hierarchy.
- Fake strategies and property tests before W42 logic: L1-L10, including recording-on/off
  equivalence and winning-fresh-only commitment.
- Pure fallback adapter around Lens(ev): prove `BookStrategyPlayer([], Lens(ev)) == Lens(ev)`
  exactly, and prove fallback dry-run does not consume live RNG or mutate state.
- In-memory `DecisionRecord` schema first; parquet only after replay/version fields are
  stable and lossless `game_state_snapshot` is distinguished from model input tensor.
- One real starter strategy (~150 LOC). `singleton_lead_to_void` is a valid first
  measurement/negative-control target, but it should not be assumed to rehabilitate
  [[w42-bookval-v1-wave2-void-creation]]. The canonical follow-position void variant remains
  the cleaner book-positive candidate if the first strategy is meant to test the strongest
  book version.
- Head-to-head measurement vs Lens(ev) (~50 LOC, reuses `w42/lens_v1/parallel_match.py`)

Phase 1 deliverables are useful even if Phase 2/3 never happen: a measurement instrument
+ replayable training-data surface + the option to spend on Phase 2/3 only after Phase 1
confirms strategies are worth measuring.

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
listing it once, and order doesn't matter. **Forced**: the canonical library is
`dict[str, Strategy]`. User-facing construction rejects duplicate names with
`DuplicateStrategyNameError` rather than silently keeping one; algebraic idempotence applies
after construction to canonical libraries.

**A2. `Arbitration` is a bounded join-semilattice.**
```
carrier   = (Strategy, priority) ∪ {⊥}
join (∨)  = argmax-priority with canonical strategy.name tie-break (with -∞ identity)
laws      = associative, commutative, idempotent
```
The framework picks the join over all applicable (strategy, priority) pairs. Bottom ⊥
exists (no strategies applicable → fall back). **Forced**: arbitration code is
`applicable.fold(max_priority_then_name, ⊥)` — three lines, no branching. Equal priorities
must not reintroduce library-order dependence. Priority values must be finite and non-NaN.

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
This includes `fallback_action` logging: fallback evaluation must use a pure choose function,
a dry-run with cloned RNG, or a cloned immutable decision context.

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
  recognizes          : (GS, PS) → 𝔹
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
wrapped in a monoid type. Shared facts are cumulative observations; scalar-looking phase,
target, and context values belong in private plan state unless wrapped as `UniqueFact`,
`SetFact`, `MaxFact`, or another lawful `FactValue` with canonical tie handling.

**A8. Hierarchical composition is a lawful wrapper `BookStrategyPlayer → Strategy`.**
```
wrap : BookStrategyPlayer → Strategy
wrap(BSP).next_action(gs, ps) = (inner action or None, Δ-from-sub-recording)
```
A sub-player can wrap as a Strategy if the wrapper exposes a lawful parent-level priority,
namespaces internal PlanState by `strategy_path`, and returns `None` rather than consuming
control when no inner strategy applies, unless explicitly configured as a terminal fallback.
The earlier strong equivalence claim (`wrap(BSP1 ∪ BSP2)` equals coordinating wrappers at
the parent) is not generally true because internal fallbacks, parent-level priorities, and
namespace visibility all matter. **Forced**: chapters can be sub-libraries, but finite
acyclic wrappers need explicit semantics; construction rejects cycles rather than providing
a recursive execution stack.

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
Falls out of A1 after canonical library construction. Raw construction from a sequence
raises on duplicate strategy names instead of silently deduping; that preserves engineering
safety while keeping the algebra over canonical libraries.

**L4 — Do not re-recognize active plans.**
```
If s.name ∈ ps.active_plans, commit_recognition(s, _, _) is not called.
```
Plans are committed once and persist until retired. A strategy may recognize again after
retirement only if the library intentionally permits a new plan instance. One-shot-per-hand
strategies require an explicit tombstone fact; they are not the framework default.

**L5 — Priority monotonicity for active plans.**
```
For every active, unblocked strategy s:
    s.priority(gs, ps) ≥ max priority of every fresh recognized strategy,
    unless s declares itself blocked, bailed, or done.
```
Active-plan continuation dominates fresh-plan recognition. The comparison is only against
fresh strategies that actually recognized the current state; inactive irrelevant strategies
are outside the law. Convention enforced as an assertion in the fake-strategy property
tests and optional debug checks.

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
action, plan-state snapshot, lossless game-state snapshot, serializer/framework/library/
fallback/forge/rules versions, player config hash, seat/partnership identity, initial
deal seed, and all RNG/forge world-sampling seeds or stream positions needed for replay.

### The core operation

```
choose_action : (GameState, PlanState) → (Action, ΔPlanState, DecisionRecord)
choose_action(gs, ps) =
    let ps₁ = retire_done(ps, library)                            -- A6 plan_done; A7 Δ_apply
        active = {s ∈ library | s.name ∈ ps₁.active_plans}
        fresh  = {s ∈ library | s.name ∉ ps₁.active_plans ∧ s.recognizes(gs, ps₁)}

        active_candidates =
            [Candidate(s, priority=s.priority(gs, ps₁),
                       commit_delta=ε, committed_state=ps₁, is_fresh=False)
             for s in active]

        fresh_candidates =
            [let δ = s.commit_recognition(gs, ps₁)                -- pure proposal only
                 ps' = ps₁ ⊕ δ
             in Candidate(s, priority=s.priority(gs, ps'),
                          commit_delta=δ, committed_state=ps', is_fresh=True)
             for s in fresh]

        candidates = active_candidates ++ fresh_candidates
        fb_action = fb.pure_choose_action(gs)                     -- pure dry-run / clone
    in  case argmax-priority-then-name(candidates) of             -- A2 ∨-fold
            ⊥          → (fb_action, ε, Record(fallback))
            Some(c)    → let ps₂ = ps₁ ⊕ c.commit_delta           -- commits winner only
                             (a, δ) = c.strategy.next_action(gs, ps₂)
                         in  case a of
                                Nothing  → (fb_action, δ, Record(bail, c.strategy))
                                Just(α)  → if α ∉ gs.legal_actions
                                           then (fb_action, ε,
                                                 Record(illegal_strategy_action, c.strategy, α))
                                           else (α, δ, Record(strategy, c.strategy, α))
```

The key point: every fresh strategy may propose a plan, but only the winning fresh strategy
becomes committed. Active plans compete directly with fresh proposals. A losing fresh
recognizer does not pollute `active_plans`.

### What the algebra forces in implementation

A handful of decisions that look like preferences are actually forced:

| design "choice" | actually forced by | what would break |
|---|---|---|
| Strategies stored as `dict[name → Strategy]`, not `list`; duplicate names rejected | A1 + L3 | silent dedupe hides implementation mistakes |
| Arbitration = `max(priority)` with canonical `strategy.name` tie-break, not weighted vote | A2 | weighted vote or order-dependent ties break commutativity (L2) |
| Priorities are finite and non-NaN | A2 | ordering laws fail |
| Recognize fresh plans separately from active execution | A6 + L4 | losing recognizers can become persistent commitments |
| Commit only the winning fresh strategy | A6 + L4 | `active_plans` fills with strategies that never won control |
| Facts must be monoid-valued | A7 + L7 + L8 | merge ordering becomes load-bearing |
| Per-hand PlanState, not global | A3 | parallel hands need locks |
| Recording and fallback dry-runs are pure-write / pure-read | A4 | recording could secretly affect play |
| Strategies hermetic in namespace | A9 + L9 | adding a strategy could break others |
| Bail preserves active plan unless explicit retire delta is emitted | L6 | a last-second fallback silently destroys plan state |
| Illegal strategy actions fall back with contract-violation records | core operation | silent fallback hides broken strategies; playing illegal actions corrupts runs |
| Hierarchical sub-player wraps as explicit Strategy in an acyclic graph | A8 | implicit wrapper equivalence reintroduces parent-priority and fallback artifacts |

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
- **Bead in progress:** `t42-zrf9` tracks Phase 1 build (framework + first strategy +
  measurement harness + recording). The wiki contract supersedes older bead wording when
  the bead description still uses `applies_to` or says all fresh recognitions commit.
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
