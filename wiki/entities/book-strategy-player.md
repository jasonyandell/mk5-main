---
title: BookStrategyPlayer — amended Phase 1 design
kind: entity
first_seen: local-2026-05-03
last_updated: local-2026-05-04
status: active
phase: amended design (Phase 1 build-ready)
supersedes: BookStrategyPlayer — multi-step strategy framework with recording
---

## What it is

BookStrategyPlayer is a player architecture that plays through a library of named
multi-step **strategies** rather than through a single per-decision utility. Each
strategy is a small pure object with an explicit lifecycle:

```text
recognition -> commitment -> execution -> completion/retirement
```

Strategies that do not apply at a decision fall back to the established strongest
fixed-utility baseline: [[w42-lens-v1-utility-head-to-head|Lens]] with EV utility.
The baseline identity is load-bearing:

```python
BookStrategyPlayer(strategies=[], fallback=Lens("ev")) == Lens("ev")
```

The framework was motivated by [[w42-book-claim-synthesis-and-ai-directions]]'s
single-decision blind spot: many book claims are multi-step plans, not one-domino
utility preferences. A planner that can express plans is the natural abstraction
level for testing the book.

Phase 1 is not primarily an attempt to beat EV immediately. Its value is that it
creates a measurement instrument: named tactics become executable plans; each
decision records strategy context; the fallback's local action is logged as a
cheap counterfactual proposal; exact contribution can later be measured with
paired-seed replay; and the same records become training data for strategy
selection and plan-success prediction.

## Critical amendments

The amended Phase 1 contract folds in the review corrections that protect the
algebraic laws and prevent measurement leakage:

1. Split fresh-plan recognition from active-plan execution.
2. Commit only the winning fresh strategy, not every strategy that recognizes an
   opportunity.
3. Require shared `facts` to be monoid-valued wrappers; reject naked scalar facts.
4. Treat `fallback_action` as a free local counterfactual proposal, not exact
   causal attribution.
5. Ensure fallback dry-runs are pure and cannot consume live RNG or mutate state.
6. Reject duplicate strategy names at library construction.
7. Reject `NaN` or non-finite priorities.
8. Weaken hierarchy claims unless wrapper semantics are explicit.
9. Define illegal strategy-action handling.
10. Add replay, version, and fingerprint fields to `DecisionRecord`.

The central semantic correction is: a fresh strategy may **recognize** an
opportunity, but only the arbitration winner may **commit** a new plan instance
into `active_plans`. Losing fresh recognizers do not become active merely because
they saw the same opportunity as the winner.

## Strategy

A strategy can recognize a fresh plan opportunity, optionally commit a plan
instance, produce one action at a time, and eventually retire.

```python
class Strategy(Protocol):
    name: str
    version: str

    def recognizes(self, gs: GameState, ps: PlanState) -> bool:
        """True when a fresh plan instance may be proposed."""
        ...

    def priority(self, gs: GameState, ps: PlanState) -> float:
        """Priority under the supplied state. Must be finite and non-NaN."""
        ...

    def commit_recognition(self, gs: GameState, ps: PlanState) -> DeltaPlanState:
        """
        Pure proposal for creating this strategy's private plan state.
        Called for fresh candidates during arbitration, but only the winner's
        delta is applied to the live PlanState.
        """
        ...

    def next_action(
        self,
        gs: GameState,
        ps: PlanState,
    ) -> tuple[Action | None, DeltaPlanState]:
        """
        Returns either a legal action or None to bail.
        A returned delta may update private plan state or shared facts.
        """
        ...

    def plan_done(self, gs: GameState, ps: PlanState) -> bool:
        """True when an active plan should be retired before arbitration."""
        ...

    def observe(
        self,
        opp_action: Action,
        opp_seat: Seat,
        gs: GameState,
        ps: PlanState,
    ) -> DeltaPlanState:
        ...
```

The old `applies_to` concept splits into fresh recognition and active-plan
continuation. A strategy does not need to newly recognize a situation in order
to continue a plan that is already active. Active plans are arbitration
candidates by virtue of being active unless retired or explicitly blocked.

## Plan state

`PlanState` is per-hand state threaded through decisions.

```python
@dataclass(frozen=True)
class PlanState:
    active_plans: dict[str, dict]
    facts: dict[str, FactValue]
```

`active_plans` is private, namespaced strategy state:

```python
ps.active_plans[strategy.name]
```

A strategy may read and write only its own private plan state. It may not read or
write another strategy's private state. Strategies communicate through shared
`facts` only.

Strategies do not mutate `PlanState` directly. They return deltas:

```python
@dataclass(frozen=True)
class DeltaPlanState:
    active_plans: dict[str, PlanPatch]
    facts: dict[str, FactValue]
```

Plan patches are namespace-keyed:

```text
dict       overwrite/update this strategy's private plan state
None       retire/delete this strategy's private plan state
absent key no change
```

The framework applies deltas. Strategies declare changes; they do not mutate the
state in place.

## Fact algebra

Shared facts must be lawful, commutative monoid values. Naked scalars are
rejected.

Bad:

```python
facts["target_suit"] = "sixes"
facts["active_context"] = "setter_defense"
facts["plan_phase"] = 2
```

Good:

```python
facts["void_suits"] = SetFact({"sixes"})
facts["opponent_lead_count"] = CounterFact({("seat_2", "trump"): 1})
facts["best_known_bid_strength"] = MaxFact(17)
facts["active_contexts"] = SetFact({("Chapter5_SetterDefense", "void_creation")})
facts["target_suit"] = UniqueFact("sixes")
```

Minimum Phase 1 fact wrappers:

```python
class FactValue(Protocol):
    def merge(self, other: Self) -> Self: ...

@dataclass(frozen=True)
class SetFact[T]:
    value: frozenset[T]
    def merge(self, other): return SetFact(self.value | other.value)

@dataclass(frozen=True)
class CounterFact[T]:
    value: Counter[T]
    def merge(self, other): return CounterFact(self.value + other.value)

@dataclass(frozen=True)
class MaxFact[T]:
    value: T
    def merge(self, other): return MaxFact(max(self.value, other.value))

@dataclass(frozen=True)
class MinFact[T]:
    value: T
    def merge(self, other): return MinFact(min(self.value, other.value))

@dataclass(frozen=True)
class BoolOrFact:
    value: bool
    def merge(self, other): return BoolOrFact(self.value or other.value)

@dataclass(frozen=True)
class BoolAndFact:
    value: bool
    def merge(self, other): return BoolAndFact(self.value and other.value)

@dataclass(frozen=True)
class UniqueFact[T]:
    value: T
    def merge(self, other):
        if self.value != other.value:
            raise FactConflictError(self.value, other.value)
        return self
```

`UniqueFact` is for scalar-like facts where conflicting values are contract
violations. If multiple values are valid, use `SetFact`. Timestamped or
priority-based scalar facts need explicit canonical tie-breaking, for example
`MaxByKeyFact`; otherwise merge order becomes behaviorally load-bearing and
violates L7/L8.

## Player loop

Each decision proceeds through five phases:

1. **Retire completed/dead plans** — call `plan_done` on active strategies and
   apply retire deltas.
2. **Propose candidates** — active strategies are candidates automatically;
   fresh strategies may recognize and propose a commit delta.
3. **Arbitrate** — choose highest-priority candidate with canonical
   `strategy.name` tie-break.
4. **Commit winning fresh plan, if any** — apply only the winner's
   `commit_recognition` delta.
5. **Act or bail** — call `next_action`; if it returns `None` or an illegal
   action, use fallback according to the contract and log the event.

Corrected core operation:

```python
def choose_action(gs: GameState, ps: PlanState) -> tuple[Action, PlanState, DecisionRecord]:
    ps1 = retire_done(ps, library, gs)

    active = {
        s for s in library
        if s.name in ps1.active_plans
    }

    fresh = {
        s for s in library
        if s.name not in ps1.active_plans
        and s.recognizes(gs, ps1)
    }

    candidates: list[Candidate] = []

    for s in active:
        p = checked_priority(s, gs, ps1)
        candidates.append(Candidate(
            strategy=s,
            priority=p,
            commit_delta=DeltaPlanState.identity(),
            committed_state=ps1,
            is_fresh=False,
        ))

    for s in fresh:
        delta = s.commit_recognition(gs, ps1)
        validate_delta_namespace(s, delta)
        proposed_ps = ps1.apply(delta)
        p = checked_priority(s, gs, proposed_ps)
        candidates.append(Candidate(
            strategy=s,
            priority=p,
            commit_delta=delta,
            committed_state=proposed_ps,
            is_fresh=True,
        ))

    fallback_action = fallback.pure_choose_action(gs)
    validate_legal_or_fail_fallback(fallback_action, gs.legal_actions)

    winner = argmax_priority_then_name(candidates)

    if winner is None:
        return fallback_action, ps1, record_fallback(gs, ps1, candidates, fallback_action)

    ps2 = ps1.apply(winner.commit_delta)
    action, delta = winner.strategy.next_action(gs, ps2)
    validate_delta_namespace(winner.strategy, delta)

    if action is None:
        ps3 = ps2.apply(delta)
        return fallback_action, ps3, record_bail(gs, ps2, winner, fallback_action, delta)

    if action not in gs.legal_actions:
        return fallback_action, ps2, record_illegal_strategy_action(
            gs=gs,
            ps=ps2,
            winner=winner,
            illegal_action=action,
            fallback_action=fallback_action,
        )

    ps3 = ps2.apply(delta)
    return action, ps3, record_strategy_action(
        gs=gs,
        ps=ps2,
        winner=winner,
        chosen_action=action,
        fallback_action=fallback_action,
        delta=delta,
    )
```

Fresh strategy deltas are speculative until arbitration is complete. Only the
winning fresh strategy's delta is committed to the live `PlanState`.

## Arbitration

Libraries are stored as dictionaries:

```python
library: dict[str, Strategy]
```

Construction rejects duplicate names. Silent deduplication is algebraically
tempting but operationally dangerous because a duplicate strategy name almost
always means an accidental collision or bug.

Arbitration is a deterministic max:

```text
max(candidates, key=(priority, canonical strategy.name tie-break))
```

The exact tie-break direction does not matter. It only must be deterministic and
independent of library insertion order.

Priority validation is mandatory:

```python
def checked_priority(s, gs, ps) -> float:
    p = s.priority(gs, ps)
    if not math.isfinite(p):
        raise InvalidPriorityError(s.name, p)
    return p
```

By convention, fresh recognition priority is `1.0-5.0` and active continuation
priority is `>= 10.0`. The formal law is narrower: active-plan continuation
priority must dominate fresh recognition priority among currently eligible
candidates unless the active strategy declares itself done, blocked, bailed, or
explicitly retires. It is not compared against irrelevant inactive strategies.

## Bail, retire, blocked, disrupted

These states remain distinct.

**Bail** means `next_action(...) -> (None, delta)`: the strategy recognized or
continued a plan but declines to choose an action at this decision. Bail does not
automatically remove the plan from `active_plans`. To retire while bailing, the
strategy must explicitly emit:

```python
DeltaPlanState(active_plans={strategy.name: None}, facts={})
```

**Retire** means the plan instance is no longer active. A strategy retires when
`plan_done(gs, ps)` is true or when it emits
`delta.active_plans[strategy.name] = None`. A strategy may be recognized again
later unless it writes a tombstone fact such as:

```python
facts["strategy_tombstones"] = SetFact({("SingletonLeadToVoid", hand_id)})
```

**Blocked** means a plan remains active but cannot currently act. A blocked plan
may return very low active priority, return `None` and keep private state,
publish a `blocked_plans` fact, or retire if permanently infeasible.

**Disrupted** means an external event invalidates the expected path. A disrupted
plan usually emits a retire delta and records completion status as `disrupted`.
Disruption is a diagnostic status, not a separate framework mechanism.

## Fallback purity

Recording must not change play. Because every `DecisionRecord` includes the
fallback action, the framework may call the fallback even when a strategy wins.
That is only safe if fallback evaluation is observationally pure:

```python
fallback.pure_choose_action(gs)
```

The pure fallback path must not mutate fallback/player state, game state, plan
state, or the recording buffer; it must not consume the live RNG stream or
advance a forge/world-sampling stream; and it must not depend on whether
recording is enabled.

If fallback needs randomness or world sampling, the framework uses a cloned
deterministic decision context or logs and restores the stream position:

```python
fallback_action = fallback.choose_action(gs, rng=decision_rng.clone())
```

This is required for the Writer-law claim that recording-on and recording-off
are behaviorally identical.

## Recording

Per-decision records are logged structurally.

```python
@dataclass(frozen=True)
class DecisionRecord:
    schema_version: str
    framework_version: str
    player_id: str
    player_config_hash: str
    library_fingerprint: str
    strategy_versions: dict[str, str]
    fallback_fingerprint: str
    forge_version: str
    rules_version: str

    match_id: str | None
    hand_id: str
    hand_seed: int | str
    initial_deal_id: str | None
    decision_rng_seed: int | str | None
    decision_rng_stream_position: int | None
    forge_oracle_seed: int | str | None
    seat_id: Seat
    partnership_config: dict

    trick_idx: int
    decision_idx: int
    absolute_decision_idx: int

    game_state_snapshot: bytes | dict
    game_state_tensor: bytes | list | None
    plan_state_snapshot: dict

    strategy_path: tuple[str, ...]

    legal_actions: tuple[Action, ...]
    applicable_strategy_names: tuple[str, ...]
    active_strategy_names: tuple[str, ...]
    fresh_strategy_names: tuple[str, ...]
    strategy_priorities: dict[str, float]
    chosen_strategy: str | None
    chosen_action: Action
    fallback_action: Action
    strategy_returned_action: Action | None

    decision_status: Literal[
        "fallback",
        "strategy_action",
        "bailed_to_fallback",
        "illegal_strategy_action_fallback",
        "fallback_illegal_error",
    ]

    applied_delta_summary: dict
    immediate_trick_outcome: dict | None
    hand_point_margin: float | None
    plan_completion_status: dict[str, Literal[
        "active",
        "completed",
        "disrupted",
        "abandoned",
        "bailed",
        "illegal_action",
    ]]
```

`game_state_tensor` is a model input and is not assumed replay-lossless.
`game_state_snapshot` is the replay input. If the tensor is actually lossless,
both fields may point to the same serialized payload; otherwise they stay
separate.

`fallback_action` is a free local counterfactual proposal, not exact causal
effect. Exact causal contribution requires paired-seed replay because one changed
domino can affect future legal actions, information, plan applicability,
opponent behavior, partner behavior, trick outcomes, and hand margin. If a
strategy diverges five times in one hand, the final hand margin cannot be
assigned independently to all five decisions without replay or a value model.

Conservative coverage buckets before replay:

- `uncovered` — no strategy candidate won; fallback played.
- `covered_same_as_fallback` — strategy won but chose the same action as fallback.
- `covered_diff_unattributed` — strategy won and diverged from fallback, but no
  replay/value attribution has run.
- `covered_diff_replay_positive` — paired replay estimates positive causal effect.
- `covered_diff_replay_negative` — paired replay estimates negative causal effect.
- `bailed` — strategy won arbitration but returned `None` and fallback played.
- `illegal_strategy_action` — strategy returned an illegal action; fallback played
  or test failed.
- `disrupted` — committed plan failed to complete because the expected path became
  infeasible.

The older `covered_diff_positive` and `covered_diff_negative` names are valid
only after replay or model-based value attribution.

## Composition modes

**State-conditioned strategy selection** is built in because `priority(gs, ps)`
takes game state and plan state. `TrumpPulling.priority` can be high early and
fade; `ThrowawayLadder.priority` can rise late. This connects to
[[w42|t42-nwuu]]: state-conditioned utility lifted from utilities to named
strategies.

**Strategy chaining through facts** lets one strategy publish a fact and another
recognize based on it. `SingletonLeadToVoid` can publish `facts.void_suits`;
`CountSteeringIntoVoid` can read it. Context facts must be monoid-valued, such
as `SetFact({("Chapter5_SetterDefense", "void_creation")})`. Plan-local context
stays in private plan state.

**Hierarchical strategies** wrap a `BookStrategyPlayer` as a `Strategy`, but the
wrapper semantics must be explicit. A chapter wrapper usually returns `None`
when no inner strategy applies instead of consuming control by playing its own
fallback. Otherwise a high-priority chapter wrapper can block other parent
strategies while merely playing fallback. Allowed hierarchy is finite, acyclic,
path-prefixed in `PlanState`, and recorded via `strategy_path`.

**Opponent-aware counter-strategies** use the optional `observe` hook to update
facts such as counters, sets, and maxima. Counter-strategies recognize when facts
cross thresholds. This is a small framework extension but not required for the
first strategy.

**Cross-hand match memory** is not Phase 1. Lifting state to match level requires
`MatchState` threaded through lifecycle methods, while the current parallel-hand
simulator treats hands independently.

## The book as meta-strategy

The book's chapter structure is naturally represented as state-conditioned
selection plus hierarchy: chapters partition state space, each chapter owns a
sub-library of tactics, chapter wrappers expose lawful parent-level priorities,
and nested decisions record a `strategy_path`.

```python
TheBook = BookStrategyPlayer(
    strategies=[
        Chapter3_BidderPlay(),
        Chapter4_PartnerSupport(),
        Chapter5_SetterDefense(),
        Chapter8_AllTricks84(),
        Chapter12_HighBidPlay(),
    ],
    fallback=Lens("ev"),
)
```

`Lens(ev) vs TheBook` is a single head-to-head measurement of how much the
encoded book is worth in points per hand, subject to the quality and coverage of
the strategy encodings. This is the correct abstraction level for testing
multi-step book claims.

## Recording to training pipeline

The same records that support measurement become training data.

**Model A — strategy selector** can eventually replace hand-coded priority.
Inputs are game-state encoding, plan-state encoding, facts, active/fresh mask,
and strategy metadata. Outputs are expected hand margin per candidate strategy
or logits over candidate strategies. Labels are not free for every strategy:
each decision observes only chosen strategy outcome and fallback local action.
Other candidate outcomes need replay or value-model estimation. The cheap
version imitates hand-crafted priorities; the strong version trains Q-values
from paired-seed replays; the mature version mixes replay labels, value-model
estimates, and uncertainty penalties.

**Model B — plan-success predictor** is likely the first learned model that pays
rent. It predicts plan completion, disruption, bail, hand margin conditioned on
choice or completion, and number of decisions consumed. It can provide early
bail signals, priority calibration, selector features, and strategy debugging.

**Model C — end-to-end policy distillation** predicts raw actions directly. In
that case, strategies become training scaffolding. If the distilled policy
consistently selects actions in regions where no encoded strategy applies, that
region becomes a candidate for new strategy discovery.

## Parked model roles

[[gus]] is the natural belief-state encoder for Model A and Model B. It turns
partial observability into a clean belief tensor consumed by the strategy
selector or plan-success head.

[[burl]] becomes plausible when the action space is named strategies rather than
raw dominoes. The question becomes "which named strategy should fire?" over a
small 15-30 strategy space with narrative rationales. Burl is gated behind
Phase 2 evidence; first prove that strategy labels and plan records are
predictive.

[[zeb]] can serve as a training-data generator, backup Model C target, or policy
head trained over strategy labels instead of raw action labels. The structured
output space is the important change.

## Why this is the cleanest path past EV

[[w42-lens-v1-utility-head-to-head]] established that EV is the ceiling for
fixed pointwise utilities at one-step lookahead.

| path | status |
|---|---|
| Multi-utility heads | dead — Lens v1 killed it |
| Soft-cliff utilities | dead — disaster confirmed EV ceiling for fixed pointwise tweaks |
| Generic MCTS | expensive build; does not directly encode book wisdom |
| Lookahead-Lens K=2 | impractical branching over multi-player plies |
| Strategy-selector from book-strategy gameplay | uses book wisdom as structured macro-action space |

The key move is changing the action abstraction from "pick a domino by pointwise
utility" to "choose among named multi-step plans, then execute one legal domino."
That directly attacks the single-decision blind spot.

## Phased build plan

**Phase 1 — framework and measurement instrument.** Deliver the strategy
framework, `PlanState`, `DeltaPlanState`, monoid-valued facts, duplicate-name
rejection, deterministic arbitration, pure Lens(ev) fallback adapter,
in-memory `DecisionRecord`, parquet only after schema stabilization,
property-based tests for L1-L10, a head-to-head harness reusing
`w42/lens_v1/parallel_match.py`, and one to three starter strategies.

Starter strategy options:

- `singleton_lead_to_void` — valid first measurement/negative-control candidate;
  do not assume it rehabilitates [[w42-bookval-v1-wave2-void-creation]].
- canonical follow-position void variant — cleaner book-positive candidate if
  the first strategy is meant to test the strongest book version.

Recommended order: framework contracts and fake strategies, property tests,
fallback adapter, recording, one real strategy, then head-to-head measurement.
Do not start with the book tactic before the framework laws are tested.

**Phase 2 — learned selector after coverage.** Gate on 5-10 strategies, about
50K labeled decisions, and basic replay diagnostics. Deliver Model B, initial
Model A, learned-selector vs hand-coded-priority A/B test, and a decision about
Burl-as-selector. Model B may be the first real learned win.

**Phase 3 — strategy discovery frontier.** Gate on Phase 2 selector lift,
high-regret uncovered regions, and a working paired replay pipeline. Cluster
fallback-invoked or uncovered decisions, identify high-tail-risk regions,
propose candidate strategies, and distill an end-to-end policy only if strategy
lift is real.

## Algebraic specification

The framework is determined by nine algebras and ten laws. The amendments keep
the algebra but tighten runtime semantics.

**A1. `Library` is a commutative idempotent monoid at the semantic level.**

```text
Library    = finite map StrategyName -> Strategy
identity   = empty map
operation  = union if names unique
```

Semantic set behavior remains. Implementation stores `dict[name -> Strategy]`
and rejects duplicate names.

**A2. `Arbitration` is a bounded join-semilattice.**

```text
carrier = Candidate union {bottom}
join    = argmax(priority, canonical strategy.name tie-break)
bottom  = no candidate
```

The laws are associative, commutative, and idempotent. Priorities must be finite
and non-NaN.

**A3. `PlanState` evolves under the State monad, per hand.**

```text
HandComputation a = State PlanState a
```

Each decision is pure: `(GameState, PlanState) -> (Action, PlanState')`.
PlanState is per-hand, immutable-or-copy-on-write, and rebound after deltas.

**A4. `Recording` is the Writer monad.**

```text
type      = WriterT [DecisionRecord] Identity
operation = tell per decision
```

Records accumulate and never read back into decision behavior. Computing
`fallback_action` for recording must be pure.

**A5. `Fallback` and `Library` form the Reader environment.**

```text
PlayerEnv = (Library, Fallback)
type      = Reader PlayerEnv
```

The player object is immutable after construction.

**A6. `Strategy` is a Sigma-algebra.**

```text
recognizes          : (GS, PS) -> Bool
priority            : (GS, PS) -> RealFinite
commit_recognition  : (GS, PS) -> DeltaPS
next_action         : (GS, PS) -> (Maybe Action, DeltaPS)
plan_done           : (GS, PS) -> Bool
observe optional    : (Action, Seat, GS, PS) -> DeltaPS
```

The framework never inspects strategy internals.

**A7. `DeltaPlanState` is a partial monoid; facts are monoid-valued.**

```text
DeltaPlanState = (Delta active_plans, Delta facts)
active_plans   = Map StrategyName (Maybe PlanPatch)
facts          = Map FactName FactValue
```

Fact merge is pointwise monoid merge. Plan merge is namespace-keyed
overwrite/delete. The facts API rejects naked scalars.

**A8. Hierarchical composition is a constrained wrapper.**

A `BookStrategyPlayer` can wrap as a strategy only if internal `PlanState` is
path-namespaced, wrapper priority is lawful at the parent level, the wrapper
returns `None` when no inner strategy applies unless explicitly terminal, the
strategy graph is finite and acyclic, and records carry `strategy_path`.

**A9. Full player is a composed monad stack.**

```text
PlayerM = ReaderT (Library, Fallback)
          (StateT PlanState
          (WriterT [DecisionRecord] Identity))
```

No IO is required in decision logic. The entire decision is testable as a pure
function from env, game state, and plan state to action, updated plan state, and
records.

## Laws

**L1 — Empty library identity.**

```text
BSP(empty, fb).choose_action(gs) == fb.choose_action(gs)
```

Adding strategies strictly extends the fallback baseline.

**L2 — Library order independence.**

```text
BSP({s1, s2}, fb).choose_action(gs) == BSP({s2, s1}, fb).choose_action(gs)
```

This falls out of canonical arbitration, not Python insertion order.

**L3 — Duplicate strategy names rejected.**

The semantic idempotence law says `BSP({s, s}, fb) == BSP({s}, fb)`.
The implementation law says constructing a library with duplicate
`Strategy.name` raises `DuplicateStrategyNameError`.

**L4 — Do not re-recognize active plans.**

If `s.name in ps.active_plans`, `s.recognizes` may be ignored and
`commit_recognition` is not called for a fresh instance. Active plans continue
through active-plan arbitration. A strategy may recognize a new instance only
after retirement unless a tombstone fact prevents it.

**L5 — Active continuation dominates fresh recognition.**

For active applicable strategies, continuation priority should exceed fresh
recognition priority among currently eligible candidates unless the active
strategy declares itself done, blocked, bailed, or retires.

**L6 — Bail and retire are orthogonal.**

Bail means `next_action` returns `None`; the plan remains active unless the delta
explicitly retires it. Retire means `plan_done` returns true or
`delta.active_plans[s.name] = None`; the plan is removed from `active_plans`.

**L7 — Fact merge is associative.**

```text
merge(merge(a, b), c) == merge(a, merge(b, c))
```

**L8 — Fact merge is commutative.**

```text
merge(a, b) == merge(b, a)
```

L7 and L8 fall out of monoid-valued fact wrappers.

**L9 — Strategy namespace hermeticity.**

A strategy may read and write only `ps.active_plans[s.name]` and `ps.facts.*`.
It may not read or write `ps.active_plans[s_prime.name]` for any other strategy.
Inter-strategy communication goes through shared facts only.

**L10 — Recording fidelity.**

```text
Records(hand) + forge oracle/seeds + player/library/fallback fingerprints
  entails Replay(hand)
```

Records must contain player identity, framework version, library fingerprint,
strategy versions, fallback fingerprint, forge/rules versions, hand seed,
decision RNG seed or stream position, game-state snapshot, plan-state snapshot,
legal actions, chosen action, fallback action, strategy path, and
deltas/statuses. Records are training data; if they are not replayable, training
labels are suspect.

## Implementation consequences

| Design choice | Forced by | Failure if ignored |
|---|---|---|
| Store strategies as `dict[name -> Strategy]` | A1/L2/L3 | order and duplicate bugs |
| Reject duplicate names | L3 amended | silent collision hides bugs |
| Deterministic `max(priority, name)` arbitration | A2/L2 | order-dependent behavior |
| Reject NaN/non-finite priority | A2 | ordering laws break |
| Commit only winning fresh strategy | recognition/commit distinction | losing strategies become active accidentally |
| Active plans eligible without fresh recognition | lifecycle semantics | active plans can strand incorrectly |
| Facts use monoid wrappers | A7/L7/L8 | merge order becomes load-bearing |
| Per-hand immutable PlanState | A3 | parallel simulation needs locks |
| Recording is pure-write | A4 | records can affect behavior |
| Fallback dry-run is pure | A4/L10 | logging `fallback_action` changes play |
| Strategy namespace hermeticity | A9/L9 | adding a strategy can break another |
| Bail preserves active plan unless explicit retire | L6 | last-second fallback silently destroys plan state |
| Illegal action handling is explicit | strategy contract | corrupt simulations or hidden fallback |
| Hierarchy is acyclic and path-namespaced | A8 | cycles/collisions/ambiguous records |

## Property tests for Phase 1

Use fake strategies and fake game states before implementing real W42 tactics.
Minimum tests:

- T1: Empty library matches fallback.
- T2: Library order does not affect chosen action.
- T3: Duplicate strategy names raise at construction.
- T4: Active strategy is not re-committed.
- T5: Fresh losing strategy is not committed.
- T6: Active continuation beats fresh recognition under priority convention.
- T7: Bail preserves active plan unless retire delta emitted.
- T8: Retire removes active plan.
- T9: Fact merge associative for all fact wrappers.
- T10: Fact merge commutative for all commutative fact wrappers.
- T11: Naked scalar facts rejected.
- T12: Strategy cannot write another strategy's namespace.
- T13: NaN priority rejected.
- T14: Illegal strategy action records/falls back or raises in strict mode.
- T15: Recording-on/off produce identical chosen actions.
- T16: Fallback dry-run does not consume live RNG.
- T17: DecisionRecord contains replay-required fields.

Property tests should exercise arbitrary library orderings, duplicate insertion
attempts, strategy bails, retire deltas, and fact merge permutations.

## Suggested module layout

One possible Phase 1 layout:

```text
w42/book_strategy/
    __init__.py
    player.py              # BookStrategyPlayer, arbitration loop
    strategy.py            # Strategy Protocol, Candidate
    plan_state.py          # PlanState, DeltaPlanState
    facts.py               # FactValue wrappers
    recording.py           # DecisionRecord, serializers
    fallback.py            # LensFallbackAdapter
    hierarchy.py           # optional wrapper utilities
    errors.py              # contract errors
    tests/
        test_laws.py
        test_facts.py
        test_recording_purity.py
        test_arbitration.py
    strategies/
        __init__.py
        singleton_lead_to_void.py
        follow_position_void.py
```

Strict/test mode: illegal strategy action, namespace violation, NaN priority, and
fallback illegal action raise. Long simulation mode may record an illegal
strategy action and fall back, but namespace violations, NaN priorities, and
illegal fallback actions remain hard failures.

## Composition limits

The framework cannot overcome hard game constraints.

**One domino per decision.** If two strategies want different dominoes,
arbitration picks one. The other plan may become infeasible. The framework can
detect and log conflicts but cannot play two actions.

**No automatic plan negotiation.** Two active plans can rest on contradictory
predictions of opponent behavior. The framework does not reason about plan
compatibility unless a strategy explicitly encodes that reasoning.

**Partnership coordination via assumed mirror only.** A strategy requiring
partner coordination can be tested by running the same player in the partner
seat. This is asymmetric self-play, and the simulator already supports it.

## Final framing

BookStrategyPlayer is a pure, per-hand, strategy-arbitrating player that strictly
extends Lens(ev). Strategies propose named multi-step plans; arbitration chooses
one candidate per decision; `PlanState` carries private plan state and lawful
shared facts; `DecisionRecord` logs enough information for replay, diagnostics,
and later selector training.

The Phase 1 goal is to build the measurement instrument that makes book tactics
executable, comparable, replayable, and trainable. The correct claim is that the
framework changes the decision abstraction from one-step utility selection to
named multi-step plan selection. Logged fallback actions provide cheap local
baselines and enable paired replay. The local fallback difference is not
automatically exact strategy contribution.

## Status

- **Designed:** 2026-05-03 in conversation with the orchestrator.
- **Amended:** 2026-05-04 after lifecycle, algebra, and recording review.
- **Built:** not yet.
- **Strategies encoded:** none yet.
- **Phase 1 readiness:** build-ready with amended semantics.
- **Bead in progress:** `t42-zrf9` tracks the framework, first strategy,
  measurement harness, and recording.

## Links

- [[w42-book-claim-synthesis-and-ai-directions]] — methodology insight that
  motivated this design.
- [[w42-book-validation-campaign]] — campaign this serves.
- [[w42-lens-v1-utility-head-to-head]] — EV-as-ceiling result constraining the
  design space.
- [[w42-bookval-v3-utility-argmax-divergence]] — Wave 4.0 measurement that gated
  the architecture branch.
- [[w42-bookval-v1-wave2-void-creation]] — prior void-creation result to avoid
  overclaiming against.
- [[gus]] — input encoder candidate.
- [[burl]] — possible Model A strategy selector.
- [[zeb]] — training-data generator / Model C target.
- [[forge]] — simulator the player runs against.
