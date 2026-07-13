---
title: BookStrategyPlayer - strategy algebra
kind: entity
first_seen: 2026-05-03
last_updated: 2026-07-13
status: superseded
phase: algebraic Phase 1 contract (never built)
supersedes: BookStrategyPlayer - amended Phase 1 design
---

## What it is

**Superseded before build.** BookStrategyPlayer was the W42 player
architecture proposed for testing book advice that is plan-shaped instead of
one-domino-shaped. The chassis was redirected from play-time strategy to the
auction by Fable's 2026-06-09 design review (see [[champion-design-review]]);
this Phase 1 design was never implemented — bead `t42-zrf9` froze
`in_progress` on 2026-05-04 and no `w42/book_strategy*` directory ever
appeared on disk. The project's best-player energy moved instead to the
auction-first [[jud]] value-net line, an unrelated mechanism.
BSP may still run someday as a cheap epilogue measurement, not a live plan —
the algebra below is preserved as a design record, not a build in progress.

It was to play through a finite library
of named strategies, each with this lifecycle:

```text
recognize -> commit -> execute -> retire
```

When no strategy controls a decision, it falls back to the strongest fixed
one-step utility baseline from [[w42-lens-v1-utility-head-to-head]]:

```python
Lens("ev")
```

The baseline identity is load-bearing:

```python
BookStrategyPlayer(strategies=[], fallback=Lens("ev")) == Lens("ev")
```

The point is not that Phase 1 should immediately beat EV. The point is that
[[w42-book-claim-synthesis-and-ai-directions]] identified a single-decision
blind spot: many [[w42]] book claims are multi-step plans, while earlier probes
mostly measured isolated actions. BookStrategyPlayer changes the action
abstraction from "choose the best domino by pointwise utility" to "choose a
named plan, then let that plan choose one legal domino."

## Core equation

The implementer-facing spec is this pure function:

```text
Env + GameState + PlanState -> Action + PlanState' + DecisionRecord
```

Where:

```text
Env       = Library + Fallback
Library   = finite map StrategyName -> Strategy
Fallback  = pure action proposal, normally Lens("ev")
PlanState = active private plans + shared monoid facts
Record    = pure Writer output; never read by decision behavior
```

Equivalently:

```text
PlayerM = ReaderT Env
          (StateT PlanState
          (WriterT [DecisionRecord] Identity))
```

No IO belongs in the decision logic. A decision may query only its supplied
environment, game state, and plan state. It returns an action, a new plan state,
and an append-only record.

The recording contract, Phase 1 build contract, and extension points were
satellite pages, now reduced to stubs with their gists absorbed below
([[book-strategy-player-recording]], [[book-strategy-player-phase-1-build]],
[[book-strategy-player-extension-points]]; full text in git history).

## Carriers

**Library.** A library is a finite map from strategy name to strategy object.
Construction rejects duplicate names. Semantically, library union is a partial
commutative operation defined only when names are disjoint:

```text
Library = Map StrategyName Strategy
empty   = {}
union   = map union, undefined on duplicate keys
```

Rejecting duplicates is deliberate. Silent deduplication hides strategy-name
collisions and makes experiments hard to audit.

**Fallback.** The fallback is a pure local proposal:

```python
fallback_action = fallback.pure_choose_action(gs)
```

It must not mutate game state, player state, plan state, recording buffers, or
live RNG streams. Recording `fallback_action` is a local counterfactual proposal,
not exact causal attribution.

**PlanState.** `PlanState` is per-hand state.

```python
@dataclass(frozen=True)
class PlanState:
    active_plans: dict[str, dict]
    facts: dict[str, FactValue]
```

`active_plans[name]` is private state for the strategy named `name`. A strategy
may read and write only its own private plan state. Inter-strategy communication
goes through shared `facts`.

**DeltaPlanState.** Strategies do not mutate `PlanState`; they return deltas.

```python
@dataclass(frozen=True)
class DeltaPlanState:
    active_plans: dict[str, PlanPatch]
    facts: dict[str, FactValue]
```

Plan patches are namespace-local state operations:

```text
dict       set or update this strategy's private state
None       retire/delete this strategy's private state
absent key no private-state change
```

Fact patches merge pointwise through `FactValue.merge`. Facts must be lawful,
commutative monoid values. Naked scalar facts are rejected.

Minimum fact wrappers:

```python
SetFact(frozenset(...))        # union
CounterFact(Counter(...))      # addition
MaxFact(value)                 # max
MinFact(value)                 # min
BoolOrFact(value)              # or
BoolAndFact(value)             # and
UniqueFact(value)              # equal values only, else conflict
```

Use `UniqueFact` only when disagreement is a contract violation. If several
values can be true at once, use `SetFact`. Timestamped or priority-based scalar
facts need an explicit canonical tie-breaker such as `MaxByKeyFact`; otherwise
merge order becomes behaviorally load-bearing.

**Candidate.** A candidate is an active or freshly recognized strategy prepared
for arbitration:

```python
@dataclass(frozen=True)
class Candidate:
    strategy: Strategy
    priority: float
    commit_delta: DeltaPlanState
    committed_state: PlanState
    is_fresh: bool
```

Fresh candidates carry speculative `commit_delta`s. Those deltas are not applied
to the live `PlanState` unless the candidate wins arbitration.

## Strategy

A strategy is a pure object with five required operations:

```python
class Strategy(Protocol):
    name: str
    version: str

    def recognizes(self, gs: GameState, ps: PlanState) -> bool:
        """True when a fresh plan instance may be proposed."""
        ...

    def priority(self, gs: GameState, ps: PlanState) -> float:
        """Finite, non-NaN priority under the supplied state."""
        ...

    def commit_recognition(self, gs: GameState, ps: PlanState) -> DeltaPlanState:
        """Pure proposal for creating this strategy's private plan state."""
        ...

    def next_action(
        self,
        gs: GameState,
        ps: PlanState,
    ) -> tuple[Action | None, DeltaPlanState]:
        """Return one legal action plus delta, or None to bail."""
        ...

    def plan_done(self, gs: GameState, ps: PlanState) -> bool:
        """True when an active plan should retire before arbitration."""
        ...
```

The old `applies_to` idea is intentionally split:

```text
recognizes(gs, ps)      fresh opportunity
name in active_plans    continuing committed plan
```

An active strategy does not need to re-recognize the current state. It is a
candidate because it is already committed.

## Decision algorithm

Each decision has five phases:

1. Retire completed active plans.
2. Build candidates from active plans and fresh recognizers.
3. Arbitrate by deterministic priority/name join.
4. Commit only the winning fresh candidate.
5. Ask the winner for an action; otherwise play fallback and record why.

Reference operation:

```python
def choose_action(gs: GameState, ps: PlanState) -> tuple[Action, PlanState, DecisionRecord]:
    ps1 = retire_done(ps, library, gs)

    candidates = []

    for s in library.values():
        if s.name in ps1.active_plans:
            candidates.append(active_candidate(s, gs, ps1))
        elif s.recognizes(gs, ps1):
            delta = s.commit_recognition(gs, ps1)
            validate_namespace(s, delta)
            proposed_ps = ps1.apply(delta)
            candidates.append(fresh_candidate(s, gs, proposed_ps, delta))

    fallback_action = fallback.pure_choose_action(gs)
    validate_legal_fallback(fallback_action, gs.legal_actions)

    winner = join_candidates(candidates)

    if winner is None:
        return fallback_action, ps1, record_fallback(...)

    ps2 = ps1.apply(winner.commit_delta)
    action, delta = winner.strategy.next_action(gs, ps2)
    validate_namespace(winner.strategy, delta)

    if action is None:
        ps3 = ps2.apply(delta)
        return fallback_action, ps3, record_bail(...)

    if action not in gs.legal_actions:
        return fallback_action, ps2, record_illegal_strategy_action(...)

    ps3 = ps2.apply(delta)
    return action, ps3, record_strategy_action(...)
```

The central correction is: fresh strategies may recognize opportunities, but
only the arbitration winner commits a new plan into `active_plans`.

## Arbitration

Arbitration is a bounded join over candidates:

```text
carrier = Candidate union {bottom}
join    = max(priority, canonical strategy.name tie-break)
bottom  = no candidate
```

Priority must be finite and non-NaN:

```python
def checked_priority(strategy, gs, ps) -> float:
    p = strategy.priority(gs, ps)
    if not math.isfinite(p):
        raise InvalidPriorityError(strategy.name, p)
    return p
```

The name tie-break direction is arbitrary, but it must be canonical and
independent of library insertion order. With finite priorities and canonical
names, the join is associative, commutative, and idempotent.

Library strategies should use this convention:

```text
fresh recognition priority: low
active continuation priority: high
```

That convention prevents accidental plan flipping. It is a strategy-library
contract, not a separate algebra.

## State outcomes

Four plan states remain distinct:

**Bail.** `next_action` returns `None`. The plan remains active unless the delta
explicitly retires it.

**Retire.** `plan_done(gs, ps)` returns true or the strategy emits
`active_plans[name] = None`. The plan is removed from `active_plans`.

**Blocked.** The plan remains active but cannot currently act. It may lower its
priority, return `None`, publish a fact, or retire if permanently infeasible.

**Disrupted.** An external event invalidates the expected path. This is recorded
as an outcome, usually alongside a retire delta. It is not a separate framework
mechanism.

Illegal strategy actions are explicit contract events. Strict/test mode raises.
Long simulation mode may record `illegal_strategy_action` and play fallback.
Namespace violations, invalid priorities, and illegal fallback actions are always
hard failures.

## Laws

**L1 - Empty library identity.**

```text
BSP(empty, fb).choose_action(gs, ps) == fb.choose_action(gs)
```

**L2 - Unique strategy names.** Constructing a library with duplicate
`Strategy.name` raises `DuplicateStrategyNameError`.

**L3 - Library order independence.** Reordering strategy construction inputs
does not change decisions or records except for irrelevant serialization order.

**L4 - Recognition/commit separation.** Fresh recognizers propose deltas; only
the winning fresh recognizer commits.

**L5 - Do not re-recognize active plans.** If `s.name in ps.active_plans`,
`recognizes` and `commit_recognition` are not used to create another instance of
that plan.

**L6 - Deterministic finite-priority arbitration.** Candidate choice is the
canonical join over `(priority, strategy.name)`; `NaN` and non-finite priorities
raise.

**L7 - Namespace hermeticity.** Strategy `s` may read and write only
`ps.active_plans[s.name]` and `ps.facts.*`. It may not inspect or patch another
strategy's private state.

**L8 - Fact merge laws.** Shared facts are monoid-valued wrappers. Their merges
are associative and commutative, or they raise an explicit conflict such as
`FactConflictError`.

**L9 - Bail and retire are orthogonal.** Bail does not retire a plan. Retire does
not require a bail. The plan leaves `active_plans` only through `plan_done` or an
explicit retire delta.

**L10 - Recording purity and replay sufficiency.** Recording is Writer-only and
cannot affect action choice. Records plus seeds, player/library/fallback
fingerprints, strategy versions, game-state snapshots, legal actions, chosen
actions, fallback proposals, strategy paths, and deltas must be enough to replay
the hand. See the recording contract below.

## Implementation surface

Phase 1 should be implemented in this order:

1. Core carriers and fake strategies.
2. Property tests for L1-L10.
3. Pure `Lens("ev")` fallback adapter.
4. Recording contract.
5. One real book strategy.
6. Head-to-head measurement against `Lens("ev")`.

Do not start with a book tactic before the framework laws pass on fake
strategies.

The first strategy can be a negative-control or measurement candidate such as
`singleton_lead_to_void`; it should not be assumed to rehabilitate
[[w42-bookval-v1-wave2-void-creation]] before measurement.

## Phase 1 build contract (absorbed)

Build order: core carriers with fake strategies → property tests for L1-L10
(seventeen named tests, T1-T17: empty-library identity, library-order
independence, duplicate-name rejection, commit/arbitration discipline,
bail/retire orthogonality, fact-merge associativity/commutativity,
naked-scalar rejection, namespace hermeticity, priority finiteness, recording
purity, replay-field presence) → pure `Lens("ev")` fallback adapter →
recording → one real strategy → head-to-head against `Lens("ev")`, reusing
`w42/lens_v1/parallel_match.py` where possible.

Error policy: strict/test mode raises on illegal strategy actions, namespace
violations, non-finite priorities, and illegal fallback actions; long
simulation mode may record-and-fallback for illegal strategy actions only.
Target module layout was `w42/book_strategy/` (never created). The first
head-to-head report was to include paired-seed point margin, bootstrap CI,
strategy firing/completion/disruption/bail rates, illegal-action count,
fallback fraction, and a sample of replayable decision records.

## Recording contract (absorbed)

Records are Writer-only, append-only `DecisionRecord`s carrying identity and
version fingerprints (player, library, fallback, framework, rules, forge,
per-strategy versions), seeds and RNG stream positions, lossless game-state
and plan-state snapshots, legal actions, strategy path, priorities, chosen and
fallback actions, applied deltas, and per-plan completion statuses. Replay
sufficiency is the acceptance test: if records cannot deterministically replay
the hand, downstream training labels are suspect.

`fallback_action` is a cheap local counterfactual proposal — a local baseline
that enables replay, never exact causal attribution (that needs paired-seed
replay or a value model). Computing it must not mutate any state or consume
live RNG. Coverage buckets stay conservative before replay (`uncovered`,
`covered_same_as_fallback`, `covered_diff_unattributed`, `bailed`,
`illegal_strategy_action`, `disrupted`); `covered_diff_replay_positive` /
`_negative` only after replay or model-based value attribution.

## Extension boundary

The core algebra admits extension without changing the decision equation:

- state-conditioned strategy selection through `priority(gs, ps)`;
- strategy chaining through lawful shared facts;
- finite acyclic hierarchy through wrapper strategies and `strategy_path`;
- opponent-aware facts through an optional observation hook;
- learned selectors or plan-success predictors after enough records exist.

Those are extension points, not Phase 1 obligations. Beyond them: cross-hand
match memory would need a `MatchState` threaded through the whole strategy
lifecycle (the design is per-hand because parallel-hand simulation treats
hands as independent), and the recorded data was to train a strategy selector,
a plan-success predictor, or an end-to-end distilled policy — [[gus]] as the
belief-state encoder, [[burl]] plausible once the action space is named
strategies, [[zeb]] as generator or backup policy target. Hard limits: one
domino per decision (arbitration picks a single plan), and no automatic
plan-compatibility negotiation between simultaneously active plans.

## Status

- **Designed:** 2026-05-03.
- **Amended:** 2026-05-04 after lifecycle, algebra, and recording review.
- **Refactored:** 2026-05-04 into compact algebra plus satellite pages.
- **Built:** never — bead `t42-zrf9` froze `in_progress` at 2026-05-04 and was
  never revisited; beads were retired project-wide 2026-06 without it closing.
- **Strategies encoded:** none.
- **Superseded:** 2026-06-09, by the auction-first [[jud]] line
  (see [[champion-design-review]]). May still run someday as a cheap epilogue
  measurement, not a live plan.

## Links

- [[w42-book-claim-synthesis-and-ai-directions]] - methodology insight that
  motivated this design.
- [[w42-book-validation-campaign]] - campaign this serves.
- [[w42-lens-v1-utility-head-to-head]] - EV-as-ceiling result constraining the
  design space.
- [[book-strategy-player-recording]] · [[book-strategy-player-phase-1-build]]
  · [[book-strategy-player-extension-points]] - satellite stubs; their
  contracts are absorbed above.
- [[w42-bookval-v1-wave2-void-creation]] - prior void-creation result to avoid
  overclaiming against.
- [[champion-design-review]] - Fable's 2026-06-09 review that redirected the
  chassis from play to the auction.
- [[jud]] - the mechanism the project's best-player energy
  moved to instead.
