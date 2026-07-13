---
title: BookStrategyPlayer Recording Contract
kind: topic
first_seen: 2026-05-04
last_updated: 2026-07-06
status: superseded
---

## Purpose

**Superseded before build.** [[book-strategy-player]] was redirected before
this schema was ever implemented — the chassis moved from play-time strategy
to the auction by Fable's 2026-06-09 design review (see
[[champion-design-review]]); this Phase 1 spec was never built. May still run
someday as a cheap epilogue measurement, not a live plan.

BookStrategyPlayer recording is the Writer side of [[book-strategy-player]].
Records are append-only decision facts. They support diagnostics, paired replay,
strategy contribution measurement, and later strategy-selector training.

Recording must never affect play. The decision function remains:

```text
Env + GameState + PlanState -> Action + PlanState' + DecisionRecord
```

The record is output, not input.

## Purity contract

Each record includes a fallback proposal even when a strategy wins:

```python
fallback_action = fallback.pure_choose_action(gs)
```

That path must not:

- mutate fallback/player state;
- mutate game state;
- mutate plan state;
- mutate or read the recording buffer;
- consume the live RNG stream;
- advance a forge/world-sampling stream;
- depend on whether recording is enabled.

If a fallback implementation needs randomness or world sampling, the framework
must use a cloned deterministic decision context or log and restore the stream
position.

## DecisionRecord schema

The in-memory schema is intentionally replay-heavy. Parquet serialization can
wait until Phase 1 stabilizes the field set.

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

`game_state_snapshot` is the replay input. `game_state_tensor` is a model input
and is not assumed replay-lossless. If a tensor is later proven lossless, both
fields may point to the same serialized payload.

## Counterfactual semantics

`fallback_action` is a cheap local proposal. It is not exact causal attribution.

Exact strategy contribution requires paired-seed replay or a value model. One
changed domino can change future legal actions, hidden information, opponent
behavior, partner behavior, plan applicability, trick outcomes, and hand margin.
If a strategy diverges five times in one hand, the final margin cannot be
assigned independently to all five divergences from local records alone.

Correct claim:

```text
fallback_action gives a local baseline and enables replay.
```

Incorrect claim:

```text
chosen_action - fallback_action is the exact strategy contribution.
```

## Coverage buckets

Use conservative bucket names before replay:

- `uncovered` - no strategy candidate won; fallback played.
- `covered_same_as_fallback` - strategy won but chose the fallback action.
- `covered_diff_unattributed` - strategy won and diverged, but no replay or
  value attribution has run.
- `covered_diff_replay_positive` - paired replay estimates positive causal
  effect.
- `covered_diff_replay_negative` - paired replay estimates negative causal
  effect.
- `bailed` - strategy won arbitration, returned `None`, and fallback played.
- `illegal_strategy_action` - strategy returned an illegal action; fallback
  played or strict mode failed.
- `disrupted` - committed plan became infeasible before normal completion.

The older names `covered_diff_positive` and `covered_diff_negative` are valid
only after replay or model-based value attribution.

## Replay obligation

Records must contain enough information to deterministically replay a hand:

- player identity and config hash;
- framework, rules, forge, fallback, and strategy versions;
- library fingerprint;
- hand seed and decision RNG seed or stream position;
- forge oracle seed or oracle fingerprint;
- lossless game-state snapshot;
- plan-state snapshot;
- legal actions;
- chosen action and fallback proposal;
- strategy path;
- applied deltas and completion statuses.

If records are not replayable, downstream training labels are suspect.
