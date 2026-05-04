---
title: BookStrategyPlayer Phase 1 Build Contract
kind: topic
first_seen: local-2026-05-04
last_updated: local-2026-05-04
status: active
---

## Purpose

[[book-strategy-player]] Phase 1 builds the framework and measurement
instrument. It should prove the algebra on fake strategies before encoding real
[[w42]] tactics.

The build order is:

1. Core contracts and fake strategies.
2. Property tests.
3. Pure fallback adapter.
4. Recording.
5. One real strategy.
6. Head-to-head measurement.

## Deliverables

- `Strategy` protocol with `recognizes`, `priority`, `commit_recognition`,
  `next_action`, and `plan_done`.
- `PlanState` and `DeltaPlanState`.
- Lawful `FactValue` wrappers and naked-scalar rejection.
- Library construction with duplicate-name rejection.
- Deterministic arbitration with canonical name tie-break.
- Pure `Lens("ev")` fallback adapter.
- In-memory `DecisionRecord`; parquet after schema stabilization.
- Fake-strategy property tests for the algebra.
- Head-to-head harness against `Lens("ev")`, reusing
  `w42/lens_v1/parallel_match.py` where possible.
- One to three starter strategies after the laws pass.

## Property tests

Minimum Phase 1 tests:

- T1: Empty library matches fallback.
- T2: Library order does not affect chosen action.
- T3: Duplicate strategy names raise at construction.
- T4: Active strategy is not re-committed.
- T5: Fresh losing strategy is not committed.
- T6: Active continuation beats fresh recognition under the library priority
  convention.
- T7: Bail preserves active plan unless a retire delta is emitted.
- T8: Retire removes active plan.
- T9: Fact merge is associative for all fact wrappers.
- T10: Fact merge is commutative for commutative fact wrappers.
- T11: Naked scalar facts are rejected.
- T12: Strategy cannot write another strategy's namespace.
- T13: NaN and non-finite priorities are rejected.
- T14: Illegal strategy action records/falls back or raises in strict mode.
- T15: Recording-on and recording-off produce identical chosen actions.
- T16: Fallback dry-run does not consume live RNG.
- T17: `DecisionRecord` contains replay-required fields.

Tests should permute library orderings, duplicate insertion attempts, strategy
bails, retire deltas, illegal actions, and fact merge order.

## Error policy

Strict/test mode:

- illegal strategy action raises;
- namespace violation raises;
- NaN or non-finite priority raises;
- illegal fallback action raises.

Long simulation mode:

- illegal strategy action may record and fall back;
- namespace violation still raises;
- NaN or non-finite priority still raises;
- illegal fallback action still raises.

## Suggested module layout

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

## Starter strategies

`singleton_lead_to_void` is a valid first measurement or negative-control
candidate, but it should not be framed as rehabilitating
[[w42-bookval-v1-wave2-void-creation]] unless measurement supports that claim.

A canonical follow-position void variant may be cleaner if the first strategy is
intended to test the strongest positive version of the book idea.

## Measurement

The first head-to-head should compare:

```python
BookStrategyPlayer([starter_strategy], fallback=Lens("ev"))
```

against:

```python
Lens("ev")
```

The report should include paired-seed point margin, bootstrap confidence
interval, strategy firing rate, plan completion rate, disruption rate, bail
count, illegal action count, fallback fraction, and a small sample of replayable
decision records.
