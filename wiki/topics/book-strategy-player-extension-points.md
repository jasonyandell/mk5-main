---
title: BookStrategyPlayer Extension Points
kind: topic
first_seen: 2026-05-04
last_updated: 2026-07-06
status: superseded
---

## Purpose

**Superseded before build.** These are extension points for a core
([[book-strategy-player]]) that was never built — the chassis was redirected
from play-time strategy to the auction by Fable's 2026-06-09 design review
(see [[champion-design-review]]). Preserved as design record only.

[[book-strategy-player]] has a small Phase 1 algebra. The following directions
are allowed by that algebra but should not complicate the core implementation.

## State-conditioned selection

`priority(gs, ps)` already makes strategy choice state-conditioned. A strategy
can be important early and fade late; another can rise as the hand develops.
This is the plan-level analogue of [[w42]] state-conditioned utility.

## Strategy chaining

Strategies can communicate through lawful shared facts. Example:

```text
SingletonLeadToVoid publishes facts.void_suits
CountSteeringIntoVoid reads facts.void_suits
```

Context facts must be monoid-valued, such as:

```python
facts["active_contexts"] = SetFact({
    ("Chapter5_SetterDefense", "void_creation"),
})
```

Plan-local context stays in private plan state.

## Hierarchy

A nested BookStrategyPlayer can be wrapped as a strategy only with explicit
semantics:

- internal state is path-namespaced;
- wrapper priority is lawful at the parent level;
- the strategy graph is finite and acyclic;
- records carry `strategy_path`;
- the wrapper returns `None` when no inner strategy applies unless explicitly
  configured as a terminal fallback.

The book's chapter structure is naturally represented this way. Chapters expose
parent-level priorities and own sub-libraries of tactics.

## Observation

Opponent-aware counter-strategies can add an optional hook:

```python
observe(opp_action, opp_seat, gs, ps) -> DeltaPlanState
```

Observation deltas should publish facts such as counters, sets, maxima, or
boolean flags. Counter-strategies then recognize when those facts cross
thresholds.

## Match memory

Cross-hand memory is not Phase 1. It requires a `MatchState` threaded through
the lifecycle:

```text
recognizes(gs, ps, ms)
priority(gs, ps, ms)
commit_recognition(gs, ps, ms)
next_action(gs, ps, ms)
plan_done(gs, ps, ms)
observe(..., gs, ps, ms)
```

The current design is per-hand because parallel-hand simulation treats hands as
independent.

## Training pipeline

Records from [[book-strategy-player-recording]] can later train:

- a strategy selector that replaces hand-coded priorities;
- a plan-success predictor that estimates completion, disruption, bail, and
  hand margin;
- an end-to-end policy distilled from strategy-labeled play.

Labels are not free for every candidate strategy. Each decision observes the
chosen strategy and a local fallback proposal; other candidates require paired
replay or value-model estimates.

[[gus]] is the natural belief-state encoder for selector and plan-success
features. [[burl]] becomes plausible when the output space is named strategies
rather than raw dominoes. [[zeb]] can remain useful as a generator, backup policy
target, or strategy-label policy head.

## Limits

The framework still plays one domino per decision. If two plans want different
actions, arbitration picks one.

It also does not negotiate plan compatibility automatically. Two active plans
can rest on contradictory predictions unless a strategy explicitly encodes that
reasoning.

Partnership coordination can be tested by running the same player in the partner
seat, but that is an experimental setup, not a new framework mechanism.
