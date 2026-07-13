---
title: Engine ADRs — the seven architecture decision records
kind: decision
first_seen: 2026-07-11
last_updated: 2026-07-11
status: active
---

The engine's seven formal ADRs, recorded Nov 2025 – Jan 2026 under `docs/adrs/` and retired
to this page at the docs→wiki consolidation (full texts: `docs/adrs/ @ 233b7dc5`). Each
decision remains in force in the engine ([[engine]], [[engine-architecture]]).

1. **System vs Player authority** (ADR-20250111) — two authority models for action
   execution: Player authority (session capabilities, `act-as-player`) for user-initiated
   actions, System authority for engine-initiated scripted execution.
2. **No protocol leaks** (ADR-20251110) — `state` and `actions` fields removed from all
   protocol messages; the server sends only the filtered `GameView` (extended with
   `transitions`). See [[multiplayer-pattern]].
3. **Single composition point** (ADR-20251110) — `createExecutionContext` may be called
   from exactly three places (production `Room.ts` plus two sanctioned test/support paths),
   enforced by ESLint rules and architecture tests. See [[engine-testing-patterns]].
4. **Connection.reply() pattern** (ADR-20251111) — no global transport routing; each
   connection is self-contained and knows how to deliver messages to itself.
5. **URL replay carries complete GameConfig** (ADR-20251111) — URL encoding captures the
   full `GameConfig` via short-code registries, so any game is replayable from its URL
   alone. See [[engine-architecture]].
6. **One-hand terminal phase via GameRules** (ADR-20251112) — mode-specific phase
   transitions go through `GameRules.getPhaseAfterHandComplete(state)`, not conditionals;
   the one-hand layer overrides it to reach a terminal phase.
7. **Layer unification** (ADR-20251124) — RuleSets and ActionTransformers unified into a
   single `Layer` concept with two orthogonal composition surfaces (execution rules +
   action generation). The founding decision of the current [[layer-system]].

## Links

[[engine]] · [[engine-architecture]] · [[layer-system]] · [[multiplayer-pattern]] ·
[[web-game]]
