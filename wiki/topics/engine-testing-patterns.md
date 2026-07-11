---
title: Engine Testing Patterns
kind: topic
first_seen: aa718769
last_updated: pending-this-ingest
status: active
---

Test infrastructure for the web-game engine: which tool for which test, the
state builders, and the guardrail tests that enforce the architecture
mechanically. Part of the [[engine-architecture]] cluster.

## Key principle: composition all the way down

Tests use the same composition paths as production. Production composes via
`Room → createExecutionContext → layers`; integration tests mirror it via
`HeadlessRoom`. Calling `createExecutionContext` directly outside
Room/HeadlessRoom/test-helpers fails the build (ESLint
`no-restricted-imports`) and `src/tests/architecture/composition.test.ts`.

## Which tool for which test

| Test type | Tool | Location |
|---|---|---|
| Unit: layer composition, rule methods | `createTestContext()` | `src/tests/helpers/executionContext.ts` |
| Integration: full game flows | `HeadlessRoom` | `src/server/HeadlessRoom.ts` |
| Simulation: minimax/PIMC loops | `createSimulationContext()` | `src/tests/helpers/executionContext.ts` |
| Scenario state construction | `StateBuilder` | `src/tests/helpers/stateBuilder.ts` |
| UI / user interaction | Playwright + real Room | `npm run test:e2e` |

Unit tests run under Vitest with `environment: 'node'` — pure logic, no DOM.

## Context helpers (verified 2026-07)

`src/tests/helpers/executionContext.ts`, all `@testOnly`:

- `createTestContext(config?)` — standard 42, human playerTypes
- `createTestContextWithLayers(names)` — explicit layer composition
- `createAITestContext(config?)` — all-AI playerTypes
- `createSimulationContext(config?)` — all-AI, `layers: ['base','speed']`,
  **no consensus layer**: `complete-trick`/`score-hand` execute immediately,
  so minimax/Monte-Carlo loops don't stall waiting for `agree-trick`
  acknowledgments ([[intermediate-ai]])

Do not use these for full game flows — that's HeadlessRoom's job.

## HeadlessRoom

`new HeadlessRoom(config, seed?)` — same composition as Room, no transport.
API: `getState()`, `getValidActions(playerIndex)`,
`executeAction(playerIndex, action)`, `replayActions(actions)`,
`getUnfilteredState()`, `getAllActions()`. Use it for game simulators
(`src/game/ai/gameSimulator.ts`), URL replay, and multi-action integration
tests. Use `Room` itself only when transport/sessions matter.

## StateBuilder

Fluent construction of states at any phase
(`src/tests/helpers/stateBuilder.ts`):

```typescript
const state = StateBuilder
  .inPlayingPhase({ type: 'suit', suit: ACES })
  .withSeed(12345)
  .withPlayerHand(0, ['6-6', '6-5', '5-5', '6-4', '3-2', '4-1', '5-0'])
  .withCurrentPlayer(1)
  .build();
```

Factories: `inBiddingPhase()`, `inTrumpSelection()`, `inPlayingPhase()`,
`withTricksPlayed()`, `inScoringPhase()`, `gameEnded()`, plus special
contracts `nelloContract()`, `splashContract()`, `plungeContract()`,
`sevensContract()`. Chainable modifiers for dealer, current player, trump,
hands, tricks, bids, scores, seed, config. Constraint-based dealing
(`src/tests/helpers/dealConstraints.ts`): `withPlayerDoubles(p, n)`,
`withPlayerConstraint(p, { minDoubles, exactDominoes, voidInSuit })`.
`gameTestHelper.ts` adds scenario-level helpers (bidding scenarios,
sequential consensus processing).

## Guardrail and architecture tests

These encode the invariants of [[engine-architecture]] as failing tests:

- `src/tests/guardrails/no-bypass.test.ts` — import boundaries for the
  dumb-client pattern: UI must not import rule logic (`rules-base.ts`,
  utility-level domino functions); AI must go through the composed
  `GameRules`, never direct imports.
- `src/tests/guardrails/projection-security.test.ts` — no hidden-state
  leaks: opponent hands never visible, rule-aware fields come only from
  server-computed view fields, capability filtering respected
  ([[multiplayer-pattern]]).
- `src/tests/guardrails/rule-contracts.test.ts` — the "Crystal Palace
  contract": `getLedSuit`, `suitsWithTrump`, `canFollow`, `rankInTrick`,
  `calculateTrickWinner`, `isTrump` behave predictably across base and every
  special contract ([[layer-system]]).
- `src/tests/architecture/composition.test.ts` — single composition point:
  only Room/HeadlessRoom/test helpers may create ExecutionContext.
- `src/tests/architecture/no-backwards-compat.test.ts` — greenfield
  enforcement: no `@deprecated`, no "backward compatibility" comments, no
  `_legacy`/`_old` suffixes anywhere. Delete, don't deprecate.

## Running

`npm test` (Vitest, all unit/integration), `npm run test:e2e` (Playwright
production tests), `npm run test:static` (typecheck + svelte-check + lint in
parallel), `npm run test:all`. Scratch/debug Playwright tests live in
`scratch/` with `.test.ts` extension and run only via
`npx playwright test --config=playwright.scratch.config.ts`.

---

*Source: docs/TESTING_PATTERNS.md @ 233b7dc5, verified against `src/tests/`
at this ingest.*
