---
title: Engine Architecture
kind: topic
first_seen: b7a32f5e
last_updated: pending-this-ingest
status: active
---

The TypeScript web-game engine's architecture as current reference. This is the
start-here page for anyone working in `src/`. History of how it got this way:
[[web-game]]. The cluster: [[layer-system]] (rule composition),
[[multiplayer-pattern]] (Socket/GameClient/Room), [[client-implementation]]
(building against the engine), [[engine-testing-patterns]] (test
infrastructure), [[intermediate-ai]] (the shipped opponents). Hub: [[engine]].

## The fundamental pattern

```
STATE → ACTION → NEW STATE
```

Everything in the system exists to (1) **generate** valid actions from a state,
(2) **execute** actions deterministically, (3) **filter** results per observer.
All transitions are pure functions; state objects are never mutated, only
transformed.

**Event sourcing** is the foundation: `state = replayActions(config, history)`.
Actions are the source of truth; state is computed. Two games with the same
config and action history are byte-identical, which buys perfect replay,
time-travel debugging, deterministic tests, and shareable games.

## URL replay

The entire game — seed (or explicit deal), action history, player types,
dealer, layers, theme — encodes into a shareable URL
(`src/game/core/url-compression.ts`; params `s` seed XOR `i` initialHands,
plus `p`, `d`, `l`, `t`, `v`, `a`). A pasted URL reproduces the exact game
state, which is the project's standard bug-report and debugging currency.
Replay utilities live in `src/game/utils/urlReplay.ts`.

## The stack

```
Svelte components            reactive views
  ↕ src/stores/gameStore.ts  Svelte facade over GameClient
GameClient                   fire-and-forget send + subscribe (43 lines)
  ↕ Socket                   send/onMessage/close (9 lines)
Room                         orchestrator; owns state, sessions, transport callback
  ↕
kernel (src/kernel/kernel.ts) pure helpers: executeKernelAction, buildKernelView,
                              buildActionsMap, processAutoExecuteActions
  ↕
Layer system                 composed GameRules + action generation ([[layer-system]])
  ↕
Core engine (src/game/core/) pure utilities, zero coupling
```

The kernel is a pure function: `newState = f(oldState, action)`, no hidden
state. The `Room` stores *unfiltered* `MultiplayerGameState` and delegates all
game logic to the kernel helpers; filtering happens per-request, per-observer
— never at rest. Clients are deliberately dumb: `buildKernelView` computes
every rule-aware field (valid plays, suit analysis, hints) server-side, and
the client renders what it is told. Details: [[multiplayer-pattern]].

## Composition

`createExecutionContext(config)` (`src/game/types/execution.ts`) is where a
`GameConfig` becomes a frozen `ExecutionContext` — enabled layers, composed
`GameRules`, composed `getValidActions`. Only `Room` and `HeadlessRoom`
constructors call it (enforced by ESLint `no-restricted-imports` and
`src/tests/architecture/composition.test.ts`). Executors contain zero
conditional logic on game mode: they call `ctx.rules.method()` and trust the
result (parametric polymorphism).

## Directory map (verified 2026-07)

- `src/game/types.ts` — GameState, GameAction, Domino, Bid, Trick, suit types,
  `CALLED = 7` (the called suit)
- `src/game/types/` — `config.ts` (GameConfig, DealOverrides), `execution.ts`
  (ExecutionContext, createExecutionContext)
- `src/game/core/` — pure engine: `actions.ts` (executors, `executeAction`),
  `state.ts` (state utilities, `getNextStates`), `rules.ts`, `scoring.ts`,
  `bidding.ts`, `setup.ts`, `dominoes.ts`, `domino-tables.ts` (precomputed
  `EFFECTIVE_SUIT` / `SUIT_MASK` / `HAS_POWER` lookup tables),
  `url-compression.ts`, `handOutcome.ts`, `suit-analysis.ts`
- `src/game/layers/` — the Layer system ([[layer-system]])
- `src/game/ai/` — the shipped PIMC opponents ([[intermediate-ai]])
- `src/game/view-projection.ts` — pure GameState → UI-projection computation
- `src/multiplayer/` — Socket, GameClient, protocol, capabilities,
  authorization, `local.ts` (createLocalGame)
- `src/server/` — `Room.ts` (production orchestrator), `HeadlessRoom.ts`
  (tools/simulation API)
- `src/kernel/kernel.ts` — pure multiplayer helpers
- `src/stores/` — Svelte stores; `src/App.svelte` — UI entry
- `src/tests/` — helpers, guardrails, architecture tests
  ([[engine-testing-patterns]])

## The algebraic boundary: tables vs dynamic computation

`domino-tables.ts` precomputes what depends only on domino + trump
configuration: which suit a domino leads (`EFFECTIVE_SUIT`), which dominoes
can follow a suit (`SUIT_MASK`), which dominoes are trump (`HAS_POWER`).
What *cannot* be precomputed is trick-context ranking: the three-tier system
(trump ≈ 200+, follows-led-suit ≈ 50+, slough = pip sum) depends on what was
led, known only once the trick starts. The same domino ranks differently
under different leads. A related precomputed artifact,
`src/game/ai/strength-table.generated.ts`, is regenerated before every build
and dev run (`npm run generate:strength-table`).

## Architectural invariants

Violation of any of these is a regression:

1. **Pure state storage** — Room stores unfiltered state; filtering is
   per-request.
2. **Server authority** — clients never revalidate, refilter, or recompute;
   they trust `validActions` completely.
3. **Capability-based access** — permission tokens, never identity checks
   ([[multiplayer-pattern]]).
4. **Single composition point** — ExecutionContext is created only in
   Room/HeadlessRoom.
5. **Zero coupling** — the core engine knows nothing of multiplayer, layers,
   or transport.
6. **Parametric execution** — executors call `rules.method()`, never
   `if (nello)`.
7. **Event sourcing** — state derivable from `replayActions(config, history)`;
   actions immutable, append-only.
8. **Clean separation** — Room orchestrates, kernel helpers execute, transport
   routes.

## Design philosophy

Correct by construction: the type system makes illegal states unrepresentable
(GameAction union, GamePhase union, discriminated `HandOutcome`). Composition
over configuration: variant behavior comes from composing layers, not flags
and conditionals. Explicit over implicit: all dependencies passed as
parameters, no globals. Every line of code is a liability — the recurring
"Crystal Palace" dedup epics ([[web-game]]) exist to enforce this.

## Commands

`npm run dev` (Vite dev server), `npm run typecheck`, `npm test` (Vitest,
node environment), `npm run test:e2e` (Playwright), `npm run check`
(svelte-check), `npm run lint`, `npm run build`.

---

*Sources: docs/ORIENTATION.md, docs/CONCEPTS.md,
docs/ARCHITECTURE_PRINCIPLES.md @ 233b7dc5, verified against `src/` at this
ingest.*
