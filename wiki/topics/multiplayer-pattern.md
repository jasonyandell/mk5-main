---
title: Multiplayer Pattern — Socket / GameClient / Room
kind: topic
first_seen: afac151c
last_updated: pending-this-ingest
status: active
---

The engine's multiplayer architecture: a deliberately minimal
Socket/GameClient/Room pattern (inspired by PartyKit, Colyseus, and
boardgame.io — the frameworks themselves were evaluated and never adopted,
see [[multiplayer-lineage]]) plus capability-token security. Part of the
[[engine-architecture]] cluster.

## Core principle

**Room handles game logic. Clients send and receive. Wiring is external.**
No promise correlation — actions are fire-and-forget, results arrive via
subscription. State updates are the universal response.

## The three pieces (verified 2026-07)

**Socket** (`src/multiplayer/Socket.ts`, 9 lines) — the only transport
abstraction: `send(data)`, `onMessage(handler)`, `close()`. Matches
WebSocket, postMessage, or any bidirectional channel.

**GameClient** (`src/multiplayer/GameClient.ts`, 43 lines) — wraps a Socket:
`send(message)` fire-and-forget, `subscribe(callback)` for `GameView`
updates, `disconnect()`. No caching, no validation, no state synthesis. It
only ever receives filtered views.

**Room** (`src/server/Room.ts`, ~420 lines) — the game authority.
`constructor(config, send)`: it takes a send callback and never creates or
manages transport. It owns the unfiltered `MultiplayerGameState`, creates the
ExecutionContext (single composition point, [[layer-system]]), manages
player sessions, routes protocol messages, and delegates all game logic to
the pure kernel helpers (`src/kernel/kernel.ts`): `executeKernelAction`,
`buildKernelView`, `buildActionsMap`, `processAutoExecuteActions`.

`HeadlessRoom` (`src/server/HeadlessRoom.ts`) is the same composition without
transport, for tools and tests ([[engine-testing-patterns]]).

## Protocol

Five message types total (`src/multiplayer/protocol.ts`):

```typescript
type ClientMessage =
  | { type: 'EXECUTE_ACTION'; action: GameAction }
  | { type: 'JOIN'; playerIndex: number; name: string }
  | { type: 'SET_CONTROL'; playerIndex: number; controlType: 'human' | 'ai' };

type ServerMessage =
  | { type: 'STATE_UPDATE'; view: GameView }
  | { type: 'ERROR'; error: string };
```

No SUBSCRIBE/UNSUBSCRIBE/GAME_CREATED ceremony. Nothing unfiltered ever
crosses the boundary.

## Capability system

Permissions are composable data tokens, never identity checks
(`src/multiplayer/types.ts`). Exactly two capability types exist:

```typescript
type Capability =
  | { type: 'act-as-player'; playerIndex: number }
  | { type: 'observe-hands'; playerIndices: number[] | 'all' };
```

A `PlayerSession` groups `playerId`, `playerIndex` (seat), `controlType`
(`human | ai`), and its capability list — separating identity from seat
(hot-seat and control-swap fall out for free).

Builders in `src/multiplayer/capabilities.ts`: `humanCapabilities(i)` and
`aiCapabilities(i)` (act as seat i + observe own hand — identical),
`spectatorCapabilities()` (observe all, act as none), and a fluent
`buildCapabilities()` for custom sets.

Capabilities drive two mechanisms:

- **Authorization** (`src/multiplayer/authorization.ts`,
  `authorizeAndExecute`): find session, generate valid actions via the
  composed context, filter by `act-as-player`, execute only if the requested
  action is in the filtered set. Actions carrying
  `meta.authority: 'system'` (scripted/auto-executed moves, e.g. from the
  speed layer) bypass capability checks but are still structurally validated.
- **Visibility** (`getVisibleStateForSession` in `capabilities.ts`): hands
  not covered by an `observe-hands` token are emptied (`hand: []`); a
  `handCount` field remains visible. Action metadata (hints) is likewise
  filtered per session.

## Dumb client / server-owned projection

The client never imports rule logic and never recomputes anything.
`buildKernelView` produces a complete `GameView` — filtered state,
`validActions` (the full legal menu for that session), transitions with UI
metadata — and the client renders it. Trust flows one way: server validates
everything, client trusts completely. This is enforced mechanically by the
guardrail tests `no-bypass` (import boundaries) and `projection-security`
(no hidden-state leaks) — [[engine-testing-patterns]].

## AI as clients

An AI player is a plain GameClient with a subscription callback
(`attachAIBehavior` in `src/multiplayer/local.ts`): on each view, it calls
`selectAIAction` ([[intermediate-ai]]) over its own filtered `validActions`
and sends `EXECUTE_ACTION`. No privileged access, no special protocol —
protocol equality is an invariant.

## Wiring

The only wiring that exists is in-process: `createLocalGame(config)`
(`src/multiplayer/local.ts`) routes messages through a
`Map<clientId, handler>` with `queueMicrotask` for async parity, creating the
Room, the player-0 client, and AI clients ([[client-implementation]]).
The pattern was designed so a WebSocket/Durable-Object wiring could reuse
Room and GameClient unchanged; that online mode was never built — the
project pivoted to ML before it mattered ([[the-wall]]).

---

*Sources: docs/MULTIPLAYER.md, docs/CAPABILITY_SYSTEM.md @ 233b7dc5,
verified against `src/multiplayer/`, `src/server/`, `src/kernel/` at this
ingest.*
