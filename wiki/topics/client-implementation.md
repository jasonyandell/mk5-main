---
title: Building a Client Against the Engine
kind: topic
first_seen: 2025-11-28
last_updated: 2026-07-11
status: active
---

How to drive the engine as a client — the working loop, the types, and the
rules of the road. Companion to [[multiplayer-pattern]] (the architecture) in
the [[engine-architecture]] cluster.

## The whole loop

```typescript
import { createLocalGame } from './src/multiplayer/local';

const { client } = createLocalGame({ playerTypes: ['human', 'ai', 'ai', 'ai'] });
client.send({ type: 'JOIN', playerIndex: 0, name: 'You' });
client.subscribe((view) => {
  const mine = view.validActions.filter(
    (a) => !('player' in a.action) || a.action.player === 0
  );
  if (mine.length > 0) {
    client.send({ type: 'EXECUTE_ACTION', action: mine[0].action });
  }
});
```

That is a complete client. The server pre-generates every legal action;
the client picks from the menu and sends it back. Three rules:

1. **Never construct actions.** During bidding the server sends every valid
   bid and pass; during trump selection every legal trump; during play every
   playable domino. Pick from `validActions`.
2. **Never validate client-side.** If it's in `validActions`, it's legal.
   No follow-suit logic, no turn-order checks in the client.
3. **No request-response.** Results of `EXECUTE_ACTION` arrive as the next
   `STATE_UPDATE`. Empty `validActions` for your player means it's not your
   turn — wait.

## createLocalGame (verified 2026-07)

`createLocalGame(config, options?)` in `src/multiplayer/local.ts` returns a
`LocalGame`:

| Field | What it is |
|---|---|
| `client` | `GameClient` for player 0 |
| `room` | the `Room` instance (direct access, e.g. action replay) |
| `createSocket(clientId)` | factory for additional connections (spectators, extra seats) |
| `attachAI()` | attach AI behavior after the fact (when `skipAIBehavior` was set, e.g. replay URLs first, then let AI move) |

`LocalGameOptions`: `aiPlayerIndexes` (default `[1,2,3]`), `skipAIBehavior`,
and `aiStrategyConfig` (`{ type: 'beginner' | 'random' }`, default beginner —
[[intermediate-ai]]). The Svelte app wires itself exactly this way
(`src/stores/gameStore.ts` → `createLocalGame`).

## GameView

Each `STATE_UPDATE` carries the complete client-side truth:

- `state: FilteredGameState` — phase, `currentPlayer`, `trump`,
  `currentTrick`, `tricks`, `bids`, `teamScores`, `teamMarks`, players.
  Hidden hands are empty arrays; `handCount` stays visible.
- `validActions: ValidAction[]` — `{ action, label, group?, recommended? }`,
  already filtered to this session's capabilities.
- `transitions` — same actions with stable `id`s for UI keying (no
  `newState`; that stays server-side).
- `players`, `metadata` — control types, connection state, gameId, layers.

## GameConfig

`src/game/types/config.ts`:

- `playerTypes: ('human' | 'ai')[]` — required
- `layers?: string[]` — names from the registry: `nello`, `splash`,
  `plunge`, `sevens`, `tournament`, `oneHand`, `speed`, `hints`,
  `consensus` ([[layer-system]]). Consensus is auto-appended to every game
  whether listed or not; it only gates when human players exist.
- `shuffleSeed?: number` — deterministic deal
- `dealOverrides?: { initialHands }` — exact 4×7 hands; overrides seed;
  serializes to the URL (teaching, challenges, bug repro)
- `theme?`, `colorOverrides?`, `aiDifficulty?`, `timeLimits?`

## Actions you'll see

Player actions (require `act-as-player`): `bid`, `pass`, `select-trump`,
`play` (by `dominoId`, format `"high-low"` e.g. `'6-4'`). Neutral actions:
`complete-trick`, `score-hand`, `redeal`. Consensus actions (when humans are
present): `agree-trick`, `agree-score` — tap-to-continue acknowledgments.
One-hand mode terminal actions: `retry-one-hand`, `new-one-hand`.

Phases (`GamePhase`): `setup`, `bidding`, `trump_selection`, `playing`,
`scoring`, `game_end`, `one-hand-complete`. Game rules themselves:
[[rules-of-42]].

## Multiple clients / spectators

Use `createSocket(clientId)` and `new GameClient(socket)` for extra
connections; capabilities determine what each sees and does
([[multiplayer-pattern]]). Errors surface as `{ type: 'ERROR' }` messages
and are console-logged by GameClient; handle them at the Socket level if you
need more.

---

*Sources: docs/CLIENT_QUICKSTART.md, docs/CLIENT_IMPLEMENTATION_GUIDE.md
@ 233b7dc5, deduplicated and verified against `src/multiplayer/` at this
ingest.*
