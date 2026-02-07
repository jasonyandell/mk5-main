# Texas 42 Client Quickstart

## TL;DR

```typescript
import { createLocalGame } from './src/multiplayer/local';

// 1. Create a game (human + 3 AI opponents)
const { client } = createLocalGame({ playerTypes: ['human', 'ai', 'ai', 'ai'] });

// 2. Join as player 0
client.send({ type: 'JOIN', playerIndex: 0, name: 'You' });

// 3. React to state updates
client.subscribe((view) => {
  if (view.validActions.length > 0) {
    // Pick an action and send it
    client.send({ type: 'EXECUTE_ACTION', action: view.validActions[0].action });
  }
});
```

That's a working client. Everything else is details.

---

## The Big Picture

Here's the mental model:

1. **Server is the boss.** It knows the rules, tracks the game, validates everything.

2. **You just react.** When you get a state update, look at `validActions`.

3. **`validActions` tells you what you can do right now.** It's a list. Pick one. Send it back.

4. **Wait for the next update.** The server processes your action and sends everyone a new state.

That's the whole loop. You don't need to:
- Track whose turn it is (server tells you via `validActions`)
- Validate if a move is legal (if it's in `validActions`, it's legal)
- Know the game rules (just pick from the menu)

---

## Current Architecture

Right now, games run in-process (no network). The architecture is designed for future WebSocket support, but today you use:

```typescript
import { createLocalGame } from './src/multiplayer/local';
import type { GameConfig } from './src/game/types/config';

const config: GameConfig = {
  playerTypes: ['human', 'ai', 'ai', 'ai'],
  layers: ['consensus'],  // optional
};

const { client, room, createSocket } = createLocalGame(config);
```

**What you get back:**
- `client` - A `GameClient` for player 0 (human)
- `room` - The `Room` instance (server-side, for advanced use)
- `createSocket` - Factory to create additional client connections

The `GameClient` wraps the low-level Socket interface and gives you:
- `send(message)` - Send a message (fire-and-forget)
- `subscribe(callback)` - Get notified on state updates
- `disconnect()` - Clean up

---

## What's in a GameView?

When you subscribe, your callback receives a `GameView`:

| Field | What it tells you |
|-------|-------------------|
| `state.phase` | Current phase: `"bidding"`, `"trump_selection"`, `"playing"`, etc. |
| `state.currentPlayer` | Whose turn (0-3) |
| `state.players[n].hand` | Your dominoes (others' hands are hidden - empty arrays) |
| `state.trump` | What's trump: `{ type: "suit", suit: 5 }` means fives are trump |
| `state.currentTrick` | Dominoes played in current trick |
| `state.teamScores` | Points this hand: `[12, 7]` |
| `state.teamMarks` | Marks (game score): `[2, 1]` |
| **`validActions`** | **What you can do right now** (the important one!) |

---

## You Never Construct Actions

This is the key insight: **you never build action objects yourself**.

The server pre-generates ALL valid actions for you:
- During bidding: every valid bid (30, 31, 32... up to 42), every mark bid, pass
- During trump selection: every valid trump choice
- During play: every domino you can legally play

You just pick from the menu:

```typescript
// The server gives you something like:
validActions = [
  { action: { type: 'bid', player: 0, bid: 'points', value: 30 }, label: 'Bid 30' },
  { action: { type: 'bid', player: 0, bid: 'points', value: 31 }, label: 'Bid 31' },
  // ... all the way to 42
  { action: { type: 'pass', player: 0 }, label: 'Pass' },
];

// You pick one and send it back:
const chosen = validActions[0];
client.send({ type: 'EXECUTE_ACTION', action: chosen.action });
```

That's it. No validation logic. No rule knowledge. Just pick and send.

---

## Minimal Bot (Complete Example)

```typescript
import { createLocalGame } from './src/multiplayer/local';
import type { GameView } from './src/multiplayer/types';

const PLAYER_INDEX = 0;

// Create game with human player and AI opponents
const { client } = createLocalGame({
  playerTypes: ['human', 'ai', 'ai', 'ai'],
});

// Join as player 0
client.send({ type: 'JOIN', playerIndex: PLAYER_INDEX, name: 'SimpleBot' });

// Subscribe to state updates
client.subscribe((view: GameView) => {
  // Filter to actions for our player (or system actions with no player field)
  const myActions = view.validActions.filter(
    (a) => !('player' in a.action) || a.action.player === PLAYER_INDEX
  );

  if (myActions.length > 0) {
    // Pick randomly
    const chosen = myActions[Math.floor(Math.random() * myActions.length)];
    console.log('Playing:', chosen.label);

    client.send({ type: 'EXECUTE_ACTION', action: chosen.action });
  }
});
```

---

## Future: WebSocket Support

The `Socket` interface is transport-agnostic:

```typescript
interface Socket {
  send(data: string): void;
  onMessage(handler: (data: string) => void): void;
  close(): void;
}
```

When WebSocket support is added, you'll be able to create a socket that wraps a real WebSocket connection. The `GameClient` and all your code stays the same - only the transport changes.

---

## Next Steps

- **Full reference**: [CLIENT_IMPLEMENTATION_GUIDE.md](./CLIENT_IMPLEMENTATION_GUIDE.md) - All message types, action fields, capabilities
- **Reference implementation**: [`src/multiplayer/GameClient.ts`](../src/multiplayer/GameClient.ts) - The actual client class (~40 lines)
- **Local game setup**: [`src/multiplayer/local.ts`](../src/multiplayer/local.ts) - How `createLocalGame` works
- **Game rules**: [rules.md](./rules.md) - How Texas 42 is played
