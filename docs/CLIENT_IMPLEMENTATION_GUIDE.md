# Texas 42 Client Implementation Guide

Complete reference for building a Texas 42 multiplayer client.

> **New here?** Start with [CLIENT_QUICKSTART.md](./CLIENT_QUICKSTART.md) for a gentler introduction.

## Overview

Texas 42 uses a simple client-server architecture:

- **Server is authoritative**: The server validates all actions and maintains game state
- **Fire-and-forget messaging**: Clients send actions without waiting for responses
- **Subscription model**: Clients receive state updates via callback/subscription
- **Trust validActions**: Don't validate client-side; use the server-provided `validActions` list

## Getting Started (Current Architecture)

Games currently run in-process. Use `createLocalGame` to set up a game:

```typescript
import { createLocalGame } from './src/multiplayer/local';
import type { GameConfig } from './src/game/types/config';
import type { GameView } from './src/multiplayer/types';

const config: GameConfig = {
  playerTypes: ['human', 'ai', 'ai', 'ai'],
  layers: ['consensus'],
};

const { client, room, createSocket } = createLocalGame(config);

// Join as player 0
client.send({ type: 'JOIN', playerIndex: 0, name: 'Alice' });

// Subscribe to state updates
client.subscribe((view: GameView) => {
  console.log('Phase:', view.state.phase);
  console.log('Valid actions:', view.validActions.length);
});
```

**What `createLocalGame` returns:**

| Field | Type | Description |
|-------|------|-------------|
| `client` | `GameClient` | Client for player 0 (human) |
| `room` | `Room` | Server-side room instance |
| `createSocket` | `(clientId: string) => Socket` | Factory for additional connections |
| `attachAI` | `() => void` | Attach AI behavior (if `skipAIBehavior` was true) |

## Socket Interface

The `Socket` interface is the transport abstraction:

```typescript
interface Socket {
  send(data: string): void;
  onMessage(handler: (data: string) => void): void;
  close(): void;
}
```

This maps to:
- **In-process**: Message routing via callbacks (current implementation)
- **WebSocket**: `ws.send()`, `ws.onmessage`, `ws.close()` (future)
- **postMessage**: `window.postMessage()`, `addEventListener('message', ...)` (future)

All data is transmitted as JSON strings.

## GameClient

`GameClient` wraps a `Socket` and provides a typed API:

```typescript
import { GameClient } from './src/multiplayer/GameClient';
import type { ClientMessage } from './src/multiplayer/protocol';
import type { GameView } from './src/multiplayer/types';

class GameClient {
  view: GameView | null;

  constructor(socket: Socket);

  /** Send a message. Fire-and-forget - results come via subscription. */
  send(message: ClientMessage): void;

  /** Subscribe to state updates. Returns unsubscribe function. */
  subscribe(callback: (view: GameView) => void): () => void;

  /** Disconnect from the game. */
  disconnect(): void;
}
```

## Protocol Messages

### Client → Server (`ClientMessage`)

```typescript
import type { ClientMessage } from './src/multiplayer/protocol';
import type { GameAction } from './src/game/types';

type ClientMessage =
  | { type: 'EXECUTE_ACTION'; action: GameAction }
  | { type: 'JOIN'; playerIndex: number; name: string }
  | { type: 'SET_CONTROL'; playerIndex: number; controlType: 'human' | 'ai' };
```

#### EXECUTE_ACTION
Execute a game action (bid, play domino, etc.)

```typescript
client.send({
  type: 'EXECUTE_ACTION',
  action: { type: 'bid', player: 0, bid: 'points', value: 30 }
});
```

#### JOIN
Associate this client with a player seat (0-3)

```typescript
client.send({
  type: 'JOIN',
  playerIndex: 0,
  name: 'Alice'
});
```

#### SET_CONTROL
Change a player's control type (human or AI)

```typescript
client.send({
  type: 'SET_CONTROL',
  playerIndex: 1,
  controlType: 'ai'
});
```

### Server → Client (`ServerMessage`)

```typescript
import type { ServerMessage } from './src/multiplayer/protocol';
import type { GameView } from './src/multiplayer/types';

type ServerMessage =
  | { type: 'STATE_UPDATE'; view: GameView }
  | { type: 'ERROR'; error: string };
```

## GameView Structure

The `STATE_UPDATE` message contains a `GameView`:

```typescript
import type { GameView, ValidAction, ViewTransition, PlayerInfo } from './src/multiplayer/types';
import type { FilteredGameState } from './src/game/types';

interface GameView {
  state: FilteredGameState;
  validActions: ValidAction[];
  transitions: ViewTransition[];
  players: PlayerInfo[];
  metadata: {
    gameId: string;
    layers?: string[];
  };
}
```

### state (`FilteredGameState`)

The current game state, filtered based on client capabilities:

```typescript
interface FilteredGameState {
  phase: GamePhase;
  currentPlayer: number;
  dealer: number;
  trump: TrumpSelection;
  teamScores: [number, number];
  teamMarks: [number, number];
  currentTrick: Play[];
  tricks: Trick[];
  bids: Bid[];
  currentBid: Bid;
  winningBidder: number;
  players: Array<{
    id: number;
    name: string;
    teamId: 0 | 1;
    marks: number;
    hand: Domino[];      // Empty array if hidden
    handCount: number;   // Always visible
    suitAnalysis?: SuitAnalysis;
  }>;
  // ... other fields
}
```

**Important**: Hidden hands appear as empty arrays (`hand: []`), but `handCount` always shows how many dominoes the player holds.

### validActions

Pre-calculated list of actions this client can execute:

```typescript
interface ValidAction {
  action: GameAction;
  label: string;
  group?: string;
  shortcut?: string;
  recommended?: boolean;
}
```

**Trust this list completely.** If an action appears here, it's valid.

### transitions

Same as `validActions` but with unique IDs for UI keying:

```typescript
interface ViewTransition {
  id: string;
  label: string;
  action: GameAction;
  group?: string;
  recommended?: boolean;
}
```

### players

Information about all players:

```typescript
interface PlayerInfo {
  playerId: number;
  controlType: 'human' | 'ai';
  sessionId?: string;
  connected: boolean;
  name?: string;
  avatar?: string;
  capabilities?: Capability[];
}
```

## Capability System

Capabilities control what a client can see and do:

```typescript
import type { Capability } from './src/multiplayer/types';

type Capability =
  | { type: 'act-as-player'; playerIndex: number }
  | { type: 'observe-hands'; playerIndices: number[] | 'all' };
```

### act-as-player
Permission to execute actions for a specific player seat.

```typescript
{ type: 'act-as-player', playerIndex: 0 }
```

### observe-hands
Permission to see specific players' hands.

```typescript
// See own hand only
{ type: 'observe-hands', playerIndices: [0] }

// See all hands (spectator)
{ type: 'observe-hands', playerIndices: 'all' }
```

### Standard Capability Sets

| Role | Capabilities | Can See | Can Act |
|------|--------------|---------|---------|
| Human Player (seat 0) | act-as-player:0, observe-hands:[0] | Own hand only | As self only |
| AI Player (seat 1) | act-as-player:1, observe-hands:[1] | Own hand only | As self only |
| Spectator | observe-hands:"all" | All hands | Nothing |

### Capability Helpers

```typescript
import { humanCapabilities, aiCapabilities, spectatorCapabilities, buildCapabilities } from './src/multiplayer/capabilities';

// Standard sets
const human = humanCapabilities(0);  // act-as-player:0, observe-hands:[0]
const ai = aiCapabilities(1);        // act-as-player:1, observe-hands:[1]
const spec = spectatorCapabilities(); // observe-hands:'all'

// Custom (fluent builder)
const custom = buildCapabilities()
  .actAsPlayer(0)
  .actAsPlayer(2)
  .observeHands([0, 2])
  .build();
```

## Game Actions Reference

> **Important**: You never construct these actions yourself. The server pre-generates ALL valid actions and sends them in `validActions`. This section documents the structure for understanding what you'll receive - not for building actions manually.

The server generates the complete menu:
- **Bidding phase**: Every valid point bid (30-42), mark bid (1-4), and pass
- **Trump selection**: Every valid trump choice for the current game mode
- **Playing phase**: Every domino you can legally play

Just pick from `validActions` and send it back.

### Player Actions

These require `act-as-player` capability for the specified player:

#### bid
Make a bid during the bidding phase.

```typescript
interface BidAction {
  type: 'bid';
  player: number;          // 0-3
  bid: BidType;            // 'pass' | 'points' | 'marks' | 'splash' | 'plunge'
  value?: number;          // Required for 'points' (30-42) and 'marks' (1-7)
}

// Examples
{ type: 'bid', player: 0, bid: 'points', value: 30 }
{ type: 'bid', player: 0, bid: 'marks', value: 2 }
```

#### pass
Decline to bid.

```typescript
interface PassAction {
  type: 'pass';
  player: number;
}

{ type: 'pass', player: 0 }
```

#### select-trump
Choose trump after winning the bid.

```typescript
interface SelectTrumpAction {
  type: 'select-trump';
  player: number;
  trump: TrumpSelection;
}

interface TrumpSelection {
  type: 'not-selected' | 'suit' | 'doubles' | 'no-trump' | 'nello' | 'sevens';
  suit?: RegularSuit;  // 0-6, only when type === 'suit'
}

// Examples
{ type: 'select-trump', player: 0, trump: { type: 'suit', suit: 5 } }  // Fives
{ type: 'select-trump', player: 0, trump: { type: 'doubles' } }
{ type: 'select-trump', player: 0, trump: { type: 'no-trump' } }
{ type: 'select-trump', player: 0, trump: { type: 'nello' } }  // Layer-dependent
```

Trump selection options:

| type | suit field | Description |
|------|------------|-------------|
| `"suit"` | 0-6 | A pip value is trump (0=blanks, 1=aces, ..., 6=sixes) |
| `"doubles"` | — | Doubles are a separate suit |
| `"no-trump"` | — | No trump (follow-me) |
| `"nello"` | — | Bidder must lose all tricks (layer-dependent) |
| `"sevens"` | — | Sevens trump (layer-dependent) |

#### play
Play a domino from your hand.

```typescript
interface PlayAction {
  type: 'play';
  player: number;
  dominoId: string;  // e.g., '5-3', '6-6'
}

{ type: 'play', player: 0, dominoId: '5-3' }
```

### System Actions

These don't require player capability and are typically auto-executed:

#### complete-trick
Finish the current trick after all 4 plays.

```typescript
{ type: 'complete-trick' }
```

#### score-hand
Score the completed hand and advance to the next.

```typescript
{ type: 'score-hand' }
```

#### redeal
Redeal when all players pass (auto-generated).

```typescript
{ type: 'redeal' }
```

### Consensus Actions

Used when the consensus layer is enabled for tap-to-continue UX:

#### agree-trick
Player confirms they've seen the trick result.

```typescript
{ type: 'agree-trick', player: 0 }
```

#### agree-score
Player confirms they've seen the hand score.

```typescript
{ type: 'agree-score', player: 0 }
```

### One-Hand Mode Actions

Available when a one-hand game ends:

#### retry-one-hand
Replay with the same deal.

```typescript
{ type: 'retry-one-hand' }
```

#### new-one-hand
Start a new one-hand game with a different deal.

```typescript
{ type: 'new-one-hand' }
```

### Complete GameAction Type

```typescript
import type { GameAction } from './src/game/types';

type GameAction =
  | { type: 'bid'; player: number; bid: BidType; value?: number }
  | { type: 'pass'; player: number }
  | { type: 'select-trump'; player: number; trump: TrumpSelection }
  | { type: 'play'; player: number; dominoId: string }
  | { type: 'complete-trick' }
  | { type: 'score-hand' }
  | { type: 'agree-trick'; player: number }
  | { type: 'agree-score'; player: number }
  | { type: 'redeal' }
  | { type: 'retry-one-hand' }
  | { type: 'new-one-hand' };
```

## Client Lifecycle

### 1. Create Game
```typescript
const { client } = createLocalGame(config);
```

### 2. Join
```typescript
client.send({ type: 'JOIN', playerIndex: 0, name: 'Alice' });
```

### 3. Subscribe to Updates
```typescript
const unsubscribe = client.subscribe((view) => {
  // Handle state update
});
```

### 4. React to validActions
```typescript
client.subscribe((view) => {
  const myActions = view.validActions.filter(
    (a) => !('player' in a.action) || a.action.player === PLAYER_INDEX
  );

  if (myActions.length > 0) {
    const chosen = myActions[0];  // Or use smarter selection
    client.send({ type: 'EXECUTE_ACTION', action: chosen.action });
  }
});
```

### 5. Handle Errors
Errors are logged to console by `GameClient`. For custom handling, use the raw Socket.

### 6. Disconnect
```typescript
client.disconnect();
```

## Room Configuration (GameConfig)

```typescript
import type { GameConfig } from './src/game/types/config';

interface GameConfig {
  /** Player control types (required) */
  playerTypes: ('human' | 'ai')[];

  /** Enabled layers */
  layers?: string[];

  /** Random seed for deterministic games */
  shuffleSeed?: number;

  /** Override dealing (for tests/teaching) */
  dealOverrides?: {
    initialHands?: Domino[][];  // Exactly 4 arrays of 7 dominoes
  };

  /** UI theme name */
  theme?: string;

  /** AI difficulty levels */
  aiDifficulty?: ('beginner' | 'intermediate' | 'expert')[];

  /** Time limits */
  timeLimits?: {
    perAction?: number;  // ms
    perHand?: number;    // ms
  };
}
```

### Available Layers

| Layer | Description |
|-------|-------------|
| `nello` | Enable nello trump (bidder must lose all tricks) |
| `splash` | Enable splash bids (3+ doubles) |
| `plunge` | Enable plunge bids (4+ doubles) |
| `sevens` | Enable sevens trump |
| `oneHand` | Single-hand mode (no full game) |
| `speed` | Auto-execute forced moves |
| `consensus` | Require player confirmation between phases |
| `hints` | Include hint metadata in actions |

## Minimal AI Client Example

```typescript
import { createLocalGame } from './src/multiplayer/local';
import type { GameView } from './src/multiplayer/types';
import type { GameAction } from './src/game/types';

const PLAYER_INDEX = 0;

const { client } = createLocalGame({
  playerTypes: ['human', 'ai', 'ai', 'ai'],
});

client.send({ type: 'JOIN', playerIndex: PLAYER_INDEX, name: 'SimpleBot' });

client.subscribe((view: GameView) => {
  // Filter to actions we can execute
  const myActions = view.validActions.filter((a) => {
    const action = a.action as GameAction;
    return !('player' in action) || action.player === PLAYER_INDEX;
  });

  if (myActions.length > 0) {
    // Pick randomly
    const chosen = myActions[Math.floor(Math.random() * myActions.length)];
    console.log(`[${view.state.phase}] Playing: ${chosen.label}`);

    client.send({ type: 'EXECUTE_ACTION', action: chosen.action });
  }
});
```

## Common Mistakes to Avoid

### Don't construct actions - pick from validActions
The server generates ALL valid actions for you. Don't build bid objects, don't construct trump selections, don't create play actions. Just pick from `validActions` and send it back:

```typescript
// WRONG - constructing your own action
client.send({ type: 'EXECUTE_ACTION', action: { type: 'bid', player: 0, bid: 'points', value: 30 } });

// RIGHT - picking from the server's menu
const chosen = view.validActions.find(a => a.label === 'Bid 30');
client.send({ type: 'EXECUTE_ACTION', action: chosen.action });
```

### Don't validate client-side
The `validActions` array contains exactly the actions you can execute. Don't implement follow-suit rules, bid validation, or turn order checking. If it's in `validActions`, it's valid.

### Don't expect request-response
After sending `EXECUTE_ACTION`, results come via the next `STATE_UPDATE`. Don't block waiting for a reply.

### Handle empty validActions
When `validActions` is empty or contains no actions for your player, it's not your turn. Just wait.

## Domino Reference

Dominoes are identified by `high-low` format (e.g., `"6-4"`, `"3-3"`).

| ID | Points |
|----|--------|
| 5-5 | 10 |
| 6-4 | 10 |
| 5-0 | 5 |
| 4-1 | 5 |
| 3-2 | 5 |
| All others | 0 |

Total points in a hand: 42 (hence the name).

## Game Phases

| Phase | Description | Valid Actions |
|-------|-------------|---------------|
| `setup` | Game initializing | None |
| `bidding` | Players bid for contract | `bid`, `pass` |
| `trump_selection` | Winner selects trump | `select-trump` |
| `playing` | Playing tricks | `play`, `complete-trick`, `agree-trick` |
| `scoring` | Tallying hand results | `score-hand`, `agree-score` |
| `game_end` | Game complete | None |
| `one-hand-complete` | One-hand mode ended | `retry-one-hand`, `new-one-hand` |

## Further Reading

- [CLIENT_QUICKSTART.md](./CLIENT_QUICKSTART.md) - Simplified getting started guide
- [rules.md](./rules.md) - Official Texas 42 game rules
- [MULTIPLAYER.md](./MULTIPLAYER.md) - Server-side architecture details
- [`src/multiplayer/GameClient.ts`](../src/multiplayer/GameClient.ts) - Reference client implementation
- [`src/multiplayer/local.ts`](../src/multiplayer/local.ts) - Local game setup
