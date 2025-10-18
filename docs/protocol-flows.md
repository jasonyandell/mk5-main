# Game Protocol Message Flows

## Overview

This document describes the message sequences for various game operations in the client-server protocol.

## Core Principles

1. **Client is dumb**: Never computes game logic, only displays pre-calculated state
2. **Server is authoritative**: All game rules and validation happen server-side
3. **Protocol is transport-agnostic**: Same messages work over WebSocket, Worker, or direct calls
4. **AI clients are just clients**: They speak the same protocol as human clients
5. **State is filtered**: Clients receive only the information they're authorized to see based on capabilities

## Message Flows

### 1. Create New Game

```
Client                          Server
  |                               |
  |-------- CREATE_GAME --------->|
  |  config: {                    |
  |    playerTypes: [...],        |
  |    variant: { ... }           |
  |  }                            |
  |                               |
  |<------- GAME_CREATED ---------|
  |  gameId: "abc123"             |
  |  view: { filteredState, ... } |
  |                               |
  |<------ PLAYER_STATUS ---------|  (Server spawns AI clients)
  |  playerId: 1                  |
  |  status: "joined"             |
  |  controlType: "ai"            |
```

### 2. Execute Action (Human Player)

```
Client                          Server
  |                               |
  |------ EXECUTE_ACTION -------->|
  |  gameId: "abc123"             |
  |  playerId: 0                  |
  |  action: { type: "bid", ... } |
  |                               |
  |<------- STATE_UPDATE ---------|  (Filtered based on capabilities)
  |  view: { filteredState, ... } |
  |  lastAction: { ... }          |
```

### 3. Execute Action (AI Client)

```
AI Client                       Server
  |                               |
  |<------- STATE_UPDATE ---------|  (AI receives filtered state like any client)
  |  view: { filteredState, ... } |  (Only sees own hand, not opponents')
  |                               |
  | [AI selects from validActions]|
  |                               |
  |------ EXECUTE_ACTION -------->|
  |  gameId: "abc123"             |
  |  playerId: 1                  |
  |  action: { type: "play", ... }|
  |                               |
  |<------- STATE_UPDATE ---------|  (Broadcast to all clients, each filtered)
  |  view: { filteredState, ... } |
```

### 4. Change Player Control (Human -> AI)

```
Client                          Server
  |                               |
  |---- SET_PLAYER_CONTROL ------>|
  |  gameId: "abc123"             |
  |  playerId: 0                  |
  |  controlType: "ai"            |
  |                               |
  |<------ PLAYER_STATUS ---------|
  |  playerId: 0                  |
  |  status: "control_changed"    |
  |  controlType: "ai"            |
  |                               |
                                  | (Server spawns AI client)
                                  |
AI Client <----- SUBSCRIBE -------|
  |                               |
  |<------- STATE_UPDATE ---------|  (Filtered for player 0's view)
  |  view: { filteredState, ... } |
```

### 5. Change Player Control (AI -> Human)

```
Client                          Server
  |                               |
  |---- SET_PLAYER_CONTROL ------>|
  |  gameId: "abc123"             |
  |  playerId: 1                  |
  |  controlType: "human"         |
  |                               |
                                  | (Server kills AI client)
                                  |
  |<------ PLAYER_STATUS ---------|
  |  playerId: 1                  |
  |  status: "control_changed"    |
  |  controlType: "human"         |
```

### 6. One-Hand Mode with Seed Finding

```
Client                          Server
  |                               |
  |-------- CREATE_GAME --------->|
  |  variant: {                   |
  |    type: "one-hand"           |
  |  }                            |
  |                               |
  |<-------- PROGRESS ------------|  (While finding seed)
  |  operation: "seed_finding"    |
  |  progress: 25                 |
  |                               |
  |<-------- PROGRESS ------------|
  |  progress: 50                 |
  |                               |
  |<------- GAME_CREATED ---------|
  |  view: { filteredState, ... } |
```

### 7. Error Handling

```
Client                          Server
  |                               |
  |------ EXECUTE_ACTION -------->|
  |  action: { invalid }          |
  |                               |
  |<--------- ERROR --------------|
  |  error: "Invalid action"      |
  |  requestType: "EXECUTE_ACTION"|
```

## State Synchronization

### Subscription Model

1. Clients explicitly subscribe to game updates via `SUBSCRIBE` message
2. Server broadcasts `STATE_UPDATE` to all subscribed clients after each action
3. Clients can unsubscribe with `UNSUBSCRIBE` message

### View Calculation

The `GameView` sent in `STATE_UPDATE` messages contains:

```typescript
{
  state: FilteredGameState,    // Filtered based on client capabilities
  validActions: [               // Pre-calculated valid moves for this client
    {
      action: { type: "bid", ... },
      label: "Bid 30",
      shortcut: "3"
    },
    ...
  ],
  players: [                    // Player metadata with capabilities
    { playerId: 0, controlType: "human", connected: true, capabilities: [...] },
    ...
  ],
  metadata: { ... }
}
```

**State Filtering**:

The `FilteredGameState` is created by `GameHost.createView()` which:
1. Calls `getVisibleStateForSession(state, session)` to filter based on capabilities
2. Hides opponent hands unless client has `observe-all-hands` capability
3. Filters action metadata based on capabilities (hints, AI intent, etc.)
4. Returns only information the client is authorized to see

**Capability-Based Views**:

Different clients receive different views of the same game state:

- **Player (default)**: Can see only their own hand
  - Capabilities: `act-as-player`, `observe-own-hand`
  - Other players' hands appear as empty arrays with `handCount` only

- **Spectator**: Can see all hands
  - Capabilities: `observe-all-hands`, `observe-full-state`
  - Full visibility for observation/coaching

- **Debug/Admin**: Can see everything including internal metadata
  - Capabilities: `observe-full-state`, `see-hints`, `see-ai-intent`
  - No filtering applied

**Implementation**: `src/game/multiplayer/capabilityUtils.ts:getVisibleStateForSession()`

## Security & Information Hiding

### Client Can Only See What They Should

The protocol enforces information hiding through type-safe filtering:

1. **Server-side filtering**: `GameHost.createView()` filters state before transmission
2. **Type safety**: `FilteredGameState` type ensures clients can't access unfiltered data
3. **Capability-based**: Different clients receive different views based on their role

### Example: Player View vs Spectator View

**Same game state, different views**:

```typescript
// Player 0's view (observe-own-hand capability)
{
  state: {
    players: [
      { id: 0, hand: [<7 dominoes>], handCount: 7 },      // Own hand visible
      { id: 1, hand: [], handCount: 7 },                   // Hidden
      { id: 2, hand: [], handCount: 7 },                   // Hidden
      { id: 3, hand: [], handCount: 7 }                    // Hidden
    ],
    // ... rest of state
  }
}

// Spectator's view (observe-all-hands capability)
{
  state: {
    players: [
      { id: 0, hand: [<7 dominoes>], handCount: 7 },      // Visible
      { id: 1, hand: [<7 dominoes>], handCount: 7 },      // Visible
      { id: 2, hand: [<7 dominoes>], handCount: 7 },      // Visible
      { id: 3, hand: [<7 dominoes>], handCount: 7 }       // Visible
    ],
    // ... rest of state
  }
}
```

### AI Clients Receive Filtered Views Too

AI clients are just clients - they receive the same filtered `GameView`:

- AI for player 1 receives view with only player 1's hand visible
- AI cannot peek at opponent hands (type system prevents it)
- AI strategies work with `FilteredGameState` (structurally compatible)

This ensures AI plays fairly without access to hidden information.

## Transport Adapters

### InProcessAdapter (Local Games)

```
NetworkGameClient -> InProcessAdapter -> GameHost
                          |
                          v
                     AIClient (spawned)
```

- Direct method calls, no serialization
- AIClient runs in same process
- Used for single-player and hot-seat multiplayer

### WorkerAdapter (Future)

```
NetworkGameClient -> WorkerAdapter -> [Worker Boundary] -> GameHost
                                                              |
                                                              v
                                                         AIClient
```

- Messages serialized across Worker boundary
- Game runs in Web Worker for performance
- AI also in Worker

### WebSocketAdapter (Future)

```
NetworkGameClient -> WebSocketAdapter -> [Network] -> Server -> GameHost
                                                                    |
                                                                    v
                                                               AIClient
```

- Messages serialized to JSON
- Game runs on remote server
- AI runs on server

## Migration Strategy

### Phase 1: InProcessAdapter
- Replace LocalGameClient with NetworkGameClient + InProcessAdapter
- Same behavior, new architecture
- Feature flag for safe rollback

### Phase 2: WorkerAdapter
- Move game to Web Worker
- Better performance for complex games
- UI remains responsive

### Phase 3: WebSocketAdapter
- Enable true multiplayer
- Server authoritative
- Multiple clients can connect