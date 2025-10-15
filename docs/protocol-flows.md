# Game Protocol Message Flows

## Overview

This document describes the message sequences for various game operations in the client-server protocol.

## Core Principles

1. **Client is dumb**: Never computes game logic, only displays pre-calculated state
2. **Server is authoritative**: All game rules and validation happen server-side
3. **Protocol is transport-agnostic**: Same messages work over WebSocket, Worker, or direct calls
4. **AI clients are just clients**: They speak the same protocol as human clients

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
  |  view: { state, validActions }|
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
  |<------- STATE_UPDATE ---------|
  |  view: { state, validActions }|
  |  lastAction: { ... }          |
```

### 3. Execute Action (AI Client)

```
AI Client                       Server
  |                               |
  |<------- STATE_UPDATE ---------|  (AI receives state like any client)
  |  view: { state, validActions }|
  |                               |
  | [AI selects from validActions]|
  |                               |
  |------ EXECUTE_ACTION -------->|
  |  gameId: "abc123"             |
  |  playerId: 1                  |
  |  action: { type: "play", ... }|
  |                               |
  |<------- STATE_UPDATE ---------|  (Broadcast to all clients)
  |  view: { state, validActions }|
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
  |<------- STATE_UPDATE ---------|
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
  |  view: { state, validActions }|
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
  state: GameState,           // Complete game state
  validActions: [              // Pre-calculated valid moves
    {
      action: { type: "bid", ... },
      label: "Bid 30",
      shortcut: "3"
    },
    ...
  ],
  players: [                   // Player metadata
    { playerId: 0, controlType: "human", connected: true },
    ...
  ],
  metadata: { ... }
}
```

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