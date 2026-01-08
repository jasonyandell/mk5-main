# Texas 42 Multiplayer Architecture

## Overview

This document defines a simple, elegant multiplayer architecture for Texas 42. The design is inspired by [PartyKit](https://docs.partykit.io/), [Colyseus](https://docs.colyseus.io/), and [boardgame.io](https://boardgame.io/) - battle-tested patterns for real-time multiplayer games.

### Core Principle

**Room handles game logic. Clients send and receive. Wiring is external.**

```
┌─────────────────────────────────────────────────────────────┐
│  CLIENTS (Human UI, AI, Spectators)                         │
│  - All use identical GameClient interface                   │
│  - Send actions, receive state updates                      │
│  - Don't know how they're connected                         │
└─────────────────────────────────────────────────────────────┘
                            ↕ Socket
┌─────────────────────────────────────────────────────────────┐
│  ROOM (Game Authority)                                      │
│  - Owns game state                                          │
│  - Handles messages from any client                         │
│  - Broadcasts state to all clients                          │
│  - Doesn't know about transport                             │
└─────────────────────────────────────────────────────────────┘
                            ↕ Pure Functions
┌─────────────────────────────────────────────────────────────┐
│  GAME ENGINE (Pure Logic)                                   │
│  - State transitions, validation, Layers                    │
│  - Zero multiplayer awareness                               │
└─────────────────────────────────────────────────────────────┘
```

### What Makes This Simple

1. **Socket is the only interface** - `send()`, `onMessage()`, `close()`
2. **Room takes a `send` function** - Doesn't create or manage transport
3. **Clients are identical** - Human, AI, spectator all use GameClient
4. **Wiring is external** - Different environments wire things differently
5. **No promise correlation** - Fire-and-forget actions, updates via subscription

---

## Interfaces

### Socket (The Only Transport Abstraction)

```typescript
interface Socket {
  send(data: string): void;
  onMessage(handler: (data: string) => void): void;
  close(): void;
}
```

This is intentionally minimal. It matches:
- WebSocket API
- postMessage API
- Any bidirectional channel

### GameClient (What UI and AI Use)

```typescript
class GameClient {
  view: GameView | null = null;
  private listeners = new Set<(view: GameView) => void>();
  private socket: Socket;

  constructor(socket: Socket) {
    this.socket = socket;
    socket.onMessage((data) => this.handleMessage(JSON.parse(data)));
  }

  /** Send an action to the server. Fire-and-forget - result comes via subscription. */
  send(message: ClientMessage): void {
    this.socket.send(JSON.stringify(message));
  }

  /** Subscribe to state updates. Returns unsubscribe function. */
  subscribe(callback: (view: GameView) => void): () => void {
    if (this.view) callback(this.view);
    this.listeners.add(callback);
    return () => this.listeners.delete(callback);
  }

  /** Disconnect from the game. */
  disconnect(): void {
    this.socket.close();
  }

  private handleMessage(message: ServerMessage): void {
    switch (message.type) {
      case 'STATE_UPDATE':
        this.view = message.view;
        this.listeners.forEach(cb => cb(this.view!));
        break;
      case 'ERROR':
        console.error('Server error:', message.error);
        break;
    }
  }
}
```

**That's the entire client.** ~30 lines. No promise queues, no caches, no complexity.

### Room (Game Authority)

```typescript
class Room {
  private state: MultiplayerGameState;
  private clients = new Map<string, boolean>();  // clientId → connected
  private send: (clientId: string, message: ServerMessage) => void;

  constructor(
    config: GameConfig,
    send: (clientId: string, message: ServerMessage) => void
  ) {
    this.send = send;
    this.state = createInitialState(config);
  }

  /** Called when a client connects. */
  handleConnect(clientId: string): void {
    this.clients.set(clientId, true);
    this.sendStateToClient(clientId);
  }

  /** Called when a client sends a message. */
  handleMessage(clientId: string, message: ClientMessage): void {
    switch (message.type) {
      case 'EXECUTE_ACTION':
        this.executeAction(clientId, message.action);
        break;
      // ... other message types
    }
  }

  /** Called when a client disconnects. */
  handleDisconnect(clientId: string): void {
    this.clients.delete(clientId);
  }

  private executeAction(clientId: string, action: GameAction): void {
    const result = authorizeAndExecute(this.state, clientId, action);
    if (result.success) {
      this.state = result.value;
      this.broadcastState();
    } else {
      this.send(clientId, { type: 'ERROR', error: result.error });
    }
  }

  private broadcastState(): void {
    for (const [clientId] of this.clients) {
      this.sendStateToClient(clientId);
    }
  }

  private sendStateToClient(clientId: string): void {
    const view = buildFilteredView(this.state, clientId);
    this.send(clientId, { type: 'STATE_UPDATE', view });
  }
}
```

**Room doesn't know HOW to send messages** - it just calls the `send` function it was given.

---

## Wiring

Different environments wire Room and Clients differently. The Room and GameClient code is identical everywhere.

### Local (In-Process)

For development and offline play. Everything runs in browser main thread.

```typescript
function createLocalGame(config: GameConfig): GameClient {
  // Message routing
  const handlers = new Map<string, (data: string) => void>();

  // Room sends via handlers map
  const room = new Room(config, (clientId, message) => {
    const handler = handlers.get(clientId);
    if (handler) handler(JSON.stringify(message));
  });

  // Factory to create sockets for this room
  function createSocket(clientId: string): Socket {
    return {
      send: (data) => {
        // Client → Room (use queueMicrotask to match async behavior)
        queueMicrotask(() => room.handleMessage(clientId, JSON.parse(data)));
      },
      onMessage: (handler) => {
        handlers.set(clientId, handler);
      },
      close: () => {
        handlers.delete(clientId);
        room.handleDisconnect(clientId);
      }
    };
  }

  // Create human client
  const humanSocket = createSocket('player-0');
  room.handleConnect('player-0');
  const humanClient = new GameClient(humanSocket);

  // Create AI clients
  for (let i = 1; i < 4; i++) {
    const aiSocket = createSocket(`ai-${i}`);
    room.handleConnect(`ai-${i}`);
    const aiClient = new GameClient(aiSocket);
    attachAIBehavior(aiClient, i);  // Subscribe and send moves when it's their turn
  }

  return humanClient;
}
```

### Cloudflare Durable Object (Future)

For online multiplayer. Room runs on edge, clients connect via WebSocket.

```typescript
// worker/GameRoom.ts
export class GameRoom implements DurableObject {
  private room: Room;
  private sockets = new Map<string, WebSocket>();

  constructor(state: DurableObjectState) {
    this.room = new Room(config, (clientId, message) => {
      const ws = this.sockets.get(clientId);
      if (ws) ws.send(JSON.stringify(message));
    });
  }

  async fetch(request: Request): Promise<Response> {
    if (request.headers.get('Upgrade') === 'websocket') {
      const pair = new WebSocketPair();
      const [client, server] = Object.values(pair);

      const clientId = crypto.randomUUID();
      this.sockets.set(clientId, server);

      server.accept();
      server.addEventListener('message', (event) => {
        this.room.handleMessage(clientId, JSON.parse(event.data as string));
      });
      server.addEventListener('close', () => {
        this.sockets.delete(clientId);
        this.room.handleDisconnect(clientId);
      });

      this.room.handleConnect(clientId);
      return new Response(null, { status: 101, webSocket: client });
    }

    return new Response('Expected WebSocket', { status: 400 });
  }
}

// Browser client just uses WebSocket
function createOnlineClient(gameId: string): GameClient {
  const ws = new WebSocket(`wss://your-worker.dev/game/${gameId}`);
  const socket: Socket = {
    send: (data) => ws.send(data),
    onMessage: (handler) => ws.addEventListener('message', (e) => handler(e.data)),
    close: () => ws.close()
  };
  return new GameClient(socket);
}
```

**Same Room class. Same GameClient class. Different wiring.**

---

## AI Clients

AI players are just GameClients with AI behavior attached. No special treatment.

```typescript
function attachAIBehavior(client: GameClient, playerIndex: number): void {
  client.subscribe((view) => {
    const chosen = selectAIAction(view.state, playerIndex, view.validActions);
    if (!chosen) return;

    client.send({ type: 'EXECUTE_ACTION', action: chosen.action });
  });
}
```

The `selectAIAction` function uses the configured AI strategy (beginner/intermediate/random). AI responds instantly - timing/pacing is a UI concern, not game logic.

For computationally expensive AI:
- **Offline**: Run AI in a Web Worker, post result back
- **Online**: Run AI in browser (your compute) or on server (Cloudflare compute)

Room doesn't care how long AI takes. It's just another client that eventually sends an action.

---

## Protocol Messages

Minimal message types. No ceremony.

```typescript
// Client → Server
type ClientMessage =
  | { type: 'EXECUTE_ACTION'; action: GameAction }
  | { type: 'JOIN'; playerIndex: number; name: string }
  | { type: 'SET_CONTROL'; playerIndex: number; controlType: 'human' | 'ai' }

// Server → Client
type ServerMessage =
  | { type: 'STATE_UPDATE'; view: GameView }
  | { type: 'ERROR'; error: string }
```

That's it. No GAME_CREATED, SUBSCRIBE, UNSUBSCRIBE, PLAYER_STATUS, PROGRESS.
State updates are the universal response. Errors are errors.

---

## What We're NOT Building

1. **Progressive enhancement** (offline → online conversion) - Start local OR online, don't convert
2. **Complex session management** - ClientId is identity, capabilities come from config
3. **Promise-based actions** - Fire-and-forget, results via subscription
4. **Transport abstraction layer** - Socket interface is enough
5. **Reconnection logic** - Handle at WebSocket level, not in GameClient

---

## File Structure

```
src/
├── multiplayer/
│   ├── index.ts           # Public API exports
│   ├── types.ts           # All shared types (GameView, Capability, PlayerSession, etc.)
│   ├── protocol.ts        # Wire protocol (ClientMessage, ServerMessage)
│   ├── Socket.ts          # Transport interface (~10 lines)
│   ├── GameClient.ts      # Client class (~44 lines)
│   ├── capabilities.ts    # Capability builders + filtering
│   ├── authorization.ts   # Action authorization
│   ├── stateLifecycle.ts  # Multiplayer state CRUD
│   └── local.ts           # Local wiring + AI behavior (~120 lines)
├── server/
│   ├── Room.ts            # Game authority (~400 lines)
│   └── HeadlessRoom.ts    # Minimal API for tools
└── stores/
    └── gameStore.ts       # Svelte facade over GameClient

# Future
worker/
├── GameRoom.ts            # Durable Object wrapper (~50 lines)
└── index.ts               # Worker routing (~20 lines)
```

---

## Success Metrics

- **NetworkGameClient**: 550 lines → 40 lines
- **Room**: Remove ~100 lines of transport/cache code
- **gameStore.wireUpGame**: 30 lines → 10 lines
- **Total multiplayer code**: ~50% reduction
- **Concepts to understand**: Transport, Connection, NetworkGameClient, AIManager → Socket, GameClient, Room
