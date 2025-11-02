# GameServer + GameKernel Architecture Refactoring Plan

## Problem Statement

The Texas 42 multiplayer architecture has evolved to have several critical issues that prevent it from scaling to different deployment modes (local, Web Worker, Cloudflare Durable Objects):

### Current Architecture Problems

1. **Circular Dependencies**: The `InProcessAdapter` creates an `InProcessGameServer`, which needs a reference back to the adapter to spawn AI clients. This circular dependency makes the code hard to test, reason about, and impossible to deploy in Web Worker or Cloudflare environments.

2. **Mixed Responsibilities**: The `InProcessGameServer` class handles too many concerns:
   - Protocol message routing
   - Game state management (via GameHost)
   - AI client lifecycle management
   - Subscription handling
   - This violates single responsibility principle and makes the code brittle.

3. **Poor Naming**: The current `GameHost` class doesn't actually "host" anything - it's just the pure game logic engine. This naming confusion makes the architecture harder to understand.

4. **Transport Coupling**: The game logic is too aware of transport concerns. AI spawning is baked into the server layer, and the adapter pattern doesn't cleanly separate transport from game logic.

5. **Deployment Mode Coupling**: The current code can't run in Web Workers (workers can't instantiate main-thread objects) or Cloudflare Durable Objects (can't spawn external processes directly).

### Vision: Clean Service Boundaries

We need an architecture where:
- The game logic engine knows nothing about networking, AI, or where it's running
- AI clients are truly independent actors that happen to run in the same process
- The same code runs identically whether deployed locally, in a Web Worker, or in Cloudflare
- There are no circular dependencies - everything has clear ownership hierarchy
- Each component has a single, well-defined responsibility

### Solution: GameServer + GameKernel Pattern

**GameKernel**: The pure game logic engine (renamed from GameHost)
- Only knows about game state, rules, and actions
- Zero knowledge of transport, AI, or deployment
- Pure functional core that could run anywhere

**GameServer**: The orchestrator that clients connect to
- Creates and manages the GameKernel
- Handles all protocol messages
- Manages AI client lifecycle
- Routes messages between clients and kernel
- IS "the server" from the client's perspective

This refactoring is a "big bang" approach - we'll update everything at once to move to the new, cleaner architecture without maintaining backwards compatibility.

## Architecture Overview

The new architecture establishes a clear hierarchy with no circular dependencies:

```
Clients → Transport → GameServer → GameKernel
                            ↓
                        AIManager → AI Clients
```

**GameServer IS the orchestrator** - it's "the server" from any client's perspective. When clients "connect to a server", they're connecting to GameServer.

### Ownership Hierarchy
- GameServer creates and owns GameKernel (one direction)
- GameServer creates and owns AIManager (one direction)
- GameServer holds reference to Transport (for broadcasting)
- Transport routes incoming messages to GameServer
- No circular dependencies!

### Creation Flow
The exact order of creation is critical:

1. `const gameServer = new GameServer(config)` - Creates GameKernel internally, spawns AI in constructor
2. `const transport = new InProcessTransport()` - Create transport layer
3. `gameServer.setTransport(transport)` - GameServer gets transport for broadcasting
4. `transport.setGameServer(gameServer)` - Transport gets GameServer for message routing
5. `const connection = transport.connect(clientId)` - Clients connect through transport

This bidirectional wiring (step 3 & 4) is acceptable because:
- GameServer depends on Transport interface (abstraction)
- InProcessTransport depends on GameServer (concrete)
- This follows Dependency Inversion Principle
- Enables easy offline→online migration

## Implementation Plan

### Phase 1: Create GameKernel from GameHost

#### 1.1 Rename and Relocate
**FROM:** `src/server/game/GameHost.ts`
**TO:** `src/kernel/GameKernel.ts`

#### 1.2 Interface Definition
```typescript
// src/kernel/GameKernel.ts
export class GameKernel {
  private mpState: MultiplayerGameState;
  private subscribers: Map<string, { perspective?: string; listener: (update: KernelUpdate) => void }>;

  constructor(gameId: string, config: GameConfig, players: PlayerSession[]) {
    // Pure initialization - no external dependencies
  }

  // Core methods (keep from existing GameHost):
  executeAction(playerId: string, action: GameAction): Result<void>
  getView(playerId?: string): GameView
  getState(): MultiplayerGameState
  subscribe(playerId: string | undefined, callback: (update: KernelUpdate) => void): () => void

  // Methods to ADD:
  updatePlayerControl(playerIndex: number, controlType: 'human' | 'ai'): void
  getPlayers(): PlayerInfo[]
}
```

#### 1.3 Changes Required
- Remove any import of adapters or transport layers
- Rename types: `HostViewUpdate` → `KernelUpdate`
- Keep all game logic, variant composition, and state management
- Ensure no AI-specific code remains

### Phase 2: Create GameServer

#### 2.1 New File Structure
```typescript
// src/server/GameServer.ts
export class GameServer {
  private kernel: GameKernel;
  private aiManager: AIManager;
  private transport?: Transport;  // GameServer holds Transport reference for broadcasting
  private connections: Map<string, Connection> = new Map();
  private gameId: string;

  constructor(config: GameConfig) {
    // 1. Generate game ID
    this.gameId = `game-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;

    // 2. Create GameKernel with initial seats
    const players = this.createInitialPlayers(config);
    this.kernel = new GameKernel(this.gameId, config, players);

    // 3. Create AIManager
    this.aiManager = new AIManager();

    // 4. Spawn AI immediately based on config (in constructor)
    for (let i = 0; i < 4; i++) {
      if (config.playerTypes[i] === 'ai') {
        this.spawnAI(i);
      }
    }

    // 5. Subscribe to kernel updates
    this.kernel.subscribe(undefined, (update) => {
      this.broadcastUpdate(update);
    });
  }

  setTransport(transport: Transport): void {
    this.transport = transport;
  }
}
```

**GameServer's Dual Role is Intentional:**
1. **Orchestrator**: Routes messages between clients and kernel
2. **AI Coordinator**: Manages AI lifecycle

This pragmatic design ensures the same code works in all deployment modes (local/Worker/Cloudflare). Alternative approaches would require different code for each deployment mode.

#### 2.2 Core Responsibilities
```typescript
class GameServer {
  // Connection Management
  acceptConnection(clientId: string): Connection
  disconnect(clientId: string): void

  // Protocol Message Handling (moved from InProcessGameServer)
  handleMessage(clientId: string, message: ClientMessage): void
  private handleCreateGame(config: GameConfig): void
  private handleExecuteAction(playerId: string, action: GameAction): void
  private handleSetPlayerControl(playerIndex: number, controlType: 'human' | 'ai'): void
  private handleSubscribe(playerId?: string): void

  // AI Lifecycle (moved from InProcessGameServer)
  private spawnAI(playerIndex: number, playerId: string): void
  private destroyAI(playerIndex: number): void

  // Broadcasting
  private broadcast(message: ServerMessage): void
  private broadcastToPlayer(playerId: string, message: ServerMessage): void
}
```

#### 2.3 Key Design Decisions
- GameServer creates GameKernel, not vice versa
- AI clients are internal to GameServer
- All protocol handling happens here
- Single source of truth for connections

### Phase 3: Refactor InProcessGameServer

#### 3.1 New Role
Transform from "does everything" to "thin local wrapper"

```typescript
// src/server/offline/InProcessGameServer.ts
export class InProcessGameServer {
  private gameServer: GameServer;

  constructor(config: GameConfig) {
    this.gameServer = new GameServer(config);
  }

  // Simple delegation to GameServer
  handleMessage(message: ClientMessage, emit: (msg: ServerMessage) => void): void {
    // Just forward to GameServer
  }
}
```

#### 3.2 What Gets Removed
- ❌ AI spawning logic (lines 283-320)
- ❌ Direct GameHost/Kernel management
- ❌ Complex message routing
- ❌ Player session management

#### 3.3 What Remains
- ✅ Wrapper for local deployment
- ✅ Simple message forwarding
- ✅ Emission callback handling

### Phase 4: Remove Adapter Layer Entirely

The adapter layer is being removed completely in favor of direct Transport usage. This simplifies the architecture and removes an unnecessary abstraction.

#### 4.1 Delete InProcessAdapter
The `InProcessAdapter` class will be deleted entirely. Clients will connect directly through the Transport layer instead.

#### 4.2 Transport Takes Its Place
```typescript
// src/server/transports/InProcessTransport.ts
export class InProcessTransport implements Transport {
  private gameServer?: GameServer;
  private clients: Map<string, (message: ServerMessage) => void> = new Map();

  setGameServer(server: GameServer): void {
    this.gameServer = server;
  }

  // Clients connect directly to transport
  connect(clientId: string): Connection {
    return {
      send: (msg: ClientMessage) => {
        // Route to GameServer
        this.gameServer?.handleMessage(clientId, msg);
      },
      onMessage: (handler: (msg: ServerMessage) => void) => {
        // Store handler for this client
        this.clients.set(clientId, handler);
      },
      disconnect: () => {
        this.clients.delete(clientId);
      }
    };
  }

  // GameServer uses this to broadcast
  send(clientId: string, message: ServerMessage): void {
    const handler = this.clients.get(clientId);
    handler?.(message);
  }
}

### Phase 5: Extract AI Management

#### 5.1 Create AIManager
```typescript
// src/server/ai/AIManager.ts
export class AIManager {
  private aiClients: Map<number, AIClient> = new Map();

  spawnAI(
    seat: number,
    gameId: string,
    onAction: (action: GameAction) => void
  ): void {
    // Create AI client
    // Wire up action callback
    // Start AI
  }

  destroyAI(seat: number): void {
    // Stop AI
    // Clean up resources
  }

  updateState(seat: number, view: GameView): void {
    // Push state update to AI
  }

  destroyAll(): void {
    // Clean shutdown of all AI
  }
}
```

#### 5.2 Integrate with GameServer
```typescript
class GameServer {
  private aiManager: AIManager;

  constructor(config: GameConfig) {
    this.aiManager = new AIManager();
    // ...
  }

  private spawnAI(seat: number): void {
    this.aiManager.spawnAI(
      seat,
      this.gameId,
      (action) => {
        // AI action comes back through callback
        this.kernel.executeAction(`ai-${seat}`, action);
      }
    );
  }
}
```

#### 5.3 Simplify AIClient
```typescript
// src/game/multiplayer/AIClient.ts
export class AIClient {
  // REMOVE: adapter reference
  // REMOVE: protocol knowledge

  // ADD: Simple callbacks
  constructor(
    seat: number,
    playerId: string,
    onAction: (action: GameAction) => void
  ) {
    // Much simpler!
  }

  updateState(view: GameView): void {
    // Receive state, decide action
    if (isMyTurn) {
      const action = selectAIAction(view);
      this.onAction(action);
    }
  }
}
```

#### 5.4 AI Lifecycle Ownership

**Critical Design Decision:** GameServer spawns AI internally, not external application code.

- **AI is spawned in GameServer constructor** based on config.playerTypes
- **AI lifecycle is tied to game lifecycle** - when game ends, AI is destroyed
- **AI behaves externally** - uses callbacks, no special privileges, no direct kernel access
- **AI is managed internally** - GameServer controls spawning/destruction for simplicity

This distinction means:
- **Architecturally**: AI is an external actor with no special code paths, uses same interface as human clients
- **Operationally**: GameServer manages AI lifecycle to ensure consistent behavior across deployment modes

The rationale:
- GameServer knows which seats need AI from config
- Can spawn AI immediately on game creation
- Ensures AI is always present when needed
- Simplifies deployment to Worker/Cloudflare (no external AI spawning needed)

### Phase 6: Create Transport Abstraction

#### 6.1 Define Transport Interface
```typescript
// src/server/transports/Transport.ts

// Connection object returned when client connects
export interface Connection {
  send: (message: ClientMessage) => void;
  onMessage: (handler: (message: ServerMessage) => void) => void;
  disconnect: () => void;
}

// Transport interface for server-side message handling
export interface Transport {
  // Server-side message broadcasting
  send(clientId: string, message: ServerMessage): void;

  // Lifecycle
  start(): Promise<void>;
  stop(): Promise<void>;
}
```

#### 6.2 Implement InProcessTransport
```typescript
// src/server/transports/InProcessTransport.ts
export class InProcessTransport implements Transport {
  private gameServer?: GameServer;
  private clients: Map<string, (message: ServerMessage) => void> = new Map();

  setGameServer(server: GameServer): void {
    this.gameServer = server;
  }

  // Client connects directly via method call - returns Connection object
  connect(clientId: string): Connection {
    return {
      send: (msg: ClientMessage) => {
        // Route to GameServer
        this.gameServer?.handleMessage(clientId, msg);
      },
      onMessage: (handler: (msg: ServerMessage) => void) => {
        // Store handler for this client
        this.clients.set(clientId, handler);
      },
      disconnect: () => {
        this.clients.delete(clientId);
        this.gameServer?.disconnect(clientId);
      }
    };
  }

  // GameServer uses this to broadcast
  send(clientId: string, message: ServerMessage): void {
    const handler = this.clients.get(clientId);
    handler?.(message);
  }
}
```

#### 6.3 Future Transport Implementations
```typescript
// src/server/transports/WorkerTransport.ts (future)
export class WorkerTransport implements Transport {
  // Uses postMessage
}

// src/server/transports/CloudflareTransport.ts (future)
export class CloudflareTransport implements Transport {
  // Uses WebSocket
}
```

#### 6.4 Transport-GameServer Relationship

**GameServer holds a Transport reference for broadcasting:**
- This is acceptable coupling because Transport is an interface (abstraction)
- Enables easy offline→online migration
- GameServer can broadcast without knowing transport implementation details
- Transport routes incoming messages to GameServer

The relationship is bidirectional but not circular:
- GameServer depends on Transport interface (abstraction) - for broadcasting
- InProcessTransport depends on GameServer (concrete) - for message routing
- This follows Dependency Inversion Principle
- No circular dependency because GameServer doesn't create Transport

This design enables:
- GameServer to broadcast to all clients without knowing how
- Transport to route messages without knowing game logic
- Easy swapping of transport implementations (local/Worker/WebSocket)

### Phase 7: Update Client Connection Flow

#### 7.1 Refactor gameStore.ts
```typescript
// src/stores/gameStore.ts

// OLD (with adapter):
const adapter = new InProcessAdapter();
const gameClient = new NetworkGameClient(adapter, config);

// NEW (direct transport, no adapter):
// Step 1: Create GameServer
const gameServer = new GameServer(config);

// Step 2: Create Transport
const transport = new InProcessTransport();

// Step 3: Wire them together (bidirectional but not circular)
gameServer.setTransport(transport);
transport.setGameServer(gameServer);

// Step 4: Client connects through transport
const connection = transport.connect('player-0');
const gameClient = new NetworkGameClient(connection);
```

#### 7.2 Remove Test Mode Hacks
- Remove URL parameter checking
- Remove hardcoded player types
- Let GameServer handle initial configuration

### Phase 8: Test Migration Strategy

#### 8.1 Test File Updates

**Files using createGameAuthority:**
- Replace with `new GameKernel()`
- Update imports from `server/game/createGameAuthority` to `kernel/GameKernel`

**Files using InProcessAdapter:**
- Create GameServer instead
- Use InProcessTransport for connection

**Example migration:**
```typescript
// OLD test:
const host = createGameAuthority(gameId, config, sessions);
host.executeAction(playerId, action);

// NEW test:
const kernel = new GameKernel(gameId, config, sessions);
kernel.executeAction(playerId, action);
```

#### 8.2 New Test Structure
```
tests/
├── kernel/
│   ├── GameKernel.test.ts      (pure logic tests)
│   └── variants.test.ts        (variant composition)
├── server/
│   ├── GameServer.test.ts      (orchestration tests)
│   └── ai/
│       └── AIManager.test.ts   (AI lifecycle)
├── transport/
│   └── InProcessTransport.test.ts
└── integration/
    └── full-game.test.ts       (end-to-end with GameServer)
```

## File Structure Transformation

### New Directory Structure
```
src/
├── kernel/                         [NEW]
│   ├── GameKernel.ts              (from server/game/GameHost.ts)
│   ├── types.ts                   (kernel-specific types)
│   └── __tests__/
│       └── GameKernel.test.ts
│
├── server/
│   ├── GameServer.ts              [NEW - main orchestrator]
│   ├── ai/                        [NEW]
│   │   └── AIManager.ts          (extracted from InProcessGameServer)
│   ├── transports/                [NEW]
│   │   ├── Transport.ts          (interface)
│   │   ├── InProcessTransport.ts
│   │   ├── WorkerTransport.ts    (future)
│   │   └── CloudflareTransport.ts (future)
│   ├── offline/
│   │   └── InProcessGameServer.ts [SIMPLIFIED - just a wrapper]
│   └── game/
│       └── createGameAuthority.ts [DELETE - replaced by GameKernel]
│
├── game/
│   └── multiplayer/
│       └── AIClient.ts            [SIMPLIFIED - no adapter dependency]
│
└── stores/
    └── gameStore.ts               [UPDATED - uses GameServer]
```

### Import Path Changes

**Before:**
```typescript
import { GameHost } from '../server/game/GameHost';
import { createGameAuthority } from '../server/game/createGameAuthority';
import { InProcessAdapter } from '../server/offline/InProcessAdapter';
```

**After:**
```typescript
import { GameKernel } from '../kernel/GameKernel';
import { GameServer } from '../server/GameServer';
import { InProcessTransport } from '../server/transports/InProcessTransport';
```

### Deleted Files
- `src/server/game/createGameAuthority.ts` - Functionality moved to GameKernel constructor
- `src/server/game/GameHost.ts` - Renamed to GameKernel
- `src/server/offline/InProcessAdapter.ts` - Removed entirely, replaced by Transport

## Implementation Order

1. **Create GameKernel** (rename GameHost, ensure pure)
2. **Create Transport abstraction** (interface + InProcessTransport)
3. **Create AIManager** (extract from InProcessGameServer)
4. **Create GameServer** (the new orchestrator)
5. **Gut InProcessGameServer** (make it a thin wrapper)
6. **Delete InProcessAdapter** (remove entirely)
7. **Update AIClient** (simplify to callbacks)
8. **Update gameStore** (use GameServer + Transport)
9. **Fix all tests** (bulk update)
10. **Delete obsolete code** (createGameAuthority, InProcessAdapter, etc.)

## Validation Checkpoints

After each step, validate:
- ✅ No circular dependencies introduced
- ✅ Tests still compile (may fail temporarily)
- ✅ No transport knowledge in GameKernel
- ✅ No game logic in transport layer

## Critical Success Factors

1. **GameKernel must be pure** - No imports from server/, no adapter references
2. **GameServer owns everything** - Creates kernel, manages AI, handles connections
3. **No circular dependencies** - Strict top-down ownership
4. **Transport agnostic** - Same GameServer code for local/Worker/Cloudflare

## Common Pitfalls to Avoid

- ❌ Don't let GameKernel know about transport
- ❌ Don't let AIClient access adapters directly
- ❌ Don't create adapter before server
- ❌ Don't mix protocol handling with game logic

## Expected Outcomes

After this refactoring:

1. **Clean Architecture**: Clear separation of concerns with no circular dependencies
2. **Deployment Flexibility**: Same code runs in local/Worker/Cloudflare modes
3. **Better Testing**: Each component can be tested in isolation
4. **Clearer Names**: GameServer and GameKernel accurately describe their roles
5. **Maintainability**: Single responsibility per component makes changes easier

The transformation is from:
```
Circular mess: Adapter ↔ Server ↔ AI
```

To:
```
Clean hierarchy: GameServer → GameKernel
                          → AIManager → AIClients
                          → Transport → Clients
```

This creates a system where each component has a clear role and the entire architecture can be deployed in any JavaScript environment without modification.