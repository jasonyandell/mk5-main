# Texas 42 Multiplayer Architecture

## Philosophy: Composition Over Complexity

The entire multiplayer system builds by composing small pure functions:

**Authorization:**
```
canPlayerExecuteAction + getValidActions → getValidActionsForPlayer
canPlayerExecuteAction + executeAction → authorizeAndExecute
```

**Transport:**
```
authorizeAndExecute + postMessage → OfflineGameClient
authorizeAndExecute + fetch/WebSocket → OnlineGameClient
```

**Variants (future):**
```
baseStateMachine + variantTransform → transformedStateMachine
```

This approach keeps the core engine unchanged while adding multiplayer as a thin, composable wrapper.

---

## What You Already Have

### Pure State Machine ✓

**Core functions** (see `src/game/core/`):
- `executeAction: (state: GameState, action: GameAction) → GameState` - actions.ts:16
- `getValidActions: (state: GameState) → GameAction[]` - gameEngine.ts:82
- `getNextStates: (state: GameState) → StateTransition[]` - gameEngine.ts:315

**Key properties:**
- Pure functions, no side effects
- Complete action history in `state.actionHistory`
- Deterministic replay via seed + actions
- Already handles 4-player turn-based flow

### Implicit Authorization ✓

**Player view filtering** (playerView.ts:8-22):
```typescript
// Logic exists but needs extraction
const validTransitions = allTransitions.filter(transition => {
  if (!('player' in action)) return true;  // Neutral actions
  return action.player === playerId;       // Player-specific actions
});
```

**What this gives you:**
- Actions already tagged with player index
- Consensus actions (agree-complete-trick, agree-score-hand) available to all
- Turn-based actions restricted to currentPlayer

### Local AI ✓

**Pure strategy functions** (controllers/strategies.ts):
- AI executes via pure functions, no controller needed
- Server will spawn these as workers
- Already hot-swappable (change `playerTypes` array)

---

## What You're Building

### 1. Core Types

Minimal types to wrap existing `GameState`:

```typescript
interface PlayerSession {
  playerId: string              // Unique identifier
  playerIndex: 0 | 1 | 2 | 3   // Which seat (null for spectators)
  isConnected: boolean
  name: string
  capabilities: Set<Capability> // Static permissions set at creation
}

interface MultiplayerGameState {
  gameId: string
  coreState: GameState          // Existing game state, unchanged
  players: readonly PlayerSession[]
  managedAI: Set<number>        // Which seats have server-spawned AI
  createdAt: number
  lastActionAt: number
}

interface ActionRequest {
  playerId: string
  action: GameAction
  timestamp: number
}

// Railway-oriented programming: explicit success/failure paths
type Result<T, E = string> =
  | { success: true; value: T }
  | { success: false; error: E }

// Helper for chaining operations that might fail
function chain<A, B>(
  result: Result<A>,
  fn: (value: A) => Result<B>
): Result<B> {
  if (!result.success) return result;
  return fn(result.value);
}

// Example usage:
chain(validateSession(playerId), session =>
  chain(authorizeAction(session, action), () =>
    executeAction(state, action)
  )
)

// Future: capabilities for spectators, coaches, etc.
type Capability =
  | 'act-as-player'       // Can execute actions for their seat
  | 'observe-own-hand'    // See own cards only
  | 'observe-all-hands'   // Spectator mode
  | 'see-hints'           // Tutorial/coach mode
```

### 2. Core Functions (Extract + Compose)

**Extract implicit authorization:**
```typescript
// Extract from playerView.ts filtering logic
canPlayerExecuteAction(
  playerIndex: number,
  action: GameAction,
  state: GameState
): boolean {
  // Neutral actions (no player field) available to all
  if (!('player' in action)) {
    return true;
  }

  // Player-specific actions must match index
  return action.player === playerIndex;
}
```

**Compose for session-aware filtering:**
```typescript
getValidActionsForPlayer(
  mpState: MultiplayerGameState,
  playerId: string
): GameAction[] {
  // Find session
  const session = mpState.players.find(p => p.playerId === playerId);
  if (!session || session.playerIndex === null) return [];

  // Get all valid actions
  const allActions = getValidActions(mpState.coreState);

  // Filter by authorization
  return allActions.filter(action =>
    canPlayerExecuteAction(session.playerIndex, action, mpState.coreState)
  );
}
```

**Compose for authorized execution:**
```typescript
authorizeAndExecute(
  mpState: MultiplayerGameState,
  request: ActionRequest
): Result<MultiplayerGameState> {
  // Find session
  const session = mpState.players.find(p => p.playerId === request.playerId);
  if (!session) {
    return { success: false, error: 'Player not found' };
  }

  // Check authorization
  if (!canPlayerExecuteAction(session.playerIndex, request.action, mpState.coreState)) {
    return { success: false, error: 'Action not authorized' };
  }

  // Execute action (pure function from actions.ts)
  const newCoreState = executeAction(mpState.coreState, request.action);

  // Return updated multiplayer state
  return {
    success: true,
    value: {
      ...mpState,
      coreState: newCoreState,
      lastActionAt: request.timestamp
    }
  };
}
```

### 3. Transport Layer

**Unified interface:**
```typescript
interface GameClient {
  // State queries
  getState(): Promise<MultiplayerGameState>

  // Action execution
  executeAction(request: ActionRequest): Promise<Result<MultiplayerGameState>>

  // Session management
  joinGame(playerId: string, name: string): Promise<Result<PlayerSession>>
  leaveGame(playerId: string): Promise<void>

  // Filtered actions for player
  getActions(playerId: string): Promise<GameAction[]>

  // Real-time updates
  subscribe(callback: (state: MultiplayerGameState) => void): () => void
}
```

**Offline implementation** (Web Worker):
```typescript
class WebWorkerGameClient implements GameClient {
  private worker: Worker;
  private state: MultiplayerGameState;

  async executeAction(request: ActionRequest): Promise<Result<MultiplayerGameState>> {
    // postMessage to worker
    // Worker runs: authorizeAndExecute(state, request)
    // Broadcast new state to all subscribers
  }

  // All other methods use postMessage communication
}
```

**Online implementation** (Cloudflare Durable Object):
```typescript
class DurableObjectGameClient implements GameClient {
  private gameId: string;
  private ws: WebSocket;

  async executeAction(request: ActionRequest): Promise<Result<MultiplayerGameState>> {
    // POST /api/games/:gameId/actions
    // Server runs: authorizeAndExecute(state, request)
    // WebSocket broadcasts new state
  }

  subscribe(callback: (state: MultiplayerGameState) => void): () => void {
    // WebSocket connection for real-time updates
  }
}
```

**Key insight:** Same `authorizeAndExecute` function in both transports. Offline uses postMessage, online uses HTTP/WebSocket.

### 4. AI Lifecycle Management

**Server tracks which seats have AI:**
```typescript
createMultiplayerGame(
  gameId: string,
  config: {
    initialState: GameState,
    playerNames: string[]
  }
): MultiplayerGameState {
  const mpState = {
    gameId,
    coreState: config.initialState,
    players: [],
    managedAI: new Set<number>(),
    createdAt: Date.now(),
    lastActionAt: Date.now()
  };

  // Spawn AI workers for AI-controlled seats
  for (let i = 0; i < 4; i++) {
    if (config.initialState.playerTypes[i] === 'ai') {
      spawnAIWorker(gameId, i, 'beginner');
      mpState.managedAI.add(i);
    }
  }

  return mpState;
}
```

**Hot-swap: Human replaces AI:**
```typescript
handleJoinRequest(
  mpState: MultiplayerGameState,
  playerId: string,
  playerIndex: number
): Result<MultiplayerGameState> {
  // Kill AI worker if seat occupied by AI
  if (mpState.managedAI.has(playerIndex)) {
    killAIWorker(mpState.gameId, playerIndex);
    mpState.managedAI.delete(playerIndex);
  }

  // Add human session
  const session: PlayerSession = {
    playerId,
    playerIndex,
    isConnected: true,
    name: `Player ${playerIndex + 1}`,
    capabilities: new Set(['act-as-player', 'observe-own-hand'])
  };

  return {
    success: true,
    value: {
      ...mpState,
      players: [...mpState.players, session]
    }
  };
}
```

**Step away: AI replaces human:**
```typescript
handleLeaveRequest(
  mpState: MultiplayerGameState,
  playerId: string
): MultiplayerGameState {
  const session = mpState.players.find(p => p.playerId === playerId);
  if (!session) return mpState;

  // Spawn AI for this seat
  spawnAIWorker(mpState.gameId, session.playerIndex, 'random');
  mpState.managedAI.add(session.playerIndex);

  // Remove human session
  return {
    ...mpState,
    players: mpState.players.filter(p => p.playerId !== playerId)
  };
}
```

**Critical:** Game engine never checks `managedAI`. Only lifecycle manager knows about AI.

### 5. Progressive Enhancement

**Take online** (one-way state migration):
```typescript
async function upgradeToOnline(
  localState: GameState,
  myPlayerId: string
): Promise<DurableObjectGameClient> {
  // Create server game with current state
  const response = await fetch('/api/games', {
    method: 'POST',
    body: JSON.stringify({
      initialState: localState,
      playerTypes: localState.playerTypes
    })
  });

  const { gameId } = await response.json();

  // Switch to online client
  const onlineClient = new DurableObjectGameClient(gameId);
  await onlineClient.joinGame(myPlayerId, 'Player 1');

  // Server already spawned AI workers during create
  return onlineClient;
}
```

**Go offline** (state handoff, single player):
```typescript
async function downgradeToOffline(
  onlineClient: DurableObjectGameClient
): Promise<WebWorkerGameClient> {
  // Fetch current state
  const mpState = await onlineClient.getState();

  // Create local client with core state
  const offlineClient = new WebWorkerGameClient();
  await offlineClient.createGame(mpState.coreState);

  // Server game abandoned (cleaned up by TTL)
  return offlineClient;
}
```

**Key insight:** Pure functional core makes migration trivial. Just pass `GameState` between transports.

---

## How It Works: Complete Flow Example

Let's trace one action through the entire system to see how all pieces compose together.

### Scenario: Alice bids 30 in an online game

**Initial state:**
- Game created with Alice (player 0), AI players at seats 1-3
- Phase: bidding, currentPlayer: 0 (Alice's turn)
- Alice connected via `DurableObjectGameClient`

### Step-by-step flow:

**1. UI Layer (Client)**
```typescript
// Alice clicks "Bid 30" button
const action: GameAction = { type: 'bid', player: 0, bid: 'points', value: 30 };

// Client creates request
const request: ActionRequest = {
  playerId: 'alice',
  action: action,
  timestamp: Date.now()
};

// Send to server
const result = await gameClient.executeAction(request);
```

**2. Transport Layer (HTTP)**
```typescript
// DurableObjectGameClient.executeAction()
async executeAction(request: ActionRequest): Promise<Result<MultiplayerGameState>> {
  const response = await fetch(`/api/games/${this.gameId}/actions`, {
    method: 'POST',
    body: JSON.stringify(request)
  });

  return response.json();  // Returns Result<MultiplayerGameState>
}
```

**3. Server: Authorization Check**
```typescript
// Durable Object receives request
async handleAction(request: ActionRequest): Promise<Result<MultiplayerGameState>> {
  // Find session
  const session = this.state.players.find(p => p.playerId === request.playerId);
  if (!session) {
    return { success: false, error: 'Player not found' };
  }

  // Check authorization
  if (!canPlayerExecuteAction(session.playerIndex, request.action, this.state.coreState)) {
    return { success: false, error: 'Not authorized to bid for player 0' };
  }

  // Continue to execution...
}
```

**4. Server: State Execution (Pure)**
```typescript
// authorizeAndExecute composition
const newCoreState = executeAction(this.state.coreState, request.action);

// executeAction (from actions.ts:16) - PURE FUNCTION
// 1. Validates bid is legal (uses rules.ts)
// 2. Adds bid to state.bids
// 3. Advances to next player (player 1)
// 4. Returns new immutable state

const newMpState: MultiplayerGameState = {
  ...this.state,
  coreState: newCoreState,
  lastActionAt: request.timestamp
};
```

**5. Server: Persist & Broadcast**
```typescript
// Save to durable storage
this.state = newMpState;

// Broadcast to all connected clients via WebSocket
for (const ws of this.connections) {
  ws.send(JSON.stringify({
    type: 'state-update',
    state: newMpState
  }));
}

// Return success
return { success: true, value: newMpState };
```

**6. Client: UI Update**
```typescript
// WebSocket handler receives broadcast
gameClient.onStateUpdate = (state: MultiplayerGameState) => {
  // Update Svelte store
  gameState.set(state.coreState);

  // UI reactively updates to show:
  // - Alice's bid in bidding table
  // - Current player now P1 (AI)
  // - Waiting indicator
};
```

**7. AI Client: Triggered**
```typescript
// AI worker for player 1 receives same state update
aiClient.onStateUpdate = async (state: MultiplayerGameState) => {
  // Check if it's our turn
  if (state.coreState.currentPlayer !== this.playerIndex) return;

  // Get valid actions
  const actions = await this.gameClient.getActions(this.playerId);

  // Run strategy (pure function)
  const chosen = this.strategy(state.coreState, this.playerIndex, actions);

  // Execute chosen action
  await this.gameClient.executeAction({
    playerId: this.playerId,
    action: chosen,
    timestamp: Date.now()
  });
};

// Chosen action: { type: 'pass', player: 1 }
```

**8. Loop: AI Action Flows Through Same Pipeline**
```
AI pass action
  → Transport (POST /api/games/123/actions)
  → Authorization (player 1 can pass? yes)
  → Execution (executeAction adds pass to bids, moves to player 2)
  → Broadcast (all clients updated)
  → Next AI (player 2) triggered
  → ...continues until bidding complete
```

### Key Observations

**Pure functions at the core:**
- `executeAction` doesn't know about multiplayer, sessions, or transport
- Same function works in offline (Worker) and online (Durable Object)
- Deterministic: same state + action = same result

**Result type keeps errors explicit:**
- Authorization failures return `{ success: false, error: '...' }`
- No exceptions thrown across async boundaries
- Errors compose through `chain` helper

**Composition throughout:**
- `canPlayerExecuteAction` + `executeAction` → `authorizeAndExecute`
- `authorizeAndExecute` + `fetch` → `DurableObjectGameClient`
- `authorizeAndExecute` + `postMessage` → `WebWorkerGameClient`

**AI is just another client:**
- Uses same `GameClient` interface
- Server doesn't know it's AI (except for `managedAI` set for lifecycle)
- AI strategy is pure function: `(state, playerIndex, actions) → action`

**State flows one direction:**
```
User Action
  ↓
Request
  ↓
Authorization
  ↓
Pure State Transition
  ↓
Broadcast
  ↓
All Clients Updated
  ↓
AI Clients Respond (if their turn)
```

---

## Extensions (Future)

### Capabilities

**Static permissions set at creation:**
```typescript
// Regular player
const playerSession: PlayerSession = {
  playerId: 'alice',
  playerIndex: 0,
  capabilities: new Set(['act-as-player', 'observe-own-hand'])
};

// Spectator
const spectatorSession: PlayerSession = {
  playerId: 'bob',
  playerIndex: null,  // Not seated
  capabilities: new Set(['observe-all-hands'])
};

// Coach/tutorial mode
const coachSession: PlayerSession = {
  playerId: 'coach',
  playerIndex: null,
  capabilities: new Set(['observe-all-hands', 'see-hints'])
};
```

**Authorization becomes:**
```typescript
canPlayerExecuteAction(session: PlayerSession, action: GameAction): boolean {
  if (!session.capabilities.has('act-as-player')) return false;
  if (!('player' in action)) return true;
  return action.player === session.playerIndex;
}
```

### Variants

**Traditional Texas 42 has many rule variants.** Examples:

**Nello:** Partner sits out after winning bid
**Plunge:** Partner leads first trick
**Near Seven:** Score by proximity to 7 pips, not trump strength

**Variants transform the state machine:**
```typescript
type Variant = (baseMachine: StateMachine) → StateMachine

interface StateMachine {
  getValidActions: (state: GameState) → GameAction[]
  executeAction: (state: GameState, action: GameAction) → GameState
}
```

**Example - Plunge variant:**
```typescript
const plungeVariant: Variant = (base) => ({
  ...base,
  executeAction: (state, action) => {
    const newState = base.executeAction(state, action);

    // After selecting trump on plunge bid, partner leads
    if (action.type === 'select-trump' &&
        newState.currentBid.type === 'plunge') {
      const partner = (newState.winningBidder + 2) % 4;
      return { ...newState, currentPlayer: partner };
    }

    return newState;
  }
});
```

**Composition:**
```typescript
// Apply multiple variants
const machine = compose(nelloVariant, plungeVariant)(baseStateMachine);

// Store variant IDs in state for serialization
interface GameState {
  // ... existing fields
  variants?: string[];  // ['nello', 'plunge']
}
```

**Variants are surgical modifications:** Intercept specific mechanics (bidding, scoring, turn order) without touching core engine.

---

## Implementation Sequence

### Phase 1: Types + Authorization Extraction (~1 hour)

**Files to create:**
- `src/game/multiplayer/types.ts` - Core types
- `src/game/multiplayer/authorization.ts` - Extract authorization logic

**Work:**
- Define `PlayerSession`, `MultiplayerGameState`, `ActionRequest`, `Result`
- Extract `canPlayerExecuteAction` from playerView.ts filtering
- Build `getValidActionsForPlayer` by composing existing functions
- Build `authorizeAndExecute` wrapper

**Test:**
- Unit tests for authorization edge cases
- Verify composition: same behavior as existing playerView filtering

### Phase 2: Offline Transport (~2 hours)

**Files to create:**
- `src/game/multiplayer/GameClient.ts` - Interface
- `src/game/multiplayer/WebWorkerGameClient.ts` - Implementation
- `src/game/multiplayer/game.worker.ts` - Worker logic

**Work:**
- Define `GameClient` interface
- Implement Web Worker transport using `postMessage`
- Worker holds `MultiplayerGameState`, runs `authorizeAndExecute`
- Subscribe pattern for state updates

**Test:**
- Local 4-player game via worker
- Verify AI lifecycle (spawn/kill)
- Test hot-swap (human ↔ AI)

### Phase 3: Online Transport + Deployment (~4 hours)

**Files to create:**
- `src/game/multiplayer/DurableObjectGameClient.ts` - Client
- `server/game-room.ts` - Durable Object class
- `server/api/games.ts` - HTTP endpoints

**Work:**
- Implement Durable Object with same `authorizeAndExecute` logic
- HTTP POST for actions, WebSocket for subscriptions
- Deploy to Cloudflare Workers

**Test:**
- Cross-device multiplayer
- Connection handling (disconnect/reconnect)
- AI spawning on server

### Phase 4: Extensions (~2-4 hours)

**Capabilities:**
- Add capability checking to authorization
- Implement spectator mode
- Add coach/tutorial hints

**Variants:**
- Define `Variant` type and `compose` helper
- Implement 2-3 variants (nello, plunge, near-seven)
- Update `GameState` to store active variants
- URL serialization for variants

---

## Architecture Invariants

**Pure functions everywhere:** All state transitions remain pure. `MultiplayerGameState` is just a wrapper.

**Server agnostic to game logic:** Server only knows about sessions and authorization. Game rules live in core engine.

**Transport independence:** `authorizeAndExecute` works identically in offline (Worker) and online (Durable Object) modes.

**Composability:** Every function builds by composing smaller functions. No monoliths.

**Progressive enhancement:** Offline → online is just state migration. No sync protocol needed.

---

## Summary

You're adding **~500 lines of multiplayer code** to wrap your existing **pure functional core**:

- Extract implicit authorization → explicit functions (100 lines)
- Define types for sessions and requests (50 lines)
- Build `GameClient` interface (50 lines)
- Implement offline transport via Worker (150 lines)
- Implement online transport via Durable Object (150 lines)

Everything else (variants, capabilities, spectators) builds on this foundation through composition.

**The core insight:** Your state machine is already multiplayer-ready. You just need to add session management and authorization as a thin, composable layer.
