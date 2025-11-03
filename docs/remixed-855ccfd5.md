# Texas 42 Multiplayer Architecture Specification

## 1. Overview

### 1.1 Purpose

This specification defines a purely functional, composable multiplayer architecture for Texas 42 where:
- Server logic is transport-agnostic (online via Cloudflare Workers or offline via Web Workers)
- AI clients are independent actors communicating through the same API as human clients
- Game variants compose via function transformers
- Information visibility and action authorization are controlled through a capability system

### 1.2 Architecture Principles

**Pure Function Composition**: All game logic expressed as pure functions that compose naturally:
```
GameState → Pure Function → GameState → Pure Function → GameState
```

**Server-Client Symmetry**: Server and clients use identical core functions. The server adds persistence and broadcasting; clients add UI and input.

**AI as External Actor**: AI clients connect as regular clients. The server has no AI-specific logic.

**Variants as Transformers**: Game rule modifications are functions that transform the state machine, stored in game state, and applied at runtime.

**Capabilities Control Access**: What players can do (actions) and see (information) is governed by composable capability tokens.

### 1.3 System Architecture

```
┌─────────────────────────────────────────┐
│         CLIENT LAYER                    │
│  Human UI | AI Clients | Spectators     │
│  All use GameClient interface           │
└─────────────────────────────────────────┘
              ↕ HTTP/WebSocket/postMessage
┌─────────────────────────────────────────┐
│         SERVER LAYER                    │
│  Online: Cloudflare Durable Object      │
│  Offline: Web Worker                    │
│  Pure state management only             │
└─────────────────────────────────────────┘
              ↕ Pure Functions
┌─────────────────────────────────────────┐
│         MULTIPLAYER LAYER               │
│  Authorization & Visibility             │
│  Pure functional combinators            │
└─────────────────────────────────────────┘
              ↕ Pure Functions
┌─────────────────────────────────────────┐
│         CORE GAME ENGINE                │
│  Existing Texas 42 logic                │
│  Zero multiplayer awareness             │
└─────────────────────────────────────────┘
```

---

## 2. Core Game Engine Layer

### 2.1 Responsibility

Pure game logic with no multiplayer awareness. This is the existing Texas 42 engine.

### 2.2 Key Functions

```typescript
// Pure state transition
executeAction: (GameState, GameAction) → GameState

// Generate all structurally possible actions
getValidActions: (GameState) → GameAction[]

// Check if specific action is valid
isValidAction: (GameState, GameAction) → boolean
```

### 2.3 Constraints

The core engine must not know about:
- Player identities or authentication
- Network transport
- Persistence
- Whether players are human or AI
- Game variants (these are applied in the multiplayer layer)

---

## 3. Multiplayer Layer

### 3.1 Data Structures

#### 3.1.1 Player Session

```typescript
interface PlayerSession {
  playerId: string              // Unique identifier (UUID, user ID, etc)
  playerIndex: 0 | 1 | 2 | 3   // Which seat (immutable after joining)
  isConnected: boolean          // Current connection status
  name: string                  // Display name
  capabilities: Capability[]    // What this player can do/see
}
```

#### 3.1.2 Multiplayer Game State

```typescript
interface MultiplayerGameState {
  gameId: string                           // Unique game identifier
  coreState: GameState                     // Core Texas 42 state
  players: readonly PlayerSession[]        // 0-4 players
  createdAt: number                        // Timestamp
  lastActionAt: number                     // Last activity timestamp
  enabledActionTransformers: ActionTransformer[]  // Active rule modifications
}
```

#### 3.1.3 Action Request

```typescript
interface ActionRequest {
  playerId: string          // Who is requesting
  action: GameAction        // What action to execute
  timestamp: number         // When requested
}
```

#### 3.1.4 Result Type

```typescript
type Result<T, E = string> = 
  | { success: true; value: T }
  | { success: false; error: E }
```

### 3.2 Core Functions

#### 3.2.1 Game Creation

```typescript
createMultiplayerGame: (
  gameId: string,
  coreState: GameState,
  players: PlayerSession[]
) → MultiplayerGameState
```

Creates immutable game state with given players and initial core state.

#### 3.2.2 Action Authorization

```typescript
canPlayerExecuteAction: (
  coreState: GameState,
  playerIndex: number,
  action: GameAction
) → boolean
```

**Purpose**: Determine if a specific player can execute a specific action.

**Rules**:
1. Actions with explicit `player` field must match the player's index
2. Consensus actions (agree-complete-trick, agree-score-hand) can be performed by any player
3. System actions (complete-trick, score-hand, redeal) can be initiated by current player when conditions are met
4. Default: action requires it to be the player's turn (currentPlayer === playerIndex)

**Example**:
```typescript
// Player 2 trying to bid when it's player 1's turn
canPlayerExecuteAction(
  state,
  2,
  { type: 'bid', player: 1, value: 30 }
) // → false

// Player 1 bidding on their turn
canPlayerExecuteAction(
  state,
  1,
  { type: 'bid', player: 1, value: 30 }
) // → true
```

#### 3.2.3 Action Filtering

```typescript
getValidActionsForPlayer: (
  mpState: MultiplayerGameState,
  playerId: string
) → GameAction[]
```

**Purpose**: Get all actions a specific player can currently execute.

**Behavior**:
1. Find player session by playerId
2. If player not found or not connected, return empty array
3. Get all valid actions from core engine
4. Apply action transformers (if any)
5. Filter by authorization (canPlayerExecuteAction)
6. Return filtered list

**Example**:
```typescript
getValidActionsForPlayer(state, 'alice')
// → [{ type: 'bid', value: 30 }, { type: 'bid', value: 32 }, { type: 'pass' }]

getValidActionsForPlayer(state, 'bob')
// → [] (not bob's turn)
```

#### 3.2.4 Authorized Execution

```typescript
authorizeAndExecute: (
  mpState: MultiplayerGameState,
  request: ActionRequest
) → Result<MultiplayerGameState>
```

Validates authorization and executes action if allowed. Returns new state or error.

#### 3.2.5 Player Management

```typescript
addPlayer: (
  mpState: MultiplayerGameState,
  playerId: string,
  name: string,
  capabilities: Capability[]
) → Result<MultiplayerGameState>

removePlayer: (
  mpState: MultiplayerGameState,
  playerId: string
) → MultiplayerGameState
```

**addPlayer**: Add new player or reconnect existing. Finds available slot or returns error if full.

**removePlayer**: Mark player as disconnected (preserves player in game state).

---

## 4. Capability System

### 4.1 Purpose

Replace boolean flags (isAI, isSpectator) with composable capability tokens that control both what actions players can execute and what information they can see.

### 4.2 Capability Types

```typescript
type Capability = 
  // Action Capabilities
  | { type: 'act-as-player'; playerIndex: number }
  | { type: 'replace-ai' }
  | { type: 'configure-variant' }
  
  // Visibility Capabilities
  | { type: 'observe-own-hand' }
  | { type: 'observe-hand'; playerIndex: number }
  | { type: 'observe-all-hands' }
  | { type: 'observe-full-state' }
  | { type: 'see-hints' }
  | { type: 'see-ai-intent' }
```

### 4.3 Standard Capability Sets

```typescript
// Human player
humanCapabilities = (playerIndex: number) → [
  { type: 'act-as-player', playerIndex },
  { type: 'observe-own-hand' }
]

// AI player (can be replaced by human)
aiCapabilities = (playerIndex: number) → [
  { type: 'act-as-player', playerIndex },
  { type: 'observe-own-hand' },
  { type: 'replace-ai' }
]

// Spectator (watch only)
spectatorCapabilities = [
  { type: 'observe-all-hands' },
  { type: 'observe-full-state' }
]

// Coach (can see student's hand)
coachCapabilities = (studentIndex: number) → [
  { type: 'observe-hand', playerIndex: studentIndex },
  { type: 'see-hints' }
]

// Tutorial student
tutorialCapabilities = (playerIndex: number) → [
  { type: 'act-as-player', playerIndex },
  { type: 'observe-own-hand' },
  { type: 'see-hints' },
  { type: 'undo-actions' }
]
```

### 4.4 Information Authorization

#### 4.4.1 State Visibility Filtering

```typescript
getVisibleState: (
  fullState: MultiplayerGameState,
  viewingPlayerId: string
) → MultiplayerGameState
```

**Purpose**: Filter game state based on what the viewing player's capabilities allow them to see.

**Behavior**:
1. Find viewing player's session
2. Check capabilities
3. Filter player hands based on observe capabilities
4. Return personalized view of state

**Example**:
```typescript
// Human player sees only their hand
getVisibleState(fullState, 'alice')
// → state where players[1,2,3].hand is empty or hidden

// Spectator sees all hands
getVisibleState(fullState, 'spectator-1')
// → full state with all hands visible

// Coach sees student's hand (index 0)
getVisibleState(fullState, 'coach-1')
// → state where players[0].hand is visible, others hidden
```

#### 4.4.2 Action Metadata Filtering

```typescript
getVisibleActions: (
  actions: GameAction[],
  viewingPlayer: PlayerSession
) → GameAction[]
```

**Purpose**: Filter action metadata (hints, AI intent, etc) based on capabilities.

**Behavior**:
1. Check player's capabilities
2. Remove metadata fields player cannot see
3. Return filtered actions

**Example**:
```typescript
// Actions with hints added by variant
const actions = [
  { type: 'bid', value: 30, hint: 'Safe bid' },
  { type: 'bid', value: 35, hint: 'Risky but strong' }
]

// Player with see-hints capability
getVisibleActions(actions, playerWithHints)
// → actions unchanged, hints visible

// Player without see-hints capability
getVisibleActions(actions, playerWithoutHints)
// → [{ type: 'bid', value: 30 }, { type: 'bid', value: 35 }]
```

### 4.5 Integration Pattern

```
1. ActionTransformers modify state machine → actions with metadata
2. Authorization filters actions → only actions player can execute
3. Visibility filters metadata → only metadata player can see
4. Client receives personalized view
```

---

## 5. ActionTransformer System

### 5.1 Purpose

Enable compositional game rule modifications that can be mixed, matched, and stored in game state.

### 5.2 Core Types

```typescript
// A state machine produces actions from state
type StateMachine = (state: GameState) → GameAction[]

// An action transformer transforms a state machine
type ActionTransformer = (baseMachine: StateMachine) → StateMachine

// Parameterized action transformers
type ActionTransformerFactory<P> = (params: P) → ActionTransformer
```

### 5.3 How ActionTransformers Work

ActionTransformers are function transformers. They take a state machine (function that produces actions) and return a modified state machine.

**Composition Pattern**:
```typescript
const baseRules = getValidActions
const modifiedRules = actionTransformer1(actionTransformer2(actionTransformer3(baseRules)))

// Or using pipe
const finalRules = pipe(
  baseRules,
  actionTransformer1,
  actionTransformer2,
  actionTransformer3
)

const actions = finalRules(state)
```

### 5.4 ActionTransformer Operations

ActionTransformers can perform four operations on actions:

1. **Filter**: Remove actions (tournament mode removes special contracts)
2. **Transform**: Modify action properties (speed mode adds autoExecute flag)
3. **Annotate**: Add metadata (showHints adds hint field)
4. **Replace**: Swap action types (single hand mode replaces score-hand with end-game)

### 5.5 Standard ActionTransformers

#### 5.5.1 Tournament Mode

**Purpose**: Disable special contracts (nello, splash, plunge).

**Implementation Pattern**:
```typescript
const tournamentMode: ActionTransformer = (baseMachine) => (state) => {
  const actions = baseMachine(state)

  return actions.filter(action =>
    action.type !== 'bid' ||
    ['points', 'marks'].includes(action.bid)
  )
}
```

**Effect**: Removes nello, splash, plunge from available bids.

#### 5.5.2 Forced Bid Minimum

**Purpose**: First bidder cannot pass, must bid at least minimum value.

**Implementation Pattern**:
```typescript
const forcedBidMinimum = (minBid: number): ActionTransformer => {
  return (baseMachine) => (state) => {
    const actions = baseMachine(state)

    if (state.phase !== 'bidding') return actions

    const hasBids = state.bids.some(b => b.type !== 'pass')
    if (hasBids) return actions

    return actions.filter(action => {
      if (action.type === 'pass') return false
      if (action.type === 'bid' && action.bid === 'points') {
        return action.value >= minBid
      }
      return true
    })
  }
}
```

**Usage**: `forcedBidMinimum(30)` requires first bid to be 30+.

#### 5.5.3 Nello Must Play Alone

**Purpose**: During nello, partner doesn't play their dominoes.

**Implementation Pattern**:
```typescript
const nelloMustPlayAlone: ActionTransformer = (baseMachine) => (state) => {
  const actions = baseMachine(state)

  if (state.phase !== 'playing') return actions

  // Check if partner bid nello
  const currentPlayerIndex = state.currentPlayer
  const partnerIndex = (currentPlayerIndex + 2) % 4
  const winningBid = state.winningBid

  if (winningBid?.player === partnerIndex && winningBid?.type === 'nello') {
    // Partner is playing nello - current player passes their turn
    return [{ type: 'auto-pass' }]
  }

  return actions
}
```

**Effect**: When a player bids and plays nello, their partner automatically passes all tricks.

#### 5.5.4 Speed Mode

**Purpose**: Auto-play when only one valid option.

**Implementation Pattern**:
```typescript
const speedMode: ActionTransformer = (baseMachine) => (state) => {
  const actions = baseMachine(state)

  if (state.phase !== 'playing') return actions

  const playActions = actions.filter(a => a.type === 'play')

  if (playActions.length === 1) {
    return [{
      ...playActions[0],
      autoExecute: true,
      delay: 300
    }]
  }

  return actions
}
```

**Effect**: Single valid play is marked for auto-execution by client.

#### 5.5.5 Single Hand Mode

**Purpose**: End game after one hand with results screen.

**Implementation Pattern**:
```typescript
const singleHandMode = (seed: string): ActionTransformer => {
  return (baseMachine) => (state) => {
    const actions = baseMachine(state)

    if (state.phase === 'scoring' &&
        state.consensus.scoreHand.size === 4) {
      return actions.map(action => {
        if (action.type === 'score-hand') {
          return {
            type: 'end-game',
            winner: determineWinner(state),
            finalScore: state.score,
            seed: seed,
            replayUrl: `/replay/${seed}`
          }
        }
        return action
      })
    }

    return actions
  }
}
```

**Usage**: `singleHandMode('tutorial-1')` or `singleHandMode('daily-2025-01-15')`.

#### 5.5.6 Show Hints

**Purpose**: Add hint metadata to actions for tutorial mode.

**Implementation Pattern**:
```typescript
const showHints: ActionTransformer = (baseMachine) => (state) => {
  const actions = baseMachine(state)

  return actions.map(action => ({
    ...action,
    hint: generateHint(state, action)
  }))
}
```

**Effect**: All actions get hint field. Visibility filtering determines who sees it.

**Integration with Capabilities**:
```typescript
// Tutorial player has see-hints capability
const tutorialPlayer = {
  capabilities: [
    { type: 'act-as-player', playerIndex: 0 },
    { type: 'see-hints' }
  ]
}

// When getting actions for tutorial player:
// 1. showHints action transformer adds hints
// 2. getVisibleActions checks see-hints capability
// 3. Hints are included in result

// When getting actions for AI:
// 1. showHints action transformer adds hints
// 2. getVisibleActions checks capabilities (no see-hints)
// 3. Hints are stripped from result
```

#### 5.5.7 Daily Challenge

**Purpose**: Single-hand game with difficulty settings and scoring.

**Implementation Pattern**:
```typescript
const dailyChallenge = (params: {
  date: string,
  difficulty: 'easy' | 'medium' | 'hard',
  starThresholds: number[]
}): ActionTransformer => {
  const seed = `daily-${params.date}-${params.difficulty}`

  return (baseMachine) => (state) => {
    let actions = baseMachine(state)

    // Difficulty modifiers during bidding
    if (params.difficulty === 'hard' && state.phase === 'bidding') {
      actions = actions.filter(a =>
        a.type !== 'bid' || a.bid !== 'nello'
      )
    }

    // End game with star rating
    if (state.phase === 'scoring' &&
        state.consensus.scoreHand.size === 4) {
      const score = state.score[0]
      const stars = calculateStars(score, params.starThresholds)

      actions = actions.map(action => {
        if (action.type === 'score-hand') {
          return {
            type: 'end-game',
            seed: seed,
            stars: stars,
            shareText: `I scored ${stars}⭐ on ${params.date}'s ${params.difficulty} challenge!`
          }
        }
        return action
      })
    }

    return actions
  }
}
```

**Usage**:
```typescript
dailyChallenge({
  date: '2025-01-15',
  difficulty: 'hard',
  starThresholds: [25, 35, 42]  // 1-star, 2-star, 3-star thresholds
})
```

### 5.6 ActionTransformer Composition

ActionTransformers compose naturally through function composition:

```typescript
// Tournament rules
const tournamentGame = {
  enabledActionTransformers: [
    tournamentMode,
    forcedBidMinimum(30)
  ]
}

// Tutorial mode
const tutorialGame = {
  enabledActionTransformers: [
    singleHandMode('tutorial-1'),
    speedMode,
    showHints,
    tournamentMode  // Simplify rules
  ]
}

// Daily challenge
const dailyGame = {
  enabledActionTransformers: [
    dailyChallenge({
      date: '2025-01-15',
      difficulty: 'hard',
      starThresholds: [25, 35, 42]
    }),
    speedMode
  ]
}
```

### 5.7 ActionTransformer Application

ActionTransformers are stored in game state and applied when generating actions:

```typescript
function getValidActionsWithActionTransformers(
  state: MultiplayerGameState
): GameAction[] {
  // Start with base state machine
  let machine: StateMachine = getValidActions

  // Apply each action transformer in order
  for (const actionTransformer of state.enabledActionTransformers) {
    machine = actionTransformer(machine)
  }

  // Execute final composed machine
  return machine(state.coreState)
}
```

**Order matters**: ActionTransformers apply left-to-right. If two action transformers conflict, the rightmost wins.

---

## 6. Transport Layer

### 6.1 Purpose

Provide I/O adapters that execute the pure multiplayer functions over different transports (HTTP, WebSocket, postMessage).

### 6.2 Unified Client Interface

All clients (online and offline) implement the same interface:

```typescript
interface GameClient {
  getState(): Promise<MultiplayerGameState>
  
  executeAction(
    request: ActionRequest
  ): Promise<Result<MultiplayerGameState>>
  
  joinGame(
    playerId: string,
    name: string,
    capabilities: Capability[]
  ): Promise<Result<PlayerSession>>
  
  leaveGame(playerId: string): Promise<void>
  
  getActions(playerId: string): Promise<GameAction[]>
  
  subscribe(
    callback: (state: MultiplayerGameState) => void
  ): () => void
}
```

### 6.3 Online Mode (Cloudflare Workers)

#### 6.3.1 Durable Object

One Durable Object instance per game. Handles:
- State persistence (Durable Object storage)
- WebSocket connections for real-time updates
- HTTP endpoints for REST operations

**Endpoints**:
- `POST /initialize` - Create new game
- `GET /state` - Get current state (filtered by viewing player)
- `POST /action` - Execute action
- `POST /join` - Add player
- `POST /leave` - Remove player
- `GET /actions?playerId=X` - Get valid actions for player
- `GET /ws` - WebSocket upgrade for real-time updates

**Pattern**:
```typescript
class GameDurableObject {
  async handleExecuteAction(request) {
    // Load state from storage
    const state = await this.storage.get('state')
    
    // Use pure function
    const result = authorizeAndExecute(state, request)
    
    if (result.success) {
      // Persist
      await this.storage.put('state', result.value)
      
      // Broadcast to all connected WebSockets
      await this.broadcast(result.value)
    }
    
    return jsonResponse(result)
  }
}
```

#### 6.3.2 HTTP Client

Browser client that communicates with Durable Object:

```typescript
class OnlineGameClient implements GameClient {
  async executeAction(request: ActionRequest) {
    const response = await fetch(
      `${this.baseUrl}/action?gameId=${this.gameId}`,
      { method: 'POST', body: JSON.stringify(request) }
    )
    return await response.json()
  }
  
  subscribe(callback) {
    const ws = new WebSocket(`${this.baseUrl}/ws?gameId=${this.gameId}`)
    ws.onmessage = (e) => {
      const data = JSON.parse(e.data)
      if (data.type === 'state-update') {
        callback(data.state)
      }
    }
    return () => ws.close()
  }
}
```

### 6.4 Offline Mode (Web Workers)

#### 6.4.1 Game Server Worker

Web Worker that holds state in memory:

```typescript
// worker.ts
let currentState: MultiplayerGameState | null = null

self.onmessage = (event) => {
  const { type, payload } = event.data
  
  switch (type) {
    case 'execute-action':
      const result = authorizeAndExecute(currentState, payload)
      if (result.success) {
        currentState = result.value
        self.postMessage({ type: 'state-update', state: currentState })
      }
      self.postMessage({ type: 'action-result', result })
      break
    
    // ... other message types
  }
}
```

#### 6.4.2 Worker Client

Browser client that communicates via postMessage:

```typescript
class OfflineGameClient implements GameClient {
  private worker: Worker
  private subscribers = new Set<Function>()
  
  constructor(workerUrl: string) {
    this.worker = new Worker(workerUrl)
    this.worker.onmessage = (e) => {
      if (e.data.type === 'state-update') {
        this.subscribers.forEach(cb => cb(e.data.state))
      }
    }
  }
  
  async executeAction(request: ActionRequest) {
    return this.sendRequest({ type: 'execute-action', payload: request })
  }
  
  subscribe(callback) {
    this.subscribers.add(callback)
    return () => this.subscribers.delete(callback)
  }
}
```

### 6.5 Client Factory

```typescript
function createGameClient(
  mode: 'online' | 'offline',
  config: {
    baseUrl?: string
    gameId: string
    workerUrl?: string
  }
): GameClient {
  if (mode === 'online') {
    return new OnlineGameClient(config.baseUrl, config.gameId)
  } else {
    return new OfflineGameClient(config.workerUrl)
  }
}
```

### 6.6 Progressive Mode

Offline games can transition to online:

```typescript
// Start offline
const offlineClient = createGameClient('offline', {
  gameId: 'local-game',
  workerUrl: '/worker.js'
})

// User clicks "Allow others to join"
async function goOnline() {
  // Get current state
  const state = await offlineClient.getState()
  
  // POST to server
  const response = await fetch('/create-game', {
    method: 'POST',
    body: JSON.stringify(state)
  })
  const { gameId } = await response.json()
  
  // Switch to online client
  const onlineClient = createGameClient('online', {
    baseUrl: WORKER_URL,
    gameId: gameId
  })
  
  // Original player continues seamlessly
  return { onlineClient, gameId, shareUrl: `${DOMAIN}/join/${gameId}` }
}
```

---

## 7. AI Client Layer

### 7.1 Purpose

Independent actors that play the game by connecting as regular clients.

### 7.2 AI Strategy

AI strategies are pure functions:

```typescript
type AIStrategy = (
  gameState: GameState,
  playerIndex: number,
  validActions: GameAction[]
) → GameAction
```

**Reference Implementation (Random)**:
```typescript
const randomStrategy: AIStrategy = (state, index, actions) => {
  return actions[Math.floor(Math.random() * actions.length)]
}
```

**Note**: More sophisticated strategies (rule-based, ML-based, etc) are out of scope for this specification. The architecture only requires the function signature.

### 7.3 AI Client

```typescript
class AIClient {
  private playerId: string
  private strategy: AIStrategy
  private gameClient: GameClient
  
  async start() {
    // Join game with AI capabilities
    await this.gameClient.joinGame(
      this.playerId,
      this.playerName,
      [
        { type: 'act-as-player', playerIndex },
        { type: 'replace-ai' }
      ]
    )
    
    // Subscribe to state changes
    this.gameClient.subscribe(state => {
      this.onStateUpdate(state)
    })
  }
  
  private async onStateUpdate(state) {
    // Get valid actions
    const actions = await this.gameClient.getActions(this.playerId)
    
    if (actions.length === 0) return
    
    // Use strategy to select
    const selected = this.strategy(state.coreState, playerIndex, actions)
    
    // Optional: delay for natural feel
    await delay(500)
    
    // Execute
    await this.gameClient.executeAction({
      playerId: this.playerId,
      action: selected,
      timestamp: Date.now()
    })
  }
}
```

### 7.4 Key Properties

**Independent**: AI client is a separate process/worker/service that connects to server like any client.

**Transport Agnostic**: AI can use online client (HTTP) or offline client (Worker) identically.

**Replaceable**: AI has `replace-ai` capability, allowing humans to take over mid-game.

**Stateless**: AI has no persistent state beyond the game state it receives from server.

### 7.5 Deployment Patterns

**Online Mode**: AI runs as separate Node.js service, Cloudflare Worker, or browser worker
```
Cloudflare Worker (game server)
  ↑
  ├─ HTTP: Human browsers
  └─ HTTP: AI service (Node.js/Deno)
```

**Offline Mode**: AI runs as browser worker
```
Main Thread (game server worker + human UI)
  ↑
  ├─ postMessage: AI worker 1
  ├─ postMessage: AI worker 2
  └─ postMessage: AI worker 3
```

---

## 8. Complete Flow Examples

### 8.1 Online Game with Humans and AIs

```typescript
// Server: Initialize game
const game = createMultiplayerGame(
  'game-123',
  initialState,
  []
)
await durableObject.storage.put('state', game)

// Human 1 joins via browser
const human1Client = createGameClient('online', {
  baseUrl: 'https://game.example.com',
  gameId: 'game-123'
})
await human1Client.joinGame('alice', 'Alice', humanCapabilities(0))

// AI service joins
const aiClient1 = new AIClient(
  'ai-1',
  'AI Player',
  randomStrategy,
  createGameClient('online', {
    baseUrl: 'https://game.example.com',
    gameId: 'game-123'
  })
)
await aiClient1.start()

// Human 2 joins
await human2Client.joinGame('bob', 'Bob', humanCapabilities(2))

// Another AI joins
await aiClient2.start()

// Game plays out:
// 1. Server broadcasts state updates to all
// 2. When it's a player's turn, they get valid actions
// 3. Human clicks action in UI → client.executeAction()
// 4. AI receives update → strategy picks action → client.executeAction()
// 5. Server validates, executes, broadcasts
```

### 8.2 Offline Tutorial Mode

```typescript
// Create offline client
const client = createGameClient('offline', {
  gameId: 'tutorial',
  workerUrl: '/worker.js'
})

// Initialize with tutorial action transformers
const game = createMultiplayerGame(
  'tutorial',
  initialState,
  [
    {
      playerId: 'student',
      playerIndex: 0,
      capabilities: [
        { type: 'act-as-player', playerIndex: 0 },
        { type: 'see-hints' }
      ],
      name: 'You'
    }
  ]
)
game.enabledActionTransformers = [
  singleHandMode('tutorial-1'),
  speedMode,
  showHints,
  tournamentMode
]

// Send to worker
await (client as OfflineGameClient).initializeGame(game)

// Add 3 AI players
for (let i = 1; i <= 3; i++) {
  const ai = new AIClient(
    `ai-${i}`,
    `AI ${i}`,
    randomStrategy,
    createGameClient('offline', {
      gameId: 'tutorial',
      workerUrl: '/worker.js'
    })
  )
  await ai.start()
}

// Student plays
// 1. UI shows valid actions (with hints because student has see-hints)
// 2. Student clicks action
// 3. Worker executes, broadcasts
// 4. AIs receive update and respond
// 5. Speed mode auto-plays when student has only one option
// 6. Single hand mode ends game after scoring
```

### 8.3 Hot-Swap AI with Human

```typescript
// Game running with AI at index 2
const state = await client.getState()
const aiPlayer = state.players[2]  // Has replace-ai capability

// Stop AI client
await aiClientPool.remove(aiPlayer.playerId)

// Human joins in same slot
await humanClient.joinGame(
  'charlie',
  'Charlie',
  [{ type: 'act-as-player', playerIndex: 2 }]
)

// Game continues seamlessly
```

### 8.4 Progressive Enhancement (Offline → Online)

```typescript
// Start offline
const offlineClient = createGameClient('offline', {
  gameId: 'local-123',
  workerUrl: '/worker.js'
})

// User wants to invite friends
async function allowOthersToJoin() {
  const state = await offlineClient.getState()
  
  // Upload to server
  const response = await fetch('https://game.example.com/create', {
    method: 'POST',
    body: JSON.stringify({ state, originalPlayerId: 'me' })
  })
  
  const { gameId } = await response.json()
  
  // Switch to online client
  const onlineClient = createGameClient('online', {
    baseUrl: 'https://game.example.com',
    gameId: gameId
  })
  
  return {
    client: onlineClient,
    shareUrl: `https://game.example.com/join/${gameId}`
  }
}
```

### 8.5 Daily Challenge Flow

```typescript
// Create daily challenge game
const todayChallenge = createMultiplayerGame(
  'daily-2025-01-15-hard',
  generateStateFromSeed('daily-2025-01-15-hard'),
  [
    {
      playerId: 'player',
      playerIndex: 0,
      capabilities: [
        { type: 'act-as-player', playerIndex: 0 },
        { type: 'see-hints' }
      ],
      name: 'Player'
    }
  ]
)

todayChallenge.enabledActionTransformers = [
  dailyChallenge({
    date: '2025-01-15',
    difficulty: 'hard',
    starThresholds: [25, 35, 42]
  }),
  speedMode
]

// Add 3 AIs
// ... add AI clients ...

// Play game
// When scoring completes, dailyChallenge action transformer replaces score-hand with:
// {
//   type: 'end-game',
//   seed: 'daily-2025-01-15-hard',
//   stars: 2,  // Player scored 36 points
//   shareText: 'I scored 2⭐ on January 15th's hard challenge!'
// }

// UI shows results screen with:
// - Star rating (2/3)
// - Share button (posts to social media)
// - Replay button (same seed, try again)
// - Next challenge button (tomorrow's challenge)
```

---

## 9. Data Flow Patterns

### 9.1 Action Execution Flow

```
Client: executeAction(request)
  ↓
Server: Receive request
  ↓
Server: Load state from storage/memory
  ↓
Server: authorizeAndExecute(state, request)
  ↓ (if success)
Server: Persist new state
  ↓
Server: getVisibleState for each connected client
  ↓
Server: Broadcast personalized states
  ↓
Clients: Receive update, notify subscribers
  ↓
AI Clients: Check if their turn, act if so
```

### 9.2 Information Flow with Capabilities

```
Base State (full information)
  ↓
Apply action transformers → Actions with metadata (hints, etc)
  ↓
Authorization filter → Actions player can execute
  ↓
Visibility filter → Remove metadata player cannot see
  ↓
Client receives personalized view
```

### 9.3 ActionTransformer Application Flow

```
Core state machine: getValidActions(state)
  ↓
ActionTransformer 1: tournament mode (removes special bids)
  ↓
ActionTransformer 2: forced bid minimum (removes low bids)
  ↓
ActionTransformer 3: show hints (adds hint metadata)
  ↓
Final action list with all transformations applied
```

---

## 10. Architectural Invariants

### 10.1 Pure Functions Everywhere

**Invariant**: All state transitions are pure functions. No function modifies its inputs.

**Implication**: Can replay any sequence of actions deterministically. Time-travel debugging works.

### 10.2 Server Agnostic to AI

**Invariant**: Server has no AI-specific code paths. AI clients are regular clients.

**Implication**: Can add/remove/replace AI without server changes. AI complexity independent of server complexity.

### 10.3 Transport Independence

**Invariant**: Core multiplayer functions work with any transport (HTTP, WebSocket, postMessage).

**Implication**: Same code runs online and offline. Can switch transports without changing game logic.

### 10.4 ActionTransformer Composability

**Invariant**: ActionTransformers are independent transformers that compose via function composition.

**Implication**: Can mix any action transformers without conflicts. Adding action transformers doesn't require modifying base game.

### 10.5 Capability-Based Access

**Invariant**: What players can do and see is determined by capability tokens, not identity checks.

**Implication**: Can add new player types (spectators, coaches, tournament organizers) without changing authorization logic.

---

## 11. Extension Points

### 11.1 New ActionTransformers

Add new rule modifications by creating action transformer functions:

```typescript
const myActionTransformer: ActionTransformer = (baseMachine) => (state) => {
  const actions = baseMachine(state)
  // Transform actions
  return modifiedActions
}
```

Add to game state:
```typescript
game.enabledActionTransformers.push(myActionTransformer)
```

### 11.2 New Capabilities

Define new capability types:

```typescript
type Capability = 
  | ExistingCapabilities
  | { type: 'my-new-capability'; params: ... }
```

Update visibility/authorization logic to respect new capability.

### 11.3 New Transport

Implement `GameClient` interface for new transport:

```typescript
class MyTransportClient implements GameClient {
  // Implement all methods using new transport
}
```

Works with all existing code.

### 11.4 New AI Strategy

Implement strategy function signature:

```typescript
const myStrategy: AIStrategy = (state, index, actions) => {
  // Your logic
  return selectedAction
}
```

Pass to AIClient constructor. No other changes needed.

### 11.5 Spectator/Coach Modes

Add capabilities and implement visibility filtering:

```typescript
const spectatorCapabilities = [
  { type: 'observe-all-hands' },
  { type: 'observe-full-state' }
]

// Update getVisibleState to respect these capabilities
```

### 11.6 Replay System

Store action log instead of state snapshots:

```typescript
interface GameLog {
  initialState: GameState
  actions: Array<{ timestamp: number; request: ActionRequest }>
}

// Replay by reducing over actions
const finalState = actions.reduce(
  (state, action) => executeAction(state, action),
  initialState
)
```

---

## 12. Glossary

**Pure Function**: Function with no side effects that always returns the same output for the same input.

**State Machine**: Function that takes current state and returns possible next states (transitions).

**ActionTransformer**: Function that transforms a state machine, used to modify game rules.

**Capability**: Token that grants permission to perform actions or see information.

**GameClient**: Interface for interacting with game server (online or offline).

**AI Client**: Independent actor that connects as regular client and uses strategy function to play.

**Durable Object**: Cloudflare Workers primitive that provides persistent storage and single-threaded execution per instance.

**Authorization**: Determining whether a player can execute an action.

**Visibility**: Determining what information a player can see.

**Hot-swap**: Replacing an AI player with a human mid-game.

**Progressive Enhancement**: Starting offline and transitioning to online mode.

---

## 13. Summary

This architecture achieves:

- **Pure functional design**: All logic is pure functions that compose naturally
- **Transport agnostic**: Same code works online (Cloudflare) and offline (Web Workers)
- **AI as external actor**: AI clients connect like humans, server has no AI logic
- **Compositional action transformers**: Game rules compose via function transformers
- **Capability-based access**: Fine-grained control over actions and visibility
- **Extensibility**: Easy to add action transformers, capabilities, transports, AI strategies
- **Testability**: Every layer tests in isolation with no mocking
- **Flexibility**: Supports tutorials, daily challenges, tournaments, spectators, coaches

The key insight: **pure function composition with separation of concerns**. Core game engine → multiplayer authorization → transport → AI clients. Each layer independent, composable, testable.