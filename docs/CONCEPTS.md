# Texas 42 Architecture: Concepts Reference

**Purpose**: Complete reference of all major architectural concepts in the Texas 42 codebase. High-level conceptual overview for onboarding and architecture understanding (not implementation details).

**How to use**: Search by concept name or section. Navigate using the table of contents below.

---

## Table of Contents

1. [Core Game Abstractions](#1-core-game-abstractions)
   - GameAction, Executor, ExecutionContext, StateTransition, GameEngine, Engine

2. [Composition Systems](#2-composition-systems)
   - GameRules, GameLayer, Layers, Variants, StateMachine

3. [Multiplayer Architecture](#3-multiplayer-architecture)
   - PlayerSession, Capability, Authorization, Visibility, Consensus

4. [Server Architecture](#4-server-architecture)
   - GameKernel, GameServer, Transport, AIManager, KernelUpdate

5. [Client Architecture](#5-client-architecture)
   - NetworkGameClient, AIClient, AIStrategy, PlayerController

6. [AI Architecture](#6-ai-architecture)
   - Strategies, Hand Analysis, Testing, Integration

7. [Protocol & Communication](#8-protocol--communication)
   - Messages, Adapters, Views

8. [View Projection (UI Layer)](#9-view-projection-ui-layer)
   - ViewProjection, Components

9. [Configuration & Initialization](#10-configuration--initialization)
   - GameConfig, Variants, Themes

10. [Event Sourcing & Replay](#11-event-sourcing--replay)
    - Action History, URL Compression, Determinism

11. [Domain Objects](#12-domain-objects)
    - Game pieces: Domino, Bid, Trump, Play, Trick, Suits, Players

12. [Architectural Patterns & Principles](#13-architectural-patterns--principles)
    - Philosophy and design principles

13. [Architectural Invariants](#14-architectural-invariants)
    - Core rules that must never be violated

14. [Concept Relationships & Interactions](#15-concept-relationships--interactions)
    - How concepts work together

15. [Quick Reference](#16-quick-reference-concepts-by-file)
    - File locations and navigation

---

## 1. Core Game Abstractions

### GameAction
**Definition**: Union type representing all possible player actions in the game.

**Types**:
- Bidding: pass, redeal, bid (30-42 points, 1-3 marks)
- Trump: select-trump (suit, doubles, no-trump)
- Playing: play (domino)
- Consensus: complete-trick, score-hand, agree-complete-trick, agree-score-hand

**Location**: `src/game/types.ts`

**Key Properties**:
- Type category identifying the action
- Player index (0-3) who is acting
- Type-specific fields (bid value, domino, etc.)

**Used By**: Executors, GameEngine, State machines, Protocol messages

**Related**: Executor, StateMachine, StateTransition

---

### Executor
**Definition**: Pure function that applies a GameAction to state, updating it deterministically.

**Location**: `src/game/core/actions.ts`

**Key Characteristics**:
- Pure: No side effects, deterministic output
- Parameterized by rules: Never inspect state to determine behavior
- Delegates to rules: Calls rule methods for all decisions, never hardcodes variant logic
- Location-agnostic: Same executor code works for all special contracts (nello, splash, plunge, sevens)

**Pattern**: Executors validate via rules methods and return updated state without side effects.

**Related**: GameRules, ExecutionContext, Actions.ts

---

### ExecutionContext
**Definition**: Immutable container bundling composed layers, rules, and valid action generator that always travel together through the system.

**Location**: `src/game/types/execution.ts` (referenced in GameKernel)

**Key Properties**:
- Enabled layers - Which special contracts are active
- Composed rules - GameRules methods with layer overrides applied
- Valid action generator - Action generation with variants applied

**Purpose**:
- Single composition point: All composition happens in GameKernel constructor
- Ensures consistency: Layers and variants composed together, never separately
- Enables polymorphism: Executors use context without inspecting state

**Created Once**: GameKernel constructor creates it and threads through entire system

**Related**: GameKernel, Layers, Variants, GameRules

---

### StateTransition
**Definition**: Represents a possible game state transition with label, action, and resulting state.

**Location**: `src/game/types.ts`

**Key Properties**:
- `label: string` - Human-readable description of action
- `action: GameAction` - The action to execute
- `state: GameState` - Resulting state after execution

**Usage**:
- AI decision-making: `getNextStates()` returns all possibilities
- UI exploration: Show available moves and their outcomes
- Testing: Enumerate all valid transitions

**Related**: GameAction, GameState, GameEngine

---

### GameEngine (Class)
**Definition**: Stateful wrapper providing history tracking and undo functionality for pure game logic.

**Location**: `src/game/core/actions.ts`

**Key Characteristics**:
- Maintains mutable state and action history
- Provides convenient wrapper around pure functions
- Supports undo/redo via state snapshots
- Legacy pattern - only used in tests

**Important**:
- **Not used in production** - GameKernel uses pure functions directly
- Breaks immutability principle
- Intended for testing and development convenience

**Related**: GameAction, executeAction (pure function), GameKernel

---

### Engine (Conceptual)
**Definition**: The collection of pure functions forming the core game logic.

**Location**: `src/game/core/` directory:
- `gameEngine.ts` - Action generation, state exploration
- `actions.ts` - Executors
- `state.ts` - State utilities
- `rules.ts` - Rule validation
- `scoring.ts` - Score calculation

**Key Functions**:
- `executeAction(state, action, rules)` - Pure state transition
- `getValidActions(state, layers, rules)` - What actions are possible
- `getNextStates(state, layers, rules)` - All possible transitions
- `actionToId(action)` - Serialize action
- `actionToLabel(action)` - Human-readable label

**Key Properties**:
- Pure: No side effects, deterministic
- Parameterized: Rules injected, not hardcoded
- Zero coupling: No awareness of multiplayer, networking, UI
- Composable: Works uniformly with any layers/variants

**Relationship to Kernel**:
- Engine = pure game logic only
- Kernel = engine + multiplayer (authorization, filtering, sessions)
- Kernel **composes** engine and wraps with multiplayer concerns

**Related**: GameKernel, Executor, Layers, Variants

---

## 2. Composition Systems

### GameRules (Interface)
**Definition**: Interface of 13 pure methods that determine HOW the game executes.

**Location**: `src/game/layers/types.ts`

**The 13 Methods** (organized by category):

**WHO (3 methods)** - Player determination:
- `getTrumpSelector(state, winningBid): number` - Which player selects trump
- `getFirstLeader(state, trumpSelector, trump): number` - First trick leader
- `getNextPlayer(state, currentPlayer): number` - Next player to act

**WHEN (2 methods)** - Timing and completion:
- `isTrickComplete(state): boolean` - Is current trick done (3 or 4 plays)
- `checkHandOutcome(state): HandOutcome | null` - Early termination (nello/plunge win conditions)

**HOW (2 methods)** - Game mechanics:
- `getLedSuit(state, domino): LedSuit` - What suit is led (for follow-suit validation)
- `calculateTrickWinner(state, trick): number` - Who won the trick

**VALIDATION (3 methods)** - Legality:
- `isValidPlay(state, domino, playerId): boolean` - Can player play this domino
- `getValidPlays(state, playerId): Domino[]` - All playable dominoes
- `isValidBid(state, bid, playerHand?): boolean` - Is bid legal

**SCORING (3 methods)** - Outcomes:
- `getBidComparisonValue(bid): number` - Numeric value for comparison
- `isValidTrump(trump): boolean` - Can this be trump
- `calculateScore(state): [number, number]` - Team scores

**Key Insight**: Executors call these methods, never inspect state. This enables polymorphism - same executor code works for all variants.

**Composed By**: Layers override specific methods, compose via `composeRules()` reduce pattern

**Related**: GameLayer, Executor, Base Layer, Special Contracts

---

### GameLayer
**Definition**: Composable layer that overrides specific rules and/or transforms actions.

**Location**: `src/game/layers/types.ts`

**Two Composition Surfaces**:

1. **Rules** (execution semantics): Override specific GameRules methods to change how execution works
   - Example: Nello overrides `isTrickComplete` to return true at 3 plays (partner out)
   - Only override what differs from base, rest pass through

2. **Action Transformation**: Filter, annotate, or replace action generation
   - Example: Nello disables certain special bid actions
   - Receives previous layer's result for chaining

**Composition Pattern**: Layers compose via reduce pattern where later layers override earlier ones without modifying core executors.

**Related**: GameRules, Variant, Layer Composition, Base Layer

---

### Layer Implementations

#### Base Layer
**Definition**: Standard Texas 42 rules.

**Location**: `src/game/layers/base.ts`

**Provides**: All 13 GameRules methods with standard 4-player Texas 42 logic

**Related**: GameRules, GameLayer

#### Nello Layer
**Definition**: Partner sits out, 3-player tricks, lose all tricks to win.

**Location**: `src/game/layers/nello.ts`

**Overrides**:
- `isTrickComplete`: Return true at 3 plays (partner out)
- `checkHandOutcome`: Check if bidder lost all tricks (wins hand)
- `calculateScore`: Nello-specific scoring

**Related**: GameLayer, HandOutcome, Special Contracts

#### Plunge Layer
**Definition**: Partner selects trump and leads, bidder must win all tricks.

**Location**: `src/game/layers/plunge.ts`

**Overrides**: Similar to Nello but reverse outcome logic

#### Splash Layer
**Definition**: Like plunge but requires 3+ doubles in hand.

**Location**: `src/game/layers/splash.ts`

**Overrides**: Builds on plunge with additional validation

#### Sevens Layer
**Definition**: Distance from 7 wins, no follow-suit rule.

**Location**: `src/game/layers/sevens.ts`

**Overrides**: Completely different scoring and led-suit logic

---

### Layer Composition
**Definition**: Pure reduce pattern that composes multiple layers into single GameRules interface.

**Location**: `src/game/layers/compose.ts`

**Algorithm**:
```typescript
export function composeRules(layers: GameLayer[]): GameRules {
  return layers.reduce((prev, layer) => ({
    // For each rule method
    getTrumpSelector: (state, bid) =>
      layer.rules?.getTrumpSelector?.(state, bid, prev) ?? prev.getTrumpSelector(state, bid),
    // ... repeat for all 13 methods
  }), baseRules);
}
```

**Key Pattern**: Each layer gets `prev` parameter - override or delegate

**Related**: GameRules, GameLayer

---

### Variant
**Definition**: Function that transforms the action state machine to change what actions are available.

**Location**: `src/game/variants/types.ts`

**Variant Operations**:
- **Filter**: Remove actions (tournament removes special contracts)
- **Annotate**: Add metadata (hints adds hint field, speed adds autoExecute flag)
- **Script**: Inject actions (oneHand scripts bidding sequence)
- **Replace**: Swap action types (oneHand replaces score-hand with end-game)

**Composition**: Variants wrap via function composition - each variant wraps the previous one, building a transformation pipeline.

**Key Insight**: Variants transform action generation independently from layers (which transform execution rules).

**Related**: StateMachine, VariantConfig, Variant Implementations

---

### Variant Implementations

#### Tournament Variant
**Definition**: Disable special contracts (nello, splash, plunge, sevens).

**Location**: `src/game/variants/tournament.ts`

**Operation**: Filter - removes special bids from action list

#### OneHand Variant
**Definition**: Single hand game with scripted actions.

**Location**: `src/game/variants/oneHand.ts`

**Operations**: Script bidding, replace score-hand with end-game

#### Hints Variant
**Definition**: Add hint metadata to actions.

**Location**: `src/game/variants/hints.ts`

**Operation**: Annotate - adds `hint` field to each action

#### Speed Variant
**Definition**: Auto-play when only one option available.

**Location**: `src/game/variants/speed.ts`

**Operation**: Annotate - adds `autoExecute: true` flag

---

### StateMachine
**Definition**: Function producing available GameActions from state.

**Signature**: `(state: GameState) => GameAction[]`

**Location**: Referenced throughout, created by `getValidActions()`

**Composition**: Variants compose StateMachines:
```typescript
const base = (state) => getValidActions(state, layers, rules);
const withTournament = tournament(base);
const withOneHand = oneHand(withTournament);
// Now withOneHand is the composed state machine
```

**Related**: Variant, getValidActions, Variant Composition

---

## 3. Multiplayer Architecture

### PlayerSession
**Definition**: Separates player identity from game seat, grouping identity, seat, control type, and capabilities.

**Location**: `src/game/multiplayer/types.ts`

**Key Properties**:
- `playerId: string` - Unique identifier (e.g., "player-0", "ai-1")
- `playerIndex: 0 | 1 | 2 | 3` - Game seat (partner with 2, opponent with 3)
- `controlType: 'human' | 'ai'` - Who controls this seat
- `capabilities: Capability[]` - Permissions for this player

**Key Insight**: Separates identity from seat:
- Same player can sit in different seats
- Multiple humans can play (hot-seat mode)
- Capabilities control what player can see/do

**Related**: Capability, Authorization, Visibility Filtering

---

### Capability
**Definition**: Composable permission token granting action or observation rights.

**Location**: `src/game/multiplayer/types.ts`

**Types**:

**Action Capabilities**:
- `{ type: 'act-as-player'; playerIndex: number }` - Execute actions for seat
- `{ type: 'replace-ai'; }` - Switch AI to human (UI button)
- `{ type: 'configure-variant'; }` - Change game settings

**Visibility Capabilities**:
- `{ type: 'observe-own-hand' }` - See own domino hand
- `{ type: 'observe-hand'; playerIndex: number }` - See specific player's hand
- `{ type: 'observe-all-hands' }` - See all hands
- `{ type: 'observe-full-state' }` - See unrestricted state
- `{ type: 'see-hints' }` - See AI recommendations
- `{ type: 'see-ai-intent' }` - See AI strategy metadata

**Replaces**: Boolean flags (isObserver, canSeeBids, etc.)

**Composed**: Multiple capabilities per session

**Related**: PlayerSession, Authorization, Visibility Filtering

---

### ActionRequest
**Definition**: Request to execute an action, with identity and metadata.

**Location**: `src/game/multiplayer/types.ts`

**Key Properties**:
- `playerId: string` - Who is requesting
- `action: GameAction` - What to execute
- `timestamp: number` - When requested

**Used By**: Protocol layer, GameKernel execution

**Related**: GameAction, PlayerSession

---

### Authorization
**Definition**: System determining what actions a session can execute.

**Location**: `src/game/multiplayer/authorization.ts`

**Key Function**: `authorizeAndExecute()`
```typescript
authorizeAndExecute(
  mpState: MultiplayerGameState,
  request: ActionRequest,
  composedStateMachine: StateMachine,
  rules: GameRules
): { success: boolean; state?: MultiplayerGameState; error?: string }
```

**Process**:
1. Find session by playerId
2. Get valid actions via composed state machine
3. Filter by session capabilities: `filterActionsForSession(session, validActions)`
4. Check if requested action is in filtered list
5. If yes, execute and broadcast; if no, reject

**Related**: PlayerSession, Capability, Visibility Filtering

---

### Visibility Filtering
**Definition**: System filtering state and actions based on session capabilities.

**Location**: `src/game/multiplayer/capabilityUtils.ts`

**Key Function**: `getVisibleStateForSession()`
```typescript
getVisibleStateForSession(state: GameState, session: PlayerSession): FilteredGameState
```

**Filtering**:
- Hand visibility: Based on `observe-*-hand` capabilities
- State visibility: Full or filtered based on `observe-*-state`
- Action visibility: Filter by `see-hints`, `see-ai-intent`

**Related**: PlayerSession, Capability, FilteredGameState

---

### FilteredGameState
**Definition**: Version of GameState with hands filtered based on visibility permissions.

**Location**: `src/game/types.ts`

**Key Difference**:
- `GameState.players[].hand` - Full hand (server only)
- `FilteredGameState.players[].hand` - Empty or filtered (client view)

**Used By**: Clients, multiplayer filtering

**Related**: GameState, Visibility Filtering

---

### Consensus
**Definition**: Set-based tracking for actions requiring all players to agree before proceeding.

**Location**: `src/game/types.ts`

**Tracked Actions**:
- `complete-trick` - All players agree current trick is complete
- `score-hand` - All players agree hand is scored and next hand can begin

**Purpose**: Provides mechanism for neutral/administrative actions requiring consensus rather than individual player decisions.

**Related**: GameAction, ActionRequest

---

### MultiplayerGameState
**Definition**: Wraps GameState with multiplayer metadata.

**Location**: `src/game/multiplayer/types.ts`

**Key Properties**:
- `gameId: string` - Unique game identifier
- `state: GameState` - Core game state
- `sessions: Map<string, PlayerSession>` - Active player sessions
- `lastActionAt: number` - Timestamp of last action
- `variantConfig: VariantConfig[]` - Active variants

**Used By**: GameKernel (stores MultiplayerGameState, not GameState)

**Related**: GameState, PlayerSession, VariantConfig

---

## 4. Server Architecture

### GameKernel
**Definition**: Pure game authority that stores state, composes layers/variants, authorizes actions, and filters views.

**Location**: `src/kernel/GameKernel.ts`

**Key Responsibilities**:
1. **State Storage**: Owns MultiplayerGameState unfiltered
2. **Composition**: Creates ExecutionContext in constructor (single composition point)
3. **Authorization**: Executes `authorizeAndExecute()` for each action
4. **Filtering**: Per-request filtering via `getView()`
5. **Lifecycle**: Manages subscriptions

**Key Methods**:
```typescript
constructor(gameId, config, players)
executeAction(playerId, action, timestamp): Result
getView(playerId): GameView  // Filtered state + valid actions
subscribe(perspective, listener): unsubscribe
notifyListeners()
```

**Zero Knowledge Of**:
- Transport/networking
- AI spawning
- WebSockets
- Multiplayer lifecycle

**Pure Game Logic**: No side effects beyond state mutation

**Related**: GameEngine, GameServer, ExecutionContext, Authorization

---

### GameServer
**Definition**: Orchestrator that creates/owns GameKernel and AIManager, routes protocol messages.

**Location**: `src/server/GameServer.ts`

**Key Responsibilities**:
1. **Creates GameKernel**: Game logic authority
2. **Creates AIManager**: AI lifecycle
3. **Routes Messages**: Protocol to GameKernel execution
4. **Broadcasts Updates**: GameKernel updates via Transport
5. **Manages Subscriptions**: GameKernel listeners

**Key Methods**:
```typescript
handleMessage(clientId, message): void
setTransport(transport): void
notifyPlayer(playerId, update): void
broadcast(update): void
```

**Architecture**:
```
GameServer
├── owns GameKernel (pure game logic)
├── owns AIManager (AI lifecycle)
├── owns Transport (message routing)
└── owns subscriptions (listener management)
```

**Related**: GameKernel, Transport, AIManager

---

### Transport
**Definition**: Message routing abstraction enabling different transport implementations.

**Location**: `src/server/transports/Transport.ts`

**Interfaces**:

**Connection**:
```typescript
interface Connection {
  send(message: ServerMessage): void;
  onMessage(handler: (message: ClientMessage) => void): void;
  disconnect(): void;
}
```

**Transport**:
```typescript
interface Transport {
  send(clientId: string, message: ServerMessage): void;
  start(): void;
  stop(): void;
}
```

**Implementations**:
- `InProcessTransport`: In-browser, single process
- `WorkerTransport`: Web Worker (planned)
- `CloudflareTransport`: Durable Objects (planned)

**Key Insight**: GameServer doesn't know transport type - abstracts away

**Related**: GameServer, InProcessTransport

---

### InProcessTransport
**Definition**: Transport implementation for in-browser, single-process game.

**Location**: `src/server/transports/InProcessTransport.ts`

**How It Works**:
1. Client calls `adapter.send(message)`
2. Transport receives message
3. Routes to server-side handler
4. Server sends response back

**No Network**: All in-process, synchronous

**Used By**: Current development and testing

**Related**: Transport, IGameAdapter

---

### AIManager
**Definition**: Centralized manager for AI client lifecycle.

**Location**: `src/server/ai/AIManager.ts`

**Key Responsibilities**:
1. Spawn AI clients when game starts
2. Connect AI clients to transport
3. Destroy AI clients when game ends
4. Map seats to AI instances

**No Game Logic**: Just lifecycle management

**Related**: GameServer, AIClient

---

### KernelUpdate
**Definition**: Filtered state update bundle sent to subscriber.

**Location**: `src/kernel/GameKernel.ts`

**Key Properties**:
- `view: GameView` - Client-facing state (hand filtered)
- `state: FilteredGameState` - Full filtered state
- `actions: ValidAction[]` - Playable actions
- `perspective: string` - Player receiving update

**Purpose**: GameKernel creates one per subscriber, customized to their capabilities

**Related**: GameView, PlayerSession, Capability

---

## 5. Client Architecture

### NetworkGameClient
**Definition**: Client-side interface that caches filtered state and trusts server.

**Location**: `src/game/multiplayer/NetworkGameClient.ts`

**Key Responsibilities**:
1. Send actions to server
2. Cache filtered state from server
3. Cache valid actions list
4. Notify UI of updates

**Trust Model**: Client trusts server's validActions list, never refilters

**Related**: Transport, GameServer, GameView

---

### AIClient
**Definition**: Independent AI actor that connects and plays via protocol.

**Location**: `src/game/multiplayer/AIClient.ts`

**How It Works**:
1. Receives GameView from server
2. Uses AIStrategy to decide action
3. Sends action via protocol
4. Server validates and executes

**Pure Logic**: Strategy function, not game logic

**Related**: AIStrategy, AIManager, GameServer

---

### AIStrategy
**Definition**: Pure function deciding AI action from game state.

**Signature**: `(state: GameState, transitions: StateTransition[]) => StateTransition`

**Location**: `src/game/ai/types.ts`

**Characteristics**:
- Pure: No side effects
- Parameterized: Difficulty passed as parameter
- Explores: Gets all possible transitions
- Decides: Returns chosen transition

**Difficulties**:
- `beginner`: Random choice
- `intermediate`: Simple heuristics
- `expert`: Game tree evaluation

**Related**: StateTransition, AIClient

---

### PlayerController
**Definition**: Abstraction for both human and AI control.

**Location**: `src/game/ai/types.ts`

**Interface**: Can choose actions (location-agnostic)

**Related**: AIStrategy, AIClient

---

## 6. AI Architecture

### AIStrategy
**Definition**: Interface for deciding which action an AI should take, given the current game state and available transitions.

**Location**: `src/game/ai/types.ts`

**Purpose**: Enables different AI difficulties and strategies while keeping core game logic unchanged.

**Key Characteristics**:
- Pure function: No side effects, deterministic output
- Stateless: Each decision is independent
- Parameterized by difficulty: Strategy changes based on AIDifficulty
- Abstracted from protocol: Doesn't know about networking or multiplayer

**Related**: StateTransition, AIClient, AIDifficulty

---

### AIStrategy Implementations

#### RandomAIStrategy
**Definition**: AI that makes random action selections.

**Location**: `src/game/ai/strategies.ts`

**Characteristics**:
- Picks random valid action (except consensus actions which execute immediately)
- Instant response (no thinking time)
- Used for testing and baseline comparison

**Related**: AIStrategy, BeginnerAIStrategy

#### BeginnerAIStrategy
**Definition**: Intelligent AI with phase-specific decision logic using hand strength analysis.

**Location**: `src/game/ai/strategies.ts`

**Decision Making by Phase**:
- **Bidding**: Evaluates hand strength and bids conservatively
- **Trump Selection**: Chooses best trump suit or doubles
- **Playing**: Leads with strong dominoes, follows smartly based on trick position
- **Thinking Time**: 800-2800ms for human-like delays

**Uses Hand Analysis**: Leverages analyzeHand, calculateLexicographicStrength, and determineBestTrump

**Related**: AIStrategy, Hand Analysis System, AIDifficulty

---

### AIDifficulty
**Definition**: Enumeration of AI skill levels.

**Location**: `src/game/multiplayer/AIClient.ts`

**Types**: `'beginner' | 'intermediate' | 'expert'`

**Purpose**: Controls which strategy implementation is used and thinking time delays.

**Current State**: Beginner implemented; intermediate and expert are placeholders for future expansion.

**Related**: AIStrategy, AIClient

---

### AIClient
**Definition**: Independent AI player that connects to the game server via the standard protocol (same as human clients).

**Location**: `src/game/multiplayer/AIClient.ts`

**Key Characteristics**:
- Protocol-speaking: Uses ClientMessage and ServerMessage like any client
- No special privileges: Actions validated by GameKernel same as humans
- Dynamic lifecycle: Can be spawned and destroyed at runtime
- Adapter-based: Communicates via IGameAdapter abstraction

**Lifecycle**:
- `start()`: Subscribe to game updates
- Waits for turn notification
- Thinks for configurable delay (based on difficulty)
- Executes action via adapter
- `destroy()`: Clean up timers and unsubscribe

**Key Insight**: AI is not a privileged system - it's a regular client that happens to make decisions algorithmically.

**Related**: AIManager, selectAIAction, IGameAdapter, GameServer

---

### AIManager
**Definition**: Server-side manager for AI client lifecycle (spawning, tracking, destroying).

**Location**: `src/server/ai/AIManager.ts`

**Responsibilities**:
- Spawn AI clients for configured AI seats at game start
- Track active AI instances per seat
- Destroy AI clients when control changes (e.g., human replaces AI)
- Provide AI information for debugging

**Key Insight**: Zero game logic - purely lifecycle management. All decisions happen in AIStrategy.

**Used By**: GameServer (creates and manages AIManager)

**Related**: AIClient, GameServer, IGameAdapter

---

### selectAIAction
**Definition**: Pure function that selects an action for an AI player using strategy pattern.

**Location**: `src/game/ai/actionSelector.ts`

**Purpose**: Composition point connecting player identity to strategy implementation.

**Algorithm**:
1. Filter available transitions to this player's actions only
2. Prioritize consensus actions (execute immediately without thinking)
3. Get strategy for player (currently all AI use BeginnerAIStrategy)
4. Delegate to strategy: `strategy.chooseAction(state, transitions)`
5. Return chosen transition

**Extensibility**: Designed to support per-player strategies or difficulty levels in the future.

**Related**: AIStrategy, StateTransition, BeginnerAIStrategy

---

## 6.1 Hand Analysis System

### analyzeHand
**Definition**: Evaluates each domino in a player's hand within its game context (what can beat it, what it can beat).

**Location**: `src/game/ai/utilities.ts`

**Purpose**: Provides context-aware domino evaluation for AI decision-making.

**Produces**: Analysis showing each domino's strength in the current trump context, which unplayed dominoes can beat it, and which it can beat.

**Used By**: BeginnerAIStrategy (for playing decisions), hand strength calculations

**Related**: GameState, DominoAnalysis, GameRules

---

### calculateLexicographicStrength
**Definition**: Evaluates hand strength by ranking dominoes by how many can beat them, then comparing lexicographically.

**Location**: `src/game/ai/lexicographic-strength.ts`

**Purpose**: Provides consistent bidding heuristic (lower score = stronger hand).

**Used For**: Bidding decisions in BeginnerAIStrategy.

**Related**: BeginnerAIStrategy, BID_THRESHOLDS, LAYDOWN_SCORE

---

### determineBestTrump
**Definition**: Determines optimal trump selection for a player's hand.

**Location**: `src/game/ai/hand-strength.ts`

**Logic**:
- Prefers doubles if hand has 3+ of them
- Otherwise selects strongest suit by count and composition
- Tie-breaker: prefers higher suits

**Used By**: BeginnerAIStrategy, calculateLexicographicStrength

**Related**: TrumpSelection, Hand Analysis

---

### BID_THRESHOLDS
**Definition**: Bidding decision thresholds used to determine when AI should bid vs pass.

**Location**: `src/game/ai/hand-strength.ts`

**Purpose**: Maps hand strength scores to bidding decisions (currently all AI is conservative, mostly bidding 30 or passing).

**Related**: calculateLexicographicStrength, BeginnerAIStrategy

---

### LAYDOWN_SCORE
**Definition**: Special constant (999) indicating a guaranteed-win hand where all 7 dominoes are unbeatable.

**Location**: `src/game/ai/hand-strength.ts`

**Purpose**: Detects when AI has perfect hand and bids appropriately.

**Related**: Hand Analysis, calculateLexicographicStrength

---

### Multi-Trump Analysis
**Definition**: System for evaluating how dominoes perform under different trump contexts (for understanding versatility).

**Location**: `src/game/ai/multi-trump-analysis.ts`

**Purpose**: Provides educational/debugging tools to understand domino strength across all possible trump selections.

**Related**: Hand Analysis, TrumpSelection

---

## 6.2 AI Testing

### GameSimulator
**Definition**: Utilities for running complete games with AI players for testing and analysis.

**Location**: `src/game/ai/gameSimulator.ts`

**Capabilities**:
- Run games to completion with AI
- Batch-run multiple games with same seed for win-rate analysis
- Search for seeds with target win rates (for balanced scenarios)

**Used For**: Testing AI strategies, seed selection for one-hand mode, regression testing.

**Related**: selectAIAction, GameEngine, getNextStates

---

## 6.3 AI Architecture Overview

### Data Flow: AI Decision to Action

```
GameServer executes action
   ↓
GameKernel broadcasts STATE_UPDATE
   ↓
AIClient receives update
   ↓ (if it's AI's turn)
selectAndExecuteAction()
   → Get available transitions
   → Call selectAIAction()
      → Call strategy.chooseAction()
         → analyzeHand() / calculateLexicographicStrength()
         → determineBestTrump()
      ← Return chosen transition
   → Send EXECUTE_ACTION
   ↓
GameServer validates and executes
```

### Composition Points

**Strategy Selection**: `selectAIAction()` determines which strategy to use (currently all AI use BeginnerAIStrategy, extensible for future per-player strategies).

**Difficulty Configuration**: AIDifficulty determines thinking time and strategy parameters.

**Hand Analysis Method**: BeginnerAIStrategy can plug in different hand strength calculations (currently uses lexicographic strength for bidding).

---

## 6.4 AI Architectural Invariants

1. **Pure Decision Functions**: AIStrategy.chooseAction() and selectAIAction() have no side effects
2. **Protocol Equality**: AI clients use identical protocol as human clients, no privileged access
3. **Zero Coupling to Core**: AI system has zero knowledge of multiplayer, networking, or core engine
4. **Separation of Concerns**: Strategy handles decisions, AIClient handles protocol, AIManager handles lifecycle
5. **Extensibility**: New strategies, difficulties, and analysis methods can be added without modifying core

---

## 8. Protocol & Communication

### ClientMessage
**Definition**: Message sent from client to server.

**Types**:
- `CREATE_GAME` - Start new game
- `EXECUTE_ACTION` - Execute action
- `JOIN_GAME` - Join existing game
- `SUBSCRIBE` - Subscribe to updates
- `UNSUBSCRIBE` - Stop receiving updates

**Related**: ServerMessage, Transport

---

### ServerMessage
**Definition**: Message sent from server to client.

**Types**:
- `GAME_CREATED` - Game initialized
- `STATE_UPDATE` - State changed
- `ACTION_CONFIRMED` - Action accepted
- `ERROR` - Something failed

**Related**: ClientMessage, Transport

---

### GameView
**Definition**: Client-facing filtered view of game state.

**Location**: Referenced in protocol

**Contains**:
- `state: FilteredGameState` - Hand-filtered
- `validActions: ValidAction[]` - Playable actions
- `metadata: { phase, turn, tricks, scoring }`

**Trust**: Client trusts this is correct

**Related**: FilteredGameState, ValidAction

---

### ValidAction
**Definition**: Action with client-visible metadata.

**Location**: Referenced in protocol

**Contains**:
- `action: GameAction` - The action
- `label: string` - Human-readable
- `hint?: string` - Optional AI recommendation
- `autoExecute?: boolean` - Auto-play flag

**Related**: GameAction, Capability

---

### IGameAdapter
**Definition**: Interface for game interaction (currently used by AI).

**Location**: `src/shared/multiplayer/protocol.ts`

**Methods**:
- `send(message: ClientMessage): Promise<void>`
- `subscribe(handler: (message: ServerMessage) => void): () => void`
- `destroy(): void`
- `isConnected(): boolean`

**Used By**: AIClient for communication

**Related**: Transport, ClientMessage, ServerMessage

---

## 9. View Projection (UI Layer)

### ViewProjection
**Definition**: Complete UI state derived from GameState via pure computation.

**Location**: `src/game/view-projection.ts`

**Key Properties**:
- Phase and turn information (current player, game status)
- Hand with playability information for each domino
- Actions grouped by category (bidding, trump, play, consensus)
- Game display data (bid statuses, tricks, current trick)
- Scoring results with perspective-aware messages
- UI metadata (hints, tooltips, display flags)

**Transformation Pipeline**:
1. GameState → FilteredGameState (hand filtering by capability)
2. FilteredGameState + ValidActions → ViewProjection (pure computation)
3. ViewProjection → Svelte stores (reactive state)
4. Svelte stores → Components (UI rendering)

**Key Insight**: Pure computation - identical state always produces identical projection

**Related**: GameState, FilteredGameState, ValidAction

---

### HandDomino
**Definition**: Domino with playability and tooltip information.

**Location**: `src/game/view-projection.ts`

**Properties**:
- `domino: Domino` - The piece
- `playable: boolean` - Can play now
- `tooltip: string` - Why/why not playable

**Used By**: UI to render hand with visual feedback

**Related**: Domino, ViewProjection

---

### BidStatus
**Definition**: Player bid status for display.

**Location**: `src/game/view-projection.ts`

**Properties**:
- `player: number` - Which player
- `bid: Bid | null` - Their bid (null if passed)
- `passed: boolean` - Explicitly passed
- `reason?: string` - Why they passed

**Used By**: UI to show bidding sequence

**Related**: Bid, ViewProjection

---

### TrickDisplay
**Definition**: Current trick visualization data.

**Location**: `src/game/view-projection.ts`

**Properties**:
- `plays: Play[]` - Plays in trick
- `winner: number | null` - Who's winning (or null if incomplete)
- `points: number` - Points in trick
- `isVisible: boolean` - Should UI display

**Used By**: UI trick display area

**Related**: Trick, Play, ViewProjection

---

### HandResults
**Definition**: Scoring results with perspective-aware messages.

**Location**: `src/game/view-projection.ts`

**Properties**:
- `biddingTeam: number` - Which team bid
- `bidTeamScore: number` - Their points this hand
- `otherTeamScore: number` - Opposition points
- `handWinner: number` - Which team won hand
- `bidSuccessful: boolean` - Bid outcome
- `messages: string[]` - Perspective-aware explanations

**Used By**: UI scoring display

**Related**: ViewProjection, GameState

---

## 10. Configuration & Initialization

### GameConfig
**Definition**: Game initialization configuration.

**Location**: `src/game/types/config.ts`

**Key Properties**:
```typescript
playerTypes: ('human' | 'ai')[]  // Seat control
shuffleSeed: number              // Deterministic shuffle
enabledLayers: string[]          // Active special contracts
enabledVariants: VariantConfig[] // Active variants
theme: string                    // DaisyUI theme
colorOverrides: Record<string, string>  // CSS variables
```

**Used By**: GameKernel constructor, Game initialization

**Related**: VariantConfig, Theme System

---

### VariantConfig
**Definition**: Configuration for enabled variant.

**Location**: `src/game/types/config.ts`

**Structure**:
```typescript
{
  type: 'tournament' | 'oneHand' | 'hints' | 'speed';
  config?: Record<string, any>  // Variant-specific options
}
```

**Related**: GameConfig, Variant

---

## 11. Event Sourcing & Replay

### Action History
**Definition**: Array of GameActions forming complete audit trail.

**Location**: Core concept, stored in MultiplayerGameState

**Purpose**:
- Complete game record
- Enable replay: `state = replayActions(config, history)`
- Share games: Encode in URL
- Testing: Deterministic scenarios

**Key Property**: Immutable - never modified, only appended

**Related**: GameAction, Event Sourcing Formula

---

### Event Sourcing Formula
**Definition**: Core principle that state is derived from actions.

**Formula**: `state = replayActions(config, history)`

**Implications**:
- Actions are source of truth
- State is computed, not stored
- Two games with same config + actions = identical states
- Enables time-travel debugging
- Enables replay from URL

**Related**: Action History, Determinism

---

### URL Compression System
**Definition**: Encode/decode game state in URL for sharing.

**Location**: `src/game/core/url-compression.ts`

**How It Works**:
- Actions → Compressed string (1 char per common event, 2 for rare)
- Seed → Base36 (compact)
- Player types → Single chars
- URL → 300-400 chars for typical 8KB game

**Compressed Format**:
```
?s={seed}&a={compressed_actions}&p={player_types}&d={dealer}&t={theme}&v={colors}&h={scenario}
```

**Related**: Event Sourcing, Action History, decodeGameUrl, encodeGameUrl

---

## 12. Domain Objects

### Domino
**Definition**: Game piece with two pips (0-6).

**Location**: `src/game/types.ts`

**Properties**:
- `high: number` - High pip (0-6)
- `low: number` - Low pip (0-6)
- `id: string` - Unique identifier
- `points: number` - Victory point value (dot sum)

**Related**: Pip, Play, Hand

---

### Bid
**Definition**: Player's contract during bidding phase.

**Location**: `src/game/types.ts`

**Types**:
- `pass` - Not bidding
- `redeal` - Ask to shuffle and deal again
- `points` - Bid 30-42 points
- `marks` - Bid 1-3 marks (doubles)
- Special: `nello`, `splash`, `plunge`, `sevens`

**Properties**:
- `type: string` - Bid category
- `value: number` - Points or marks
- `player: number` - Who bid

**Related**: TrumpSelection, bidding phase

---

### TrumpSelection
**Definition**: Trump contract for hand.

**Location**: `src/game/types.ts`

**Types**:
- `not-selected` - Not yet chosen
- Regular suit (0-5): blanks, ones, twos, threes, fours, fives, sixes
- `doubles` (7) - All doubles are trump
- `no-trump` (8) - No trump, follow suit always
- Special: `nello`, `plunge`, `splash`, `sevens`

**Properties**:
- `type: string` - Trump type
- `suit?: number` - Suit if applicable (0-6)

**Related**: Bid, GameState

---

### Play
**Definition**: Single domino played by a player.

**Location**: `src/game/types.ts`

**Properties**:
- `player: number` - Who played
- `domino: Domino` - Which domino

**Related**: Domino, Trick

---

### Trick
**Definition**: Four dominoes played in sequence, with winner and points.

**Location**: `src/game/types.ts`

**Properties**:
- `plays: Play[]` - The plays (1-4)
- `winner: number` - Who won
- `points: number` - Points scored
- `ledSuit: LedSuit` - Suit led

**Related**: Play, GameState

---

### Suits

#### RegularSuit
**Definition**: Standard domino suits 0-6.

**Location**: `src/game/types.ts`

**Types**:
- `0` - Blanks
- `1` - Ones
- `2` - Twos
- `3` - Threes
- `4` - Fours
- `5` - Fives
- `6` - Sixes

**Related**: LedSuit, TrumpSuit

#### LedSuit
**Definition**: Suit led in trick (regular suit or doubles).

**Location**: `src/game/types.ts`

**Types**: RegularSuit (0-6) or `7` (doubles)

**Related**: RegularSuit, Trick

#### TrumpSuit
**Definition**: Trump suit (regular suit, doubles, or no-trump).

**Location**: `src/game/types.ts`

**Types**: RegularSuit (0-6), `7` (doubles), `8` (no-trump)

**Related**: RegularSuit, TrumpSelection

---

### Players & Teams

#### Player
**Definition**: Game piece representing a seat player.

**Location**: `src/game/types.ts`

**Properties**:
- `id: string` - Unique identifier
- `name: string` - Display name
- `hand: Domino[]` - Current hand
- `teamId: 0 | 1` - Team (partner on seat 2)
- `marks: number` - Marks scored this game
- `suitAnalysis: SuitAnalysis` - Hand composition

**Related**: Team, PlayerSession, Hand

#### PublicPlayer
**Definition**: Player without hand visibility.

**Location**: `src/game/types.ts`

**Use**: Filtering state for other players

**Related**: Player, FilteredGameState

#### Team
**Definition**: Group of two players.

**Structure**:
- Team 0: Seats 0 and 2
- Team 1: Seats 1 and 3

**Related**: Player, teamId

---

### SuitAnalysis
**Definition**: Breakdown of player's hand by suit showing composition and strength.

**Location**: `src/game/types.ts`

**Provides**:
- Count of dominoes per suit
- List of dominoes in each suit
- Highest pip value in each suit

**Used By**: AI strategy for bidding, hand evaluation, trump selection

**Related**: Player, Hand

---

## 13. Architectural Patterns & Principles

### Pure Functions
**Definition**: Functions with no side effects, deterministic output.

**Applied To**:
- Executors: State transitions
- Composition: Layer and variant composition
- Filtering: State and action visibility
- Utilities: Suit following, score calculation

**Benefit**: Testability, reproducibility, composability

**Related**: Immutability, Determinism

---

### Parametric Polymorphism
**Definition**: Behavior determined by injected parameters (rules, layers), not state inspection.

**Applied To**: Executors never inspect state, always call `rules.method()`

**Benefit**: Same executor code works for all variants without conditional logic or state inspection.

**Key Pattern**: Executors call rules methods rather than inspecting state for variant-specific logic. Layer composition determines behavior at initialization time, executors never know.

**Related**: Composition Over Configuration, GameRules

---

### Composition Over Configuration
**Definition**: Achieve behavior via function composition, not conditional flags.

**Applied To**:
- Layers override methods (not flags)
- Variants wrap state machines (not switches)
- GameKernel composes in constructor

**Benefit**: Eliminates conditional complexity, enables unlimited combinations

**Related**: Parametric Polymorphism, Layer Composition

---

### Correct by Construction
**Definition**: Use type system to prevent bugs at compile time.

**Applied To**:
- GameAction union type (can't create invalid actions)
- GamePhase enum (can't have invalid phase)
- Domino.id uniqueness (checked at type level)
- Capability types (type-safe permissions)

**Benefit**: Compile errors catch bugs before runtime

**Related**: Type Safety, TypeScript

---

### Event Sourcing
**Definition**: Actions are source of truth, state is derived.

**Applied To**: Action history enables complete replay

**Benefit**:
- Time-travel debugging
- Sharing via URL
- Deterministic testing
- Complete audit trail

**Related**: Action History, Determinism

---

### Capability-Based Security
**Definition**: Permissions via composable tokens, not identity checks.

**Applied To**: PlayerSession capabilities determine action/visibility rights

**Benefit**:
- Fine-grained permissions
- Composable (multiple capabilities)
- Transparent (tokens are data)
- Flexible (easy to extend)

**Key Pattern**: Check capability tokens (observe-all-hands, see-hints, etc.) rather than player identity (playerId comparison). Capabilities are composable and transparently represent permissions.

**Related**: Capability, PlayerSession, Authorization

---

### Separation of Concerns
**Definition**: Each layer has clear, narrow responsibility.

**Layers**:
1. **Engine**: Pure game logic
2. **Kernel**: Game logic + multiplayer
3. **Server**: Orchestration + lifecycle
4. **Transport**: Message routing
5. **UI**: Rendering + interaction

**Benefit**: Changes in one layer don't cascade

**Related**: Architecture Stack

---

## 14. Architectural Invariants

**VIOLATION OF ANY = REGRESSION**

1. **Pure State Storage**
   - GameKernel stores unfiltered GameState
   - Filtering happens per-request, not at rest

2. **Server Authority**
   - Client trusts server's validActions list
   - Client never refilters or revalidates

3. **Capability-Based Access**
   - Permissions via capability tokens
   - Never identity checks (playerId comparison)

4. **Single Composition Point**
   - GameKernel constructor ONLY place layers/variants compose
   - ExecutionContext created once, used everywhere

5. **Zero Coupling**
   - Core engine has zero multiplayer awareness
   - Core engine has zero networking awareness
   - Layers/variants don't reference multiplayer

6. **Parametric Polymorphism**
   - Executors call `rules.method()`, never inspect state
   - No conditional logic based on state type checks

7. **Event Sourcing**
   - State must be derivable from `replayActions(config, history)`
   - Actions are immutable source of truth

8. **Clean Separation**
   - GameServer orchestrates, GameKernel executes, Transport routes
   - Each component has single responsibility

---

## 15. Concept Relationships & Interactions

### Request-Response Flow

**Client sends action**: User clicks button or AI makes decision

**Transport routes**: Network/IPC layer delivers message to server

**GameServer processes**: Orchestrator receives message and routes to GameKernel

**GameKernel executes**:
1. Find PlayerSession by player ID
2. Get valid actions via composed state machine (with layers and variants)
3. Filter actions by session capabilities (observe-hands, act-as-player, etc.)
4. Validate requested action against filtered set
5. Execute action via executor (which delegates to rules methods)
6. Process any auto-execute actions (speed variant, oneHand variant, etc.)

**GameKernel notifies**: For each subscriber with their perspective:
1. Filter state by capabilities (which hands are visible)
2. Filter actions by capabilities (hints visible? ai-intent visible?)
3. Build KernelUpdate with custom view

**GameServer broadcasts**: Send updates to all connected clients via Transport

**NetworkGameClient receives**: Cache filtered state and action list locally

**UI updates reactively**: Svelte stores receive update → ViewProjection recomputes → Components re-render

### Composition Pipeline

**GameKernel initialization** (single composition point):

1. **Get enabled layers**: Read from GameConfig which special contracts (nello, splash, plunge, sevens) are active

2. **Compose layers → GameRules**: Apply reduce pattern where each layer overrides specific rule methods. Non-overridden rules pass through unchanged.

3. **Create base state machine**: Build action generator using composed rules

4. **Apply variants to actions**: Function composition wraps the base state machine. Tournament filters, OneHand scripts, Hints annotates.

5. **Bundle ExecutionContext**: Group layers, rules, and action generator into immutable container

6. **Use everywhere**: Executors receive context and call rule methods - they never inspect state or know about specific variants

**Result**: Same executor code works for all variant combinations with zero conditional logic.

### State Transformation Pipeline

**GameState** (core) → Unfiltered, server-only, source of truth

**↓ Validation** → GameKernel validates action against capabilities and rules

**GameState** (updated) → State changes, now needs distribution

**↓ Capability-based filtering** → For each subscriber, filter hands and action metadata based on their permissions

**FilteredGameState** (hand-filtered) → Client-safe version ready for transmission

**↓ Pure computation** → ViewProjection transforms into UI-ready format with grouped actions and labels

**ViewProjection** (UI data) → Turn info, hand with playability, action choices, scoring

**↓ Svelte stores** → Reactive state management in browser

**Component state** → Derived stores compute UI metadata

**↓ Template rendering** → Components reactively update

**DOM elements** → User sees current game state

**Key Pattern**: GameKernel filters once per subscriber perspective. Transport delivers. Client trusts server's filtered data without revalidating.

---

## 16. Quick Reference: Concepts by File

### src/game/types.ts
- GameAction, GameState, Domino, Bid, Play, Trick, Player, Suit types

### src/game/core/
- gameEngine.ts: GameEngine class, getValidActions, getNextStates
- actions.ts: Executors (executeBid, executePlay, etc.)
- state.ts: State utilities
- rules.ts: Rule validation

### src/game/layers/
- types.ts: GameRules (13 methods), GameLayer, HandOutcome
- compose.ts: Layer composition via reduce
- base.ts: Base Layer (standard Texas 42)
- nello.ts, splash.ts, plunge.ts, sevens.ts: Special contracts

### src/game/variants/
- types.ts: Variant, StateMachine
- registry.ts: applyVariants composition
- tournament.ts, oneHand.ts, hints.ts, speed.ts: Implementations

### src/game/multiplayer/
- types.ts: PlayerSession, Capability, MultiplayerGameState
- authorization.ts: authorizeAndExecute
- capabilityUtils.ts: Filtering functions
- NetworkGameClient.ts: Client interface
- AIClient.ts: AI player

### src/game/ai/
- types.ts: AIStrategy, PlayerController, AIDifficulty
- strategies.ts: RandomAIStrategy, BeginnerAIStrategy
- actionSelector.ts: selectAIAction
- utilities.ts: analyzeHand
- hand-strength.ts: calculateHandStrengthWithTrump, determineBestTrump, BID_THRESHOLDS, LAYDOWN_SCORE
- lexicographic-strength.ts: calculateLexicographicStrength
- domino-strength.ts: analyzeDomino, DominoStrength
- multi-trump-analysis.ts: MultiTrumpAnalysis
- gameSimulator.ts: Game simulation utilities

### src/kernel/
- GameKernel.ts: Authority, composition, filtering, lifecycle

### src/server/
- GameServer.ts: Orchestrator
- transports/: Transport abstractions
- ai/: AIManager

### src/game/view-projection.ts
- ViewProjection, BidStatus, HandDomino, TrickDisplay, HandResults

---

## How to Navigate This Document

**By Role**:
- **Game Designer**: See Variants, Layers, Domain Objects
- **Backend Engineer**: See Server Architecture, Authorization, Event Sourcing, AI Architecture
- **Frontend Engineer**: See View Projection, Client Architecture, Protocol
- **AI Developer**: See AI Architecture, Hand Analysis System, GameSimulator
- **Architect**: See Architectural Patterns, Invariants, Concept Relationships

**By Task**:
- **Add new special contract**: Layers, Layer Implementations, GameRules
- **Add new game variant**: Variants, StateMachine, Variant Implementations
- **Fix authorization bug**: Authorization, Capability, PlayerSession
- **Debug UI issue**: ViewProjection, FilteredGameState, Protocol
- **Understand event sourcing**: Action History, Event Sourcing Formula, URL Compression
- **Debug action execution**: Executor, ExecutionContext, GameRules
- **Improve AI strategy**: AI Architecture, Hand Analysis System, BeginnerAIStrategy
- **Add new AI difficulty level**: AIDifficulty, AIStrategy, GameSimulator

**By Question**:
- "What's the difference between X and Y?" → See Concept Relationships
- "Where is X defined?" → See Quick Reference
- "How does X work?" → See concept definition
- "Why did we design X this way?" → See Architectural Patterns

---

**Last Updated**: October 2025
**Architecture Version**: mk5-tailwind (post-refactoring)
