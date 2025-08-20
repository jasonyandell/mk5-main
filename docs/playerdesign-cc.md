# Player Perspective Layer Design

## Overview
This document outlines the design for a player perspective layer that provides controlled visibility and action access for each player in the Texas 42 game. The system generates player-specific views of the game state while maintaining comprehensive debugging capabilities for development.

## Core Requirements

### 1. Multi-Perspective State Generation
- Generate all 4 player perspectives simultaneously from a single GameState
- Each perspective contains only information that player should see:
  - Their own hand (full visibility)
  - Public game state (tricks, scores, current bid, trump)
  - Opponent hands are hidden (shown as count only)
  - Played dominoes are visible to all

### 2. Player-Specific Action Filtering
- Each player perspective includes only valid actions they can take
- Actions are filtered from the full set returned by `getValidActions()`
- Only the current player has available actions
- System actions (complete-trick, score-hand) are handled automatically

### 3. Developer Debug Interface
- Full visibility mode for development and testing
- Ability to view all 4 perspectives simultaneously
- Step through game history from any player's perspective
- Validate perspective generation correctness
- Export/import perspective snapshots for debugging

## Architecture Components

### Type Definitions

```typescript
// Player's view of the game state
interface PlayerPerspective {
  playerId: number;                    // Which player this perspective is for
  visibleState: PlayerVisibleState;    // What this player can see
  availableActions: GameAction[];      // Actions this player can take
  hand: Domino[];                      // Player's own hand (full visibility)
  opponentHandCounts: Map<number, number>; // Number of dominoes each opponent has
  publicState: PublicGameState;        // Shared visible state
}

// State visible to all players
interface PublicGameState {
  phase: GamePhase;
  currentPlayer: number;
  dealer: number;
  bids: Bid[];
  currentBid: Bid;
  winningBidder: number;
  trump: TrumpSelection;
  tricks: Trick[];
  currentTrick: Play[];
  teamScores: [number, number];
  teamMarks: [number, number];
  gameTarget: number;
}

// What a specific player can see
interface PlayerVisibleState extends PublicGameState {
  myHand: Domino[];
  myTeammate: number;
  mySuitAnalysis: SuitAnalysis;
}

// Manager for all perspectives
interface PerspectiveManager {
  generatePerspectives(state: GameState): Map<number, PlayerPerspective>;
  getPerspective(playerId: number): PlayerPerspective;
  getHistoricalPerspective(playerId: number, actionIndex: number): PlayerPerspective;
  getAllPerspectives(): Map<number, PlayerPerspective>;
  validatePerspectives(): ValidationResult;
}

// Debug information with full visibility
interface PerspectiveDebugInfo {
  fullState: GameState;
  perspectives: Map<number, PlayerPerspective>;
  actionHistory: GameAction[];
  stateHistory: GameState[];
  validationErrors: string[];
}
```

## Implementation Design

### Core Modules

#### 1. `perspectiveGenerator.ts`
Pure functions for generating perspectives:

```typescript
// Generate all 4 player perspectives from a game state
function generateAllPerspectives(state: GameState): Map<number, PlayerPerspective>

// Filter state to what a player can see
function filterVisibleState(state: GameState, playerId: number): PlayerVisibleState

// Get actions available to a specific player
function filterAvailableActions(actions: GameAction[], playerId: number): GameAction[]

// Replace opponent hands with masked data
function maskOpponentHands(state: GameState, playerId: number): GameState
```

#### 2. `perspectiveManager.ts`
Runtime management service:

```typescript
class PerspectiveManager {
  private perspectives: Map<number, PlayerPerspective>
  private history: PerspectiveSnapshot[]
  
  // Update all perspectives when state changes
  updatePerspectives(newState: GameState): void
  
  // Get current perspective for a player
  getPerspective(playerId: number): PlayerPerspective
  
  // Get perspective at a specific point in history
  getHistoricalPerspective(playerId: number, index: number): PlayerPerspective
  
  // Enable/disable debug mode
  setDebugMode(enabled: boolean): void
}
```

#### 3. `perspectiveDebugger.ts`
Development and debugging utilities:

```typescript
class PerspectiveDebugger {
  // View all 4 perspectives side by side
  getAllPerspectives(): DebugPerspectiveView
  
  // Replay game from specific player's view
  replayFromPerspective(playerId: number, actions: GameAction[]): ReplayResult
  
  // Validate perspective generation
  validatePerspective(perspective: PlayerPerspective, fullState: GameState): ValidationResult
  
  // Export for analysis
  exportSnapshot(): PerspectiveDebugInfo
  
  // Import saved snapshot
  importSnapshot(data: PerspectiveDebugInfo): void
}
```

### Integration Points

#### GameEngine Integration
```typescript
// Extended GameEngine with perspective support
class GameEngineWithPerspectives extends GameEngine {
  private perspectiveManager: PerspectiveManager
  
  executeAction(action: GameAction): void {
    super.executeAction(action)
    this.perspectiveManager.updatePerspectives(this.getState())
  }
  
  getPlayerPerspective(playerId: number): PlayerPerspective {
    return this.perspectiveManager.getPerspective(playerId)
  }
}
```

#### Store Integration
```typescript
// Svelte stores for UI
export const currentPlayerPerspective = derived(
  [gameState, currentPlayerId],
  ([$gameState, $playerId]) => generatePerspective($gameState, $playerId)
)

export const debugPerspectives = derived(
  [gameState, debugMode],
  ([$gameState, $debugMode]) => 
    $debugMode ? generateAllPerspectives($gameState) : null
)
```

#### URL Compression Integration
```typescript
// Extended URL data with perspective support
interface URLDataWithPerspective extends URLData {
  p?: number;  // Current perspective player ID
  d?: boolean; // Debug mode enabled
}

// Replay from specific perspective
function replayFromPerspective(url: string, playerId: number): PerspectiveReplayResult
```

## Usage Examples

### Basic Usage
```typescript
// Create game engine with perspectives
const engine = new GameEngineWithPerspectives(initialState)

// Get perspective for player 0
const player0View = engine.getPlayerPerspective(0)
console.log(player0View.hand)              // Full hand visible
console.log(player0View.availableActions)  // Only if it's player 0's turn

// Execute an action
const action = player0View.availableActions[0]
engine.executeAction(action)

// Perspectives automatically updated
const updatedView = engine.getPlayerPerspective(0)
```

### Debug Mode
```typescript
// Enable debug mode
const debugger = new PerspectiveDebugger(engine)
debugger.setDebugMode(true)

// View all perspectives
const allViews = debugger.getAllPerspectives()
allViews.forEach((view, playerId) => {
  console.log(`Player ${playerId} sees:`, view)
})

// Validate perspectives
const validation = debugger.validatePerspective(player0View, engine.getState())
if (!validation.isValid) {
  console.error('Perspective errors:', validation.errors)
}
```

### Historical Replay
```typescript
// Replay game from player 2's perspective
const replay = debugger.replayFromPerspective(2, actionHistory)

// Step through each state
replay.states.forEach((state, index) => {
  console.log(`Action ${index}: Player 2 saw:`, state)
})

// Export for analysis
const snapshot = debugger.exportSnapshot()
localStorage.setItem('debug-snapshot', JSON.stringify(snapshot))
```

## Testing Strategy

### Unit Tests
- Test perspective generation for all game phases
- Verify action filtering correctness
- Ensure opponent hands are properly masked
- Validate public state consistency

### Integration Tests
- Test perspective updates through full game flow
- Verify URL compression/decompression with perspectives
- Test historical replay from different perspectives
- Validate perspective switching in debug mode

### E2E Tests
- Test UI displays correct perspective
- Verify action availability matches perspective
- Test debug mode shows all perspectives
- Validate perspective consistency across game flow

## Benefits

1. **Security**: Players can only see and do what they should
2. **Debugging**: Complete visibility for development
3. **Deterministic**: Same state always produces same perspectives
4. **Testable**: Easy to validate perspective correctness
5. **Extensible**: Ready for multiplayer/network play
6. **Pure**: No side effects, follows functional programming principles

## Future Enhancements

1. **Network Support**: Send only player-specific perspectives to clients
2. **Spectator Mode**: Read-only perspective with full visibility
3. **Replay Analysis**: Compare perspectives to understand decisions
4. **AI Training**: Use perspectives to train AI from player viewpoint
5. **Perspective Validation**: Server-side validation in multiplayer
6. **Compression**: Optimize perspective data for network transmission

## Migration Path

### Phase 1: Core Implementation
- Implement type definitions
- Build perspective generator
- Create perspective manager
- Add debug utilities

### Phase 2: Integration
- Integrate with GameEngine
- Update stores for UI
- Modify components to use perspectives
- Add to URL compression

### Phase 3: Testing & Validation
- Complete unit test coverage
- Integration test scenarios
- E2E test player interactions
- Performance optimization

### Phase 4: Production Ready
- Enable in production mode
- Add telemetry for perspective usage
- Documentation and examples
- Performance monitoring