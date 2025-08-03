# Functional Programming Improvements for Texas 42

## Executive Summary

This document outlines functional programming improvements that will make invalid states unrepresentable while preserving JSON serialization and maintaining compatibility with the existing UI. The current implementation is already largely functional, but these improvements will provide stronger compile-time guarantees and better support for networked play.

## Current Architecture Analysis

The game currently uses:
- Immutable state updates through cloning
- Pure functions for state transitions
- Event sourcing with action history
- Dynamic action calculation via `getNextStates()`
- Simple JSON-serializable types

## Proposed Improvements

### 1. Phase-Based Discriminated Union (Highest Impact)

**Current:**
```typescript
interface GameState {
  phase: GamePhase;
  winningBidder: number | null;
  trump: Trump | null;
  currentBid: Bid | null;
  // ... many nullable fields
}
```

**Improved:**
```typescript
type GameState = 
  | { phase: 'bidding'; currentBidder: PlayerIndex; bids: Bid[]; hands: PlayerHands }
  | { phase: 'trump-selection'; selector: PlayerIndex; winningBid: Bid; hands: PlayerHands }
  | { phase: 'playing'; currentPlayer: PlayerIndex; trump: Trump; tricks: Trick[]; currentTrick: InProgressTrick; hands: PlayerHands }
  | { phase: 'scoring'; finalTricks: CompletedTrick[]; roundResult: RoundResult }
  | { phase: 'game-end'; winner: TeamIndex; finalScores: [TeamScore, TeamScore] }
```

**Benefits:**
- Impossible to access `trump` during bidding phase
- Clear who can act at any moment
- No null checks needed
- Type narrowing guides correct code

**JSON Serialization:** Trivial - discriminated unions serialize naturally

### 2. Embedded Valid Actions

**Current:**
```typescript
// UI must call getNextStates() separately
const actions = getNextStates(gameState);
```

**Improved:**
```typescript
type PlayingPhase = {
  phase: 'playing'
  currentPlayer: PlayerIndex
  validActions: PlayAction[]  // Pre-calculated valid plays
  // ... rest of state
}
```

**Benefits:**
- UI can't attempt invalid actions
- Single source of truth
- Reduces round trips in networked play
- Actions become part of the authoritative state

### 3. Player-Perspective Views

**Current:**
```typescript
// All players see full state
gameState.players[0].hand // Visible to everyone
```

**Improved:**
```typescript
type PlayerView = {
  playerId: PlayerIndex
  myHand: Domino[]
  publicState: PublicGameState
  availableActions: Action[]
  recentEvents: GameEvent[]
}

type PublicGameState = {
  phase: GamePhase
  visibleHands: Map<PlayerIndex, number> // Just counts
  currentTrick: Play[]
  // ... only public info
}
```

**Benefits:**
- Natural client-server separation
- Prevents cheating in networked play
- UI receives only what it needs
- Push-based updates via events

### 4. Branded Types for Domain Safety

**Current:**
```typescript
currentPlayer: number
dealer: number
teamId: 0 | 1
```

**Improved:**
```typescript
type PlayerIndex = number & { readonly _brand: 'PlayerIndex' }
type TeamIndex = number & { readonly _brand: 'TeamIndex' }
type SuitValue = number & { readonly _brand: 'SuitValue' }

// Constructor functions ensure validity
function PlayerIndex(n: number): PlayerIndex {
  if (n < 0 || n > 3) throw new Error('Invalid player index')
  return n as PlayerIndex
}
```

**Benefits:**
- Can't accidentally use player index as team index
- Type errors catch logic bugs
- Zero runtime overhead
- Serializes as plain numbers

### 5. Trick State Machine

**Current:**
```typescript
currentTrick: Play[]  // Could be 0-4 plays
```

**Improved:**
```typescript
type TrickState =
  | { status: 'waiting'; lead: PlayerIndex }
  | { status: 'one-played'; lead: PlayerIndex; plays: [Play] }
  | { status: 'two-played'; lead: PlayerIndex; plays: [Play, Play] }
  | { status: 'three-played'; lead: PlayerIndex; plays: [Play, Play, Play] }
  | { status: 'complete'; plays: [Play, Play, Play, Play]; winner: PlayerIndex; points: Points }
```

**Benefits:**
- Can't access winner before trick completes
- Tuple types ensure exactly 4 plays
- Clear state progression
- Type-safe trick evaluation

### 6. Event Sourcing with Causality

**Current:**
```typescript
actionHistory: StateTransition[]
```

**Improved:**
```typescript
type GameEvent = 
  | { type: 'game-started'; dealer: PlayerIndex; hands: PlayerHands; timestamp: number }
  | { type: 'bid-made'; player: PlayerIndex; bid: Bid; timestamp: number }
  | { type: 'trump-selected'; player: PlayerIndex; trump: Trump; timestamp: number }
  | { type: 'domino-played'; player: PlayerIndex; play: Play; resulting: TrickState; timestamp: number }
  | { type: 'trick-completed'; winner: PlayerIndex; points: Points; timestamp: number }

type GameLog = {
  events: GameEvent[]
  currentState: GameState
  stateCache: Map<number, GameState>  // Event index -> resulting state
}
```

**Benefits:**
- Natural replay and undo
- Network synchronization
- Spectator mode
- Debug time travel

## Migration Strategy

### Phase 1: Add Types Without Breaking Changes
1. Add branded type aliases alongside existing types
2. Add discriminated union types as additional exports
3. Keep existing `GameState` interface

### Phase 2: Dual API
1. Add new `getPlayerView()` alongside existing API
2. Add `getTypedState()` that returns discriminated union
3. Existing code continues working

### Phase 3: Gradual Migration
1. Update UI components one at a time
2. Use type guards to narrow discriminated unions
3. Replace nullable field access with phase-specific access

### Phase 4: Cleanup
1. Remove old types
2. Make new types the default export
3. Update documentation

## Backwards Compatibility

### Existing Code
```typescript
// This still works
if (gameState.trump !== null) {
  // Use trump
}
```

### New Code
```typescript
// But this is better
if (gameState.phase === 'playing') {
  // TypeScript knows trump exists
  console.log(gameState.trump)
}
```

### Serialization Compatibility
```typescript
// Add conversion functions
function toClassicState(state: NewGameState): ClassicGameState
function fromClassicState(state: ClassicGameState): NewGameState
```

## Example Implementation

```typescript
// New phase-based state that's UI-friendly
type GameState = BiddingPhase | TrumpPhase | PlayingPhase | ScoringPhase

// Helper to convert to flat UI state
function toUIState(state: GameState): UIGameState {
  const base = {
    phase: state.phase,
    players: getPlayers(state),
    dealer: getDealer(state),
    teamScores: getScores(state),
    teamMarks: getMarks(state),
  }
  
  switch (state.phase) {
    case 'bidding':
      return { ...base, currentPlayer: state.currentBidder, bids: state.bids }
    case 'playing':
      return { ...base, currentPlayer: state.currentPlayer, trump: state.trump }
    // ... etc
  }
}
```

## Impact Assessment

### Will Current Implementation Still Work?
**Yes**, with a transition period. The improvements can be added alongside existing code, allowing gradual migration.

### Performance Impact
- Slightly larger state objects (embedded actions)
- Faster UI updates (no action calculation)
- Better network efficiency (player views)

### Developer Experience
- Stronger type safety catches bugs earlier
- Clearer code with less null checking
- Better IDE autocomplete

### UI Impact
- Simpler components (actions provided)
- Natural reactive updates
- Clearer player perspective

## Recommendations

1. **Start with Phase-Based State** - Highest impact, easiest migration
2. **Add Embedded Actions** - Immediate UI benefits
3. **Implement Player Views** - Essential for networked play
4. **Add Branded Types** - Gradual safety improvements
5. **Event Sourcing** - Future-proofing for features

## Conclusion

These improvements maintain the functional nature of your game while adding compile-time guarantees that make bugs nearly impossible. The migration can be done gradually without breaking existing code, and JSON serialization remains trivial throughout.

The key insight is that **making invalid states unrepresentable** doesn't require exotic type systems - TypeScript's discriminated unions and branded types provide enormous safety improvements while keeping the code simple and the data JSON-friendly.