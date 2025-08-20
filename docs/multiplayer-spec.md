# Multiplayer Architecture Specification

## Overview
Minimal architecture for secure player views and future AI/multiplayer capabilities. Every line of code is a liability. This design is correct by construction.

## Part 1: Secure Player Views (R9 Design)

### Core Principle
Type-safe whitelist projection. Security by construction, not runtime checks.

### The Types
```typescript
// Public info only - type cannot hold private data
interface PublicPlayer {
  id: number;
  name: string;
  teamId: 0 | 1;
  handCount: number;  // Count only, no hand field exists
}

// Self info - the only place hands exist
interface SelfPlayer {
  id: number;
  hand: Domino[];
}

// Complete player view - whitelist only what UI needs
interface PlayerView {
  playerId: number;
  phase: GamePhase;
  // ... essential state fields ...
  self: SelfPlayer;           // Only place with hand data
  players: PublicPlayer[];    // Cannot contain hands by type
  validActions: GameAction[]; // Pre-filtered for this player
}
```

### The Single Function
```typescript
function getPlayerView(state: GameState, playerId: number, isHost = false): PlayerView {
  // Build from scratch - whitelist projection
  // Filter actions to this player
  // Gate neutral actions by isHost flag
}
```

### Security Guarantees
1. **Type-enforced**: `PublicPlayer` has no `hand` field
2. **Whitelist projection**: Build views from scratch
3. **Action filtering**: Each player sees only their valid actions
4. **Host gating**: Neutral actions require `isHost === true`

## Part 2: Minimal Event System

### Purpose
Optional wrapper for actions that enables future features without breaking existing code.

### The Event Type
```typescript
interface GameEvent {
  action: GameAction;      // Existing action type
  delay: number;           // 0 for instant, >0 for future
  metadata?: {
    source?: 'human' | 'ai';
    playerId?: number;
    comment?: string;      // AI chatter
    // Extensible for CEO's future plans
  };
}
```

### The Queue (~30 lines)
```typescript
class EventQueue {
  push(event: GameEvent): void
  process(): void  // Executes queued actions
  clear(): void    // Skip all delays
}
```

### Key Properties
- **Non-breaking**: Direct action execution still works
- **Opt-in**: Use events only when needed
- **Deterministic**: Same order every time
- **Extensible**: Metadata field for future features

## Part 3: Player Configuration

### Minimal Store Additions
```typescript
// Who is viewing (0-3)
const currentPlayerId = writable(0);

// Do they control neutral actions
const isHost = writable(true);

// Secure view derived from game state
const playerView = derived([gameState, currentPlayerId, isHost], 
  ([$state, $playerId, $isHost]) => getPlayerView($state, $playerId, $isHost)
);
```

### AI Integration (Future)
```typescript
// Each seat can be human or AI
interface PlayerSeat {
  id: number;
  type: 'human' | 'ai';
}

// AI takes over when type === 'ai'
if (seat.type === 'ai' && playerView.currentPlayer === seat.id) {
  const action = selectAIAction(playerView);
  eventQueue.push({ action, delay: 500 });
}
```

## Usage Examples

### Current: Human plays directly
```typescript
// Existing code unchanged
gameActions.executeAction(transition);
```

### With Events: AI plays with delay
```typescript
eventQueue.push({
  action: transition,
  delay: 500,
  metadata: {
    source: 'ai',
    playerId: 2,
    comment: "I'll take that trick!"
  }
});
```

### AI Conversation (CEO's vision)
```typescript
// P1 bids
eventQueue.push({
  action: bidAction,
  delay: 1000,
  metadata: { 
    playerId: 1, 
    comment: "30, and that's generous" 
  }
});

// P3 responds
eventQueue.push({
  action: bidAction,
  delay: 1500,
  metadata: { 
    playerId: 3, 
    comment: "31, partner's got my back" 
  }
});

// Player can skip with: eventQueue.clear()
```

### Spectator Mode
```typescript
// Watch player 2's perspective
currentPlayerId.set(2);
isHost.set(false);  // Can't control neutral actions

// Debug mode - see all hands
if (debugMode) {
  // Show raw gameState instead of playerView
}
```

## Implementation Phases

### Phase 1: Core Security (Required)
1. Implement `getPlayerView()` function
2. Add player view store
3. Update UI to use `playerView` instead of `gameState`
4. Verify no information leaks

### Phase 2: Event System (Recommended)
1. Add 30-line event queue
2. Keep direct execution as default
3. Use events only where needed

### Phase 3: AI Players (When Needed)
1. Add seat configuration
2. Trigger AI when it's their turn
3. Reuse existing quickplay logic

### Phase 4: External Systems (CEO's Plans)
1. Add data to event metadata
2. External systems read metadata
3. External systems inject responses

## Testing Strategy

### Security Tests
```typescript
test('no hand leakage', () => {
  const view = getPlayerView(state, 0);
  // Type system prevents: view.players[1].hand
  expect((view.players[1] as any).hand).toBeUndefined();
});
```

### Event Tests
```typescript
test('events execute in order', () => {
  eventQueue.push({ action: action1, delay: 0 });
  eventQueue.push({ action: action2, delay: 0 });
  // Verify action1 executes before action2
});
```

## Migration Path
1. Implement alongside existing system
2. No breaking changes to game logic
3. UI components migrate gradually
4. Old tests continue to pass

## Benefits
- **Minimal**: ~100 lines for core functionality
- **Secure**: Type system prevents leaks
- **Flexible**: Event metadata enables any future feature
- **Non-breaking**: Existing code continues to work
- **Testable**: Each part in isolation

## What This Enables (Without Writing Now)
- Multiplayer: Each client gets their `playerView`
- AI personalities: Different delays and comments
- Replay with timing: Store and replay events
- External AI: Read/write through metadata
- Spectator mode: Set `currentPlayerId` to watch
- Debug mode: Show different view levels

## Key Insight
The event system is just a queue. It doesn't change game logic, it just delays it. This separation means:
- Game logic stays pure
- Events can be ignored (instant mode)
- Future features just add metadata
- Everything is deterministic

This is the minimal foundation that enables maximum future flexibility.