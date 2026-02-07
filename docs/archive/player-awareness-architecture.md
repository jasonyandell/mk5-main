# Player Awareness & AI Architecture

## Overview
This document outlines the architecture for transitioning from developer mode (where you control all players) to a proper game mode with human and AI players.

## Core Principles
- **No changes to game logic** - Pure functions remain pure
- **Event-driven system** - All actions go through an event queue
- **Instant now, timed later** - Infrastructure for delays without using them yet
- **Personality-driven AI** - Different AI players behave differently

## Existing Foundation
The system builds on existing code:
- **Action system**: `GameAction` type from `src/game/types.ts`
- **Game engine**: `GameEngine` class from `src/game/core/gameEngine.ts`
- **Action history**: Already tracked, see `src/game/core/url-compression.ts` for how actions are compressed
- **AI logic**: Reuse `quickplayActions` from `src/stores/quickplayStore.ts`

## Architecture Components

### 1. Event System
The event system wraps existing `GameAction` (from `src/game/types.ts`) with timing and metadata.

```typescript
interface GameEvent {
  action: GameAction;      // Existing type from types.ts
  duration: number;        // How long this takes (0 for now)
  source: 'human' | 'ai' | 'system';
  metadata?: {             // For future personality features
    thinkingTime?: number;
    comment?: string;
  };
}
```

Events are generated based on:
- Action type (bids take longer to think about than obvious plays)
- Game state (tough decisions take longer)
- AI personality (some players are faster than others)

### 2. Player Configuration
Simple configuration determines who plays what:

```typescript
interface PlayerConfig {
  playerTypes: ('human' | 'ai')[];  // e.g., ['human', 'ai', 'ai', 'ai']
  humanSeat: number;                 // Which position human plays (0-3)
  showAllHands: boolean;             // Debug mode toggle
}
```

### 3. AI Personalities
Each AI has a personality that determines:
- How long they think about different actions
- What comments they might make
- Their play style preferences

The AI will reuse the existing decision logic from `quickplayActions.selectBestAction()` in `src/stores/quickplayStore.ts`, but add personality-based timing.

Example personality traits:
- **Fast Freddy**: Quick decisions, minimal delay
- **Thoughtful Theo**: Takes time on important decisions
- **Chatty Cathy**: Comments on plays (future feature)

### 4. Event Orchestrator
Manages the queue of events and processes them:
- In development: `instantMode = true` (all events execute immediately)
- In production: Events wait for their duration
- Testing: Can skip all animations

Key features:
- Deterministic execution order
- No race conditions
- Can be paused/resumed
- Supports fast-forward

### 5. Turn Controller
Monitors game state and triggers AI actions:
- Detects when it's an AI player's turn
- Uses `getValidActions()` from `src/game/core/gameEngine.ts` to get available moves
- Asks AI to select an action (reusing quickplay logic)
- Generates events based on AI personality
- Enqueues events for processing

Special handling for automatic actions:
- `complete-trick` and `score-hand` execute instantly (see `actionToId()` in `gameEngine.ts`)
- No personality delay for system actions

### 6. View Layer
Controls what each player can see. Updates needed in:
- `src/lib/components/PlayingArea.svelte` - Show only current player's hand
- `src/lib/components/ActionPanel.svelte` - Show actions only for human player
- `src/stores/gameStore.ts` - Add view filtering

**Human player sees:**
- Their own hand
- All played dominoes
- Current game state
- Available actions when it's their turn

**Debug mode shows:**
- All player hands
- All available actions
- AI decision process (future)

## Implementation Phases

### Phase 1: Core Infrastructure (Current)
- Event system with duration = 0
- Player configuration (human = P0, rest = AI)
- Basic turn controller
- View filtering

### Phase 2: AI Integration
- Reuse quickplay AI logic for decisions
- Auto-play for AI players
- Instant execution

### Phase 3: Polish (Future)
- Enable durations > 0
- Add personality configs
- Thinking indicators
- Comments/reactions

### Phase 4: Multiplayer Ready (Future)
- Each client gets their perspective
- Actions validated server-side
- Synchronized event playback

## File Structure
```
src/
  game/
    events/
      gameEvents.ts        # Event types and interfaces
      eventOrchestrator.ts # Queue management
      aiPersonality.ts     # Personality definitions
  services/
    aiPlayer.ts           # AI decision making
    turnController.ts     # Turn flow management
  stores/
    playerConfig.ts       # Player configuration
    viewStore.ts          # View filtering
```

## Integration with Existing Systems
- **URL sharing**: Action history already compressed in `url-compression.ts` - events don't change this
- **Game replay**: Can replay through event system using compressed actions from URLs
- **Testing**: E2E tests in `src/tests/e2e/` can use instant mode
- **Game state**: `GameState` type from `src/game/types.ts` unchanged

## Key Benefits
1. **Clean separation** - Events wrap actions, don't replace them
2. **Testable** - Can run in instant mode for tests
3. **Extensible** - Easy to add personalities, delays, effects
4. **Deterministic** - Reproducible gameplay (like existing URL replay)
5. **Future-proof** - Ready for multiplayer architecture

## Testing Strategy
- Unit tests run with `instantMode = true`
- E2E tests can skip animations
- Personality tests verify timing calculations
- Integration tests check turn flow

## Migration Path
1. Start with all players as AI except P0
2. Add toggle for human control of any seat
3. Add network layer for remote players (future)
4. Add spectator mode (future)