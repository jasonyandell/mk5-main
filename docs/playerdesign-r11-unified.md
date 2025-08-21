# Player Perspective Layer — R11 Unified Implementation

## Core Principle
Pure functional core with practical state transitions. Type-safe by construction. Generate only valid actions. Every line of code is a liability.

## Implementation Status

✅ **Already Implemented**
- StateTransition pattern (`src/game/types.ts:114-118`)
- getNextStates() (`src/game/core/gameEngine.ts:89-136`)
- URL compression (`src/game/core/url-compression.ts`)
- Game store with validation (`src/stores/gameStore.ts`)
- Action-based UI (`src/lib/components/ActionPanel.svelte`)

🚧 **Needs Building**
- Consensus as first-class actions
- PlayerView with type-safe privacy
- AI auto-agreement for consensus
- Pure executeAction() with history

## The Three-Layer Architecture

### Layer 1: Pure Game Logic (Core)
```typescript
// ✅ ALREADY EXISTS: src/game/types.ts:14-100
export interface GameState {
  players: Player[];
  phase: GamePhase;
  currentPlayer: number;
  currentTrick: Play[];
  tricks: Trick[];
  // 🚧 TO ADD: Consensus tracking
  consensus: {
    completeTrick: Set<number>;
    scoreHand: Set<number>;
  };
  // 🚧 TO ADD: Action history for replay
  actionHistory: GameAction[];
  // ... other game state
}

// ✅ PARTIALLY EXISTS: src/game/types.ts:102-107 (as StateTransition)
// 🚧 TO BUILD: Separate GameAction type for pure actions
export type GameAction = 
  | { type: 'play-domino'; player: number; domino: Domino }
  | { type: 'bid'; player: number; bid: number }
  | { type: 'agree-complete-trick'; player: number }  // 🚧 NEW
  | { type: 'complete-trick' }  // 🚧 NEW
  // ... etc

// 🚧 TO BUILD: Pure state transition (currently mixed with getNextStates)
// Will replace parts of src/game/core/gameEngine.ts:applyAction
export function executeAction(state: GameState, action: GameAction): GameState {
  // Always append to history
  const newState = {
    ...state,
    actionHistory: [...state.actionHistory, action]
  };
  
  // Process action
  switch (action.type) {
    case 'agree-complete-trick':
      return recordAgreement(newState, action.player, 'completeTrick');
    case 'complete-trick':
      if (newState.consensus.completeTrick.size !== 4) {
        return newState; // Invalid, but recorded
      }
      return completeTrick(newState); // Clears consensus internally
    // ... other actions
  }
}
```

### Layer 2: State Transitions (Bridge)
```typescript
// ✅ ALREADY EXISTS: src/game/types.ts:114-118
export interface StateTransition {
  id: string;           // Compressed action ID for URLs
  label: string;        // Human-readable label
  // 🚧 TO ADD: action field
  action: GameAction;   // The actual action
  newState: GameState;  // Pre-computed resulting state
}

// ✅ ALREADY EXISTS: src/game/core/gameEngine.ts:89-136
// Currently implements this pattern but needs refactoring for consensus
export function getNextStates(state: GameState): StateTransition[] {
  const transitions: StateTransition[] = [];
  const validActions = getValidActions(state);
  
  for (const action of validActions) {
    transitions.push({
      id: actionToId(action),
      label: actionToLabel(action),
      action: action,  // 🚧 TO ADD
      newState: executeAction(state, action) // Pre-compute
    });
  }
  
  return transitions;
}

// ✅ PARTIALLY EXISTS: src/game/core/gameEngine.ts:138-180
// Currently in getValidActions() but needs consensus additions
function getValidActions(state: GameState): GameAction[] {
  const actions: GameAction[] = [];
  
  // ✅ ALREADY IMPLEMENTED: Player-specific actions
  if (state.phase === 'playing') {
    const player = state.players[state.currentPlayer];
    const validPlays = getValidPlays(player.hand, state);
    for (const domino of validPlays) {
      actions.push({ type: 'play-domino', player: state.currentPlayer, domino });
    }
  }
  
  // 🚧 TO ADD: Consensus actions - any player can agree
  if (state.phase === 'trick-complete') {
    for (let i = 0; i < 4; i++) {
      if (!state.consensus.completeTrick.has(i)) {
        actions.push({ type: 'agree-complete-trick', player: i });
      }
    }
  }
  
  return actions;
}
```

### Layer 3: Player View (Privacy)
```typescript
// 🚧 TO BUILD: Type-safe public player (currently Player has hands visible)
// Will replace src/game/types.ts:Player for public views
export interface PublicPlayer {
  id: number;
  name: string;
  teamId: 0 | 1;
  handCount: number;  // No hand field exists in type!
}

// 🚧 TO BUILD: Player-specific view with privacy
// New type to add alongside GameState
export interface PlayerView {
  // Core state (filtered)
  playerId: number;
  phase: GamePhase;
  self: { id: number; hand: Domino[] };  // Only self has hands
  players: PublicPlayer[];  // Others have no hand field
  
  // Available transitions for this player
  validTransitions: StateTransition[];
  
  // Consensus visibility
  consensus: {
    completeTrick: Set<number>;
    scoreHand: Set<number>;
  };
  
  // Recent events for UI
  recentEvents: GameEvent[];
}

// 🚧 TO BUILD: Player view projection
// New function to add to src/game/core/
export function getPlayerView(state: GameState, playerId: number): PlayerView {
  const allTransitions = getNextStates(state);
  
  return {
    playerId,
    phase: state.phase,
    
    // Private data only for self
    self: {
      id: playerId,
      hand: state.players[playerId].hand
    },
    
    // Public info for all players (type-safe, no hands!)
    players: state.players.map(p => ({
      id: p.id,
      name: p.name,
      teamId: p.teamId,
      handCount: p.hand.length
    })),
    
    // Filter transitions to this player
    validTransitions: allTransitions.filter(t => {
      if ('player' in t.action && t.action.player !== playerId) {
        return false;
      }
      return true;
    }),
    
    consensus: state.consensus,
    recentEvents: computeRecentEvents(state.actionHistory)
  };
}
```

## The Svelte Store (Orchestration)

```typescript
// ✅ ALREADY EXISTS: src/stores/gameStore.ts
// Current implementation with notes on what to add
import { writable, derived } from 'svelte/store';

// ✅ ALREADY EXISTS: src/stores/gameStore.ts:113-115
export const gameState = writable<GameState>(createInitialState());
export const actionHistory = writable<StateTransition[]>([]);

// 🚧 TO BUILD: Player view store
// Will replace direct gameState access in components
export const playerView = derived(
  [gameState, currentPlayerId],
  ([$state, $playerId]) => getPlayerView($state, $playerId)
);

// ✅ ALREADY EXISTS: src/stores/gameStore.ts:143-147
// Currently derived from gameState, will derive from playerView
export const availableActions = derived(
  playerView,
  $view => $view.validTransitions
);

// ✅ ALREADY EXISTS: src/stores/gameStore.ts:211-226
// Current executeAction, needs AI response handling
export const gameActions = {
  executeTransition: (transition: StateTransition) => {
    // ✅ ALREADY: Record in history
    actionHistory.update(h => [...h, transition]);
    
    // ✅ ALREADY: Update state (already computed!)
    gameState.set(transition.newState);
    
    // ✅ ALREADY: Update URL
    updateURL(get(actionHistory));
    
    // 🚧 TO ADD: Trigger AI responses if needed
    processAIResponses(transition.newState);
  }
};

// 🚧 TO BUILD: AI response handler
// New function to add to gameStore.ts
async function processAIResponses(state: GameState) {
  for (let i = 0; i < 4; i++) {
    if (!isHumanPlayer(i)) {
      const view = getPlayerView(state, i);
      const aiAction = chooseAIAction(view.validTransitions);
      if (aiAction) {
        gameActions.executeTransition(aiAction);
      }
    }
  }
}
```

## The UI (Can Only Do Valid Things)

```svelte
<!-- ✅ ALREADY EXISTS: src/lib/components/ActionPanel.svelte:30-50 -->
<script lang="ts">
  import { availableActions, gameActions } from '../../stores/gameStore';
</script>

<!-- ✅ ALREADY IMPLEMENTED: Can only click valid actions -->
{#each $availableActions as transition}
  <button on:click={() => gameActions.executeAction(transition)}>
    {transition.label}
  </button>
{/each}

<!-- No way to create invalid actions! -->
```

## URL Compression (Efficient Sharing)

```typescript
// ✅ ALREADY EXISTS: src/game/core/url-compression.ts
// ✅ ALREADY INTEGRATED: src/stores/gameStore.ts:73-103
import { encodeURLData, decodeURLData } from '@/game/core/url-compression';

// ✅ ALREADY IMPLEMENTED: src/stores/gameStore.ts:73-103
export function saveToURL(history: StateTransition[]): string {
  const urlData = {
    v: 1,
    s: { s: getInitialSeed() },
    a: history.map(t => ({ i: t.id }))
  };
  return encodeURLData(urlData);
}

// ✅ ALREADY IMPLEMENTED: src/stores/gameStore.ts:279-317
export function loadFromURL(encoded: string): GameState {
  const data = decodeURLData(encoded);
  let state = createInitialState(data.s.s);
  
  // Replay all actions
  for (const compressed of data.a) {
    const action = parseActionId(compressed.i);
    state = executeAction(state, action);
  }
  
  return state;
}
```

## Consensus Without Special Cases

```typescript
// 🚧 TO BUILD: Consensus handling with AI auto-agreement
// In single-player, AIs auto-agree instantly
function processAIResponses(state: GameState) {
  const view = getPlayerView(state, 0); // Human view
  
  // Find consensus actions that need AI agreement
  const consensusActions = view.validTransitions.filter(t =>
    t.action.type === 'agree-complete-trick' ||
    t.action.type === 'agree-score-hand'
  );
  
  // AIs agree immediately (no waiting)
  for (const transition of consensusActions) {
    if (transition.action.player !== 0) { // Not human
      gameActions.executeTransition(transition);
    }
  }
}
```

## Complete Flow Example

```typescript
// 1. Human clicks "Play 6-6" button
UI → gameActions.executeTransition(play66transition)

// 2. Store updates state
actionHistory.push(play66transition)
gameState = play66transition.newState

// 3. AI responds
view = getPlayerView(newState, 1)
aiTransition = pickBest(view.validTransitions)
gameActions.executeTransition(aiTransition)

// 4. Repeat until trick complete
// 5. Human clicks "agree-complete-trick"
// 6. AIs auto-agree instantly
// 7. Complete-trick executes
// 8. Consensus clears, next trick begins
```

## Key Insights

1. **Pure Core**: `executeAction()` is truly pure, no mutations
2. **Pre-computed Transitions**: No need to recalculate states
3. **Single Source of Truth**: `getValidActions()` defines what's legal
4. **Type Safety**: `PublicPlayer` literally can't have hands
5. **No Invalid States**: UI can only trigger valid transitions
6. **Consensus is Just Actions**: No special handling needed
7. **AI as External**: Just picks from valid transitions

## What Makes This Unified

- **Core logic** matches R10's pure functional design
- **Practical implementation** matches actual codebase patterns
- **StateTransition** bridges pure functions with efficient UI
- **Consensus** works identically for single/multiplayer
- **URL compression** built in from the start
- **Type safety** prevents bugs at compile time

## Testing

```typescript
test('pure state transitions', () => {
  const state = createInitialState();
  const action = { type: 'bid', player: 0, bid: 30 };
  const newState = executeAction(state, action);
  
  expect(newState.actionHistory).toContain(action);
  expect(newState).not.toBe(state); // New object
});

test('only valid actions available', () => {
  const state = createInitialState();
  const transitions = getNextStates(state);
  
  // Should only have bids and pass for player 0
  expect(transitions.every(t => 
    t.action.type === 'bid' || t.action.type === 'pass'
  )).toBe(true);
});

test('consensus flow', () => {
  let state = setupTrickComplete();
  
  // All 4 players agree
  for (let i = 0; i < 4; i++) {
    state = executeAction(state, { 
      type: 'agree-complete-trick', 
      player: i 
    });
  }
  
  // Complete trick executes
  state = executeAction(state, { type: 'complete-trick' });
  
  // Consensus cleared
  expect(state.consensus.completeTrick.size).toBe(0);
});
```

## Implementation Roadmap

### Phase 1: Core Types (Foundation)
1. Add `consensus` field to GameState (`src/game/types.ts`)
2. Add `actionHistory` field to GameState
3. Create `GameAction` type separate from StateTransition
4. Create `PublicPlayer` and `PlayerView` interfaces

### Phase 2: Pure Functions
1. Build `executeAction()` pure function (`src/game/core/actions.ts`)
2. Add consensus actions to `getValidActions()` (`src/game/core/gameEngine.ts:138`)
3. Implement `getPlayerView()` function (`src/game/core/playerView.ts`)
4. Add `action` field to StateTransition

### Phase 3: Store Integration  
1. Add `playerView` derived store (`src/stores/gameStore.ts`)
2. Implement `processAIResponses()` for consensus
3. Update `availableActions` to use playerView
4. Add consensus state management

### Phase 4: Testing
1. Test pure state transitions
2. Test type-safe privacy (PublicPlayer can't have hands)
3. Test consensus flow with AI auto-agreement
4. Test URL compression with new action types

## File References

### Already Implemented ✅
- `src/game/types.ts` - Core types (GameState, StateTransition)
- `src/game/core/gameEngine.ts` - getNextStates, applyAction
- `src/game/core/url-compression.ts` - URL encoding/decoding
- `src/stores/gameStore.ts` - Store orchestration
- `src/lib/components/ActionPanel.svelte` - UI that shows only valid actions

### To Be Created 🚧
- `src/game/core/actions.ts` - Pure executeAction function
- `src/game/core/playerView.ts` - Player view projection
- `src/game/types/PlayerView.ts` - PlayerView and PublicPlayer types

## Summary

~200 lines of pure, type-safe, deterministic code:
1. **Pure core** processes actions immutably
2. **Transitions** pre-compute valid next states  
3. **Views** provide player-specific projections
4. **Store** orchestrates core, UI, and AI
5. **UI** can only trigger valid transitions

Everything else is outside the core. No special cases. No mutations. No invalid states possible.