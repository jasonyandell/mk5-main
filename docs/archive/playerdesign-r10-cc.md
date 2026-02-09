# Player Perspective Layer — R10 Final Unified

## Core Principle
Type-safe whitelist projection. Security by construction, not runtime checks. Every line of code is a liability. Everything is an action.

## The Types (Leak-Proof by Construction)
```typescript
// Public info only - type cannot hold private data
export interface PublicPlayer {
  id: number;
  name: string;
  teamId: 0 | 1;
  handCount: number;  // Count only, no hand field exists
}

// Self info - the only place hands exist
export interface SelfPlayer {
  id: number;
  hand: Domino[];
}

// Complete player view
export interface PlayerView {
  // Essential state (whitelist only what UI needs)
  playerId: number;
  phase: GameState['phase'];
  dealer: number;
  currentPlayer: number;
  trump: TrumpSelection | null;
  currentSuit: number | null;
  bids: Bid[];
  currentTrick: Trick['plays'];
  tricks: Trick[];
  teamScores: [number, number];
  teamMarks: [number, number];
  gameTarget: number;
  
  // Player data with type-enforced privacy
  self: SelfPlayer;           // Only place with hand data
  players: PublicPlayer[];    // Cannot contain hands by type
  
  // Actions this player can take right now
  validActions: GameAction[];
  
  // Consensus state (who has agreed to proceed)
  consensus: {
    completeTrick: Set<number>;   // Players who agreed to complete trick
    scoreHand: Set<number>;        // Players who agreed to score hand
  };
  
  // Recent events for animations/UI (computed from recent actions)
  recentEvents: GameEvent[];
}

// Events computed from recent actions for UI/animations
export interface GameEvent {
  type: string;
  player?: number;
  details: any;
}
```

## First-Class Action Types
```typescript
// All actions are equal - no special cases
type GameAction = 
  // Player actions
  | { type: 'bid'; player: number; bid: number }
  | { type: 'play-domino'; player: number; domino: Domino }
  | { type: 'select-trump'; player: number; suit: number }
  
  // Consensus actions (first-class, not special)
  | { type: 'agree-complete-trick'; player: number }
  | { type: 'agree-score-hand'; player: number }
  
  // Actions that execute when consensus reached
  | { type: 'complete-trick' }
  | { type: 'score-hand' }
  | { type: 'new-game' }
  | { type: 'redeal' };
```

## The Single Function
```typescript
export function getPlayerView(
  state: GameState, 
  playerId: number
): PlayerView {
  // Build view from scratch - whitelist projection
  return {
    // Essential state
    playerId,
    phase: state.phase,
    dealer: state.dealer,
    currentPlayer: state.currentPlayer,
    trump: state.trump,
    currentSuit: state.currentSuit,
    bids: state.bids,
    currentTrick: state.currentTrick.plays,
    tricks: state.tricks,
    teamScores: state.teamScores,
    teamMarks: state.teamMarks,
    gameTarget: state.gameTarget,
    
    // Self data - only place with hands
    self: {
      id: playerId,
      hand: state.players[playerId].hand
    },
    
    // Others - type guarantees no hand field
    players: state.players.map(p => ({
      id: p.id,
      name: p.name,
      teamId: p.teamId,
      handCount: p.hand.length
    })),
    
    // Valid actions for this player
    validActions: getValidActionsForPlayer(state, playerId),
    
    // Consensus state from game state
    consensus: {
      completeTrick: state.consensus?.completeTrick || new Set(),
      scoreHand: state.consensus?.scoreHand || new Set()
    },
    
    // Recent events computed from action history
    recentEvents: computeRecentEvents(state.actionHistory || [])
  };
}

// Get valid actions for a specific player
function getValidActionsForPlayer(state: GameState, playerId: number): GameAction[] {
  const actions: GameAction[] = [];
  
  // Player-specific actions (bid, play, etc.)
  if (state.currentPlayer === playerId) {
    // Add bids, plays, trump selection based on phase
    // ... game logic ...
  }
  
  // Consensus actions - any player can agree
  if (state.phase === 'trick-complete' && 
      !state.consensus.completeTrick.has(playerId)) {
    actions.push({ type: 'agree-complete-trick', player: playerId });
  }
  
  if (state.phase === 'hand-complete' && 
      !state.consensus.scoreHand.has(playerId)) {
    actions.push({ type: 'agree-score-hand', player: playerId });
  }
  
  return actions;
}
```

## Pure Action Processing (No Special Cases)
```typescript
// Core state with consensus tracking
export interface GameState {
  // ... game fields ...
  consensus: {
    completeTrick: Set<number>;
    scoreHand: Set<number>;
  };
  actionHistory: GameAction[];  // Full action history for replay/events
}

// Single pure function handles ALL actions
export function executeAction(state: GameState, action: GameAction): GameState {
  // Always append action to history
  const stateWithHistory = {
    ...state,
    actionHistory: [...state.actionHistory, action]
  };
  
  switch (action.type) {
    // Consensus agreement actions
    case 'agree-complete-trick':
      return recordAgreement(stateWithHistory, action.player, 'completeTrick');
      
    case 'agree-score-hand':
      return recordAgreement(stateWithHistory, action.player, 'scoreHand');
      
    // Game flow actions (require consensus)
    case 'complete-trick':
      if (state.consensus.completeTrick.size !== 4) {
        return stateWithHistory; // Invalid - but still record attempt
      }
      return completeTrick(stateWithHistory);
      
    case 'score-hand':
      if (state.consensus.scoreHand.size !== 4) {
        return stateWithHistory; // Invalid - but still record attempt
      }
      return scoreHand(stateWithHistory);
      
    // Regular player actions
    case 'play-domino':
      return playDomino(stateWithHistory, action.player, action.domino);
      
    // ... other actions ...
    
    default:
      return stateWithHistory; // Unknown action still gets recorded
  }
}

// Game flow functions clear consensus internally
function completeTrick(state: GameState): GameState {
  // ... perform trick completion logic ...
  return {
    ...performTrickCompletion(state),
    consensus: {
      ...state.consensus,
      completeTrick: new Set()  // Clear consensus after completion
    }
  };
}

function scoreHand(state: GameState): GameState {
  // ... perform hand scoring logic ...
  return {
    ...performHandScoring(state),
    consensus: {
      ...state.consensus,
      scoreHand: new Set()  // Clear consensus after scoring
    }
  };
}

// Helper: Record consensus agreement (pure - no mutations)
function recordAgreement(
  state: GameState,
  player: number,
  type: 'completeTrick' | 'scoreHand'
): GameState {
  return {
    ...state,
    consensus: {
      ...state.consensus,
      [type]: new Set([...state.consensus[type], player])
    }
  };
}

// Compute events from recent action history (for UI/animations)
function computeRecentEvents(actions: GameAction[]): GameEvent[] {
  // Only keep last 10 for UI performance
  return actions.slice(-10).map(action => ({
    type: action.type,
    player: 'player' in action ? action.player : undefined,
    details: action
  }));
}
```

## The Orchestrator Pattern (Not Core Code)
```typescript
// Concept: Orchestrate humans and AIs emitting actions
// Human: clicks button → emits action
// AI: reads state → emits action  
// Core: receives action → updates state

// Example orchestrator (your implementation may vary)
function orchestrateGame(state: GameState, humanAction: GameAction): GameState {
  // 1. Apply human action
  let newState = executeAction(state, humanAction);
  
  // 2. Let AIs respond (they're just external agents)
  for (const ai of getAIs()) {
    const view = getPlayerView(newState, ai.id);
    const action = ai.chooseAction(view);
    if (action) {
      newState = executeAction(newState, action);
    }
  }
  
  // 3. Check if consensus triggered follow-up actions
  if (newState.consensus.completeTrick.size === 4) {
    newState = executeAction(newState, { type: 'complete-trick' });
    // Consensus cleared inside completeTrick() function
  }
  
  if (newState.consensus.scoreHand.size === 4) {
    newState = executeAction(newState, { type: 'score-hand' });
    // Consensus cleared inside scoreHand() function
  }
  
  return newState;
}
```

## Svelte Integration
```typescript
// One derived store for the UI
export const playerView = derived(
  [gameState, currentPlayerId],
  ([$state, $playerId]) => 
    getPlayerView($state, $playerId)
);
```

## Reproducibility (Core)
```typescript
// Minimal state package
export interface StatePackage {
  seed: string;
  actions: GameAction[];
}

// Reconstruct from seed + actions
export function reconstruct(pkg: StatePackage): GameState {
  let state = initGame(pkg.seed);
  for (const action of pkg.actions) {
    state = executeAction(state, action);
  }
  return state;
}
```

## Transport Helpers (Outside Core)
```typescript
import { encodeURLData, decodeURLData, compressActionId } from '@/game/core/url-compression';

// URL encoding using compression module
export function toURL(seed: string, actions: GameAction[]): string {
  const urlData = {
    v: 1,
    s: { s: hashSeed(seed) },  // Minimal state with seed
    a: actions.map(a => ({ 
      i: compressActionId(formatActionId(a)) 
    }))
  };
  return encodeURLData(urlData);
}

export function fromURL(encoded: string): StatePackage {
  const data = decodeURLData(encoded);
  return {
    seed: data.s.s.toString(),
    actions: data.a.map(a => parseActionId(a.i))
  };
}
```

## Debug Tools (Development Only)
```typescript
export const debugTools = {
  // Get all 4 perspectives at once
  getAllViews(state: GameState): PlayerView[] {
    return [0, 1, 2, 3].map(p => getPlayerView(state, p));
  },
  
  // Replay from URL (for bug reproduction)
  replayFromURL(url: string, stopAt?: number): PlayerView[][] {
    const { seed, actions } = fromURL(url);
    const sliced = stopAt ? actions.slice(0, stopAt) : actions;
    
    const views: PlayerView[][] = [];
    let state = initGame(seed);
    views.push(this.getAllViews(state));
    
    for (const action of sliced) {
      state = executeAction(state, action);
      views.push(this.getAllViews(state));
    }
    
    return views;
  }
};
```

## Tests
```typescript
test('type-safe privacy', () => {
  const state = initGame('test-seed');
  const view = getPlayerView(state, 0);
  
  // Self has hand
  expect(view.self.hand).toBeDefined();
  
  // Players array has no hand field (enforced by type)
  // This won't compile: view.players[0].hand
  expect((view.players[0] as any).hand).toBeUndefined();
});

test('consensus through pure actions', () => {
  let state = initGame('test');
  state.phase = 'trick-complete';
  
  // Players agree one by one (no special handling)
  state = executeAction(state, { type: 'agree-complete-trick', player: 0 });
  expect(state.consensus.completeTrick.size).toBe(1);
  
  state = executeAction(state, { type: 'agree-complete-trick', player: 1 });
  expect(state.consensus.completeTrick.size).toBe(2);
  
  state = executeAction(state, { type: 'agree-complete-trick', player: 2 });
  expect(state.consensus.completeTrick.size).toBe(3);
  
  state = executeAction(state, { type: 'agree-complete-trick', player: 3 });
  expect(state.consensus.completeTrick.size).toBe(4);
  
  // Now the actual action can execute
  state = executeAction(state, { type: 'complete-trick' });
  expect(state.phase).toBe('new-trick');
  expect(state.consensus.completeTrick.size).toBe(0); // Cleared
});

test('complete action stream', () => {
  // Everything is just actions
  const actions: GameAction[] = [
    { type: 'play-domino', player: 0, domino: [6, 6] },
    { type: 'play-domino', player: 1, domino: [6, 4] },
    { type: 'play-domino', player: 2, domino: [4, 4] },
    { type: 'play-domino', player: 3, domino: [4, 0] },
    { type: 'agree-complete-trick', player: 0 },
    { type: 'agree-complete-trick', player: 1 },
    { type: 'agree-complete-trick', player: 2 },
    { type: 'agree-complete-trick', player: 3 },
    { type: 'complete-trick' }
  ];
  
  // Pure replay
  let state = initGame('test');
  for (const action of actions) {
    state = executeAction(state, action);
  }
  
  expect(state.tricks.length).toBe(1);
});
```

## What We Achieved
- **Type-safe privacy**: `PublicPlayer` type cannot hold hands
- **Pure action-based core**: Everything is just actions, no special cases
- **No AI knowledge in core**: AIs are external agents that emit actions
- **First-class consensus**: Agreement actions are regular actions
- **Minimal core**: ~100 lines total

## Core Guarantees
1. **Type-enforced privacy**: `PublicPlayer` has no `hand` field
2. **Pure functions**: No side effects, no special cases
3. **Action-based**: Everything is an action in a stream
4. **Deterministic**: Same actions → same state
5. **No AI coupling**: Core doesn't know about AI vs human

## Implementation Checklist
- [ ] Core types: `PublicPlayer`, `SelfPlayer`, `PlayerView`
- [ ] Action types: All actions as first-class types
- [ ] Core function: `getPlayerView()` with action filtering
- [ ] Pure `executeAction()`: Single function for all actions
- [ ] Tests: Type safety, consensus flow, action streams

## Usage Examples
```typescript
// Production: Single player view
const view = getPlayerView(gameState, playerId);

// Development: Debug all perspectives
const allViews = debugTools.getAllViews(gameState);
console.log('Player 0 sees:', allViews[0]);
console.log('Player 0 hand:', allViews[0].self.hand);
console.log('Others see P0 as:', allViews[1].players[0]); // No hand field

// Bug reproduction from URL
const url = 'localhost:5173/?g=...';
const replay = debugTools.replayFromURL(url, 87);
console.log('At action 87, player 2 saw:', replay[87][2]);

// UI rendering (Svelte) - All actions equal
{#each $playerView.validActions as action}
  <button 
    on:click={() => handleAction(action)}
    class:agreed={isAgreed(action, $playerView)}
  >
    {formatActionLabel(action)}
    {#if action.type.includes('agree')}
      ({getConsensusCount(action, $playerView)}/4)
    {/if}
  </button>
{/each}
```

## Pure Action Flow

### Everything is Just Actions
```typescript
// Example: Complete trick with consensus
{ type: 'play-domino', player: 0, domino: [6, 6] }
{ type: 'play-domino', player: 1, domino: [6, 4] }
{ type: 'play-domino', player: 2, domino: [4, 4] }
{ type: 'play-domino', player: 3, domino: [4, 0] }
{ type: 'agree-complete-trick', player: 0 }  // Human clicks
{ type: 'agree-complete-trick', player: 1 }  // AI emits
{ type: 'agree-complete-trick', player: 2 }  // AI emits
{ type: 'agree-complete-trick', player: 3 }  // AI emits
{ type: 'complete-trick' }                   // Consensus triggers
```

### Note on Errors
Invalid actions are no-ops (return unchanged state). This keeps the core simple and makes consensus impossible to get wrong. No distributed error handling complexity.

## Summary

The entire system is ~100 lines of **truly pure** functions:
1. **Types** prevent hand leakage at compile time
2. **getPlayerView()** projects state for each player  
3. **executeAction()** always appends to history, processes all actions uniformly
4. **State transitions** handle their own cleanup (consensus clearing)
5. **URL compression** via `@/game/core/url-compression` for efficient sharing

Everything else (orchestration, AI, UI) lives outside the pure core. Invalid actions still get recorded but don't change game state. All state transitions are immutable - no mutations anywhere.