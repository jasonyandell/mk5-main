# Player Perspective Layer — R9 Final Unified

## Core Principle
Type-safe whitelist projection. Security by construction, not runtime checks. Every line of code is a liability.

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
}
```

## The Single Function
```typescript
export function getPlayerView(
  state: GameState, 
  playerId: number,
  isHost = false
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
    
    // Valid actions filtered to this player
    validActions: getValidActions(state).filter(a => {
      // Player owns this action
      if ('player' in a && a.player !== playerId) return false;
      // Neutral actions only for host
      if (!isHost && isNeutralAction(a)) return false;
      return true;
    })
  };
}

// Helper - neutral detection in one place
function isNeutralAction(action: GameAction): boolean {
  return ['complete-trick', 'score-hand', 'redeal', 'new-game'].includes(action.type);
}
```

## Svelte Integration
```typescript
// One derived store for the UI
export const playerView = derived(
  [gameState, currentPlayerId, isHost],
  ([$state, $playerId, $isHost]) => 
    getPlayerView($state, $playerId, $isHost)
);

// UI derives convenience flags
// const isDealer = $playerView.dealer === $playerView.playerId;
// const isMyTurn = $playerView.currentPlayer === $playerView.playerId;
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
// URL encoding (no private data)
export function toURL(pkg: StatePackage): string {
  return btoa(JSON.stringify(pkg));
}

export function fromURL(url: string): StatePackage {
  return JSON.parse(atob(url));
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
  },
  
  // Validate no information leakage
  validateSecurity(state: GameState): string[] {
    const errors: string[] = [];
    for (let p = 0; p < 4; p++) {
      const view = getPlayerView(state, p);
      
      // Type system prevents hand leakage, but validate runtime
      if ((view.players as any).some((pl: any) => pl.hand !== undefined)) {
        errors.push(`Player ${p} view has hand data in players array!`);
      }
      
      // Check action ownership
      view.validActions.forEach(action => {
        if ('player' in action && action.player !== p) {
          errors.push(`Player ${p} has action for player ${action.player}`);
        }
      });
    }
    return errors;
  }
};
```

## Security Test
```typescript
test('type-safe privacy + neutral gating', () => {
  const state = initGame('test-seed');
  
  for (let i = 0; i < 4; i++) {
    const view = getPlayerView(state, i);
    
    // Self has hand
    expect(view.self.id).toBe(i);
    expect(view.self.hand).toEqual(state.players[i].hand);
    
    // Players array has no hand field (enforced by type)
    view.players.forEach(p => {
      // This would be a TypeScript compile error:
      // expect(p.hand).toBeUndefined();
      // The field doesn't exist on the type
      expect((p as any).hand).toBeUndefined();
    });
    
    // Can only take own actions
    view.validActions.forEach(a => {
      if ('player' in a) expect(a.player).toBe(i);
    });
    
    // Non-host has no neutral actions
    expect(view.validActions.some(isNeutralAction)).toBe(false);
    
    // Host gets neutral actions (when appropriate)
    const hostView = getPlayerView(state, i, true);
    if (state.phase === 'trick-complete' || state.phase === 'hand-complete') {
      expect(hostView.validActions.some(isNeutralAction)).toBe(true);
    }
  }
});
```

## What We Achieved
- **Type-safe privacy**: `PublicPlayer` type cannot hold hands
- **Explicit self/others split**: Clear in types, not just runtime
- **Minimal core**: ~70 lines total
- **Zero runtime privacy checks needed**: Types enforce it
- **Clean separation**: Core vs transport vs debug tools

## Security Guarantees
1. **Type-enforced privacy**: `PublicPlayer` has no `hand` field
2. **Whitelist projection**: Build views from scratch
3. **Deterministic**: Same state → same view
4. **Stateless**: No caches or hidden state
5. **Host gating**: Neutral actions only when `isHost === true`
6. **URL safety**: Private data never in URLs

## Implementation Checklist
- [ ] Core types: `PublicPlayer`, `SelfPlayer`, `PlayerView`
- [ ] Core function: `getPlayerView()` with host gating
- [ ] Helper: `isNeutralAction()`
- [ ] Reconstruct: `reconstruct()` from seed + actions
- [ ] Transport: `toURL()`, `fromURL()` (separate module)
- [ ] Debug: `getAllViews()`, `replayFromURL()` (dev only)
- [ ] Tests: Type safety, neutral gating, determinism

## Usage Examples
```typescript
// Production: Single player view
const view = getPlayerView(gameState, playerId, isHost);

// Development: Debug all perspectives
const allViews = debugTools.getAllViews(gameState);
console.log('Player 0 sees:', allViews[0]);
console.log('Player 0 hand:', allViews[0].self.hand);
console.log('Others see P0 as:', allViews[1].players[0]); // No hand field

// Bug reproduction from URL
const url = 'localhost:5173/?g=...';
const replay = debugTools.replayFromURL(url, 87);
console.log('At action 87, player 2 saw:', replay[87][2]);

// UI rendering (Svelte)
{#each $playerView.validActions as action}
  <button on:click={() => executeAction(action)}>
    {action.type}
  </button>
{/each}
```

## Migration from R7/R8
- Replace `players: Array<{hand?: Domino[]}>` with `self` + `players` split
- Change `p.hand` checks to use `view.self.hand` for self
- Update UI to use `self.hand` instead of finding self in players array
- Type system will catch any accidental hand access attempts

## That's It
Type-safe, minimal, leak-proof by construction. The type system is our security guard.