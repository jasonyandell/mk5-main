# Unified State Architecture - Game, Coordination, and View State

## Current Issues

1. **Consensus embedded in GameState** - Mixing game logic with coordination
2. **UI state scattered in components** - Each component manages its own view state locally  
3. **No persistence of UI preferences** - Panel states, tab selections lost on refresh
4. **No unified state transitions** - Different update patterns for different state types

## Proposed Architecture: Three-Layer State System

```typescript
interface AppState {
  game: GameState;           // Pure game mechanics
  coordination: CoordinationState;  // Proposals & consensus
  view: ViewState;            // UI presentation state
}
```

### Layer 1: Pure Game State (existing, cleaned up)

The current GameState in `src/game/types.ts` mixes game mechanics with consensus:

```diff
interface GameState {
  phase: GamePhase;
  players: Player[];
  currentPlayer: number;
  // ... other game fields ...
- consensus: {
-   completeTrick: Set<number>;
-   scoreHand: Set<number>;
- };
  actionHistory: GameAction[];
  aiSchedule: Record<number, AIScheduleEntry>;
  currentTick: number;
}
```

### Layer 2: Coordination State (new)

Extract consensus logic into a separate coordination layer:

```typescript
interface CoordinationState {
  proposals: Map<string, Proposal>;
  playerReadiness: Map<number, PlayerReadiness>;
}

interface Proposal {
  id: string;
  type: ProposalType;
  initiator: number;
  participants: Set<number>;
  votes: Map<number, boolean>;
  deadline?: number;  // Game tick for timeout
  metadata?: Record<string, unknown>;
}

type ProposalType = 
  | 'complete-trick'   // Currently in consensus.completeTrick
  | 'score-hand'       // Currently in consensus.scoreHand
  | 'new-game'         // Future: start new game
  | 'accept-renege';   // Future: handle rule violations
```

### Layer 3: View State (new)

Currently scattered across multiple components:

```svelte
<!-- PlayingArea.svelte -->
let showTrickHistory = false;
let drawerState = $state<'collapsed' | 'expanded'>('collapsed');

<!-- SettingsPanel.svelte -->
let activeTab = $state('theme');

<!-- Header.svelte -->
let menuOpen = $state(false);
```

Should be centralized:

```typescript
interface ViewState {
  panels: {
    trickHistory: 'collapsed' | 'expanded';
    settings: 'closed' | 'open';
    actionPanel: 'visible' | 'hidden';
    debugPanel?: 'closed' | 'open';
  };
  
  settings: {
    activeTab: 'theme' | 'game' | 'debug';
  };
  
  animations: {
    playedDominoIds: Set<string>;
    lastActionTime: number;
  };
  
  preferences: {
    theme: 'light' | 'dark' | 'auto';
    soundEnabled: boolean;
  };
}
```

## Implementation Plan

### 1. Create Unified State Store

Instead of the current `gameStore.ts` with just game state:

```diff
// src/stores/unifiedStore.ts
+ export const appState = writable<AppState>({
+   game: initialGameState,
+   coordination: initialCoordinationState,
+   view: initialViewState
+ });

// Derived stores for backward compatibility
- export const gameState = writable<GameState>(firstInitialState);
+ export const gameState = derived(appState, $app => $app.game);
+ export const viewState = derived(appState, $app => $app.view);
```

### 2. Pure Update Functions

Following the existing pattern from `src/game/core/actions.ts`:

```typescript
// Example: Toggle panel in view state
function togglePanel(state: ViewState, panel: keyof ViewState['panels']): ViewState {
  const current = state.panels[panel];
  const next = current === 'expanded' ? 'collapsed' : 'expanded';
  
  return {
    ...state,
    panels: {
      ...state.panels,
      [panel]: next
    }
  };
}

// Example: Add vote to proposal
function addVote(state: CoordinationState, proposalId: string, player: number, vote: boolean): CoordinationState {
  const proposal = state.proposals.get(proposalId);
  if (!proposal) return state;
  
  const newProposal = {
    ...proposal,
    votes: new Map(proposal.votes).set(player, vote)
  };
  
  return {
    ...state,
    proposals: new Map(state.proposals).set(proposalId, newProposal)
  };
}
```

### 3. Unified Action System

Current action handling in `gameStore.ts:executeAction`:

```typescript
// Current: Only game actions
executeAction: (transition: StateTransition) => {
  const actions = get(actionHistory);
  let finalActions = [...actions, transition];
  actionHistory.set(finalActions);
  gameState.set(transition.newState);
  // ...
}
```

Proposed unified system:

```typescript
type AppAction = 
  | { type: 'game', action: GameAction }
  | { type: 'coordination', action: CoordinationAction }
  | { type: 'view', action: ViewAction }
  | { type: 'batch', actions: AppAction[] };

// Single dispatch with proper typing
function dispatch<T extends AppAction>(action: T): void {
  appState.update(state => {
    switch (action.type) {
      case 'game':
        // Reuse existing executeGameAction
        return { ...state, game: executeGameAction(state.game, action.action) };
      case 'coordination':
        return { ...state, coordination: updateCoordination(state.coordination, action.action) };
      case 'view':
        return { ...state, view: updateView(state.view, action.action) };
      case 'batch':
        // Sequential application of multiple actions
        return action.actions.reduce((acc, a) => {
          // Recursively apply each action
          return applyAction(acc, a);
        }, state);
      default:
        return state;
    }
  });
}
```

### 4. View State Actions

Similar to existing GameAction types:

```typescript
type ViewAction =
  | { type: 'toggle-panel'; panel: keyof ViewState['panels'] }
  | { type: 'set-tab'; tab: string }
  | { type: 'set-preference'; key: string; value: unknown }
  | { type: 'add-animation'; animationId: string }
  | { type: 'restore-view-state'; state: ViewState };
```

### 5. Persistence Layer

Extend existing URL handling from `gameStore.ts`:

```typescript
// Current: Only game state in URL
function updateURLWithState(initialState: GameState, actions: StateTransition[]) {
  const newURL = window.location.pathname + encodeGameUrl(
    initialState.shuffleSeed,
    actions.map(a => a.id),
    // ...
  );
}

// Proposed: Split persistence by concern
function persistState(state: AppState): void {
  // Game state -> URL (existing)
  updateURLWithGameState(state.game);
  
  // View state -> localStorage
  localStorage.setItem('texas42-view', JSON.stringify({
    panels: state.view.panels,
    preferences: state.view.preferences
  }));
  
  // Coordination state -> sessionStorage (temporary)
  sessionStorage.setItem('texas42-coordination', JSON.stringify(state.coordination));
}
```

### 6. Component Integration

Transform from local to store-based state:

```diff
// TrickHistoryDrawer.svelte
<script>
- let drawerState = $state<'collapsed' | 'expanded'>('collapsed');
+ import { viewState, dispatch } from '../stores/unifiedStore';

  function toggleDrawer() {
-   drawerState = drawerState === 'collapsed' ? 'expanded' : 'collapsed';
+   dispatch({ 
+     type: 'view', 
+     action: { type: 'toggle-panel', panel: 'trickHistory' }
+   });
  }
  
+ $: drawerState = $viewState.panels.trickHistory;
</script>
```

```diff
// SettingsPanel.svelte
<script>
- let activeTab = $state('theme');
+ import { viewState, dispatch } from '../stores/unifiedStore';
  
  // Keep local state for debug-only features
  let showDiff = $state(false);
  let showTreeView = $state(true);

  function changeTab(tab: string) {
-   activeTab = tab;
+   dispatch({
+     type: 'view',
+     action: { type: 'set-tab', tab }
+   });
  }
  
+ $: activeTab = $viewState.settings.activeTab;
</script>
```

## Benefits

1. **Pure Functional**: All state transitions are pure functions
2. **Time Travel**: Can replay any sequence of actions
3. **Testable**: Each layer can be tested independently
4. **Persistent UI**: View preferences survive refreshes
5. **Unified Pattern**: Same update pattern for all state types
6. **Scalable**: Easy to add new proposal types or view states
7. **Debuggable**: Single state tree, clear action log

## Migration Strategy

1. **Phase 1**: Implement ViewState store alongside existing code
2. **Phase 2**: Move component state to ViewState one by one
3. **Phase 3**: Extract consensus to CoordinationState
4. **Phase 4**: Unify into single AppState
5. **Phase 5**: Add persistence and time-travel debugging

## Testing Approach

Following the existing test patterns from the codebase:

```typescript
// Pure function test (similar to gameEngine.test.ts)
describe('View State Updates', () => {
  test('toggle panel', () => {
    const initial: ViewState = {
      panels: { 
        trickHistory: 'collapsed',
        settings: 'closed',
        actionPanel: 'visible'
      },
      // ... other fields
    };
    
    const updated = togglePanel(initial, 'trickHistory');
    expect(updated.panels.trickHistory).toBe('expanded');
    // Original state unchanged (immutability)
    expect(initial.panels.trickHistory).toBe('collapsed');
  });
});

// Coordination test
describe('Consensus Mechanism', () => {
  test('proposal completes when all vote', () => {
    const proposal: Proposal = {
      id: 'complete-trick-1',
      type: 'complete-trick',
      initiator: 0,
      participants: new Set([0, 1, 2, 3]),
      votes: new Map([[0, true], [1, true], [2, true]]),
      deadline: 100
    };
    
    // Add final vote
    const updated = addVote(coordinationState, proposal.id, 3, true);
    
    // Check if consensus reached
    expect(hasConsensus(updated, proposal.id)).toBe(true);
  });
});
```

## Key Design Principles

### Separation of Concerns

- **GameState**: Only contains game mechanics, rules, and player data
- **CoordinationState**: Handles all multi-player agreements and proposals
- **ViewState**: Manages UI presentation and user preferences

### Immutability

Following the pattern from `src/game/core/actions.ts`:

```typescript
// ❌ Never mutate
state.panels.trickHistory = 'expanded';

// ✅ Always return new state (from executePlay function)
return {
  ...state,
  players: newPlayers,
  currentTrick: newCurrentTrick,
  currentSuit: newCurrentSuit
};
```

### Event Sourcing Pattern

Every state change is an action:
- Actions are serializable
- State can be reconstructed from action history
- Enables undo/redo and time-travel debugging

### Pure Functions

All update functions are pure:
- Same input always produces same output
- No side effects
- Highly testable

## Example: Adding a New Consensus Type

To add a "request undo" consensus:

```diff
// 1. Extend ProposalType
type ProposalType = 
  | 'complete-trick'
  | 'score-hand' 
  | 'new-game'
  | 'accept-renege'
+ | 'request-undo';

// 2. Handle proposal creation
function proposeUndo(state: CoordinationState, player: number): CoordinationState {
  const proposal: Proposal = {
    id: `undo-${Date.now()}`,
    type: 'request-undo',
    initiator: player,
    participants: new Set([0, 1, 2, 3]),
    votes: new Map([[player, true]]), // Initiator auto-votes yes
    deadline: state.currentTick + 300  // 5 second timeout
  };
  
  return {
    ...state,
    proposals: new Map(state.proposals).set(proposal.id, proposal)
  };
}

// 3. Check consensus and trigger action
function checkProposalConsensus(state: AppState, proposalId: string): AppState {
  const proposal = state.coordination.proposals.get(proposalId);
  if (!proposal || proposal.votes.size < proposal.participants.size) {
    return state;
  }
  
  // All voted - check if approved
  const approved = Array.from(proposal.votes.values()).every(v => v);
  if (approved && proposal.type === 'request-undo') {
    // Trigger undo and clear proposal
    return {
      ...state,
      game: undoLastAction(state.game),
      coordination: removeProposal(state.coordination, proposalId)
    };
  }
  
  return state;
}
```

## Future Enhancements

1. **Network Sync**: CoordinationState naturally supports multiplayer
2. **Replay System**: Complete state reconstruction from actions
3. **Analytics**: Track UI patterns and player behavior
4. **A/B Testing**: Different view states for different users
5. **Spectator Mode**: Read-only view state for observers

This architecture provides complete separation of concerns while maintaining pure functional state management throughout the application.