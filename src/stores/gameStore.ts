import { writable, derived, get } from 'svelte/store';
import type { GameState, StateTransition } from '../game/types';
import { createInitialState, getNextStates } from '../game';

// Helper function to update URL with current state and debug snapshot
function updateURLWithState(state: GameState) {
  if (typeof window !== 'undefined') {
    const currentSnapshot = get(debugSnapshot);
    
    if (currentSnapshot && currentSnapshot.actions && currentSnapshot.actions.length > 0) {
      // Use snapshot + actions format
      const snapshotData = {
        baseState: currentSnapshot.baseState,
        actions: currentSnapshot.actions.map((action: StateTransition) => ({
          id: action.id,
          label: action.label
        })),
        reason: currentSnapshot.snapshotReason
      };
      const snapshotParam = encodeURIComponent(JSON.stringify(snapshotData));
      const newURL = `${window.location.pathname}?snapshot=${snapshotParam}`;
      window.history.replaceState(null, '', newURL);
    } else {
      // Fallback to single state format
      const stateParam = encodeURIComponent(JSON.stringify(state));
      const newURL = `${window.location.pathname}?state=${stateParam}`;
      window.history.replaceState(null, '', newURL);
    }
  }
}

// Core game state store
export const gameState = writable<GameState>(createInitialState());

// History store for undo functionality
export const gameHistory = writable<GameState[]>([]);

// Debug snapshot - tracks all actions from initial state
interface DebugSnapshot {
  baseState: GameState; // Always the initial state
  actions: StateTransition[]; // All actions since initial state
  snapshotReason?: string; // Optional reason for snapshot
}

export const debugSnapshot = writable<DebugSnapshot | null>(null);

// Track all actions from the beginning
let allActions: StateTransition[] = [];
let initialGameState: GameState | null = null;

// Available actions store
export const availableActions = derived(
  gameState,
  ($gameState) => getNextStates($gameState)
);

// Current player store
export const currentPlayer = derived(
  gameState,
  ($gameState) => $gameState.players[$gameState.currentPlayer]
);

// Game phase store
export const gamePhase = derived(
  gameState,
  ($gameState) => $gameState.phase
);

// Bidding information
export const biddingInfo = derived(
  gameState,
  ($gameState) => ({
    currentBid: $gameState.currentBid,
    bids: $gameState.bids,
    winningBidder: $gameState.winningBidder
  })
);

// Team scores and marks
export const teamInfo = derived(
  gameState,
  ($gameState) => ({
    scores: $gameState.teamScores,
    marks: $gameState.teamMarks,
    target: $gameState.gameTarget
  })
);

// Current trick information
export const trickInfo = derived(
  gameState,
  ($gameState) => ({
    currentTrick: $gameState.currentTrick,
    completedTricks: $gameState.tricks,
    trump: $gameState.trump
  })
);

// Game actions
export const gameActions = {
  executeAction: (transition: StateTransition) => {
    // Add current state to history before executing action
    gameState.update(currentState => {
      gameHistory.update(history => [...history, currentState]);
      const newState = transition.newState;
      
      // Set initial state on first action
      if (initialGameState === null) {
        initialGameState = JSON.parse(JSON.stringify(createInitialState()));
      }
      
      // Add action to all actions list
      allActions.push(transition);
      
      // Update debug snapshot with all actions from initial state
      if (initialGameState) {
        debugSnapshot.set({
          baseState: initialGameState,
          actions: [...allActions] // Copy array
        });
      }
      
      updateURLWithState(newState);
      return newState;
    });
  },
  
  resetGame: () => {
    const initialState = createInitialState();
    gameState.set(initialState);
    gameHistory.set([]);
    debugSnapshot.set(null);
    allActions = [];
    initialGameState = null;
    updateURLWithState(initialState);
  },
  
  loadState: (state: GameState) => {
    gameState.set(state);
    gameHistory.set([]);
    debugSnapshot.set(null);
    allActions = [];
    initialGameState = null;
    updateURLWithState(state);
  },

  loadStateWithActionReplay: (baseState: GameState, actions: Array<{id: string, label: string}>) => {
    // Set up initial state and tracking
    gameState.set(baseState);
    const history: GameState[] = [];
    allActions = [];
    initialGameState = JSON.parse(JSON.stringify(baseState));
    
    let currentState = baseState;
    
    // Replay each action in sequence, building history as we go
    for (const actionData of actions) {
      const availableTransitions = getNextStates(currentState);
      const matchingTransition = availableTransitions.find(t => t.id === actionData.id);
      
      if (matchingTransition) {
        // Add current state to history before advancing (same as executeAction)
        history.push(currentState);
        
        // Add to action history
        allActions.push(matchingTransition);
        
        // Update current state
        currentState = matchingTransition.newState;
      } else {
        // Invalid action - set state and debug snapshot before throwing
        gameState.set(currentState);
        gameHistory.set(history);
        
        if (allActions.length > 0 && initialGameState !== null) {
          debugSnapshot.set({
            baseState: initialGameState,
            actions: [...allActions]
          });
        }
        
        updateURLWithState(currentState);
        
        // Now throw error to fail tests and expose broken logic
        const availableIds = availableTransitions.map(t => t.id).join(', ');
        throw new Error(`Invalid action replay: Action "${actionData.id}" is not available in current state. Available actions: [${availableIds}]`);
      }
    }
    
    // Set final state, history, and debug snapshot for successful replay
    gameState.set(currentState);
    gameHistory.set(history);
    
    if (allActions.length > 0 && initialGameState !== null) {
      debugSnapshot.set({
        baseState: initialGameState,
        actions: [...allActions]
      });
    }
    
    updateURLWithState(currentState);
  },
  
  undo: () => {
    gameHistory.update(history => {
      if (history.length > 0) {
        const previousState = history[history.length - 1];
        gameState.set(previousState);
        
        // Remove last action from all actions
        if (allActions.length > 0) {
          allActions.pop();
        }
        
        // Update debug snapshot
        if (allActions.length > 0 && initialGameState) {
          debugSnapshot.set({
            baseState: initialGameState,
            actions: [...allActions]
          });
        } else {
          debugSnapshot.set(null);
        }
        
        updateURLWithState(previousState);
        return history.slice(0, -1);
      }
      return history;
    });
  },
  
  generateBugReport: (): string => {
    const currentState = get(gameState);
    const currentSnapshot = get(debugSnapshot);
    
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');

    if (currentSnapshot === null) {
      // Fallback to current state if no snapshot available
      const stateJson = JSON.stringify(currentState, null, 2);
      return `import { test, expect } from '@playwright/test';
import { playwrightHelper } from './helpers/playwrightHelper';

test('Bug report - ${timestamp} (no actions)', async ({ page }) => {
  await page.goto('/');
  
  // Current game state (no action history available)
  const testState = ${stateJson};
  
  await playwrightHelper.loadState(page, testState);
  await expect(page.locator('[data-testid="game-phase"]')).toContainText('${currentState.phase}');
});`;
    }

    const baseStateJson = JSON.stringify(currentSnapshot.baseState, null, 2);
    const actionsJson = JSON.stringify(
      currentSnapshot.actions ? currentSnapshot.actions.map((action: StateTransition) => ({
        id: action.id,
        label: action.label
      })) : [], 
      null, 
      2
    );

    const testTemplate = `import { test, expect } from '@playwright/test';
import { playwrightHelper } from './helpers/playwrightHelper';
import { getNextStates } from '../../../game';

test('Bug report - ${timestamp}', async ({ page }) => {
  await page.goto('/');
  
  // Initial game state 
  const baseState = ${baseStateJson};
  
  // All actions from initial state to current state
  const actionSequence = ${actionsJson};
  
  await playwrightHelper.loadState(page, baseState);
  
  // Replay each action and validate state transitions
  for (let i = 0; i < actionSequence.length; i++) {
    const action = actionSequence[i];
    
    // Get available actions from current state
    const availableActions = await playwrightHelper.getAvailableActions(page);
    
    // Verify the action we're about to perform is valid
    const validAction = availableActions.find(a => a.id === action.id);
    if (!validAction) {
      throw new Error(\`Invalid action at step \${i + 1}: \${action.id} (\${action.label}) is not available. Available: \${availableActions.map(a => a.id).join(', ')}\`);
    }
    
    // Execute the action
    await playwrightHelper.clickAction(page, action.id);
    
    // Optional: Add specific assertions for this step
    console.log(\`Step \${i + 1}: Executed \${action.label}\`);
  }
  
  // Final state should match current state
  await expect(page.locator('[data-testid="game-phase"]')).toContainText('${currentState.phase}');
  
  // Add your specific bug assertions here
  // Example: Check if trick winner is correct
  // const trickWinner = await playwrightHelper.getTrickWinner(page);
  // expect(trickWinner).toBe(expectedWinner);
});`;

    return testTemplate;
  }
};