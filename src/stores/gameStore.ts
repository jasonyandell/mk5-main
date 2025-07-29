import { writable, derived, get } from 'svelte/store';
import type { GameState, StateTransition } from '../game/types';
import { createInitialState, getNextStates } from '../game';
import { 
  compressGameState, 
  expandMinimalState, 
  compressActionId, 
  decompressActionId,
  encodeURLData,
  decodeURLData,
  type URLData 
} from '../game/core/url-compression';

// Helper to deep compare two objects
function deepCompare(obj1: unknown, obj2: unknown, path: string = ''): string[] {
  const differences: string[] = [];
  
  if (obj1 === obj2) return differences;
  
  if (obj1 === null || obj2 === null || obj1 === undefined || obj2 === undefined) {
    differences.push(`${path}: ${JSON.stringify(obj1)} !== ${JSON.stringify(obj2)}`);
    return differences;
  }
  
  if (typeof obj1 !== typeof obj2) {
    differences.push(`${path}: type mismatch - ${typeof obj1} !== ${typeof obj2}`);
    return differences;
  }
  
  if (typeof obj1 !== 'object') {
    if (obj1 !== obj2) {
      differences.push(`${path}: ${JSON.stringify(obj1)} !== ${JSON.stringify(obj2)}`);
    }
    return differences;
  }
  
  if (Array.isArray(obj1) !== Array.isArray(obj2)) {
    differences.push(`${path}: array mismatch - one is array, other is not`);
    return differences;
  }
  
  if (Array.isArray(obj1)) {
    if (Array.isArray(obj2)) {
      if (obj1.length !== obj2.length) {
        differences.push(`${path}: array length mismatch - ${obj1.length} !== ${obj2.length}`);
      }
      const maxLen = Math.max(obj1.length, obj2.length);
      for (let i = 0; i < maxLen; i++) {
        differences.push(...deepCompare(obj1[i], obj2[i], `${path}[${i}]`));
      }
    }
  } else {
    const keys1 = Object.keys(obj1 as Record<string, unknown>).sort();
    const keys2 = Object.keys(obj2 as Record<string, unknown>).sort();
    
    // Check for missing/extra keys
    const allKeys = new Set([...keys1, ...keys2]);
    for (const key of allKeys) {
      if (!keys1.includes(key)) {
        differences.push(`${path}.${key}: missing in first object`);
      } else if (!keys2.includes(key)) {
        differences.push(`${path}.${key}: missing in second object`);
      } else {
        differences.push(...deepCompare((obj1 as Record<string, unknown>)[key], (obj2 as Record<string, unknown>)[key], path ? `${path}.${key}` : key));
      }
    }
  }
  
  return differences;
}

// Helper function to update URL with initial state and actions
function updateURLWithState(initialState: GameState, actions: StateTransition[]) {
  if (typeof window !== 'undefined') {
    // If no actions, clear the URL
    if (actions.length === 0) {
      window.history.replaceState(null, '', window.location.pathname);
      return;
    }
    
    // Use compressed format
    const urlData: URLData = {
      v: 1,
      s: compressGameState(initialState),
      a: actions.map(a => ({ i: compressActionId(a.id) }))
    };
    
    const encoded = encodeURLData(urlData);
    const newURL = `${window.location.pathname}?d=${encoded}`;
    window.history.replaceState(null, '', newURL);
  }
}

// Create the initial state once
const firstInitialState = createInitialState();

// Core game state store
export const gameState = writable<GameState>(firstInitialState);

// Store the initial state (snapshot) - deep clone to prevent mutations
export const initialState = writable<GameState>(JSON.parse(JSON.stringify(firstInitialState)));

// Store all actions taken from initial state
export const actionHistory = writable<StateTransition[]>([]);

// Store for validation errors
export const stateValidationError = writable<string | null>(null);

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

// Recompute state from initial + actions and validate
function validateState() {
  const initial = get(initialState);
  const actions = get(actionHistory);
  const currentState = get(gameState);
  
  // Recompute state from scratch
  let computedState = initial;
  
  for (const action of actions) {
    const availableTransitions = getNextStates(computedState);
    const matchingTransition = availableTransitions.find(t => t.id === action.id);
    
    if (!matchingTransition) {
      stateValidationError.set(
        `ERROR: Invalid action sequence at "${action.label}" (${action.id})\n` +
        `Available actions were: ${availableTransitions.map(t => t.id).join(', ')}\n` +
        `This indicates a bug in the game logic.`
      );
      return;
    }
    
    computedState = matchingTransition.newState;
  }
  
  // Deep compare computed vs actual state
  const differences = deepCompare(computedState, currentState);
  
  if (differences.length > 0) {
    const errorMessage = 
      `ERROR: State mismatch detected!\n` +
      `After ${actions.length} actions, the computed state differs from actual state:\n\n` +
      differences.join('\n') +
      `\n\nThis indicates a bug in the state management.`;
    stateValidationError.set(errorMessage);
  } else {
    stateValidationError.set(null);
  }
}

// Game actions
export const gameActions = {
  executeAction: (transition: StateTransition) => {
    const actions = get(actionHistory);
    
    // Add action to history
    actionHistory.set([...actions, transition]);
    
    // Update to new state
    gameState.set(transition.newState);
    
    // Validate state matches computed state
    validateState();
    
    // Update URL with initial state and actions
    updateURLWithState(get(initialState), [...actions, transition]);
  },
  
  resetGame: () => {
    const newInitialState = createInitialState();
    // Deep clone to prevent mutations
    initialState.set(JSON.parse(JSON.stringify(newInitialState)));
    gameState.set(newInitialState);
    actionHistory.set([]);
    stateValidationError.set(null);
    updateURLWithState(newInitialState, []);
  },
  
  loadState: (state: GameState) => {
    // When loading a state directly, make it the new initial state
    // Deep clone to prevent mutations
    initialState.set(JSON.parse(JSON.stringify(state)));
    gameState.set(state);
    actionHistory.set([]);
    stateValidationError.set(null);
    updateURLWithState(state, []);
  },
  
  loadFromURL: () => {
    if (typeof window !== 'undefined') {
      /* eslint-disable-next-line no-undef */
      const urlParams = new URLSearchParams(window.location.search);
      
      // Try new compressed format first
      const compressedParam = urlParams.get('d');
      if (compressedParam) {
        try {
          const urlData = decodeURLData(compressedParam);
          
          if (urlData.v === 1) {
            // Expand minimal state
            const expandedInitial = expandMinimalState(urlData.s);
            initialState.set(JSON.parse(JSON.stringify(expandedInitial)));
            
            let currentState = expandedInitial;
            const validActions: StateTransition[] = [];
            
            // Replay compressed actions
            for (const compressedAction of urlData.a) {
              const actionId = decompressActionId(compressedAction.i);
              const availableTransitions = getNextStates(currentState);
              const matchingTransition = availableTransitions.find(t => t.id === actionId);
              
              if (matchingTransition) {
                validActions.push(matchingTransition);
                currentState = matchingTransition.newState;
              } else {
                console.error(`Invalid action in URL: ${actionId}`);
                break;
              }
            }
            
            gameState.set(currentState);
            actionHistory.set(validActions);
            validateState();
            return;
          }
        } catch (e) {
          console.error('Failed to load compressed URL:', e);
        }
      }
      
      // Fallback to old format
      const dataParam = urlParams.get('data');
      if (dataParam) {
        try {
          const urlData = JSON.parse(decodeURIComponent(dataParam));
          
          if (urlData.initial && urlData.actions) {
            // Event sourcing: load initial state and replay actions
            // Deep clone to prevent mutations
            initialState.set(JSON.parse(JSON.stringify(urlData.initial)));
            
            let currentState = urlData.initial;
            const validActions: StateTransition[] = [];
            
            for (const actionData of urlData.actions) {
              const availableTransitions = getNextStates(currentState);
              const matchingTransition = availableTransitions.find(t => t.id === actionData.id);
              
              if (matchingTransition) {
                validActions.push(matchingTransition);
                currentState = matchingTransition.newState;
              } else {
                console.error(`Invalid action in URL: ${actionData.id}`);
                break;
              }
            }
            
            gameState.set(currentState);
            actionHistory.set(validActions);
            validateState();
          } else if (urlData.current) {
            // Backward compatibility: just load the current state
            gameActions.loadState(urlData.current);
          }
        } catch (e) {
          console.error('Failed to load from URL:', e);
        }
      }
    }
  },
  
  undo: () => {
    const actions = get(actionHistory);
    if (actions.length > 0) {
      // Remove last action
      const newActions = actions.slice(0, -1);
      actionHistory.set(newActions);
      
      // Recompute state from initial + remaining actions
      let currentState = get(initialState);
      
      for (const action of newActions) {
        const availableTransitions = getNextStates(currentState);
        const matchingTransition = availableTransitions.find(t => t.id === action.id);
        
        if (matchingTransition) {
          currentState = matchingTransition.newState;
        }
      }
      
      gameState.set(currentState);
      validateState();
      updateURLWithState(get(initialState), newActions);
    }
  },
  
  generateBugReport: (): string => {
    const initial = get(initialState);
    const actions = get(actionHistory);
    const currentState = get(gameState);
    const validationError = get(stateValidationError);
    
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    
    const initialJson = JSON.stringify(initial, null, 2);
    const actionsJson = JSON.stringify(
      actions.map(a => ({ id: a.id, label: a.label })), 
      null, 
      2
    );
    
    let errorSection = '';
    if (validationError) {
      errorSection = `
  // VALIDATION ERROR DETECTED:
  /*
${validationError.split('\n').map(line => '  ' + line).join('\n')}
  */
`;
    }
    
    return `import { test, expect } from '@playwright/test';
import { playwrightHelper } from './helpers/playwrightHelper';

test('Bug report - ${timestamp}', async ({ page }) => {
  await page.goto('/');
  ${errorSection}
  // All actions from initial state to current state
  // Initial game state
  const baseState = ${initialJson};
  
  const actionSequence = ${actionsJson};
  
  // Load initial state
  await playwrightHelper.loadState(page, baseState);
  
  // Replay and validate each action
  let currentGameState = baseState;
  for (let i = 0; i < actionSequence.length; i++) {
    const action = actionSequence[i];
    
    // Get available actions at this step
    const availableActions = await playwrightHelper.getAvailableActions(page, currentGameState);
    const isActionAvailable = availableActions.some(a => a.id === action.id);
    
    if (!isActionAvailable) {
      throw new Error(\`Invalid action at step \${i + 1}: "\${action.id}" not available. Available actions: \${availableActions.map(a => a.id).join(', ')}\`);
    }
    
    await playwrightHelper.clickAction(page, action.id);
    
    // Update current state for next iteration (would need state tracking)
    // currentGameState = await playwrightHelper.getCurrentState(page);
  }
  
  // Verify we're in the expected phase
  await expect(page.locator('[data-testid="game-phase"]')).toContainText('${currentState.phase}');
  
  // Add your specific bug assertions here
});`;
  }
};