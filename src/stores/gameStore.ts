import { writable, derived, get } from 'svelte/store';
import type { GameState, StateTransition } from '../game/types';
import { createInitialState, getNextStates } from '../game';
import { ControllerManager } from '../game/controllers';
import { 
  compressGameState, 
  expandMinimalState, 
  compressActionId, 
  decompressActionId,
  encodeURLData,
  decodeURLData,
  type URLData 
} from '../game/core/url-compression';
import { tickGame, skipAIDelays as skipAIDelaysPure, resetAISchedule } from '../game/core/ai-scheduler';
import { createViewProjection, type ViewProjection } from '../game/view-projection';

// Helper to deep clone an object preserving Sets
function deepClone<T>(obj: T): T {
  if (obj === null || obj === undefined) return obj;
  if (obj instanceof Set) return new Set(obj) as T;
  if (obj instanceof Date) return new Date(obj.getTime()) as T;
  if (obj instanceof Array) return obj.map(item => deepClone(item)) as T;
  if (typeof obj === 'object') {
    const cloned = {} as T;
    for (const key in obj) {
      if (Object.prototype.hasOwnProperty.call(obj, key)) {
        cloned[key] = deepClone(obj[key]);
      }
    }
    return cloned;
  }
  return obj;
}

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
function updateURLWithState(initialState: GameState, actions: StateTransition[], usePushState = false) {
  if (typeof window !== 'undefined') {
    // If no actions, clear the URL
    if (actions.length === 0) {
      const historyState = { initialState, actions: [], timestamp: Date.now() };
      if (usePushState) {
        window.history.pushState(historyState, '', window.location.pathname);
      } else {
        window.history.replaceState(historyState, '', window.location.pathname);
      }
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
    
    // Store state in history for easy access
    const historyState = { initialState, actions: actions.map(a => a.id), timestamp: Date.now() };
    
    if (usePushState) {
      window.history.pushState(historyState, '', newURL);
    } else {
      window.history.replaceState(historyState, '', newURL);
    }
  }
}

// Create the initial state once
const urlParams = typeof window !== 'undefined' ? 
  new URLSearchParams(window.location.search) : null;
const testMode = urlParams?.get('testMode') === 'true';
const firstInitialState = createInitialState({
  playerTypes: testMode ? ['human', 'human', 'human', 'human'] : ['human', 'ai', 'ai', 'ai']
});

// Core game state store
export const gameState = writable<GameState>(firstInitialState);

// Store the initial state (snapshot) - deep clone to prevent mutations
export const initialState = writable<GameState>(deepClone(firstInitialState));

// Store all actions taken from initial state
export const actionHistory = writable<StateTransition[]>([]);

// Store for validation errors
export const stateValidationError = writable<string | null>(null);

// Track which players are controlled by humans on this client
export const humanControlledPlayers = writable<Set<number>>(new Set([0]));

// Current player ID for primary view (can be changed for spectating)
export const currentPlayerId = writable<number>(0);

// Available actions store - filtered for privacy
export const availableActions = derived(
  [gameState, currentPlayerId],
  ([$gameState, _$playerId]) => {
    const allActions = getNextStates($gameState);
    
    // In test mode, show all actions for current player in game state
    if (testMode) {
      return allActions;
    }
    
    // In normal mode, only show actions for player 0
    // Filter to only actions that player 0 can take
    return allActions.filter(action => {
      // Actions without a player field are neutral (like complete-trick, score-hand)
      if (!('player' in action.action)) {
        return true;
      }
      // Only show actions for player 0
      return action.action.player === 0;
    });
  }
);

// Unified view projection for all UI rendering needs
export const viewProjection = derived<
  [typeof gameState, typeof availableActions],
  ViewProjection
>(
  [gameState, availableActions],
  ([$gameState, $availableActions]) => {
    const urlParams = typeof window !== 'undefined' ? 
      new URLSearchParams(window.location.search) : null;
    const testMode = urlParams?.get('testMode') === 'true' || false;
    
    return createViewProjection(
      $gameState,
      $availableActions,
      testMode,
      (player: number) => controllerManager.isAIControlled(player)
    );
  }
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

// Forward declaration for circular reference
let controllerManager: ControllerManager;

// Helper function to execute AI moves immediately (for tests/replay)
// Returns the new state and the AI actions taken
function executeAllAIImmediate(state: GameState): { state: GameState; aiActions: StateTransition[] } {
  let currentState = state;
  const aiActions: StateTransition[] = [];
  
  // Keep executing AI until no AI player needs to move
  while (currentState.playerTypes[currentState.currentPlayer] === 'ai') {
    const availableTransitions = getNextStates(currentState);
    
    // Find the best AI action (using existing AI logic)
    let aiTransition: StateTransition | undefined;
    
    // Try to find a play action first
    aiTransition = availableTransitions.find(t => 
      t.action.type === 'play' && 
      'player' in t.action && 
      t.action.player === currentState.currentPlayer
    );
    
    // If no play action, try bid/pass
    if (!aiTransition) {
      aiTransition = availableTransitions.find(t => 
        (t.action.type === 'bid' || t.action.type === 'pass') && 
        'player' in t.action && 
        t.action.player === currentState.currentPlayer
      );
    }
    
    // If no player action, try trump selection
    if (!aiTransition) {
      aiTransition = availableTransitions.find(t => 
        t.action.type === 'select-trump' && 
        currentState.winningBidder === currentState.currentPlayer
      );
    }
    
    // If no action found, break
    if (!aiTransition) {
      break;
    }
    
    currentState = aiTransition.newState;
    aiActions.push(aiTransition);
  }
  
  return { state: currentState, aiActions };
}

// Game actions
export const gameActions = {
  executeAction: (transition: StateTransition) => {
    const actions = get(actionHistory);
    
    // Add action to history
    let finalActions = [...actions, transition];
    actionHistory.set(finalActions);
    
    // Update to new state
    let newState = transition.newState;
    
    // In test mode, execute AI immediately after human actions
    // This ensures deterministic behavior for tests
    if (testMode && newState.playerTypes[newState.currentPlayer] === 'ai') {
      const result = executeAllAIImmediate(newState);
      newState = result.state;
      // Add AI actions to history
      if (result.aiActions.length > 0) {
        finalActions = [...finalActions, ...result.aiActions];
        actionHistory.set(finalActions);
      }
    }
    
    gameState.set(newState);
    
    // Validate state matches computed state
    validateState();
    
    // Update URL with initial state and actions (including any AI actions)
    // Pure approach: every action always updates the URL
    updateURLWithState(get(initialState), finalActions, true);
    
    // Notify all controllers of state change
    controllerManager.onStateChange(newState);
  },
  
  skipAIDelays: () => {
    // Use pure function to skip AI delays
    const currentState = get(gameState);
    const newState = skipAIDelaysPure(currentState);
    if (newState !== currentState) {
      gameState.set(newState);
    }
  },
  
  resetGame: () => {
    const oldActionCount = get(actionHistory).length;
    const newInitialState = createInitialState({
      playerTypes: testMode ? ['human', 'human', 'human', 'human'] : ['human', 'ai', 'ai', 'ai']
    });
    // Deep clone to prevent mutations
    initialState.set(deepClone(newInitialState));
    gameState.set(newInitialState);
    actionHistory.set([]);
    stateValidationError.set(null);
    updateURLWithState(newInitialState, [], true);
    
    // Notify controllers of reset
    controllerManager.onStateChange(newInitialState);
    
    // Debug logging
    if (typeof window !== 'undefined' && window.location.hostname === 'localhost') {
      const newActionCount = get(actionHistory).length;
      console.log('[GameStore] Game reset - action history cleared from', oldActionCount, 'to', newActionCount);
    }
  },
  
  loadState: (state: GameState) => {
    // When loading a state directly, make it the new initial state
    // Deep clone to prevent mutations
    initialState.set(deepClone(state));
    gameState.set(state);
    actionHistory.set([]);
    stateValidationError.set(null);
    updateURLWithState(state, [], true);
    
    // Notify controllers of new state
    controllerManager.onStateChange(state);
  },
  
  loadFromURL: () => {
    if (typeof window !== 'undefined') {
      const urlParams = new URLSearchParams(window.location.search);
      
      // Try new compressed format first
      const compressedParam = urlParams.get('d');
      if (compressedParam) {
        try {
          const urlData = decodeURLData(compressedParam);
          
          if (urlData.v === 1) {
            // Expand minimal state
            // In test mode, override player types to be all human unless explicitly specified
            if (testMode && !urlData.s.p) {
              urlData.s.p = ['h', 'h', 'h', 'h'];
            }
            const expandedInitial = expandMinimalState(urlData.s);
            initialState.set(deepClone(expandedInitial));
            
            // CRITICAL: Must deep clone here to avoid mutating the stored initial state
            let currentState = deepClone(expandedInitial);
            const validActions: StateTransition[] = [];
            
            // Replay compressed actions
            let invalidActionFound = false;
            for (const compressedAction of urlData.a) {
              const actionId = decompressActionId(compressedAction.i);
              const availableTransitions = getNextStates(currentState);
              const matchingTransition = availableTransitions.find(t => t.id === actionId);
              
              if (matchingTransition) {
                validActions.push(matchingTransition);
                currentState = matchingTransition.newState;
              } else {
                // Log warning but continue loading what we can
                const availableActionIds = availableTransitions.map(t => t.id).join(', ');
                console.warn(`Invalid action in URL: "${actionId}". Available actions: [${availableActionIds}]. Current phase: ${currentState.phase}`);
                console.warn('Stopping replay at this point - game will continue from here');
                invalidActionFound = true;
                break; // Stop processing further actions
              }
            }
            
            // After replaying all actions, if in test mode and current player is AI, execute AI
            if (testMode && currentState.playerTypes[currentState.currentPlayer] === 'ai') {
              const result = executeAllAIImmediate(currentState);
              currentState = result.state;
              // Add AI actions to the valid actions list
              validActions.push(...result.aiActions);
            }
            
            // Reset AI scheduling for clean state
            currentState = resetAISchedule(currentState);
            
            gameState.set(currentState);
            actionHistory.set(validActions);
            
            // If we had invalid actions, update the URL to reflect the valid state
            if (invalidActionFound) {
              updateURLWithState(get(initialState), validActions, false);
            }
            
            validateState();
            
            // Notify controllers of the state change so AI can take action
            controllerManager.onStateChange(currentState);
            return;
          }
        } catch (e) {
          console.error('Failed to load compressed URL:', e);
          console.warn('Starting fresh game instead');
          // Ensure we have a clean fresh game state
          const freshState = createInitialState({
            playerTypes: testMode ? ['human', 'human', 'human', 'human'] : ['human', 'ai', 'ai', 'ai']
          });
          initialState.set(deepClone(freshState));
          gameState.set(freshState);
          actionHistory.set([]);
          stateValidationError.set(null);
          // Don't update URL - leave the invalid param there but game starts fresh
          // Notify controllers of the fresh state
          controllerManager.onStateChange(freshState);
          return; // Important: return here to avoid fallthrough
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
      updateURLWithState(get(initialState), newActions, true);
    }
  },
  
  enableAI: () => {
    const state = get(gameState);
    const newState = { ...state, playerTypes: ['human', 'ai', 'ai', 'ai'] as ('human' | 'ai')[] };
    
    // Update state first
    gameState.set(newState);
    
    // If current player is now AI, execute immediately in test mode
    if (testMode && newState.playerTypes[newState.currentPlayer] === 'ai') {
      const result = executeAllAIImmediate(newState);
      gameState.set(result.state);
      // Add AI actions to history
      const currentHistory = get(actionHistory);
      const newHistory = [...currentHistory, ...result.aiActions];
      actionHistory.set(newHistory);
      // Update URL with new actions - use pushState for pure approach
      updateURLWithState(get(initialState), newHistory, true);
    }
    
    // Notify controllers of state change
    controllerManager.onStateChange(get(gameState));
  },
  
  loadFromHistoryState: (historyState: { initialState: GameState; actions: string[] }) => {
    if (historyState && historyState.initialState && historyState.actions) {
      // Deep clone to prevent mutations
      initialState.set(deepClone(historyState.initialState));
      
      // CRITICAL: Must deep clone here to avoid mutating the stored initial state
      let currentState = deepClone(historyState.initialState);
      const validActions: StateTransition[] = [];
      
      for (const actionId of historyState.actions) {
        const availableTransitions = getNextStates(currentState);
        const matchingTransition = availableTransitions.find(t => t.id === actionId);
        
        if (matchingTransition) {
          validActions.push(matchingTransition);
          currentState = matchingTransition.newState;
        } else {
          const availableActionIds = availableTransitions.map(t => t.id).join(', ');
          throw new Error(`Invalid action in history: "${actionId}". Available actions: [${availableActionIds}]. Current phase: ${currentState.phase}`);
        }
      }
      
      // After replaying all actions, if in test mode and current player is AI, execute AI
      if (testMode && currentState.playerTypes[currentState.currentPlayer] === 'ai') {
        const result = executeAllAIImmediate(currentState);
        currentState = result.state;
        // Add AI actions to the valid actions list
        validActions.push(...result.aiActions);
      }
      
      // Reset AI scheduling for clean navigation
      currentState = resetAISchedule(currentState);
      
      gameState.set(currentState);
      actionHistory.set(validActions);
      validateState();
      
      // Notify controllers of the state change so AI can take action
      controllerManager.onStateChange(currentState);
    }
  }
};

// Initialize controller manager after gameActions is defined
controllerManager = new ControllerManager((transition) => {
  // Mark this as coming from a controller (AI or human controller)
  gameActions.executeAction(transition);
});

// Initialize controllers with default configuration
if (typeof window !== 'undefined') {
  if (testMode) {
    // In test mode, all players are human-controlled for deterministic testing
    controllerManager.setupLocalGame([
      { type: 'human' },
      { type: 'human' },
      { type: 'human' },
      { type: 'human' }
    ]);
  } else {
    // Normal game: default configuration
    controllerManager.setupLocalGame();
  }
}

// Export controller manager
export { controllerManager };

// Pure game loop - advances game ticks and executes AI decisions
let animationFrame: number | null = null;
let gameLoopRunning = false;

const runGameLoop = () => {
  const currentState = get(gameState);
  
  // Use pure tick function to advance game state
  const result = tickGame(currentState);
  
  // Update tick state if it changed
  if (result.state.currentTick !== currentState.currentTick || 
      result.state.aiSchedule !== currentState.aiSchedule) {
    gameState.set(result.state);
  }
  
  // Execute any AI action through the proper channel
  if (result.action) {
    // Execute through gameActions to ensure it's recorded in history
    gameActions.executeAction(result.action);
  }
  
  // Schedule next frame if still running
  if (gameLoopRunning) {
    animationFrame = requestAnimationFrame(runGameLoop);
  }
};

// Export function to start game loop on demand
export function startGameLoop(): void {
  // Don't start in test mode or if already running
  if (testMode || gameLoopRunning) {
    return;
  }
  
  gameLoopRunning = true;
  animationFrame = requestAnimationFrame(runGameLoop);
}

// Export function to stop game loop
export function stopGameLoop(): void {
  gameLoopRunning = false;
  if (animationFrame !== null) {
    cancelAnimationFrame(animationFrame);
    animationFrame = null;
  }
}

// Clean up on page unload
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    stopGameLoop();
  });
}