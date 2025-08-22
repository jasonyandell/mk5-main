import { writable, derived, get } from 'svelte/store';
import type { GameState, StateTransition } from '../game/types';
import { createInitialState, getNextStates, getPlayerView } from '../game';
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
const firstInitialState = createInitialState();

// Core game state store
export const gameState = writable<GameState>(firstInitialState);

// Store the initial state (snapshot) - deep clone to prevent mutations
export const initialState = writable<GameState>(deepClone(firstInitialState));

// Store all actions taken from initial state
export const actionHistory = writable<StateTransition[]>([]);

// Store for validation errors
export const stateValidationError = writable<string | null>(null);

// Current player store
export const currentPlayer = derived(
  gameState,
  ($gameState) => $gameState.players[$gameState.currentPlayer]
);

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

// Player view store - provides privacy-safe view for current player
export const playerView = derived(
  [gameState, currentPlayerId],
  ([$gameState, $playerId]) => getPlayerView($gameState, $playerId)
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

// Deterministic UI state based on game state and available actions
export const uiState = derived(
  [gameState, availableActions],
  ([$gameState, $availableActions]) => {
    // Determine if player 0 is waiting (has no actions or it's not their turn)
    const isPlayer0Turn = $gameState.currentPlayer === 0;
    
    // Filter actions that are relevant for determining waiting state
    const playerActions = $availableActions.filter(a => {
      // Include bidding actions
      if (a.id.startsWith('bid-') || a.id === 'pass' || a.id === 'redeal') {
        return true;
      }
      // Include trump selection actions
      if (a.id.startsWith('trump-')) {
        return true;
      }
      // Include play actions
      if (a.id.startsWith('play-')) {
        return true;
      }
      // Exclude consensus actions from waiting determination
      if (a.id === 'complete-trick' || a.id === 'score-hand' || a.id.startsWith('agree-')) {
        return false;
      }
      // Include other actions
      return true;
    });
    
    const hasActions = playerActions.length > 0;
    
    // Determine waiting state based on phase
    const isWaitingDuringBidding = $gameState.phase === 'bidding' && (!isPlayer0Turn || !hasActions);
    const isWaitingDuringTrump = $gameState.phase === 'trump_selection' && (!isPlayer0Turn || !hasActions);
    const isWaitingDuringPlay = $gameState.phase === 'playing' && (!isPlayer0Turn || !hasActions);
    
    // Determine who we're waiting on
    const waitingOnPlayer = (!isPlayer0Turn || !hasActions) ? $gameState.currentPlayer : -1;
    
    return {
      isPlayer0Turn,
      hasActions,
      playerActions,
      isWaitingDuringBidding,
      isWaitingDuringTrump,
      isWaitingDuringPlay,
      isWaiting: isWaitingDuringBidding || isWaitingDuringTrump || isWaitingDuringPlay,
      waitingOnPlayer,
      showBiddingTable: isWaitingDuringBidding && !testMode,
      showActionPanel: $gameState.phase === 'bidding' || $gameState.phase === 'trump_selection'
    };
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

// Track if we're in a popstate response phase (controllers responding to navigation)
let inPopstateResponse = false;

// Game actions
export const gameActions = {
  executeAction: (transition: StateTransition, isFromController = false) => {
    const actions = get(actionHistory);
    
    // Add action to history
    actionHistory.set([...actions, transition]);
    
    
    // Update to new state
    gameState.set(transition.newState);
    
    // Validate state matches computed state
    validateState();
    
    // Update URL with initial state and actions
    // Skip URL updates for controller actions after popstate to avoid corrupting history
    if (inPopstateResponse && isFromController) {
      // Controller action during popstate response - skip URL update
    } else {
      // User action or normal flow - update URL
      updateURLWithState(get(initialState), [...actions, transition], true);
      // Clear popstate response phase on user action
      if (!isFromController) {
        inPopstateResponse = false;
      }
    }
    
    // Notify all controllers of state change
    controllerManager.onStateChange(transition.newState);
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
    const newInitialState = createInitialState();
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
          // Don't re-throw - just start with a fresh game
          // The game is already in its initial state
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
      
      // Reset AI scheduling for clean navigation
      currentState = resetAISchedule(currentState);
      
      gameState.set(currentState);
      actionHistory.set(validActions);
      validateState();
      
      // Enter popstate response phase - controller actions won't update URL
      inPopstateResponse = true;
      
      // Notify controllers of the state change so AI can take action
      controllerManager.onStateChange(currentState);
      
      // Clear flag synchronously - only affects controller actions taken synchronously
      inPopstateResponse = false;
    }
  }
};

// Initialize controller manager after gameActions is defined
controllerManager = new ControllerManager((transition) => {
  // Mark this as coming from a controller (AI or human controller)
  gameActions.executeAction(transition, true);
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
if (typeof window !== 'undefined') {
  let animationFrame: number | null = null;
  
  const runGameLoop = () => {
    const currentState = get(gameState);
    
    // Use pure tick function to advance game state
    const newState = tickGame(currentState);
    
    // Only update store if state actually changed
    if (newState !== currentState) {
      // Set the new state without triggering URL updates or controller notifications
      // (the tick already handled AI decisions purely)
      gameState.set(newState);
      
      // Validate the new state
      validateState();
    }
    
    // Schedule next frame
    animationFrame = requestAnimationFrame(runGameLoop);
  };
  
  // Start the game loop
  animationFrame = requestAnimationFrame(runGameLoop);
  
  // Clean up on page unload
  window.addEventListener('beforeunload', () => {
    if (animationFrame !== null) {
      cancelAnimationFrame(animationFrame);
    }
  });
}