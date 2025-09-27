/* global HTMLStyleElement, getComputedStyle */
import { writable, derived, get } from 'svelte/store';
import { converter } from 'culori';
import type { GameState, StateTransition } from '../game/types';
import { createInitialState, getNextStates } from '../game';
import { ControllerManager } from '../game/controllers';
import { TransitionDispatcher } from '../game/core/dispatcher';
import { decodeGameUrl } from '../game/core/url-compression';
import { selectAIAction, resetAISchedule, setAISpeedProfile } from '../game/core/ai-scheduler';
import { startSection } from '../game/core/sectionRunner';
import { oneHand as oneHandPreset, fromSlug as sectionFromSlug } from '../game/core/sectionPresets';
import { createViewProjection, type ViewProjection } from '../game/view-projection';
import { findBalancedSeed } from '../game/core/seedFinder';
import { seedFinderStore } from './seedFinderStore';

// Import extracted utilities
import { deepClone, deepCompare } from './utils/deepUtils';
import { updateURLWithState, setURLToMinimal, setCurrentScenario, beginUrlBatch, endUrlBatch } from './utils/urlManager';
import { incrementAttempts } from './utils/oneHandStats';
import { executeAllAIImmediate, injectConsensusIfNeeded } from './utils/aiHelpers';
import { prepareDeterministicHand, buildActionsToPlayingFromState, buildOverlayPayload } from './utils/sectionHelpers';


// Re-export for backwards compatibility
export { setCurrentScenario, beginUrlBatch, endUrlBatch };

// Helper function for seed finding and confirmation flow
async function findAndConfirmSeed(): Promise<number> {
  // Start seed search UI
  seedFinderStore.startSearch();

  try {
    const result = await findBalancedSeed(
      (progress) => {
        seedFinderStore.updateProgress(progress);
      },
      () => seedFinderStore.shouldUseBest()
    );

    // Show confirmation modal with the already-calculated win rate
    seedFinderStore.setSeedFound(result.seed, result.winRate);

    // Wait for user confirmation or regeneration
    return new Promise((resolve) => {
      const checkConfirmation = () => {
        const unsubscribe = seedFinderStore.subscribe(state => {
          if (!state.waitingForConfirmation) {
            unsubscribe();
            // Always resolve with the seed - user can't cancel out of the confirmation modal
            resolve(result.seed);
          }
        });
      };

      // Listen for regenerate event
      const handleRegenerate = () => {
        window.removeEventListener('regenerate-seed', handleRegenerate);
        // Recursively find another seed
        findAndConfirmSeed().then(resolve);
      };
      window.addEventListener('regenerate-seed', handleRegenerate);

      checkConfirmation();
    });
  } catch (error) {
    // If there's an error during seed search (shouldn't happen in normal flow)
    // Use fallback seed
    seedFinderStore.stopSearch();
    console.error('Error during seed search:', error);
    return 424242; // Fallback seed
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

// Section completion overlay state
export const sectionOverlay = writable<
  | null
  | {
      type: 'oneHand';
      phase: GameState['phase'];
      seed: number;
      canChallenge?: boolean;
      attemptsForWin?: number;
      attemptsCount?: number;
      weWon?: boolean;
      usScore?: number;
      themScore?: number;
    }
>(null);


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


// Game actions
export const gameActions = {
  executeAction: (transition: StateTransition) => {
    // Validate transition was offered by the game engine
    const currentState = get(gameState);
    const validTransitions = getNextStates(currentState);
    const validIds = validTransitions.map(t => t.id);
    
    if (!validIds.includes(transition.id)) {
      throw new Error(
        `Invalid transition attempted: "${transition.id}". ` +
        `Valid transitions are: [${validIds.join(', ')}]`
      );
    }
    
    // Use the fresh transition to avoid stale state issues
    const freshTransition = validTransitions.find(t => t.id === transition.id)!;
    
    const actions = get(actionHistory);
    
    // Add action to history (use fresh transition)
    let finalActions = [...actions, freshTransition];
    actionHistory.set(finalActions);
    
    // Update to new state (use fresh state)
    let newState = freshTransition.newState;
    
    // In test mode, execute AI immediately after human actions
    // This ensures deterministic behavior for tests
    // Skip this immediate chaining if a custom gate is active (SectionRunner)
    if (testMode && !dispatcher.hasCustomGate() && newState.playerTypes[newState.currentPlayer] === 'ai') {
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
    // Execute all scheduled transitions immediately
    dispatcher.executeAllScheduled();
  },
  
  resetGame: () => {
    const oldActionCount = get(actionHistory).length;
    const currentState = get(gameState);
    
    // Preserve theme settings when resetting
    const newInitialState = createInitialState({
      playerTypes: testMode ? ['human', 'human', 'human', 'human'] : ['human', 'ai', 'ai', 'ai'],
      theme: currentState.theme,
      colorOverrides: currentState.colorOverrides
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
    gameState.set(deepClone(state));
    actionHistory.set([]);
    stateValidationError.set(null);
    updateURLWithState(state, [], true);
    
    // Notify controllers of new state
    controllerManager.onStateChange(state);
  },
  
  loadFromURL: async () => {
    if (typeof window !== 'undefined') {
      try {
        const params = new URLSearchParams(window.location.search);
        if (!params.get('s')) {
          // Supply a seed if missing and update URL (preserving scenario if present)
          // Reuse existing URL decode to capture theme/overrides even without seed
          const { theme: urlTheme, colorOverrides: urlColorOverrides } = decodeGameUrl(window.location.search);
          const scenarioParam = params.get('h') || '';
          
          // For one_hand scenario without seed, use balanced seed finder
          let seedToUse: number | undefined;
          if (scenarioParam === 'one_hand') {
            if (testMode) {
              // In test mode, skip seed finder and use default seed
              seedToUse = 424242;
            } else {
              // Find new balanced seed
              const balancedSeed = await findAndConfirmSeed();
              // findAndConfirmSeed now always returns a seed (user can't cancel out of the flow)
              seedToUse = balancedSeed;
            }
          }
          
          const supplied = createInitialState({
            playerTypes: testMode ? ['human', 'human', 'human', 'human'] : ['human', 'ai', 'ai', 'ai'],
            theme: urlTheme,
            colorOverrides: urlColorOverrides,
            ...(seedToUse !== undefined && { shuffleSeed: seedToUse }) // Only include if defined
          });
          setCurrentScenario(scenarioParam || null);

          // Initialize stores
          initialState.set(deepClone(supplied));
          gameState.set(supplied);
          actionHistory.set([]);
          stateValidationError.set(null);

          // Only update URL with a supplied seed if a scenario is requested
          if (scenarioParam) {
            updateURLWithState(supplied, [], false);
          }

          // If a scenario is provided, prepare and start it now
          if (scenarioParam) {
            const preset = sectionFromSlug(scenarioParam);
            if (preset) {
              if (scenarioParam === 'one_hand') {
                // Prevent AI from running ahead while preparing and before runner is active
                stopGameLoop();
                dispatcher.setFrozen(true);

                const actionIds = buildActionsToPlayingFromState(supplied);
                gameActions.loadFromHistoryState({ initialState: supplied, actions: actionIds });
                updateURLWithState(get(initialState), get(actionHistory), false);
                // Increment attempts for this seed
                incrementAttempts(supplied.shuffleSeed);
              }
              const runner = startSection(preset());
              if (scenarioParam === 'one_hand') {
                // Now allow progression
                dispatcher.setFrozen(false);
                startGameLoop();
              }
              if (scenarioParam === 'one_hand') {
                // HACK: Direct state monitoring for URL-triggered sections
                // The section runner's .done promise doesn't resolve properly for URL-triggered sections
                // because the initial actions (buildActionsToPlayingFromState) are applied before the
                // section runner's dispatcher listeners are set up. This causes the section runner to
                // never "see" the state transitions during gameplay.
                //
                // TODO: Fix the root cause by either:
                // 1. Ensuring all state transitions go through the dispatcher (including loadFromHistoryState)
                // 2. Starting the section runner before applying initial actions
                // 3. Restructuring how URL-triggered sections initialize vs user-triggered sections
                //
                // For now, we directly monitor the game state for completion:
                let completed = false;
                const checkCompletion = (state: GameState) => {
                  if (!completed && (state.phase === 'scoring' || state.phase === 'game_end')) {
                    completed = true;
                    sectionOverlay.set(buildOverlayPayload(state, get(initialState)));
                    setURLToMinimal(get(initialState));
                  }
                };
                
                // Check immediately in case we're already at scoring
                checkCompletion(get(gameState));
                
                // Then subscribe for future changes
                const unsubscribe = gameState.subscribe(checkCompletion);
                
                // TODO: Handle consensus as part of game state machine
                runner.done.then((result) => {
                  stopGameLoop();
                  dispatcher.setFrozen(true);
                  sectionOverlay.set(buildOverlayPayload(result.state, get(initialState)));
                  setURLToMinimal(get(initialState));
                }).finally(() => unsubscribe());
              }
            }
          }

          controllerManager.onStateChange(get(gameState));
          return;
        }

        const { seed, actions, playerTypes, dealer, tournamentMode, theme, colorOverrides, scenario } = decodeGameUrl(window.location.search);
        
        // If no seed, there's no game to load
        if (!seed) {
          return;
        }
        
        // Theme will be applied reactively by App.svelte $effect
        
        // Use the player types from the URL
        const finalPlayerTypes = playerTypes;
        
        // Create initial state with ALL properties including theme
        const newInitialState = createInitialState({
          shuffleSeed: seed,
          playerTypes: finalPlayerTypes,
          dealer,
          tournamentMode,
          theme,
          colorOverrides
        });
        initialState.set(deepClone(newInitialState));
        
        // CRITICAL: Must deep clone here to avoid mutating the stored initial state
        let currentState = deepClone(newInitialState);
        const validActions: StateTransition[] = [];
        
        // Replay actions (supports compact consensus)
        let invalidActionFound = false;
        for (const actionId of actions) {
          if (actionId === 'complete-trick' || actionId === 'score-hand') {
            currentState = injectConsensusIfNeeded(currentState, actionId, validActions);
          }
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
        // Reflect scenario for ongoing URL updates
        setCurrentScenario(scenario || null);
        
        // If we had invalid actions, update the URL to reflect the valid state
        if (invalidActionFound) {
          updateURLWithState(get(initialState), validActions, false);
        }
        
        validateState();
        
        // Notify controllers of the state change so AI can take action
        controllerManager.onStateChange(currentState);

        // If scenario is provided, start the corresponding section.
        // If no actions were supplied and scenario needs a prepared state (e.g., one_hand),
        // build a deterministic action list up to that point and redirect URL accordingly.
        if (scenario) {
          const preset = sectionFromSlug(scenario);
          if (preset) {
            if (scenario === 'one_hand' && actions.length === 0) {
              // Prevent AI from running ahead while preparing and before runner is active
              stopGameLoop();
              dispatcher.setFrozen(true);
              // Build actions to reach playing from this initial state and load them as history
              const actionIds = buildActionsToPlayingFromState(newInitialState);
              gameActions.loadFromHistoryState({ initialState: newInitialState, actions: actionIds });
              // Ensure URL reflects the actions up to this point
              updateURLWithState(get(initialState), get(actionHistory), false);
              // Increment attempts
              incrementAttempts(newInitialState.shuffleSeed);
            }
            const runner = startSection(preset());
            if (scenario === 'one_hand') {
              // Now allow progression
              dispatcher.setFrozen(false);
              startGameLoop();
            }
            if (scenario === 'one_hand') {
              runner.done.then((result) => {
                stopGameLoop();
                dispatcher.setFrozen(true);
                sectionOverlay.set(buildOverlayPayload(result.state, get(initialState)));
                setURLToMinimal(get(initialState));
              });
            }
          }
        }
        return;
      } catch (e: unknown) {
        const error = e as Error;
        if (error.message && error.message.includes('outdated format')) {
          window.alert(error.message);
          return;
        }
        if (error.message && (error.message.includes('Invalid URL') || error.message.includes('Missing version'))) {
          // Not an error - just no game to load
          return;
        }
        console.error('Failed to load URL:', e);
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
  
  updateTheme: (theme: string, colorOverrides: Record<string, string> = {}) => {
    const currentState = get(gameState);
    // Minimize overrides to only changed-from-default values
    let minimalOverrides = colorOverrides;
    try {
      if (typeof window !== 'undefined') {
        const styleEl = document.getElementById('theme-overrides') as HTMLStyleElement | null;
        const prevDisabled = styleEl ? styleEl.disabled : undefined;
        if (styleEl) styleEl.disabled = true;
        const styles = getComputedStyle(document.documentElement);
        const toOklch = converter('oklch');
        const approxEqual = (a: string, b: string): boolean => {
          const parse = (val: string): [number, number, number] | null => {
            const v = val.trim();
            if (!v) return null;
            const parts = v.split(/\s+/);
            if (parts.length !== 3) return null;
            if (parts[0] && parts[0].includes('%')) {
              const l = parseFloat(parts[0].replace('%', ''));
              const c = parseFloat(parts[1] || '0');
              const h = parseFloat(parts[2] || '0');
              if (Number.isFinite(l) && Number.isFinite(c) && Number.isFinite(h)) return [l, c, h];
              return null;
            }
            const h = parseFloat(parts[0] || '0');
            const s = parseFloat((parts[1] || '0').replace('%', '')) / 100;
            const l = parseFloat((parts[2] || '0').replace('%', '')) / 100;
            if (!Number.isFinite(h) || !Number.isFinite(s) || !Number.isFinite(l)) return null;
            const ok = toOklch({ mode: 'hsl', h, s, l });
            if (!ok) return null;
            return [ (ok.l || 0) * 100, ok.c || 0, ok.h || 0 ];
          };
          const A = parse(a);
          const B = parse(b);
          if (!A || !B) return false;
          const [l1, c1, h1] = A;
          const [l2, c2, h2] = B;
          return Math.abs(l1 - l2) < 0.02 && Math.abs(c1 - c2) < 0.0005 && Math.abs(h1 - h2) < 0.1;
        };
        const out: Record<string, string> = {};
        for (const [varName, value] of Object.entries(colorOverrides)) {
          const base = styles.getPropertyValue(varName).trim();
          if (!base || !approxEqual(base, String(value).trim())) {
            out[varName] = value;
          }
        }
        minimalOverrides = out;
        if (styleEl && prevDisabled !== undefined) styleEl.disabled = prevDisabled;
      }
    } catch {
      // Ignore; keep provided overrides
    }
    const newState = {
      ...currentState,
      theme,
      colorOverrides: minimalOverrides
    };
    
    // Theme will be applied reactively by App.svelte $effect
    
    // Update state
    gameState.set(newState);
    
    // Update initial state too (so it persists through resets)
    const currentInitial = get(initialState);
    initialState.set({
      ...currentInitial,
      theme,
      colorOverrides: minimalOverrides
    });
    
    // Update URL on next frame so App.svelte can apply data-theme first
    if (typeof window !== 'undefined') {
      requestAnimationFrame(() => {
        updateURLWithState(get(initialState), get(actionHistory), false);
      });
    } else {
      updateURLWithState(get(initialState), get(actionHistory), false);
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
        // Inflate compact consensus if needed right before boundary actions
        if (actionId === 'complete-trick' || actionId === 'score-hand') {
          currentState = injectConsensusIfNeeded(currentState, actionId, validActions);
        }
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

// Unified dispatcher: single entry for executing transitions
export const dispatcher = new TransitionDispatcher(
  (t) => gameActions.executeAction(t),
  () => get(gameState)
);

// Initialize controller manager after gameActions is defined
controllerManager = new ControllerManager((transition) => {
  // Route through unified dispatcher as a UI-originated transition
  dispatcher.requestTransition(transition, 'ui');
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

// Helper function to wrap getNextStates with delay attachment
function getNextStatesWithDelays(state: GameState): (StateTransition & { delayTicks?: number })[] {
  const transitions = getNextStates(state);
  return transitions.map(t => ({
    ...t,
    delayTicks: getDelayTicksForAction(t.action, state)
  }));
}

// Determines delay based on action and state context
function getDelayTicksForAction(action: StateTransition['action'], state: GameState): number {
  // AI delays (at 60fps: 30 ticks = ~500ms)
  if ('player' in action && state.playerTypes[action.player] === 'ai') {
    switch (action.type) {
      case 'bid': return 30;           // ~500ms
      case 'pass': return 30;          // ~500ms
      case 'play': return 18;          // ~300ms
      case 'select-trump': return 60;  // ~1000ms
      case 'agree-complete-trick': return 0;  // Instant consensus
      case 'agree-score-hand': return 0;      // Instant consensus
      default: return 12;  // ~200ms
    }
  }
  
  // Human actions and system actions - no delay
  return 0;
}

// Check if AI should act
function shouldAIAct(state: GameState): boolean {
  // AI should act if:
  // 1. It's an AI player's turn
  // 2. Not already scheduled
  // 3. Game is in playable state
  if (state.phase === 'game_end') return false;
  
  // Check if current player is AI
  const currentPlayerType = state.playerTypes[state.currentPlayer];
  if (currentPlayerType !== 'ai') return false;
  
  // Check if already has scheduled action
  if (dispatcher.hasScheduledAction(state.currentPlayer)) return false;
  
  return true;
}

// Pure game loop - advances game ticks and executes AI decisions
let animationFrame: number | null = null;
let gameLoopRunning = false;

const runGameLoop = () => {
  // Advance dispatcher's internal tick and process scheduled transitions
  dispatcher.advanceTick();
  
  let currentState = get(gameState);
  
  // Check if AI needs to make a decision
  if (shouldAIAct(currentState)) {
    // Get available actions with delays attached
    const transitions = getNextStatesWithDelays(currentState);
    
    // Ask AI for its decision (pure computation)
    const choice = selectAIAction(currentState, currentState.currentPlayer, transitions);
    
    if (choice) {
      // Decision already has delayTicks from getNextStatesWithDelays
      dispatcher.requestTransition(choice, 'ai');
    }
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


// High-level section actions
export const sectionActions = {
  startOneHand: async () => {
    let balancedSeed: number;
    
    if (testMode) {
      // In test mode, skip seed finder and use default seed
      balancedSeed = 424242;
    } else {
      // Find new balanced seed
      const foundSeed = await findAndConfirmSeed();
      // findAndConfirmSeed now always returns a seed (user can't cancel out of the flow)
      balancedSeed = foundSeed;
    }
    
    // Prepare a deterministic hand with the balanced seed, preserving current theme settings
    const prepared = await prepareDeterministicHand(balancedSeed);
    const currentTheme = get(gameState).theme;
    const currentOverrides = get(gameState).colorOverrides;
    const preparedWithTheme = { ...prepared, theme: currentTheme, colorOverrides: currentOverrides };
    // Set scenario before load so initial URL already includes it
    setCurrentScenario('oneHand');
    // Track attempts for this seed
    incrementAttempts(prepared.shuffleSeed);
    gameActions.loadState(preparedWithTheme);

    // Start a one-hand section
    // TODO: Handle consensus as part of game state machine
    const runner = startSection(oneHandPreset());
    // Allow progression now that runner is listening
    dispatcher.setFrozen(false);
    startGameLoop();
    // Show overlay as soon as we reach scoring/game_end
    let completed = false;
    const checkCompletion = (state: GameState) => {
      if (!completed && (state.phase === 'scoring' || state.phase === 'game_end')) {
        completed = true;
        sectionOverlay.set(buildOverlayPayload(state, get(initialState)));
        setURLToMinimal(get(initialState));
      }
    };
    // Immediate check + subscribe
    checkCompletion(get(gameState));
    const unsubscribe = gameState.subscribe(checkCompletion);
    const result = await runner.done;
    // Freeze the game at completion and update overlay with final phase
    stopGameLoop();
    dispatcher.setFrozen(true);
    sectionOverlay.set(buildOverlayPayload(result.state, get(initialState)));
    setURLToMinimal(get(initialState));
    unsubscribe();
  },
  clearOverlay: (unfreeze = false) => {
    sectionOverlay.set(null);
    if (unfreeze) {
      // Ensure AI speed returns to normal when resuming general play
      setAISpeedProfile('normal');
      dispatcher.setFrozen(false);
      dispatcher.clearGate();
      startGameLoop();
    }
  },
  restartOneHand: async (seedOverride?: number) => {
    // Clear overlay and prevent AI from running ahead during setup
    sectionOverlay.set(null);
    dispatcher.clearGate();
    stopGameLoop();
    dispatcher.setFrozen(true);
    // Use current seed by default
    const seed = seedOverride ?? get(initialState).shuffleSeed;
    setCurrentScenario('oneHand');
    // Load fresh initial with this seed, preserving theme
    const currentTheme = get(gameState).theme;
    const currentOverrides = get(gameState).colorOverrides;
    const fresh = createInitialState({ shuffleSeed: seed, playerTypes: ['human', 'ai', 'ai', 'ai'], theme: currentTheme, colorOverrides: currentOverrides });
    initialState.set(deepClone(fresh));
    gameState.set(fresh);
    actionHistory.set([]);
    stateValidationError.set(null);
    updateURLWithState(fresh, [], false);
    // Build actions to playing and load
    const actionIds = buildActionsToPlayingFromState(fresh);
    gameActions.loadFromHistoryState({ initialState: fresh, actions: actionIds });
    updateURLWithState(get(initialState), get(actionHistory), false);
    // Increment attempts for this seed
    incrementAttempts(seed);
    // Start section and handle completion overlay
    const runner = startSection(oneHandPreset());
    if (typeof window !== 'undefined' && window.location.hostname === 'localhost') {
      console.log('[OneHand Restart] Runner started');
    }
    // Allow progression now that runner is listening
    dispatcher.setFrozen(false);
    startGameLoop();
    // Show overlay as soon as we reach scoring/game_end
    let completed = false;
    const checkCompletion = (state: GameState) => {
      if (!completed && (state.phase === 'scoring' || state.phase === 'game_end')) {
        completed = true;
        sectionOverlay.set(buildOverlayPayload(state, get(initialState)));
        setURLToMinimal(get(initialState));
      }
    };
    // Immediate check + subscribe
    checkCompletion(get(gameState));
    const unsubscribe = gameState.subscribe(checkCompletion);
    const result = await runner.done;
    if (typeof window !== 'undefined' && window.location.hostname === 'localhost') {
      console.log('[OneHand Restart] Runner done with phase', result.state.phase);
    }
    stopGameLoop();
    dispatcher.setFrozen(true);
    sectionOverlay.set(buildOverlayPayload(result.state, get(initialState)));
    setURLToMinimal(get(initialState));
    unsubscribe();
  },
  newOneHand: async () => {
    // Clear overlay first to prevent immediate re-triggering
    sectionOverlay.set(null);
    // Then start the seed finder flow
    await sectionActions.startOneHand();
  }
};
