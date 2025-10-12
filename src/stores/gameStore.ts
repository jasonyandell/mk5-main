import { writable, derived, get } from 'svelte/store';
import type { GameState, GameAction, StateTransition } from '../game/types';
import { createInitialState, getNextStates } from '../game';
import { LocalGameClient } from '../game/multiplayer/LocalGameClient';
import type { MultiplayerGameState } from '../game/multiplayer/types';
import { createViewProjection, type ViewProjection } from '../game/view-projection';

/**
 * New gameStore - ruthlessly simplified.
 * Everything flows through GameClient - the ONLY source of truth.
 */

// Detect test mode
const urlParams = typeof window !== 'undefined' ?
  new URLSearchParams(window.location.search) : null;
const testMode = urlParams?.get('testMode') === 'true';

// Create initial state
const initialGameState = createInitialState({
  playerTypes: testMode ? ['human', 'human', 'human', 'human'] : ['human', 'ai', 'ai', 'ai']
});

// Create the GameClient - single source of truth
export const gameClient = new LocalGameClient(
  initialGameState,
  testMode ? ['human', 'human', 'human', 'human'] : ['human', 'ai', 'ai', 'ai']
);

// Writable store that tracks GameClient state
const clientState = writable<MultiplayerGameState>(gameClient.getState());

// Subscribe to GameClient updates
gameClient.subscribe(state => {
  clientState.set(state);
});

// Derived store for just the GameState (for backwards compatibility)
export const gameState = derived(clientState, $clientState => $clientState.state);

// Derived store for player sessions
export const playerSessions = derived(clientState, $clientState => $clientState.sessions);

// Current player ID for primary view (typically 0 for single-device play)
export const currentPlayerId = writable<number>(0);

// Initial state export (for compatibility)
export const initialState = writable<GameState>(initialGameState);

// Action history (deprecated - GameClient tracks this internally via actionHistory in state)
export const actionHistory = derived(gameState, $gameState => {
  // Convert actionHistory to StateTransitions for compatibility
  return $gameState.actionHistory.map(action => {
    const allTransitions = getNextStates($gameState);
    // Find matching transition (this is a simplification)
    const found = allTransitions.find(t => t.action.type === action.type);
    return found || {
      id: action.type,
      label: action.type,
      action: action,
      newState: $gameState
    };
  });
});

// View projection - derived from gameState only
export const viewProjection = derived<typeof gameState, ViewProjection>(
  gameState,
  ($gameState) => {
    const allTransitions = getNextStates($gameState);

    // In test mode, show all actions for current player
    // In normal mode, only show actions for player 0
    const availableActions = testMode
      ? allTransitions
      : allTransitions.filter(action => {
          if (!('player' in action.action)) return true;
          return action.action.player === 0;
        });

    return createViewProjection(
      $gameState,
      availableActions,
      testMode,
      (player: number) => {
        const sessions = get(playerSessions);
        const session = sessions[player];
        return session ? session.type === 'ai' : false;
      }
    );
  }
);

/**
 * Game actions - simplified wrapper for GameClient
 */
export const gameActions = {
  /**
   * Execute an action via StateTransition (for compatibility with old UI)
   */
  executeAction: async (transition: StateTransition): Promise<void> => {
    const playerId = 'player' in transition.action ? transition.action.player : 0;
    const result = await gameClient.requestAction(playerId, transition.action);

    if (!result.ok) {
      console.error('Action failed:', result.error);
      throw new Error(result.error);
    }
  },

  /**
   * Execute action directly (simpler API)
   */
  requestAction: async (playerId: number, action: GameAction): Promise<void> => {
    const result = await gameClient.requestAction(playerId, action);
    if (!result.ok) {
      console.error('Action failed:', result.error);
      throw new Error(result.error);
    }
  },

  /**
   * Reset the game (create new client with fresh state)
   */
  resetGame: () => {
    // Destroy old client
    gameClient.destroy();

    // Create new initial state
    const newInitialState = createInitialState({
      playerTypes: testMode ? ['human', 'human', 'human', 'human'] : ['human', 'ai', 'ai', 'ai']
    });

    // Create new client
    const newClient = new LocalGameClient(
      newInitialState,
      testMode ? ['human', 'human', 'human', 'human'] : ['human', 'ai', 'ai', 'ai']
    );

    // Replace global client reference
    Object.assign(gameClient, newClient);

    // Update stores
    initialState.set(newInitialState);
    clientState.set(gameClient.getState());
  },

  /**
   * Switch player control type
   */
  setPlayerControl: async (playerId: number, type: 'human' | 'ai'): Promise<void> => {
    await gameClient.setPlayerControl(playerId, type);
  },

  /**
   * Enable AI for all non-human players
   */
  enableAI: async () => {
    await gameClient.setPlayerControl(1, 'ai');
    await gameClient.setPlayerControl(2, 'ai');
    await gameClient.setPlayerControl(3, 'ai');
  },

  // Deprecated stubs for backwards compatibility
  undo: () => console.warn('undo() not supported'),
  loadFromURL: async () => console.warn('loadFromURL() deprecated'),
  loadState: (_state: GameState) => console.warn('loadState() deprecated'),
  updateTheme: (_theme: string, _overrides?: Record<string, string>) => console.warn('updateTheme() deprecated'),
};

// Deprecated exports for backwards compatibility
export const sectionOverlay = writable<null>(null);
export const sectionActions = {
  startOneHand: async () => {},
  clearOverlay: () => {},
  restartOneHand: async () => {},
  newOneHand: async () => {}
};
export const stateValidationError = writable<string | null>(null);

// Clean up on page unload
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    gameClient.destroy();
  });
}
