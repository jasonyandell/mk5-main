import { writable, derived, get, type Readable } from 'svelte/store';
import type { GameAction, StateTransition, FilteredGameState } from '../game/types';
import { getNextStates } from '../game';
import { NetworkGameClient } from '../game/multiplayer/NetworkGameClient';
import { InProcessAdapter } from '../server/offline/InProcessAdapter';
import type { GameClient } from '../game/multiplayer/GameClient';
import type { MultiplayerGameState } from '../game/multiplayer/types';
import { hasCapabilityType } from '../game/multiplayer/types';
import type { GameConfig } from '../game/types/config';
import { createViewProjection, type ViewProjection } from '../game/view-projection';

/**
 * Pure client/server gameStore.
 * Everything flows through protocol-based GameClient.
 * Client is dumb - server handles ALL game logic via variants.
 */

// Detect test mode
const urlParams = typeof window !== 'undefined' ?
  new URLSearchParams(window.location.search) : null;
const testMode = urlParams?.get('testMode') === 'true';

// Player types configuration
const playerTypes = testMode ?
  ['human', 'human', 'human', 'human'] as ('human' | 'ai')[] :
  ['human', 'ai', 'ai', 'ai'] as ('human' | 'ai')[];

function actionKey(action: GameAction): string {
  return JSON.stringify(action);
}

// Create the GameClient with new architecture
let gameClient: GameClient;
const adapter = new InProcessAdapter();
const config: GameConfig = {
  playerTypes,
  shuffleSeed: Math.floor(Math.random() * 1000000)
};

gameClient = new NetworkGameClient(adapter, config);

// Export as const to prevent reassignment
export { gameClient };

// Writable store that tracks GameClient state
export const clientState = writable<MultiplayerGameState>(gameClient.getState());

// Subscribe to GameClient updates
gameClient.subscribe(state => {
  clientState.set(state);
});

void setPerspective(DEFAULT_SESSION_ID);

// Derived store for just the GameState (filtered view)
export const gameState: Readable<FilteredGameState> = derived(clientState, $clientState => $clientState.state);

// Derived store for player sessions
export const playerSessions = derived(clientState, $clientState => $clientState.sessions);

const DEFAULT_SESSION_ID = 'player-0';
const currentSessionIdStore = writable<string>(DEFAULT_SESSION_ID);

export const currentSessionId = derived(currentSessionIdStore, (value) => value);

export const currentSession = derived(
  [playerSessions, currentSessionId],
  ([$sessions, $sessionId]) => $sessions.find(session => session.playerId === $sessionId)
);

export const availablePerspectives = derived(playerSessions, ($sessions) =>
  $sessions.map((session) => ({
    id: session.playerId,
    label: session.name ? `${session.name}` : `P${session.playerIndex}`,
    session
  }))
);

export async function setPerspective(sessionId: string): Promise<void> {
  const current = get(currentSessionIdStore);
  if (current !== sessionId) {
    currentSessionIdStore.set(sessionId);
  }

  await (gameClient as NetworkGameClient).setPlayerId(sessionId);
}

playerSessions.subscribe(($sessions) => {
  if ($sessions.length === 0) {
    return;
  }

  const current = get(currentSessionIdStore);
  if (!$sessions.some(session => session.playerId === current)) {
    void setPerspective($sessions[0].playerId);
  }
});

// Current player ID for primary view (typically 0 for single-device play)

// View projection - derived from gameState only
export const viewProjection = derived<[typeof gameState, typeof playerSessions, typeof currentSessionId], ViewProjection>(
  [gameState, playerSessions, currentSessionId],
  ([$gameState, $sessions, $sessionId]) => {
    const session = $sessions.find(s => s.playerId === $sessionId) ?? $sessions[0];
    const perspectiveIndex = session?.playerIndex ?? 0;
    const canAct = !!(session && hasCapabilityType(session, 'act-as-player'));

    const allowedActions = (gameClient as NetworkGameClient).getValidActions(session ? session.playerIndex : undefined);
    const allowedKeys = new Set(allowedActions.map(actionKey));

    const allTransitions = getNextStates($gameState);
    const usedTransitions = testMode
      ? allTransitions
      : allTransitions.filter(transition => allowedKeys.has(actionKey(transition.action)));

    return createViewProjection(
      $gameState,
      usedTransitions,
      {
        isTestMode: testMode,
        viewingPlayerIndex: session?.playerIndex,
        canAct,
        isAIControlled: (player: number) => {
          const seat = $sessions.find(s => s.playerIndex === player);
          return seat ? seat.controlType === 'ai' : false;
        }
      }
    );
  }
);

// Store for tracking current game variant
export const gameVariant = derived(clientState, () => {
  // Extract variant from game view metadata if available
  const client = gameClient as NetworkGameClient;
  const view = client['cachedView']; // Access cached view
  return view?.metadata?.variant;
});

// Store for one-hand mode state (derived from game state)
export const oneHandState = derived(
  [gameState, gameVariant],
  ([$gameState, $gameVariant]) => {
    const isOneHand = $gameVariant?.type === 'one-hand';
    const isComplete = isOneHand && $gameState.phase === 'game_end';
    const seed = $gameVariant?.config?.originalSeed;
    const attempts = $gameVariant?.config?.attempts || 1;

    return {
      active: isOneHand,
      complete: isComplete,
      seed,
      attempts,
      scores: isComplete ? {
        us: $gameState.teamScores[0],
        them: $gameState.teamScores[1],
        won: $gameState.teamScores[0] > $gameState.teamScores[1]
      } : null
    };
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
    const session = get(currentSession);
    if (!session || !hasCapabilityType(session, 'act-as-player')) {
      throw new Error('Current perspective cannot execute actions');
    }

    if ('player' in transition.action && transition.action.player !== session.playerIndex) {
      throw new Error('Action not available for this perspective');
    }

    const result = await gameClient.requestAction(session.playerIndex, transition.action);

    if (!result.ok) {
      console.error('Action failed:', result.error);
      throw new Error(result.error);
    }
  },

  /**
   * Execute action directly (simpler API)
   */
  requestAction: async (_playerId: number, action: GameAction): Promise<void> => {
    const session = get(currentSession);
    if (!session || !hasCapabilityType(session, 'act-as-player')) {
      throw new Error('Current perspective cannot execute actions');
    }

    const preparedAction = { ...action } as GameAction;
    if ('player' in preparedAction) {
      preparedAction.player = session.playerIndex;
    }

    const result = await gameClient.requestAction(session.playerIndex, preparedAction);
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

    // Create new adapter and client
    const newAdapter = new InProcessAdapter();
    const newConfig: GameConfig = {
      playerTypes,
      shuffleSeed: Math.floor(Math.random() * 1000000)
    };

    const newClient = new NetworkGameClient(newAdapter, newConfig);

    // Replace global client reference
    Object.assign(gameClient, newClient);

    // Subscribe to updates
    newClient.subscribe(state => {
      clientState.set(state);
    });

    // Update stores
    clientState.set(newClient.getState());
    void setPerspective(DEFAULT_SESSION_ID);
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
  }
};

/**
 * Game variant actions - for special game modes
 */
export const gameVariants = {
  /**
   * Start a one-hand challenge game
   */
  startOneHand: async (seed?: number) => {
    // Destroy current game
    gameClient.destroy();

    // Create new game with one-hand variant
    const newAdapter = new InProcessAdapter();
    const newConfig: GameConfig = {
      playerTypes,
      ...(seed ? { shuffleSeed: seed } : {}),  // Only add if seed is defined
      variant: {
        type: 'one-hand',
        config: seed ? {
          targetHand: 1,
          maxAttempts: 999,
          originalSeed: seed
        } : {
          targetHand: 1,
          maxAttempts: 999
        }
      }
    };

    const newClient = new NetworkGameClient(newAdapter, newConfig);

    // Replace global client
    Object.assign(gameClient, newClient);

    // Subscribe to updates
    newClient.subscribe(state => {
      clientState.set(state);
    });

    // Update stores
    clientState.set(newClient.getState());
    void setPerspective(DEFAULT_SESSION_ID);
  },

  /**
   * Retry the same one-hand challenge
   */
  retryOneHand: async () => {
    const state = get(oneHandState);
    if (!state.seed) {
      console.error('No seed to retry');
      return;
    }

    // Start new game with same seed, incremented attempts
    await gameVariants.startOneHand(state.seed);
  },

  /**
   * Exit variant mode and return to standard game
   */
  exitVariant: async () => {
    await gameActions.resetGame();
  }
};

// Store to track if we're finding a seed (derived from adapter state)
export const findingSeed = writable(false);

// Clean up on page unload
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    gameClient.destroy();
  });
}
