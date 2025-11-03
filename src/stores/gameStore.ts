import { writable, derived, get, type Readable } from 'svelte/store';
import type { GameAction, StateTransition, FilteredGameState, GameState } from '../game/types';
import { getNextStates } from '../game';
import { NetworkGameClient } from '../game/multiplayer/NetworkGameClient';
import type { GameClient } from '../game/multiplayer/GameClient';
import type { MultiplayerGameState } from '../game/multiplayer/types';
import { hasCapabilityType } from '../game/multiplayer/types';
import type { GameConfig } from '../game/types/config';
import { createViewProjection, type ViewProjection } from '../game/view-projection';
import { getVisibleStateForSession } from '../game/multiplayer/capabilityUtils';
import { createSetupState } from '../game/core/state';
import { humanCapabilities, aiCapabilities } from '../game/multiplayer/capabilities';
import type { ValidAction } from '../shared/multiplayer/protocol';

/**
 * Pure client/server gameStore.
 * Everything flows through protocol-based GameClient.
 * Client is dumb - server handles ALL game logic via action transformers.
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

function convertToFilteredState(state: GameState): FilteredGameState {
  return {
    ...state,
    players: state.players.map(player => ({
      ...player,
      hand: [...player.hand],
      handCount: player.hand.length,
      ...(player.suitAnalysis ? { suitAnalysis: player.suitAnalysis } : {})
    }))
  };
}

function createPendingState(): MultiplayerGameState {
  const placeholderState = createSetupState({ playerTypes });
  const sessions = playerTypes.map((type, index) => {
    const idx = index as 0 | 1 | 2 | 3;
    const capabilities = type === 'human'
      ? humanCapabilities(idx)
      : aiCapabilities(idx);

    return {
      playerId: `${type === 'human' ? 'player' : 'ai'}-${index}`,
      playerIndex: idx,
      controlType: type,
      isConnected: true,
      name: `Player ${index + 1}`,
      capabilities: capabilities.map(cap => ({ ...cap }))
    };
  });

  return {
    gameId: 'initializing',
    coreState: placeholderState,
    players: sessions,
    createdAt: Date.now(),
    lastActionAt: Date.now(),
    enabledVariants: []
  };
}

// Create the GameClient with new architecture
let gameClient: GameClient | undefined;
let networkClient: NetworkGameClient | undefined;

// Default session ID constant
const DEFAULT_SESSION_ID = 'player-0';

// Writable stores for state and per-player actions
export const clientState = writable<MultiplayerGameState>(createPendingState());
const actionsByPlayer = writable<Record<string, ValidAction[]>>({});
let unsubscribeFromClient: (() => void) | undefined;

// Initialize function that creates GameServer + Transport + Client
async function initializeGameClient(): Promise<void> {
  const { GameServer } = await import('../server/GameServer');
  const { InProcessTransport } = await import('../server/transports/InProcessTransport');

  const config: GameConfig = {
    playerTypes,
    shuffleSeed: Math.floor(Math.random() * 1000000)
  };

  // Create game ID
  const gameId = `game-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;

  // Create player sessions from config
  const sessions = playerTypes.map((type, index) => {
    const idx = index as 0 | 1 | 2 | 3;
    const capabilities = type === 'human'
      ? humanCapabilities(idx)
      : aiCapabilities(idx);

    return {
      playerId: `${type === 'human' ? 'player' : 'ai'}-${index}`,
      playerIndex: idx,
      controlType: type,
      isConnected: true,
      name: `Player ${index + 1}`,
      capabilities: capabilities.map((cap: any) => ({ ...cap }))
    };
  });

  // Create dummy adapter (GameServer still requires one, but it's only used for AI)
  const dummyAdapter = {
    send: async () => {},
    subscribe: () => () => {},
    destroy: () => {},
    isConnected: () => true,
    getMetadata: () => ({ type: 'in-process' as const })
  };

  // 1. Create GameServer
  const gameServer = new GameServer(gameId, config, dummyAdapter, sessions);

  // 2. Create Transport
  const transport = new InProcessTransport();

  // 3. Wire them together
  gameServer.setTransport(transport);
  transport.setGameServer(gameServer);

  // 4. Create adapter from transport that implements IGameAdapter
  const adapter = transport.createAdapter('player-0');
  gameClient = new NetworkGameClient(adapter, config);
  networkClient = gameClient as NetworkGameClient;

  // Now set up state listeners
  try {
    const state = await gameClient.getState();
    clientState.set(state);
    actionsByPlayer.set(networkClient!.getCachedActionsMap());

    // Subscribe to GameClient updates
    unsubscribeFromClient = gameClient.subscribe(state => {
      clientState.set(state);
      actionsByPlayer.set(networkClient!.getCachedActionsMap());
    });

    await setPerspective(DEFAULT_SESSION_ID);
  } catch (error) {
    console.error('Failed to initialize game client:', error);
  }
}

// Initialize immediately
void initializeGameClient();

async function installNewClient(newClient: NetworkGameClient): Promise<void> {
  if (networkClient) {
    networkClient.destroy();
  }
  unsubscribeFromClient?.();

  gameClient = newClient;
  networkClient = newClient;

  clientState.set(createPendingState());
  actionsByPlayer.set({});

  try {
    const state = await newClient.getState();
    clientState.set(state);
    actionsByPlayer.set(networkClient!.getCachedActionsMap());
  } catch (error) {
    console.error('Failed to initialize game state:', error);
  }

  unsubscribeFromClient = newClient.subscribe(state => {
    clientState.set(state);
    actionsByPlayer.set(networkClient!.getCachedActionsMap());
  });

  await setPerspective(DEFAULT_SESSION_ID);
}

// Export getter that ensures client is initialized
export function getGameClient(): GameClient {
  if (!gameClient) {
    throw new Error('Game client not yet initialized');
  }
  return gameClient;
}

// gameClient is exported as a getter function, use getGameClient() instead
// For backward compatibility, we also export the variable (will be undefined until initialized)
export { gameClient };

// Derived store for player sessions
export const playerSessions = derived(clientState, $clientState => Array.from($clientState.players));

const currentSessionIdStore = writable<string>(DEFAULT_SESSION_ID);

export const currentSessionId = derived(currentSessionIdStore, (value) => value);

export const currentSession = derived(
  [playerSessions, currentSessionId],
  ([$sessions, $sessionId]) => $sessions.find(session => session.playerId === $sessionId)
);

export const gameState: Readable<FilteredGameState> = derived(
  [clientState, currentSession],
  ([$clientState, $session]) => {
    if ($session) {
      return getVisibleStateForSession($clientState.coreState, $session);
    }
    return convertToFilteredState($clientState.coreState);
  }
);

const allowedActionsStore = derived(
  [actionsByPlayer, currentSessionId],
  ([$actions, $sessionId]) => $actions[$sessionId] ?? $actions['__unfiltered__'] ?? []
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

  if (!networkClient) {
    throw new Error('Game client not yet initialized');
  }

  await networkClient.setPlayerId(sessionId);
  actionsByPlayer.set(networkClient.getCachedActionsMap());
}

playerSessions.subscribe(($sessions) => {
  if ($sessions.length === 0) {
    return;
  }

  const current = get(currentSessionIdStore);
  const firstSession = $sessions[0];
  if (!$sessions.some(session => session.playerId === current) && firstSession) {
    void setPerspective(firstSession.playerId);
  }
});

// Current player ID for primary view (typically 0 for single-device play)

// View projection - derived from gameState only
export const viewProjection = derived<
  [typeof gameState, typeof playerSessions, typeof currentSessionId, typeof allowedActionsStore],
  ViewProjection
>([
  gameState,
  playerSessions,
  currentSessionId,
  allowedActionsStore
],
  ([$gameState, $sessions, $sessionId, $allowedActions]) => {
    const session = $sessions.find(s => s.playerId === $sessionId) ?? $sessions[0];
    const canAct = !!(session && hasCapabilityType(session, 'act-as-player'));

    const allowedKeys = new Set($allowedActions.map(valid => actionKey(valid.action)));

    const allTransitions = getNextStates($gameState);
    const usedTransitions = testMode
      ? allTransitions
      : allowedKeys.size === 0
        ? allTransitions
        : allTransitions.filter(transition => allowedKeys.has(actionKey(transition.action)));

    const viewProjectionOptions: {
      isTestMode?: boolean;
      viewingPlayerIndex?: number;
      canAct?: boolean;
      isAIControlled?: (player: number) => boolean;
    } = {
      isTestMode: testMode,
      canAct,
      isAIControlled: (player: number) => {
        const seat = $sessions.find(s => s.playerIndex === player);
        return seat ? seat.controlType === 'ai' : false;
      }
    };

    // Only add viewingPlayerIndex if it's defined
    if (session?.playerIndex !== undefined) {
      viewProjectionOptions.viewingPlayerIndex = session.playerIndex;
    }

    return createViewProjection($gameState, usedTransitions, viewProjectionOptions);
  }
);

// Store for tracking current game action transformer
export const gameActionTransformer = derived(clientState, () => {
  // Extract action transformer from game view metadata if available
  if (!networkClient) return undefined;
  const view = networkClient.getCachedView();
  return view?.metadata?.variant;
});

// Store for one-hand mode state (derived from game state)
export const oneHandState = derived(
  [gameState, gameActionTransformer],
  ([$gameState, $gameActionTransformer]) => {
    const isOneHand = $gameActionTransformer?.type === 'one-hand';
    const isComplete = isOneHand && $gameState.phase === 'game_end';
    const seed = $gameActionTransformer?.config?.originalSeed;
    const attempts = $gameActionTransformer?.config?.attempts || 1;

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

    const result = await getGameClient().executeAction({
      playerId: session.playerId,
      action: transition.action,
      timestamp: Date.now()
    });

    if (!result.success) {
      console.error('Action failed:', result.error);
      throw new Error(result.error);
    }
  },

  /**
   * Execute action directly (simpler API)
   */
  requestAction: async (_playerId: string, action: GameAction): Promise<void> => {
    const session = get(currentSession);
    if (!session || !hasCapabilityType(session, 'act-as-player')) {
      throw new Error('Current perspective cannot execute actions');
    }

    const preparedAction = { ...action } as GameAction;
    if ('player' in preparedAction) {
      preparedAction.player = session.playerIndex;
    }

    const result = await getGameClient().executeAction({
      playerId: session.playerId,
      action: preparedAction,
      timestamp: Date.now()
    });
    if (!result.success) {
      console.error('Action failed:', result.error);
      throw new Error(result.error);
    }
  },

  /**
   * Reset the game (create new client with fresh state)
   */
  resetGame: async () => {
    const { GameServer } = await import('../server/GameServer');
    const { InProcessTransport } = await import('../server/transports/InProcessTransport');

    const newConfig: GameConfig = {
      playerTypes,
      shuffleSeed: Math.floor(Math.random() * 1000000)
    };

    // Create game ID
    const gameId = `game-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;

    // Create player sessions from config
    const sessions = playerTypes.map((type, index) => {
      const idx = index as 0 | 1 | 2 | 3;
      const capabilities = type === 'human'
        ? humanCapabilities(idx)
        : aiCapabilities(idx);

      return {
        playerId: `${type === 'human' ? 'player' : 'ai'}-${index}`,
        playerIndex: idx,
        controlType: type,
        isConnected: true,
        name: `Player ${index + 1}`,
        capabilities: capabilities.map((cap: any) => ({ ...cap }))
      };
    });

    // Create dummy adapter
    const dummyAdapter = {
      send: async () => {},
      subscribe: () => () => {},
      destroy: () => {},
      isConnected: () => true,
      getMetadata: () => ({ type: 'in-process' as const })
    };

    // 1. Create GameServer
    const gameServer = new GameServer(gameId, newConfig, dummyAdapter, sessions);

    // 2. Create Transport
    const transport = new InProcessTransport();

    // 3. Wire them together
    gameServer.setTransport(transport);
    transport.setGameServer(gameServer);

    // 4. Create adapter from transport that implements IGameAdapter
    const adapter = transport.createAdapter('player-0');
    const newClient = new NetworkGameClient(adapter, newConfig);
    await installNewClient(newClient);
  },

  /**
   * Switch player control type
   */
  setPlayerControl: async (playerId: number, type: 'human' | 'ai'): Promise<void> => {
    await getGameClient().setPlayerControl(playerId, type);
  },

  /**
   * Enable AI for all non-human players
   */
  enableAI: async () => {
    await getGameClient().setPlayerControl(1, 'ai');
    await getGameClient().setPlayerControl(2, 'ai');
    await getGameClient().setPlayerControl(3, 'ai');
  }
};

/**
 * Game action transformer actions - for special game modes
 */
export const gameActionTransformers = {
  /**
   * Start a one-hand challenge game
   */
  startOneHand: async (seed?: number) => {
    const { GameServer } = await import('../server/GameServer');
    const { InProcessTransport } = await import('../server/transports/InProcessTransport');

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

    // Create game ID
    const gameId = `game-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;

    // Create player sessions from config
    const sessions = playerTypes.map((type, index) => {
      const idx = index as 0 | 1 | 2 | 3;
      const capabilities = type === 'human'
        ? humanCapabilities(idx)
        : aiCapabilities(idx);

      return {
        playerId: `${type === 'human' ? 'player' : 'ai'}-${index}`,
        playerIndex: idx,
        controlType: type,
        isConnected: true,
        name: `Player ${index + 1}`,
        capabilities: capabilities.map((cap: any) => ({ ...cap }))
      };
    });

    // Create dummy adapter
    const dummyAdapter = {
      send: async () => {},
      subscribe: () => () => {},
      destroy: () => {},
      isConnected: () => true,
      getMetadata: () => ({ type: 'in-process' as const })
    };

    // 1. Create GameServer
    const gameServer = new GameServer(gameId, newConfig, dummyAdapter, sessions);

    // 2. Create Transport
    const transport = new InProcessTransport();

    // 3. Wire them together
    gameServer.setTransport(transport);
    transport.setGameServer(gameServer);

    // 4. Create adapter from transport that implements IGameAdapter
    const adapter = transport.createAdapter('player-0');
    const newClient = new NetworkGameClient(adapter, newConfig);
    await installNewClient(newClient);
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
    await gameActionTransformers.startOneHand(state.seed);
  },

  /**
   * Exit action transformer mode and return to standard game
   */
  exitActionTransformer: async () => {
    await gameActions.resetGame();
  }
};

// Backward compatibility aliases
export const gameVariants = gameActionTransformers;
export const gameVariant = gameActionTransformer;

// Store to track if we're finding a seed (derived from adapter state)
export const findingSeed = writable(false);

// Clean up on page unload
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    if (gameClient) {
      gameClient.destroy();
    }
  });
}
