import { writable, derived, get, type Readable } from 'svelte/store';
import type { GameAction, FilteredGameState, GameState } from '../game/types';
import { getNextStates } from '../game';
import { NetworkGameClient } from '../game/multiplayer/NetworkGameClient';
import type { MultiplayerGameState, Capability } from '../game/multiplayer/types';
import { hasCapabilityType } from '../game/multiplayer/types';
import type { GameConfig } from '../game/types/config';
import { createViewProjection, type ViewProjection } from '../game/view-projection';
import { getVisibleStateForSession } from '../game/multiplayer/capabilityUtils';
import { createSetupState } from '../game/core/state';
import { humanCapabilities, aiCapabilities } from '../game/multiplayer/capabilities';
import type { ValidAction } from '../shared/multiplayer/protocol';

/**
 * GameStore - Clean facade over Room/Transport/NetworkGameClient complexity.
 *
 * Philosophy:
 * - Hide all Room/Transport/Connection wiring
 * - Export minimal surface area (7 items total)
 * - Single method for Room creation (no duplication)
 * - Client is dumb - server handles ALL game logic
 */

// Detect test mode
const urlParams = typeof window !== 'undefined' ?
  new URLSearchParams(window.location.search) : null;
const testMode = urlParams?.get('testMode') === 'true';

// Player types configuration
const playerTypes = testMode ?
  ['human', 'human', 'human', 'human'] as ('human' | 'ai')[] :
  ['human', 'ai', 'ai', 'ai'] as ('human' | 'ai')[];

const DEFAULT_SESSION_ID = 'player-0';

// ============================================================================
// PRIVATE IMPLEMENTATION
// ============================================================================

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
    players: sessions
  };
}

/**
 * Private GameStore implementation - encapsulates all complexity.
 */
class GameStoreImpl {
  // Internal state
  private client?: NetworkGameClient;
  private unsubscribe?: (() => void) | undefined;

  // Internal stores
  private clientState = writable<MultiplayerGameState>(createPendingState());
  private actionsByPlayer = writable<Record<string, ValidAction[]>>({});
  private currentSessionIdStore = writable<string>(DEFAULT_SESSION_ID);
  private findingSeedStore = writable<boolean>(false);

  // Public reactive stores
  public readonly gameState: Readable<FilteredGameState>;
  public readonly viewProjection: Readable<ViewProjection>;
  public readonly currentPerspective: Readable<string>;
  public readonly availablePerspectives: Readable<Array<{ id: string; label: string }>>;

  constructor() {
    // Derived: current session
    const playerSessions = derived(this.clientState, $state => $state.players);

    const currentSession = derived(
      [playerSessions, this.currentSessionIdStore],
      ([$sessions, $sessionId]) => $sessions.find(s => s.playerId === $sessionId)
    );

    // Derived: gameState (filtered by current perspective)
    this.gameState = derived(
      [this.clientState, currentSession],
      ([$clientState, $session]) => {
        if ($session) {
          return getVisibleStateForSession($clientState.coreState, $session);
        }
        return convertToFilteredState($clientState.coreState);
      }
    );

    // Derived: allowed actions for current perspective
    const allowedActionsStore = derived(
      [this.actionsByPlayer, this.currentSessionIdStore],
      ([$actions, $sessionId]) => $actions[$sessionId] ?? $actions['__unfiltered__'] ?? []
    );

    // Derived: available perspectives
    this.availablePerspectives = derived(playerSessions, ($sessions) =>
      $sessions.map((session) => ({
        id: session.playerId,
        label: session.name ? `${session.name}` : `P${session.playerIndex}`
      }))
    );

    // Derived: current perspective ID
    this.currentPerspective = derived(this.currentSessionIdStore, (value) => value);

    // Derived: viewProjection (UI-ready projection)
    this.viewProjection = derived(
      [this.gameState, playerSessions, this.currentSessionIdStore, allowedActionsStore],
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

        const viewProjectionOptions = {
          isTestMode: testMode,
          canAct,
          isAIControlled: (player: number) => {
            const playerSession = $sessions.find(s => s.playerIndex === player);
            return playerSession?.controlType === 'ai';
          },
          ...(session ? { viewingPlayerIndex: session.playerIndex } : {})
        };

        return createViewProjection($gameState, usedTransitions, viewProjectionOptions);
      }
    );

    // Auto-correct perspective if current becomes invalid
    playerSessions.subscribe(($sessions) => {
      if ($sessions.length === 0) return;

      const current = get(this.currentSessionIdStore);
      const firstSession = $sessions[0];
      if (!$sessions.some(session => session.playerId === current) && firstSession) {
        void this.setPerspective(firstSession.playerId);
      }
    });

    // Initialize immediately
    void this.wireUpGame({
      playerTypes,
      shuffleSeed: Math.floor(Math.random() * 1000000)
    });
  }

  /**
   * Core method: Wire up Room + Transport + NetworkGameClient.
   * This is the ONLY place where Room/Transport/Client are created.
   */
  private async wireUpGame(config: GameConfig): Promise<void> {
    // Dynamic imports
    const { Room } = await import('../server/Room');
    const { InProcessTransport } = await import('../server/transports/InProcessTransport');

    // Create game ID
    const gameId = `game-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;

    // Create player sessions from config
    const sessions = (config.playerTypes ?? playerTypes).map((type, index) => {
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
        capabilities: capabilities.map((cap: Capability) => ({ ...cap }))
      };
    });

    // 1. Create Room
    const room = new Room(gameId, config, sessions);

    // 2. Create Transport
    const transport = new InProcessTransport();

    // 3. Wire them together
    room.setTransport(transport);
    transport.setRoom(room);

    // 4. Create connection from transport
    const connection = transport.connect('player-0');

    // 5. Create NetworkGameClient
    const newClient = new NetworkGameClient(connection, config);

    // 6. Install new client
    await this.installClient(newClient);
  }

  /**
   * Install a new NetworkGameClient and set up subscriptions.
   */
  private async installClient(newClient: NetworkGameClient): Promise<void> {
    // Unsubscribe from old client
    if (this.unsubscribe) {
      this.unsubscribe();
      this.unsubscribe = undefined;
    }

    // Store new client
    this.client = newClient;

    // Get initial state
    try {
      const state = await newClient.getState();
      this.clientState.set(state);
      this.actionsByPlayer.set(newClient.getCachedActionsMap());

      // Subscribe to updates
      this.unsubscribe = newClient.subscribe(state => {
        this.clientState.set(state);
        this.actionsByPlayer.set(newClient.getCachedActionsMap());
      });

      // Set perspective
      await this.setPerspective(DEFAULT_SESSION_ID);
    } catch (error) {
      console.error('Failed to initialize game client:', error);
    }
  }

  // ========================================================================
  // PUBLIC API
  // ========================================================================

  async executeAction(action: GameAction): Promise<void> {
    if (!this.client) throw new Error('Game not initialized');

    const session = get(this.currentPerspective);
    const sessions = get(this.clientState).players;
    const currentSession = sessions.find(s => s.playerId === session);

    if (!currentSession || !hasCapabilityType(currentSession, 'act-as-player')) {
      throw new Error('Current perspective cannot execute actions');
    }

    const preparedAction = { ...action } as GameAction;
    if ('player' in preparedAction) {
      preparedAction.player = currentSession.playerIndex;
    }

    const result = await this.client.executeAction({
      playerId: currentSession.playerId,
      action: preparedAction
    });

    if (!result.success) {
      console.error('Action failed:', result.error);
      throw new Error(result.error);
    }
  }

  async createGame(config?: GameConfig): Promise<void> {
    const finalConfig = config ?? {
      playerTypes,
      shuffleSeed: Math.floor(Math.random() * 1000000)
    };
    await this.wireUpGame(finalConfig);
  }

  async resetGame(): Promise<void> {
    await this.createGame();
  }

  async setPerspective(playerId: string): Promise<void> {
    const current = get(this.currentSessionIdStore);
    if (current !== playerId) {
      this.currentSessionIdStore.set(playerId);
    }

    if (!this.client) {
      throw new Error('Game client not yet initialized');
    }

    await this.client.setPlayerId(playerId);
    this.actionsByPlayer.set(this.client.getCachedActionsMap());
  }

  async setPlayerControl(playerIndex: number, type: 'human' | 'ai'): Promise<void> {
    if (!this.client) throw new Error('Game not initialized');
    await this.client.setPlayerControl(playerIndex, type);
  }

  async startOneHand(seed?: number): Promise<void> {
    const config: GameConfig = {
      playerTypes,
      variants: [{ type: 'one-hand', config: { seed } }],
      shuffleSeed: seed ?? Math.floor(Math.random() * 1000000)
    };

    this.findingSeedStore.set(false);
    await this.wireUpGame(config);
  }

  async retryOneHand(): Promise<void> {
    const currentState = get(this.clientState).coreState;
    if (!currentState.shuffleSeed) {
      throw new Error('No seed available to retry');
    }
    await this.startOneHand(currentState.shuffleSeed);
  }

  async exitOneHand(): Promise<void> {
    await this.resetGame();
  }

  // One-hand state (derived)
  get oneHandState() {
    return derived(this.gameState, ($gameState) => {
      const isComplete = $gameState.phase === 'scoring';
      const attempts = 1; // Could track this if needed

      return {
        active: false, // TODO: Track when in one-hand mode
        complete: isComplete,
        seed: $gameState.shuffleSeed,
        attempts,
        scores: isComplete ? {
          us: $gameState.teamScores[0],
          them: $gameState.teamScores[1],
          won: $gameState.teamScores[0] > $gameState.teamScores[1]
        } : null
      };
    });
  }

  get findingSeed() {
    return this.findingSeedStore;
  }
}

// ============================================================================
// SINGLETON INSTANCE
// ============================================================================

const store = new GameStoreImpl();

// ============================================================================
// PUBLIC EXPORTS (9 items total)
// ============================================================================

// Reactive state (5 stores)
export const gameState = store.gameState;
export const viewProjection = store.viewProjection;
export const currentPerspective = store.currentPerspective;
export const availablePerspectives = store.availablePerspectives;
export const oneHandState = store.oneHandState;

// Commands (1 object)
export const game = {
  executeAction: (action: GameAction) => store.executeAction(action),
  createGame: (config?: GameConfig) => store.createGame(config),
  resetGame: () => store.resetGame(),
  setPerspective: (playerId: string) => store.setPerspective(playerId),
  setPlayerControl: (playerIndex: number, type: 'human' | 'ai') => store.setPlayerControl(playerIndex, type)
};

// Special modes (1 object)
export const modes = {
  oneHand: {
    start: (seed?: number) => store.startOneHand(seed),
    retry: () => store.retryOneHand(),
    exit: () => store.exitOneHand()
  }
};

// Utility (2 exports)
export const findingSeed = store.findingSeed;

// Legacy exports for compatibility with main.ts
export const gameActions = game;
export const gameClient = store['client'];
