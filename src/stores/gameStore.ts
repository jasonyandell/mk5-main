import { writable, derived, get, type Readable } from 'svelte/store';
import type { GameAction, FilteredGameState } from '../game/types';
import { NetworkGameClient } from '../game/multiplayer/NetworkGameClient';
import type { Capability } from '../game/multiplayer/types';
import type { GameConfig } from '../game/types/config';
import { createViewProjection, type ViewProjection } from '../game/view-projection';
import { createSetupState } from '../game/core/state';
import { humanCapabilities, aiCapabilities } from '../game/multiplayer/capabilities';
import type { GameView, PlayerInfo } from '../shared/multiplayer/protocol';

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

function createPendingView(): GameView {
  const placeholderState = createSetupState({ playerTypes });

  const players: PlayerInfo[] = playerTypes.map((type, index) => ({
    playerId: index,
    controlType: type,
    sessionId: `${type === 'human' ? 'player' : 'ai'}-${index}`,
    connected: true,
    name: `Player ${index + 1}`,
    capabilities: []
  }));

  const filteredState: FilteredGameState = {
    ...placeholderState,
    players: placeholderState.players.map(player => ({
      ...player,
      hand: [...player.hand],
      handCount: player.hand.length,
      ...(player.suitAnalysis ? { suitAnalysis: player.suitAnalysis } : {})
    }))
  };

  return {
    state: filteredState,
    validActions: [],
    transitions: [],
    players,
    metadata: {
      gameId: 'initializing'
    }
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
  private clientView = writable<GameView>(createPendingView());
  private currentSessionIdStore = writable<string>(DEFAULT_SESSION_ID);
  private findingSeedStore = writable<boolean>(false);

  // Public reactive stores
  public readonly gameState: Readable<FilteredGameState>;
  public readonly viewProjection: Readable<ViewProjection>;
  public readonly currentPerspective: Readable<string>;
  public readonly availablePerspectives: Readable<Array<{ id: string; label: string }>>;

  constructor() {
    // Derived: game state from view
    this.gameState = derived(this.clientView, $view => $view.state);

    // Derived: available perspectives
    this.availablePerspectives = derived(this.clientView, ($view) =>
      $view.players.map((player) => ({
        // PlayerInfo.sessionId is protocol-defined as optional to support different
        // session identification schemes. Fall back to player index for perspectives.
        id: player.sessionId ?? `player-${player.playerId}`,
        label: player.name ? `${player.name}` : `P${player.playerId}`
      }))
    );

    // Derived: current perspective ID
    this.currentPerspective = derived(this.currentSessionIdStore, (value) => value);

    // Derived: viewProjection (UI-ready projection)
    // Uses transitions from GameView (server-computed)
    this.viewProjection = derived(
      [this.clientView, this.currentSessionIdStore],
      ([$view, $sessionId]) => {
        const playerInfo = $view.players.find(p => (p.sessionId ?? `player-${p.playerId}`) === $sessionId);
        const canAct = !!(playerInfo && playerInfo.capabilities?.some(cap => cap.type === 'act-as-player'));

        // Convert ViewTransitions to StateTransitions for createViewProjection
        const stateTransitions = $view.transitions.map(t => ({
          id: t.id,
          label: t.label,
          action: t.action,
          newState: $view.state as any  // Placeholder - not used by createViewProjection
        }));

        const viewProjectionOptions = {
          isTestMode: testMode,
          canAct,
          isAIControlled: (player: number) => {
            const playerInfo = $view.players.find(p => p.playerId === player);
            return playerInfo?.controlType === 'ai';
          },
          ...(playerInfo ? { viewingPlayerIndex: playerInfo.playerId } : {})
        };

        return createViewProjection($view.state, stateTransitions, viewProjectionOptions);
      }
    );

    // Auto-correct perspective if current becomes invalid
    this.clientView.subscribe(($view) => {
      if ($view.players.length === 0) return;

      const current = get(this.currentSessionIdStore);
      const firstPlayer = $view.players[0];
      const firstSessionId = firstPlayer?.sessionId ?? `player-${firstPlayer?.playerId}`;
      if (!$view.players.some(p => (p.sessionId ?? `player-${p.playerId}`) === current) && firstPlayer) {
        void this.setPerspective(firstSessionId);
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

    // Get initial view
    try {
      const view = await newClient.getView();
      this.clientView.set(view);

      // Subscribe to updates
      this.unsubscribe = newClient.subscribe(view => {
        this.clientView.set(view);
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

    const sessionId = get(this.currentPerspective);
    const view = get(this.clientView);
    const currentPlayer = view.players.find(p => (p.sessionId ?? `player-${p.playerId}`) === sessionId);

    if (!currentPlayer || !currentPlayer.capabilities?.some(cap => cap.type === 'act-as-player')) {
      throw new Error('Current perspective cannot execute actions');
    }

    const preparedAction = { ...action } as GameAction;
    if ('player' in preparedAction) {
      preparedAction.player = currentPlayer.playerId;
    }

    const result = await this.client.executeAction({
      playerId: sessionId,
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
  }

  async setPlayerControl(playerIndex: number, type: 'human' | 'ai'): Promise<void> {
    if (!this.client) throw new Error('Game not initialized');
    await this.client.setPlayerControl(playerIndex, type);
  }

  async startOneHand(seed?: number): Promise<void> {
    const config: GameConfig = {
      playerTypes,
      actionTransformers: [{ type: 'one-hand', config: { seed } }],
      shuffleSeed: seed ?? Math.floor(Math.random() * 1000000)
    };

    this.findingSeedStore.set(false);
    await this.wireUpGame(config);
  }

  async retryOneHand(): Promise<void> {
    const view = get(this.clientView);
    if (!view.state.shuffleSeed) {
      throw new Error('No seed available to retry');
    }
    await this.startOneHand(view.state.shuffleSeed);
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

/**
 * Internal client accessor for window API development/testing tools only.
 * DO NOT use in application code - use the `game` commands instead.
 * @internal
 */
export function getInternalClient() {
  return store['client'];
}
