import { writable, derived, get, type Readable } from 'svelte/store';
import type { GameAction, FilteredGameState } from '../game/types';
import type { GameConfig } from '../game/types/config';
import { createViewProjection, type ViewProjection } from '../game/view-projection';
import { createSetupState } from '../game/core/state';
import type { GameView, PlayerInfo } from '../multiplayer/types';
import { decodeGameUrl, stateToUrl } from '../game/core/url-compression';
import { createLocalGame, type LocalGame } from '../multiplayer/local';
import { GameClient } from '../multiplayer/GameClient';
import type { Room } from '../server/Room';
import { resolveActionIds } from '../game/core/action-resolution';

/**
 * GameStore - Clean facade over the simplified Room/Socket/GameClient architecture.
 *
 * Philosophy:
 * - Uses createLocalGame() for all game wiring
 * - Fire-and-forget actions, updates via subscription
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
      handCount: player.hand.length
    }))
  };

  return {
    state: filteredState,
    validActions: [],
    transitions: [],
    players,
    metadata: {
      gameId: 'initializing'
    },
    derived: {
      currentTrickWinner: -1,
      handDominoMeta: [],
      currentHandPoints: [0, 0] as [number, number]
    }
  };
}

/**
 * Private GameStore implementation - encapsulates all complexity.
 */
class GameStoreImpl {
  // Internal state - new simplified architecture
  private localGame?: LocalGame;
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
    // Uses transitions and derived fields from GameView (server-computed, dumb client pattern)
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
          newState: $view.state  // Placeholder - not used by createViewProjection
        }));

        const viewProjectionOptions = {
          isTestMode: testMode,
          canAct,
          isAIControlled: (player: number) => {
            const pInfo = $view.players.find(p => p.playerId === player);
            return pInfo?.controlType === 'ai';
          },
          // Pass server-computed derived fields (dumb client pattern)
          derived: $view.derived,
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

    // Reactive URL sync: keep browser URL in sync with game state
    // Uses replaceState to avoid polluting browser history
    this.clientView.subscribe(($view) => {
      if (typeof window === 'undefined') return;
      if ($view.metadata.gameId === 'initializing') return;

      // Generate URL from current state
      const url = stateToUrl($view.state);
      const fullUrl = window.location.pathname + url;

      // Update browser URL without adding history entry
      window.history.replaceState(
        { timestamp: Date.now() },
        '',
        fullUrl
      );
    });

    // Initialize from URL or with default config
    void this.initializeFromURL();
  }

  /**
   * Initialize game from URL parameters or create new game
   */
  private async initializeFromURL(): Promise<void> {
    try {
      // Check if we have URL parameters to load from
      if (typeof window === 'undefined') {
        // SSR - initialize with default config
        this.wireUpGame({
          playerTypes,
          shuffleSeed: Math.floor(Math.random() * 1000000)
        });
        return;
      }

      const urlParams = new URLSearchParams(window.location.search);
      const hasSeed = urlParams.has('s');
      const hasInitialHands = urlParams.has('i');

      if (hasSeed || hasInitialHands) {
        // URL has game state - decode and initialize
        const urlData = decodeGameUrl(window.location.search);

        // Build config from URL data
        // initialHands and seed are mutually exclusive
        const config: GameConfig = {
          playerTypes: urlData.playerTypes,
          ...(urlData.initialHands
            ? { dealOverrides: { initialHands: urlData.initialHands } }
            : { shuffleSeed: urlData.seed || Math.floor(Math.random() * 1000000) }),
          ...(urlData.dealer !== undefined && urlData.dealer !== 3 ? { dealer: urlData.dealer } : {}),
          ...(urlData.theme ? { theme: urlData.theme } : {}),
          ...(urlData.colorOverrides && Object.keys(urlData.colorOverrides).length > 0
            ? { colorOverrides: urlData.colorOverrides }
            : {}),
          ...(urlData.layers ? { layers: urlData.layers } : {})
        };

        // Initialize game with URL config and replay actions if present
        this.wireUpGame(config, urlData.actions.length > 0 ? urlData.actions : undefined);
      } else {
        // No URL state - initialize with default config
        this.wireUpGame({
          playerTypes,
          shuffleSeed: Math.floor(Math.random() * 1000000)
        });
      }
    } catch (error) {
      console.error('Failed to initialize from URL:', error);
      // Fallback to default initialization
      this.wireUpGame({
        playerTypes,
        shuffleSeed: Math.floor(Math.random() * 1000000)
      });
    }
  }

  /**
   * Core method: Create game using new simplified architecture.
   * Uses createLocalGame() which handles Room + Socket wiring.
   * @param config - Game configuration
   * @param actionIds - Optional action IDs to replay after Room creation
   */
  private wireUpGame(config: GameConfig, actionIds?: string[]): void {
    // Clean up old game
    if (this.unsubscribe) {
      this.unsubscribe();
      this.unsubscribe = undefined;
    }
    if (this.localGame) {
      this.localGame.client.disconnect();
    }

    // Determine AI configuration based on test mode
    // In test mode, all players are human (no AI)
    const configPlayerTypes = config.playerTypes ?? playerTypes;
    const aiPlayerIndexes = configPlayerTypes
      .map((type, index) => type === 'ai' ? index : -1)
      .filter(index => index >= 0);

    // Create new game using simplified architecture
    // If replaying actions, skip AI behavior until after replay completes
    const hasActionsToReplay = actionIds !== undefined && actionIds.length > 0;
    this.localGame = createLocalGame(config, {
      aiPlayerIndexes,
      skipAIBehavior: hasActionsToReplay
    });

    // Replay actions if provided (before subscribing to avoid intermediate updates)
    if (hasActionsToReplay) {
      this.replayActionsInRoom(this.localGame.room, actionIds, config);
      // Now attach AI behavior so they see the post-replay state
      this.localGame.attachAI();
    }

    // Subscribe to view updates
    this.installClient(this.localGame.client);

    // Send JOIN to associate with player-0
    this.localGame.client.send({ type: 'JOIN', playerIndex: 0, name: 'Player 1' });
  }

  /**
   * Install a GameClient and set up subscriptions.
   */
  private installClient(client: GameClient): void {
    // Unsubscribe from old client
    if (this.unsubscribe) {
      this.unsubscribe();
      this.unsubscribe = undefined;
    }

    // Subscribe to updates
    this.unsubscribe = client.subscribe((view: GameView) => {
      this.clientView.set(view);
    });

    // Set initial view if available
    if (client.view) {
      this.clientView.set(client.view);
    }
  }

  /**
   * Replay actions directly in the Room (before client connection).
   * Uses the same approach as HeadlessRoom replay for deterministic state restoration.
   */
  private replayActionsInRoom(room: Room, actionIds: string[], config: GameConfig): void {
    try {
      const seed = config.shuffleSeed;
      if (!seed) {
        console.error('[gameStore] Cannot replay actions: no shuffle seed in config');
        return;
      }

      // Resolve action IDs to GameActions (uses HeadlessRoom internally)
      const actions = resolveActionIds(actionIds, config, seed);

      // Execute each action directly in the Room
      for (let i = 0; i < actions.length; i++) {
        const action = actions[i];
        if (!action) {
          console.error('[gameStore] Action is undefined at index', i);
          break;
        }

        // Determine which player should execute this action
        const playerIndex = this.getPlayerIndexForAction(action);
        const playerId = `player-${playerIndex}`;

        // Execute the action in the Room
        const result = room.executeAction(playerId, action);

        if (!result.success) {
          console.error('[gameStore] Action replay failed:', result.error);
          break;
        }
      }
    } catch (error) {
      console.error('[gameStore] Failed to replay actions:', error);
    }
  }

  /**
   * Get the player index that should execute a given action.
   */
  private getPlayerIndexForAction(action: GameAction): number {
    // Actions with explicit player field
    if ('player' in action && typeof action.player === 'number') {
      return action.player;
    }

    // Consensus actions can be executed by any player
    // For URL replay, we use player 0
    if (action.type === 'complete-trick' ||
        action.type === 'score-hand' ||
        action.type === 'redeal') {
      return 0;
    }

    // Default to player 0
    return 0;
  }

  // ========================================================================
  // PUBLIC API
  // ========================================================================

  /**
   * Execute a game action. Fire-and-forget - result comes via subscription.
   */
  executeAction(action: GameAction): void {
    if (!this.localGame) throw new Error('Game not initialized');

    const view = get(this.clientView);
    const sessionId = get(this.currentPerspective);
    const currentPlayer = view.players.find(p => (p.sessionId ?? `player-${p.playerId}`) === sessionId);

    if (!currentPlayer || !currentPlayer.capabilities?.some(cap => cap.type === 'act-as-player')) {
      throw new Error('Current perspective cannot execute actions');
    }

    const preparedAction = { ...action } as GameAction;
    if ('player' in preparedAction) {
      preparedAction.player = currentPlayer.playerId;
    }

    // Fire-and-forget - result comes via subscription
    this.localGame.client.send({ type: 'EXECUTE_ACTION', action: preparedAction });
  }

  /**
   * Create a new game with optional config.
   */
  createGame(config?: GameConfig): void {
    const finalConfig = config ?? {
      playerTypes,
      shuffleSeed: Math.floor(Math.random() * 1000000)
    };
    this.wireUpGame(finalConfig);
  }

  /**
   * Reset the game with default config.
   */
  resetGame(): void {
    this.createGame();
  }

  /**
   * Change the current viewing perspective.
   * Sends JOIN message to Room to switch player association.
   */
  setPerspective(playerId: string): void {
    const current = get(this.currentSessionIdStore);
    if (current !== playerId) {
      this.currentSessionIdStore.set(playerId);
    }

    if (!this.localGame) {
      throw new Error('Game client not yet initialized');
    }

    // Extract player index from playerId (e.g., "player-2" -> 2)
    const match = playerId.match(/player-(\d+)/);
    if (match && match[1]) {
      const playerIndex = parseInt(match[1], 10);
      this.localGame.client.send({ type: 'JOIN', playerIndex, name: `Player ${playerIndex + 1}` });
    }
  }

  /**
   * Change a player's control type (human/AI).
   */
  setPlayerControl(playerIndex: number, type: 'human' | 'ai'): void {
    if (!this.localGame) throw new Error('Game not initialized');
    this.localGame.client.send({ type: 'SET_CONTROL', playerIndex, controlType: type });
  }

  /**
   * Start one-hand mode with optional seed.
   */
  startOneHand(seed?: number): void {
    // oneHand layer handles bidding automation, terminal phase, and action generation
    const config: GameConfig = {
      playerTypes,
      layers: ['oneHand'],
      shuffleSeed: seed ?? Math.floor(Math.random() * 1000000)
    };

    this.findingSeedStore.set(false);
    this.wireUpGame(config);
  }

  /**
   * Retry one-hand mode with the same seed.
   */
  retryOneHand(): void {
    const view = get(this.clientView);
    if (!view.state.shuffleSeed) {
      throw new Error('No seed available to retry');
    }
    this.startOneHand(view.state.shuffleSeed);
  }

  /**
   * Exit one-hand mode and return to normal game.
   */
  exitOneHand(): void {
    this.resetGame();
  }

  // One-hand state (derived)
  get oneHandState() {
    return derived(this.gameState, ($gameState) => {
      // Check for one-hand-complete phase (terminal state set by oneHandLayer)
      const isComplete = $gameState.phase === 'one-hand-complete';
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

  // Internal accessor for testing
  get client(): GameClient | undefined {
    return this.localGame?.client;
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
  return store.client;
}

/**
 * Get current game view for E2E testing.
 * Returns the ViewProjection with additional state information.
 * @internal
 */
export function getGameView() {
  const view = get(store['clientView']);
  const projection = get(store.viewProjection);

  return {
    ...projection,
    state: view.state,
    isProcessing: projection.ui.isAIThinking,
    validActions: view.validActions,
    players: view.players
  };
}
