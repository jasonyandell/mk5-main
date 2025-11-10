import type {
  ClientMessage,
  ServerMessage,
  GameView,
  ValidAction,
} from '../../shared/multiplayer/protocol';
import type { MultiplayerGameState, PlayerSession } from '../../game/multiplayer/types';
import type { Connection } from '../../server/transports/Transport';

/**
 * Mock connection for testing that provides pre-configured game states.
 *
 * This connection simulates server behavior without running any game logic.
 * It's useful for:
 * - Fast, deterministic UI tests
 * - Testing UI responses to specific game states
 * - Testing error handling
 * - Testing loading states
 *
 * @example
 * ```typescript
 * const biddingState = createBiddingPhaseView();
 * const connection = new MockConnection([biddingState]);
 * const client = new NetworkGameClient(connection);
 * // Client will receive biddingState when created
 * ```
 */
export class MockConnection implements Connection {
  private handlers = new Set<(message: ServerMessage) => void>();
  private states: GameView[];
  private currentStateIndex = 0;
  private gameId: string;
  private connected = true;
  private sentMessages: ClientMessage[] = [];
  private receivedMessages: ServerMessage[] = [];
  private autoAdvance: boolean;
  private simulateLatency: number;

  /**
   * Create a mock connection with pre-configured states.
   *
   * @param states - Array of GameView objects representing state sequence
   * @param options - Configuration options
   * @param options.gameId - Game ID to use (default: 'mock-game-123')
   * @param options.autoAdvance - Automatically advance to next state on action (default: true)
   * @param options.simulateLatency - Simulate network latency in ms (default: 0)
   */
  constructor(
    states: GameView[],
    options: {
      gameId?: string;
      autoAdvance?: boolean;
      simulateLatency?: number;
    } = {}
  ) {
    if (states.length === 0) {
      throw new Error('MockConnection requires at least one state');
    }

    this.states = states;
    this.gameId = options.gameId || 'mock-game-123';
    this.autoAdvance = options.autoAdvance !== false;
    this.simulateLatency = options.simulateLatency || 0;
  }

  send(message: ClientMessage): void {
    if (!this.connected) {
      throw new Error('MockConnection: Not connected');
    }

    this.sentMessages.push(message);

    // Simulate latency if configured (using setTimeout without await)
    const handleMessage = () => {
      switch (message.type) {
        case 'CREATE_GAME':
          this.handleCreateGame();
          break;

        case 'EXECUTE_ACTION':
          this.handleExecuteAction(message);
          break;

        case 'SET_PLAYER_CONTROL':
          this.handleSetPlayerControl(message);
          break;

        case 'SUBSCRIBE':
          // Mock connection automatically subscribes, no-op
          break;

        case 'UNSUBSCRIBE':
          // Mock connection doesn't support unsubscribe, no-op
          break;

        case 'JOIN_GAME':
          // Mock connection doesn't support multiplayer, no-op
          break;

        default:
          throw new Error(`MockConnection: Unsupported message type: ${(message as { type: string }).type}`);
      }
    };

    if (this.simulateLatency > 0) {
      setTimeout(handleMessage, this.simulateLatency);
    } else {
      handleMessage();
    }
  }

  onMessage(handler: (message: ServerMessage) => void): void {
    this.handlers.add(handler);
  }

  disconnect(): void {
    this.connected = false;
    this.handlers.clear();
  }

  // === Test Utilities ===

  /**
   * Get all messages sent by the client.
   */
  getSentMessages(): ClientMessage[] {
    return [...this.sentMessages];
  }

  /**
   * Get all messages received by the client.
   */
  getReceivedMessages(): ServerMessage[] {
    return [...this.receivedMessages];
  }

  /**
   * Clear message history (useful for testing specific sequences).
   */
  clearMessageHistory(): void {
    this.sentMessages = [];
    this.receivedMessages = [];
  }

  /**
   * Manually advance to the next state (useful when autoAdvance is false).
   */
  advanceState(): void {
    if (this.currentStateIndex < this.states.length - 1) {
      this.currentStateIndex++;
      const view = this.states[this.currentStateIndex]!;
      this.broadcast({
        type: 'STATE_UPDATE',
        gameId: this.gameId,
        view,
        state: this.viewToMultiplayerState(view),
        actions: this.buildActionsFromView(view),
        ...(view.players[0]?.sessionId !== undefined ? { perspective: view.players[0]!.sessionId! } : {})
      });
    }
  }

  /**
   * Jump to a specific state index.
   */
  setState(index: number): void {
    if (index < 0 || index >= this.states.length) {
      throw new Error(`Invalid state index: ${index} (max: ${this.states.length - 1})`);
    }
    this.currentStateIndex = index;
    const view = this.states[index]!;
    this.broadcast({
      type: 'STATE_UPDATE',
      gameId: this.gameId,
      view,
      state: this.viewToMultiplayerState(view),
      actions: this.buildActionsFromView(view),
      ...(view.players[0]?.sessionId !== undefined ? { perspective: view.players[0]!.sessionId! } : {})
    });
  }

  /**
   * Get the current state index.
   */
  getCurrentStateIndex(): number {
    return this.currentStateIndex;
  }

  /**
   * Check if we're at the last state.
   */
  isAtLastState(): boolean {
    return this.currentStateIndex === this.states.length - 1;
  }

  /**
   * Simulate a server error.
   */
  simulateError(code: string, message: string): void {
    this.broadcast({
      type: 'ERROR',
      error: `${code}: ${message}`,
      requestType: 'CREATE_GAME', // Default to CREATE_GAME for generic errors
    });
  }

  /**
   * Simulate disconnection.
   */
  simulateDisconnect(): void {
    this.connected = false;
  }

  /**
   * Simulate reconnection.
   */
  simulateReconnect(): void {
    this.connected = true;
  }

  // === Private Methods ===

  private handleCreateGame(): void {
    this.currentStateIndex = 0;
    const view = this.states[0]!;
    this.broadcast({
      type: 'GAME_CREATED',
      gameId: this.gameId,
      view,
      state: this.viewToMultiplayerState(view),
      actions: this.buildActionsFromView(view)
    });
  }

  private handleExecuteAction(_message: ClientMessage & { type: 'EXECUTE_ACTION' }): void {
    if (this.autoAdvance && this.currentStateIndex < this.states.length - 1) {
      this.currentStateIndex++;
      const view = this.states[this.currentStateIndex]!;
      this.broadcast({
        type: 'STATE_UPDATE',
        gameId: this.gameId,
        view,
        state: this.viewToMultiplayerState(view),
        actions: this.buildActionsFromView(view),
        ...(view.players[0]?.sessionId !== undefined ? { perspective: view.players[0]!.sessionId! } : {})
      });
    } else {
      // Stay on current state, just re-broadcast it
      const view = this.states[this.currentStateIndex]!;
      this.broadcast({
        type: 'STATE_UPDATE',
        gameId: this.gameId,
        view,
        state: this.viewToMultiplayerState(view),
        actions: this.buildActionsFromView(view),
        ...(view.players[0]?.sessionId !== undefined ? { perspective: view.players[0]!.sessionId! } : {})
      });
    }
  }

  private handleSetPlayerControl(message: ClientMessage & { type: 'SET_PLAYER_CONTROL' }): void {
    // Mock connection doesn't track player control changes, just acknowledge
    this.broadcast({
      type: 'PLAYER_STATUS',
      gameId: this.gameId,
      playerId: message.playerId,
      sessionId: `player-${message.playerId}`,
      status: 'control_changed',
      controlType: message.controlType,
      capabilities: []
    });
  }

  private viewToMultiplayerState(view: GameView): MultiplayerGameState {
    const players: PlayerSession[] = view.players.map(player => {
      const session: PlayerSession = {
        playerId: player.sessionId ?? `player-${player.playerId}`,
        playerIndex: player.playerId as 0 | 1 | 2 | 3,
        controlType: player.controlType,
        isConnected: player.connected,
        capabilities: player.capabilities?.map(cap => ({ ...cap })) ?? []
      };

      if (player.name !== undefined) {
        session.name = player.name;
      }

      return session;
    });

    return {
      gameId: view.metadata.gameId || this.gameId,
      coreState: view.state as unknown as MultiplayerGameState['coreState'],
      players
    };
  }

  private buildActionsFromView(view: GameView): Record<string, ValidAction[]> {
    const actions: Record<string, ValidAction[]> = {};
    const perspectiveActions = view.validActions.map(valid => ({
      ...valid,
      action: { ...valid.action }
    }));

    for (const player of view.players) {
      const sessionId = player.sessionId ?? `player-${player.playerId}`;
      actions[sessionId] = perspectiveActions;
    }

    actions['__unfiltered__'] = perspectiveActions;
    return actions;
  }

  private broadcast(message: ServerMessage): void {
    this.receivedMessages.push(message);
    for (const handler of this.handlers) {
      try {
        handler(message);
      } catch (error) {
        console.error('MockConnection: Error in message handler:', error);
      }
    }
  }
}
