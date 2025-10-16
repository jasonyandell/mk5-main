import type {
  IGameAdapter,
  ClientMessage,
  ServerMessage,
  GameView,
} from '../../shared/multiplayer/protocol';

/**
 * Mock adapter for testing that provides pre-configured game states.
 *
 * This adapter simulates server behavior without running any game logic.
 * It's useful for:
 * - Fast, deterministic UI tests
 * - Testing UI responses to specific game states
 * - Testing error handling
 * - Testing loading states
 *
 * @example
 * ```typescript
 * const biddingState = createBiddingPhaseView();
 * const adapter = new MockAdapter([biddingState]);
 * const client = new NetworkGameClient(adapter);
 * // Client will receive biddingState when created
 * ```
 */
export class MockAdapter implements IGameAdapter {
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
   * Create a mock adapter with pre-configured states.
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
      throw new Error('MockAdapter requires at least one state');
    }

    this.states = states;
    this.gameId = options.gameId || 'mock-game-123';
    this.autoAdvance = options.autoAdvance !== false;
    this.simulateLatency = options.simulateLatency || 0;
  }

  async send(message: ClientMessage): Promise<void> {
    if (!this.connected) {
      throw new Error('MockAdapter: Not connected');
    }

    this.sentMessages.push(message);

    // Simulate latency if configured
    if (this.simulateLatency > 0) {
      await new Promise(resolve => setTimeout(resolve, this.simulateLatency));
    }

    switch (message.type) {
      case 'CREATE_GAME':
        await this.handleCreateGame();
        break;

      case 'EXECUTE_ACTION':
        await this.handleExecuteAction(message);
        break;

      case 'SET_PLAYER_CONTROL':
        await this.handleSetPlayerControl(message);
        break;

      case 'SUBSCRIBE':
        // Mock adapter automatically subscribes, no-op
        break;

      case 'UNSUBSCRIBE':
        // Mock adapter doesn't support unsubscribe, no-op
        break;

      case 'JOIN_GAME':
        // Mock adapter doesn't support multiplayer, no-op
        break;

      default:
        throw new Error(`MockAdapter: Unsupported message type: ${(message as any).type}`);
    }
  }

  subscribe(handler: (message: ServerMessage) => void): () => void {
    this.handlers.add(handler);
    return () => this.handlers.delete(handler);
  }

  destroy(): void {
    this.connected = false;
    this.handlers.clear();
  }

  isConnected(): boolean {
    return this.connected;
  }

  getMetadata() {
    return {
      type: 'in-process' as const,
      gameId: this.gameId,
    };
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
      this.broadcast({
        type: 'STATE_UPDATE',
        gameId: this.gameId,
        view: this.states[this.currentStateIndex]!,
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
    this.broadcast({
      type: 'STATE_UPDATE',
      gameId: this.gameId,
      view: this.states[index]!,
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

  private async handleCreateGame(): Promise<void> {
    this.currentStateIndex = 0;
    this.broadcast({
      type: 'GAME_CREATED',
      gameId: this.gameId,
      view: this.states[0]!,
    });
  }

  private async handleExecuteAction(_message: ClientMessage & { type: 'EXECUTE_ACTION' }): Promise<void> {
    if (this.autoAdvance && this.currentStateIndex < this.states.length - 1) {
      this.currentStateIndex++;
      this.broadcast({
        type: 'STATE_UPDATE',
        gameId: this.gameId,
        view: this.states[this.currentStateIndex]!,
      });
    } else {
      // Stay on current state, just re-broadcast it
      this.broadcast({
        type: 'STATE_UPDATE',
        gameId: this.gameId,
        view: this.states[this.currentStateIndex]!,
      });
    }
  }

  private async handleSetPlayerControl(message: ClientMessage & { type: 'SET_PLAYER_CONTROL' }): Promise<void> {
    // Mock adapter doesn't track player control changes, just acknowledge
    this.broadcast({
      type: 'PLAYER_STATUS',
      gameId: this.gameId,
      playerId: message.playerId,
      status: 'control_changed',
      controlType: message.controlType,
    });
  }

  private broadcast(message: ServerMessage): void {
    this.receivedMessages.push(message);
    for (const handler of this.handlers) {
      try {
        handler(message);
      } catch (error) {
        console.error('MockAdapter: Error in message handler:', error);
      }
    }
  }
}
