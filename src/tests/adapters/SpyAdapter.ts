import type {
  IGameAdapter,
  ClientMessage,
  ServerMessage,
} from '../../shared/multiplayer/protocol';

/**
 * Spy adapter that wraps another adapter and records all messages.
 *
 * This adapter is useful for protocol verification tests where you want to:
 * - Verify the client sends correct messages
 * - Verify the UI responds correctly to server messages
 * - Test the actual game logic (via wrapped InProcessAdapter)
 * - Assert on message sequences
 *
 * @example
 * ```typescript
 * const realAdapter = new InProcessAdapter();
 * const spy = new SpyAdapter(realAdapter);
 * const client = new NetworkGameClient(spy);
 *
 * await client.executeAction({ playerId: 'player-0', action: { type: 'bid', ... }, timestamp: Date.now() });
 *
 * // Verify protocol
 * expect(spy.getSentMessages()).toContainEqual({
 *   type: 'EXECUTE_ACTION',
 *   playerId: 'player-0',
 *   action: { type: 'bid', ... }
 * });
 * ```
 */
export class SpyAdapter implements IGameAdapter {
  private wrappedAdapter: IGameAdapter;
  private sentMessages: ClientMessage[] = [];
  private receivedMessages: ServerMessage[] = [];
  private messageHandlers = new Set<(message: ServerMessage) => void>();
  private unsubscribeFromWrapped?: () => void;

  constructor(adapter: IGameAdapter) {
    this.wrappedAdapter = adapter;
    this.setupSpying();
  }

  async send(message: ClientMessage): Promise<void> {
    this.sentMessages.push(message);
    await this.wrappedAdapter.send(message);
  }

  subscribe(handler: (message: ServerMessage) => void): () => void {
    this.messageHandlers.add(handler);
    return () => this.messageHandlers.delete(handler);
  }

  destroy(): void {
    if (this.unsubscribeFromWrapped) {
      this.unsubscribeFromWrapped();
    }
    this.wrappedAdapter.destroy();
    this.messageHandlers.clear();
  }

  isConnected(): boolean {
    return this.wrappedAdapter.isConnected();
  }

  getMetadata() {
    const metadata = this.wrappedAdapter.getMetadata?.();
    return {
      ...metadata,
      type: 'in-process' as const,
      isSpy: true,
    };
  }

  // === Test Utilities ===

  /**
   * Get all messages sent by the client to the server.
   */
  getSentMessages(): ClientMessage[] {
    return [...this.sentMessages];
  }

  /**
   * Get all messages received from the server.
   */
  getReceivedMessages(): ServerMessage[] {
    return [...this.receivedMessages];
  }

  /**
   * Get the last message sent by the client.
   */
  getLastSentMessage(): ClientMessage | undefined {
    return this.sentMessages[this.sentMessages.length - 1];
  }

  /**
   * Get the last message received from the server.
   */
  getLastReceivedMessage(): ServerMessage | undefined {
    return this.receivedMessages[this.receivedMessages.length - 1];
  }

  /**
   * Get all sent messages of a specific type.
   */
  getSentMessagesOfType<T extends ClientMessage['type']>(
    type: T
  ): Extract<ClientMessage, { type: T }>[] {
    return this.sentMessages.filter(msg => msg.type === type) as Extract<ClientMessage, { type: T }>[];
  }

  /**
   * Get all received messages of a specific type.
   */
  getReceivedMessagesOfType<T extends ServerMessage['type']>(
    type: T
  ): Extract<ServerMessage, { type: T }>[] {
    return this.receivedMessages.filter(msg => msg.type === type) as Extract<ServerMessage, { type: T }>[];
  }

  /**
   * Clear message history (useful for testing specific sequences).
   */
  clearMessageHistory(): void {
    this.sentMessages = [];
    this.receivedMessages = [];
  }

  /**
   * Wait for a specific server message type.
   * Resolves with the message when it arrives.
   */
  waitForMessage<T extends ServerMessage['type']>(
    type: T,
    timeoutMs = 5000
  ): Promise<Extract<ServerMessage, { type: T }>> {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error(`Timeout waiting for message type: ${type}`));
      }, timeoutMs);

      const handler = (message: ServerMessage) => {
        if (message.type === type) {
          clearTimeout(timeout);
          this.messageHandlers.delete(handler);
          resolve(message as Extract<ServerMessage, { type: T }>);
        }
      };

      this.messageHandlers.add(handler);
    });
  }

  /**
   * Wait for N messages to be received.
   */
  waitForMessageCount(count: number, timeoutMs = 5000): Promise<void> {
    if (this.receivedMessages.length >= count) {
      return Promise.resolve();
    }

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(
          new Error(
            `Timeout waiting for ${count} messages (received ${this.receivedMessages.length})`
          )
        );
      }, timeoutMs);

      const handler = () => {
        if (this.receivedMessages.length >= count) {
          clearTimeout(timeout);
          this.messageHandlers.delete(handler);
          resolve();
        }
      };

      this.messageHandlers.add(handler);
    });
  }

  /**
   * Get the wrapped adapter (useful for type-specific operations).
   */
  getWrappedAdapter(): IGameAdapter {
    return this.wrappedAdapter;
  }

  // === Private Methods ===

  private setupSpying(): void {
    // Subscribe to wrapped adapter to intercept messages
    this.unsubscribeFromWrapped = this.wrappedAdapter.subscribe(message => {
      this.receivedMessages.push(message);
      // Forward to our handlers
      for (const handler of this.messageHandlers) {
        try {
          handler(message);
        } catch (error) {
          console.error('SpyAdapter: Error in message handler:', error);
        }
      }
    });
  }
}
