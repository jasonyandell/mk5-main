import type {
  ClientMessage,
  ServerMessage,
} from '../../shared/multiplayer/protocol';
import type { Connection } from '../../server/transports/Transport';

/**
 * Spy connection that wraps another connection and records all messages.
 *
 * This connection is useful for protocol verification tests where you want to:
 * - Verify the client sends correct messages
 * - Verify the UI responds correctly to server messages
 * - Test the actual game logic (via wrapped connection implementation)
 * - Assert on message sequences
 *
 * @example
 * ```typescript
 * const transport = new InProcessTransport();
 * const connection = transport.connect('player-0');
 * const spy = new SpyConnection(connection);
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
export class SpyConnection implements Connection {
  private wrappedConnection: Connection;
  private sentMessages: ClientMessage[] = [];
  private receivedMessages: ServerMessage[] = [];
  private messageHandlers = new Set<(message: ServerMessage) => void>();

  constructor(connection: Connection) {
    this.wrappedConnection = connection;
    this.setupSpying();
  }

  send(message: ClientMessage): void {
    this.sentMessages.push(message);
    this.wrappedConnection.send(message);
  }

  onMessage(handler: (message: ServerMessage) => void): void {
    this.messageHandlers.add(handler);
  }

  disconnect(): void {
    this.wrappedConnection.disconnect();
    this.messageHandlers.clear();
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
   * Get the wrapped connection (useful for type-specific operations).
   */
  getWrappedConnection(): Connection {
    return this.wrappedConnection;
  }

  // === Private Methods ===

  private setupSpying(): void {
    // Subscribe to wrapped connection to intercept messages
    this.wrappedConnection.onMessage(message => {
      this.receivedMessages.push(message);
      // Forward to our handlers
      for (const handler of this.messageHandlers) {
        try {
          handler(message);
        } catch (error) {
          console.error('SpyConnection: Error in message handler:', error);
        }
      }
    });
  }
}
