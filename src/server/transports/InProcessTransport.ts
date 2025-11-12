/**
 * InProcessTransport - Local, synchronous transport for in-browser game engine.
 *
 * This transport runs in the same process/thread as the game logic.
 * Perfect for local development and single-user testing.
 *
 * Message flow:
 * Client → Transport.connect() → receives Connection
 * Client sends: connection.send(ClientMessage)
 *   → Transport routes to Room.handleMessage(clientId, message)
 * Room broadcasts: transport.send(clientId, ServerMessage)
 *   → Transport calls registered onMessage handler for that client
 * Client receives via: onMessage handler
 */

import type { ClientMessage, ServerMessage } from '../../shared/multiplayer/protocol';
import type { Transport, Connection } from './Transport';

export class InProcessTransport implements Transport {
  private room?: import('../Room').Room | undefined;
  private clients: Map<string, (message: ServerMessage) => void> = new Map();
  private connectedClients: Set<string> = new Set();

  /**
   * Set the Room reference so we can route messages to it.
   * Called during initialization to establish the server connection.
   */
  setRoom(server: import('../Room').Room): void {
    this.room = server;
  }

  /**
   * Client connects to this transport.
   * Returns a Connection object with send/onMessage/reply/disconnect methods.
   *
   * This is how clients get their bidirectional communication channel.
   */
  connect(clientId: string): Connection {
    this.connectedClients.add(clientId);

    const connection: Connection = {
      /**
       * Client sends a message through this method.
       * Transport routes it to Room.
       *
       * CRITICAL: Uses queueMicrotask to break synchronous call chain.
       * Without this, consensus actions cause stack overflow:
       * - Consensus actions (complete-trick, score-hand, redeal) have no player field
       * - All AI clients see them as available and execute simultaneously
       * - Synchronous delivery creates exponential broadcast loop
       * - Async delivery ensures messages are processed sequentially
       */
      send: (message: ClientMessage) => {
        queueMicrotask(() => {
          if (!this.room) {
            throw new Error('InProcessTransport: Room not set');
          }
          this.room.handleMessage(clientId, message, connection);
        });
      },

      /**
       * Client registers a message handler with this method.
       * We save it so we can call it when server broadcasts.
       */
      onMessage: (handler: (message: ServerMessage) => void) => {
        this.clients.set(clientId, handler);
      },

      /**
       * Server replies to this specific client.
       * Connection knows how to deliver messages to itself.
       *
       * CRITICAL: Uses queueMicrotask to break synchronous call chain.
       * Matches the async behavior of real transports (Cloudflare Workers, WebSocket).
       */
      reply: (message: ServerMessage) => {
        queueMicrotask(() => {
          const handler = this.clients.get(clientId);
          if (handler) {
            handler(message);
          }
        });
      },

      /**
       * Client disconnects with this method.
       * Notifies server and cleans up.
       */
      disconnect: () => {
        this.clients.delete(clientId);
        this.connectedClients.delete(clientId);
        if (this.room) {
          this.room.handleClientDisconnect(clientId);
        }
      }
    };

    return connection;
  }

  /**
   * Room calls this to send a message to a specific client.
   * We look up the client's registered handler and call it.
   */
  send(clientId: string, message: ServerMessage): void {
    const handler = this.clients.get(clientId);
    if (handler) {
      handler(message);
    }
  }

  /**
   * Broadcast a message to all connected clients.
   * Called by Room when it needs to notify everyone.
   */
  broadcast(message: ServerMessage): void {
    for (const handler of this.clients.values()) {
      handler(message);
    }
  }

  /**
   * Get list of currently connected client IDs.
   */
  getConnectedClients(): string[] {
    return Array.from(this.connectedClients);
  }

  /**
   * Check if a specific client is connected.
   */
  isConnected(clientId: string): boolean {
    return this.connectedClients.has(clientId);
  }

  /**
   * Disconnect a client from the server side.
   */
  disconnectClient(clientId: string): void {
    this.clients.delete(clientId);
    this.connectedClients.delete(clientId);
  }

  /**
   * Start the transport.
   * For InProcess, this is a no-op since we're synchronous.
   */
  async start(): Promise<void> {
    // Nothing to do for in-process transport
  }

  /**
   * Stop the transport.
   * For InProcess, we just clear all connections.
   */
  async stop(): Promise<void> {
    this.clients.clear();
    this.connectedClients.clear();
    this.room = undefined;
  }
}
