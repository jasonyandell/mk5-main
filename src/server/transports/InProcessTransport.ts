/**
 * InProcessTransport - Local, synchronous transport for in-browser game engine.
 *
 * This transport runs in the same process/thread as the game logic.
 * Perfect for local development and single-user testing.
 *
 * Message flow:
 * Client → Transport.connect() → receives Connection
 * Client sends: connection.send(ClientMessage)
 *   → Transport routes to GameServer.handleMessage(clientId, message)
 * GameServer broadcasts: transport.send(clientId, ServerMessage)
 *   → Transport calls registered onMessage handler for that client
 * Client receives via: onMessage handler
 */

import type { ClientMessage, ServerMessage, IGameAdapter } from '../../shared/multiplayer/protocol';
import type { Transport, Connection } from './Transport';

export class InProcessTransport implements Transport {
  private gameServer?: import('../GameServer').GameServer | undefined;
  private clients: Map<string, (message: ServerMessage) => void> = new Map();
  private connectedClients: Set<string> = new Set();

  /**
   * Set the GameServer reference so we can route messages to it.
   * Called during initialization to establish the server connection.
   */
  setGameServer(server: import('../GameServer').GameServer): void {
    this.gameServer = server;
  }

  /**
   * Create an IGameAdapter that wraps a Connection.
   * This allows legacy code that expects IGameAdapter to work with the new Transport.
   */
  createAdapter(clientId: string): IGameAdapter {
    const connection = this.connect(clientId);
    return new ConnectionAdapter(connection);
  }

  /**
   * Client connects to this transport.
   * Returns a Connection object with send/onMessage/disconnect methods.
   *
   * This is how clients get their bidirectional communication channel.
   */
  connect(clientId: string): Connection {
    this.connectedClients.add(clientId);

    return {
      /**
       * Client sends a message through this method.
       * Transport routes it to GameServer.
       */
      send: (message: ClientMessage) => {
        if (!this.gameServer) {
          console.error('InProcessTransport.send: GameServer not set');
          return;
        }
        this.gameServer.handleMessage(clientId, message);
      },

      /**
       * Client registers a message handler with this method.
       * We save it so we can call it when server broadcasts.
       */
      onMessage: (handler: (message: ServerMessage) => void) => {
        this.clients.set(clientId, handler);
      },

      /**
       * Client disconnects with this method.
       * Notifies server and cleans up.
       */
      disconnect: () => {
        this.clients.delete(clientId);
        this.connectedClients.delete(clientId);
        if (this.gameServer) {
          this.gameServer.handleClientDisconnect(clientId);
        }
      }
    };
  }

  /**
   * GameServer calls this to send a message to a specific client.
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
   * Called by GameServer when it needs to notify everyone.
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
    this.gameServer = undefined;
  }
}

/**
 * Adapter that wraps a Connection to provide IGameAdapter interface.
 * This allows NetworkGameClient to work with Transport connections.
 */
class ConnectionAdapter implements IGameAdapter {
  private connection: Connection;
  private messageHandlers = new Set<(message: ServerMessage) => void>();
  private isConnected_ = true;

  constructor(connection: Connection) {
    this.connection = connection;
    // Register our handler to forward messages to all subscribers
    this.connection.onMessage((message) => {
      for (const handler of this.messageHandlers) {
        handler(message);
      }
    });
  }

  async send(message: ClientMessage): Promise<void> {
    this.connection.send(message);
  }

  subscribe(handler: (message: ServerMessage) => void): () => void {
    this.messageHandlers.add(handler);
    return () => {
      this.messageHandlers.delete(handler);
    };
  }

  destroy(): void {
    this.connection.disconnect();
    this.messageHandlers.clear();
    this.isConnected_ = false;
  }

  isConnected(): boolean {
    return this.isConnected_;
  }

  getMetadata?() {
    return {
      type: 'in-process' as const,
      latency: 0
    };
  }
}
