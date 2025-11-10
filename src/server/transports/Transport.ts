/**
 * Transport Abstraction Layer
 *
 * Defines the interface for different transport mechanisms:
 * - InProcessTransport: In-browser, synchronous (local development)
 * - WorkerTransport: Web Worker communication (future)
 * - CloudflareTransport: Durable Objects via WebSocket (future)
 *
 * Transport is responsible for:
 * - Client connections and disconnections
 * - Message routing between clients and Room
 * - Broadcasting server messages to clients
 *
 * Transport is NOT responsible for:
 * - Game logic
 * - AI lifecycle
 * - Protocol message validation
 */

import type { ClientMessage, ServerMessage } from '../../shared/multiplayer/protocol';

/**
 * Connection object returned when a client connects to transport.
 * Represents the bidirectional communication channel for one client.
 */
export interface Connection {
  /**
   * Send a message from client to server.
   * Called when client wants to send a message.
   */
  send: (message: ClientMessage) => void;

  /**
   * Register a handler for messages from server to client.
   * Transport will call this handler when server broadcasts to this client.
   */
  onMessage: (handler: (message: ServerMessage) => void) => void;

  /**
   * Disconnect this client from the transport.
   * Notifies transport that client is gone.
   */
  disconnect: () => void;
}

/**
 * Transport interface that Room uses.
 * Abstracts away how messages are sent to clients.
 */
export interface Transport {
  /**
   * Send a message to a specific client.
   * Called by Room to broadcast state updates.
   */
  send(clientId: string, message: ServerMessage): void;

  /**
   * Create a client connection and return a Connection object.
   * Used for creating connections for AI clients.
   */
  connect(clientId: string): Connection;

  /**
   * Lifecycle: Start the transport (initialize, open ports, etc).
   * Called once when transport is ready to accept connections.
   */
  start(): Promise<void>;

  /**
   * Lifecycle: Stop the transport (cleanup, close ports, etc).
   * Called when shutting down.
   */
  stop(): Promise<void>;
}

/**
 * Server-side interface for transport implementations.
 * Allows Room to interact with transport for routing and connections.
 */
export interface TransportServer {
  /**
   * Get all connected client IDs.
   */
  getConnectedClients(): string[];

  /**
   * Check if a specific client is connected.
   */
  isConnected(clientId: string): boolean;

  /**
   * Disconnect a specific client (from server side).
   */
  disconnectClient(clientId: string): void;

  /**
   * Broadcast to all connected clients.
   */
  broadcast(message: ServerMessage): void;
}
