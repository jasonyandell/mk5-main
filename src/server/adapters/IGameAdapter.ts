/**
 * IGameAdapter - Transport interface for game communication.
 *
 * This interface abstracts the transport layer, allowing the same
 * protocol to work over different transports (in-process, Worker, WebSocket).
 *
 * Implementations handle:
 * - Message serialization (if needed)
 * - Connection management
 * - Subscription/broadcasting
 *
 * The interface is defined in shared/multiplayer/protocol.ts
 */

// Re-export from protocol
export type { IGameAdapter } from '../../shared/multiplayer/protocol';

/**
 * Factory type for creating adapters
 */
import type { IGameAdapter } from '../../shared/multiplayer/protocol';
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type AdapterFactory = (config?: any) => IGameAdapter;