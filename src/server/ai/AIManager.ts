/**
 * AIManager - Manages AI client lifecycle for Room.
 *
 * Design:
 * - AIClient uses in-process connections (created from Transport)
 * - AIClient speaks the protocol (SUBSCRIBE, EXECUTE_ACTION, etc)
 * - Room creates connections on-demand and passes to spawnAI()
 * - AIManager holds the AIClients and manages their lifecycle
 *
 * Responsibilities:
 * - Spawn AI clients with provided connections
 * - Destroy AI clients when control changes
 * - Manage AI lifecycle
 */

import type { Connection } from '../transports/Transport';
import { AIClient } from '../../game/multiplayer/AIClient';
import type { AIDifficulty } from '../../game/multiplayer/AIClient';

interface AIEntry {
  client: AIClient;
  seat: number;
}

export class AIManager {
  private aiClients: Map<number, AIEntry> = new Map();
  private difficulty: AIDifficulty = 'beginner';

  constructor(difficulty: AIDifficulty = 'beginner') {
    this.difficulty = difficulty;
  }

  /**
   * Spawn an AI client for a given seat.
   *
   * @param seat - Player seat index (0-3)
   * @param gameId - Game ID
   * @param playerId - Player ID (should be `ai-${seat}`)
   * @param connection - Connection for this AI client
   */
  spawnAI(seat: number, gameId: string, playerId: string, connection: Connection): void {
    // Don't spawn if already exists
    if (this.aiClients.has(seat)) {
      console.warn(`AIManager: Seat ${seat} already has AI`);
      return;
    }

    const client = new AIClient(
      gameId,
      seat,
      connection,  // Pass connection to AIClient
      playerId,
      this.difficulty
    );

    // Store entry
    const entry: AIEntry = {
      client,
      seat
    };

    this.aiClients.set(seat, entry);

    // Start the AI
    client.start();
  }

  /**
   * Destroy an AI client for a given seat.
   */
  destroyAI(seat: number): void {
    const entry = this.aiClients.get(seat);
    if (!entry) {
      return;
    }

    entry.client.destroy();
    this.aiClients.delete(seat);
  }

  /**
   * Destroy all AI clients.
   * Called when game is shutting down.
   */
  destroyAll(): void {
    for (const entry of this.aiClients.values()) {
      entry.client.destroy();
    }
    this.aiClients.clear();
  }

  /**
   * Check if a seat has an AI client.
   */
  hasAI(seat: number): boolean {
    return this.aiClients.has(seat);
  }

  /**
   * Get all AI seats.
   */
  getAISeats(): number[] {
    return Array.from(this.aiClients.keys());
  }

  /**
   * Get AI client info for debugging.
   */
  getAIInfo(seat: number) {
    const entry = this.aiClients.get(seat);
    if (!entry) {
      return null;
    }
    return entry.client.getInfo();
  }
}
