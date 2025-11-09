/**
 * AIManager - Manages AI client lifecycle for Room.
 *
 * Current design:
 * - AIClient uses the real adapter (like any other client)
 * - AIClient speaks the protocol (SUBSCRIBE, EXECUTE_ACTION, etc)
 * - Room passes real adapter to AIManager
 * - AIManager holds the AIClients and manages their lifecycle
 *
 * Future design:
 * - Callback-based: AIClient takes (seat, playerId, onAction) callbacks
 * - No adapter dependency
 * - Simpler to test and deploy
 *
 * Responsibilities:
 * - Spawn AI clients when needed
 * - Destroy AI clients when control changes
 * - Manage AI lifecycle
 */

import type { IGameAdapter } from '../adapters/IGameAdapter';
import { AIClient } from '../../game/multiplayer/AIClient';
import type { AIDifficulty } from '../../game/multiplayer/AIClient';

interface AIEntry {
  client: AIClient;
  seat: number;
}

export class AIManager {
  private aiClients: Map<number, AIEntry> = new Map();
  private adapter: IGameAdapter;
  private difficulty: AIDifficulty = 'beginner';

  constructor(adapter: IGameAdapter, difficulty: AIDifficulty = 'beginner') {
    this.adapter = adapter;
    this.difficulty = difficulty;
  }

  /**
   * Spawn an AI client for a given seat.
   *
   * @param seat - Player seat index (0-3)
   * @param gameId - Game ID
   * @param playerId - Player ID (should be `ai-${seat}`)
   */
  spawnAI(seat: number, gameId: string, playerId: string): void {
    // Don't spawn if already exists
    if (this.aiClients.has(seat)) {
      console.warn(`AIManager: Seat ${seat} already has AI`);
      return;
    }

    const client = new AIClient(
      gameId,
      seat,
      this.adapter,  // Use real adapter like any other client
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
