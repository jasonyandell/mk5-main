/**
 * InProcessGameServer - Thin wrapper for backwards compatibility.
 *
 * This class is kept for backward compatibility with legacy code.
 * New code should use GameServer + Transport directly.
 *
 * This implementation simply delegates to InProcessAdapter, which was the old pattern.
 */

import type { ClientMessage, ServerMessage } from '../../shared/multiplayer/protocol';
import type { IGameAdapter } from '../adapters/IGameAdapter';

export class InProcessGameServer {
  private adapter: IGameAdapter;

  constructor(adapter: IGameAdapter) {
    this.adapter = adapter;
  }

  /**
   * Process a client message and call emit callback with response(s)
   */
  async handleMessage(
    message: ClientMessage,
    emit: (message: ServerMessage) => void
  ): Promise<void> {
    // Subscribe to messages from adapter
    const unsubscribe = this.adapter.subscribe((serverMessage) => {
      emit(serverMessage);
    });

    // Send message through adapter
    try {
      await this.adapter.send(message);
    } finally {
      unsubscribe();
    }
  }

  /**
   * Clean up resources
   */
  destroy(): void {
    this.adapter.destroy();
  }
}
