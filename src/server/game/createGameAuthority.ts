/**
 * Factory function for creating game authorities.
 * Replaces GameRegistry pattern with clean factory.
 * Per vision document - adapters call this directly.
 */

import { GameHost } from './GameHost';
import type { GameConfig } from '../../game/types/config';
import type { PlayerSession } from '../../game/multiplayer/types';
import { humanCapabilities, aiCapabilities } from '../../game/multiplayer/capabilities';

/**
 * Create a new game authority with given configuration
 */
export function createGameAuthority(
  gameId: string,
  config: GameConfig,
  sessions?: PlayerSession[]
): GameHost {
  // Generate sessions if not provided
  const playerSessions = sessions ?? generateDefaultSessions(config.playerTypes);

  return new GameHost(gameId, config, playerSessions);
}

/**
 * Generate default player sessions from player types.
 * Uses standard capability builders from vision spec §4.3
 */
function generateDefaultSessions(
  playerTypes: ('human' | 'ai')[]
): PlayerSession[] {
  return playerTypes.map((type, i) => {
    const idx = i as 0 | 1 | 2 | 3;
    const capabilities = type === 'human'
      ? humanCapabilities(idx)
      : aiCapabilities(idx);

    return {
      playerId: `${type === 'human' ? 'player' : 'ai'}-${i}`,
      playerIndex: idx,
      controlType: type,
      isConnected: true,
      name: `Player ${i + 1}`,
      capabilities
    };
  });
}
