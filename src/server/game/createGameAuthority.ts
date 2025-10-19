/**
 * Factory function for creating game authorities.
 * Replaces GameRegistry pattern with clean factory.
 * Per vision document - adapters call this directly.
 */

import { GameHost } from './GameHost';
import type { GameConfig } from '../../game/types/config';
import type { PlayerSession, Capability } from '../../game/multiplayer/types';

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
 * Generate default player sessions from player types
 */
function generateDefaultSessions(
  playerTypes: ('human' | 'ai')[]
): PlayerSession[] {
  return playerTypes.map((type, i) => {
    const baseCapabilities: Capability[] = [
      { type: 'act-as-player', playerIndex: i as 0 | 1 | 2 | 3 },
      { type: 'observe-own-hand' }
    ];

    if (type === 'ai') {
      baseCapabilities.push({ type: 'replace-ai' });
    }

    return {
      playerId: `${type === 'human' ? 'player' : 'ai'}-${i}`,
      playerIndex: i as 0 | 1 | 2 | 3,
      controlType: type,
      isConnected: true,
      name: `Player ${i + 1}`,
      capabilities: baseCapabilities
    };
  });
}
