import { describe, it, expect } from 'vitest';
import type { GameConfig } from '../../game/types/config';
import { Room } from '../../server/Room';

// Tournament layer filters out special contracts (nello, splash, plunge)
// These tests verify that at the Room level (action generation)

/** Create a Room with no-op send callback for testing */
function createRoom(config: GameConfig): Room {
  return new Room(config, () => {});
}

describe('Tournament Layer Authority', () => {
  it('prevents executing bids that tournament layer filters', () => {
    const config: GameConfig = {
      playerTypes: ['human', 'ai', 'ai', 'ai'],
      layers: ['tournament']
    };

    const room = createRoom(config);

    const currentPlayer = room.getView('player-0').state.currentPlayer;

    // Nello is not a bid type - it's a trump selection
    // Tournament mode filters nello trump selections, not nello bids
    // Attempt to force a splash bid which tournament layer filters
    const result = room.executeAction(`player-${currentPlayer}`, {
      type: 'bid',
      player: currentPlayer,
      bid: 'splash',
      value: 2
    });

    expect(result.success).toBe(false);

    const validActions = room.getView(`player-${currentPlayer}`).validActions;
    const hasSplash = validActions.some(a => a.action.type === 'bid' && a.action.bid === 'splash');
    expect(hasSplash).toBe(false);
  });
});
