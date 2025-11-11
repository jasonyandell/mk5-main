import { describe, it, expect } from 'vitest';
import type { GameConfig } from '../../game/types/config';
import { Room } from '../../server/Room';

// Tournament ruleset filters out special contracts (nello, splash, plunge)
// These tests verify that at the Room level (action generation)

describe('Tournament RuleSet Authority', () => {
  it('prevents executing bids that tournament ruleset filters', () => {
    const config: GameConfig = {
      playerTypes: ['human', 'ai', 'ai', 'ai'],
      enabledRuleSets: ['tournament']
    };

    const players = config.playerTypes.map((type, i) => ({
      playerId: type === 'human' ? `player-${i}` : `ai-${i}`,
      playerIndex: i as 0 | 1 | 2 | 3,
      controlType: type,
      isConnected: true,
      name: `Player ${i + 1}`,
      capabilities: [
        { type: 'act-as-player' as const, playerIndex: i as 0 | 1 | 2 | 3 },
        { type: 'observe-hands' as const, playerIndices: [i] }
      ]
    }));

    const room = new Room('tournament-test', config, players);

    const currentPlayer = room.getView('player-0').state.currentPlayer;

    // Nello is not a bid type - it's a trump selection
    // Tournament mode filters nello trump selections, not nello bids
    // Attempt to force a splash bid which tournament ruleset filters
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
