import { describe, it, expect } from 'vitest';
import type { GameConfig } from '../../game/types/config';
import { Room } from '../../server/Room';

// Tournament rules (baseRuleSet only) do not allow special contracts
// These tests verify that at the Room level (action generation)

// Dummy adapter for tests
const dummyAdapter = {
  send: async () => {},
  subscribe: () => () => {},
  destroy: () => {},
  isConnected: () => true,
  getMetadata: () => ({ type: 'in-process' as const })
};

describe('Tournament Action Transformer Authority', () => {
  it('prevents executing bids that action transformers remove', () => {
    const config: GameConfig = {
      playerTypes: ['human', 'ai', 'ai', 'ai'],
      variants: [{ type: 'tournament' }]
    };

    const players = config.playerTypes.map((type, i) => ({
      playerId: type === 'human' ? `player-${i}` : `ai-${i}`,
      playerIndex: i as 0 | 1 | 2 | 3,
      controlType: type,
      isConnected: true,
      name: `Player ${i + 1}`,
      capabilities: [
        { type: 'act-as-player' as const, playerIndex: i as 0 | 1 | 2 | 3 },
        { type: 'observe-own-hand' as const },
        ...(type === 'ai' ? [{ type: 'replace-ai' as const }] : [])
      ]
    }));

    const room = new Room('tournament-test', config, dummyAdapter, players);

    const currentPlayer = room.getView('player-0').state.currentPlayer;

    // Attempt to force a nello bid which tournament action transformer removes
    const result = room.executeAction(`player-${currentPlayer}`, {
      type: 'bid',
      player: currentPlayer,
      bid: 'nello',
      value: 1
    });

    expect(result.success).toBe(false);

    const validActions = room.getView(`player-${currentPlayer}`).validActions;
    const hasNello = validActions.some(a => a.action.type === 'bid' && a.action.bid === 'nello');
    expect(hasNello).toBe(false);
  });
});
