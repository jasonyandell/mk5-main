import { describe, it, expect } from 'vitest';
import type { GameConfig } from '../../game/types/config';
import { GameHost } from '../../server/game/GameHost';

// Tournament rules (baseLayer only) do not allow special contracts
// These tests verify that at the GameHost level (action generation)

describe('Tournament Variant Authority', () => {
  it('prevents executing bids that variants remove', () => {
    const config: GameConfig = {
      playerTypes: ['human', 'ai', 'ai', 'ai'],
      variants: [{ type: 'tournament' }]
    };

    const players = config.playerTypes.map((type, i) => ({
      playerId: type === 'human' ? `player-${i}` : `ai-${i}`,
      playerIndex: i as 0 | 1 | 2 | 3,
      controlType: type,
      capabilities: [
        { type: 'act-as-player' as const, playerIndex: i as 0 | 1 | 2 | 3 },
        { type: 'observe-own-hand' as const },
        ...(type === 'ai' ? [{ type: 'replace-ai' as const }] : [])
      ]
    }));

    const host = new GameHost('tournament-test', config, players);

    const currentPlayer = host.getView('player-0').state.currentPlayer;

    // Attempt to force a nello bid which tournament variant removes
    const result = host.executeAction(`player-${currentPlayer}`, {
      type: 'bid',
      player: currentPlayer,
      bid: 'nello',
      value: 1
    }, Date.now());

    expect(result.success).toBe(false);

    const validActions = host.getView(`player-${currentPlayer}`).validActions;
    const hasNello = validActions.some(a => a.action.type === 'bid' && a.action.bid === 'nello');
    expect(hasNello).toBe(false);
  });
});
