import { describe, it, expect } from 'vitest';
import type { GameConfig } from '../../game/types/config';
import { Room } from '../../server/Room';

function createPlayers(config: GameConfig) {
  return config.playerTypes.map((type, i) => ({
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
}

// Dummy adapter for tests
const dummyAdapter = {
  send: async () => {},
  subscribe: () => () => {},
  destroy: () => {},
  isConnected: () => true,
  getMetadata: () => ({ type: 'in-process' as const })
};

describe('Room auto-execute', () => {
  it('runs introductory scripts for one-hand action transformer automatically', () => {
    const config: GameConfig = {
      playerTypes: ['human', 'ai', 'ai', 'ai'],
      variants: [{ type: 'one-hand' }],
      shuffleSeed: 777777
    };

    const room = new Room('one-hand-autoplay', config, dummyAdapter, createPlayers(config));
    const view = room.getView('player-0');

    expect(view.state.phase).toBe('playing');
    expect(view.state.bids.length).toBeGreaterThanOrEqual(4);
    expect(view.state.actionHistory.length).toBeGreaterThan(0);

    // No bidding actions should remain once the scripted intro finishes
    const hasBidOption = view.validActions.some(a => a.action.type === 'bid');
    expect(hasBidOption).toBe(false);
  });
});
