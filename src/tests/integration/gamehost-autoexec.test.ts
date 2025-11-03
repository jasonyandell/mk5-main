import { describe, it, expect } from 'vitest';
import type { GameConfig } from '../../game/types/config';
import { GameKernel } from '../../kernel/GameKernel';

function createPlayers(config: GameConfig) {
  return config.playerTypes.map((type, i) => ({
    playerId: type === 'human' ? `player-${i}` : `ai-${i}`,
    playerIndex: i as 0 | 1 | 2 | 3,
    controlType: type,
    capabilities: [
      { type: 'act-as-player' as const, playerIndex: i as 0 | 1 | 2 | 3 },
      { type: 'observe-own-hand' as const },
      ...(type === 'ai' ? [{ type: 'replace-ai' as const }] : [])
    ]
  }));
}

describe('GameKernel auto-execute', () => {
  it('runs introductory scripts for one-hand action transformer automatically', () => {
    const config: GameConfig = {
      playerTypes: ['human', 'ai', 'ai', 'ai'],
      variants: [{ type: 'one-hand' }],
      shuffleSeed: 777777
    };

    const kernel = new GameKernel('one-hand-autoplay', config, createPlayers(config));
    const view = kernel.getView('player-0');

    expect(view.state.phase).toBe('playing');
    expect(view.state.bids.length).toBeGreaterThanOrEqual(4);
    expect(view.state.actionHistory.length).toBeGreaterThan(0);

    // No bidding actions should remain once the scripted intro finishes
    const hasBidOption = view.validActions.some(a => a.action.type === 'bid');
    expect(hasBidOption).toBe(false);
  });
});
