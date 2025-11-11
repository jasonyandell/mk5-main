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
      { type: 'observe-hands' as const, playerIndices: [i] }
    ]
  }));
}

describe('Room auto-execute', () => {
  it('runs introductory scripts for one-hand action transformer automatically', () => {
    const config: GameConfig = {
      playerTypes: ['human', 'ai', 'ai', 'ai'],
      actionTransformers: [{ type: 'one-hand' }],
      shuffleSeed: 777777
    };

    const room = new Room('one-hand-autoplay', config, createPlayers(config));
    const view = room.getView('player-0');

    expect(view.state.phase).toBe('playing');
    expect(view.state.bids.length).toBeGreaterThanOrEqual(4);
    expect(view.state.actionHistory.length).toBeGreaterThan(0);

    // No bidding actions should remain once the scripted intro finishes
    const hasBidOption = view.validActions.some(a => a.action.type === 'bid');
    expect(hasBidOption).toBe(false);
  });

  it('one-hand actions have system authority in their meta', () => {
    const config: GameConfig = {
      playerTypes: ['human', 'ai', 'ai', 'ai'],
      actionTransformers: [{ type: 'one-hand' }],
      shuffleSeed: 777777
    };

    const room = new Room('one-hand-system-auth', config, createPlayers(config));
    const view = room.getView('player-0');

    // Check action history for scripted actions with system authority
    const scriptedActions = view.state.actionHistory.filter(action => {
      const meta = (action as { meta?: unknown }).meta;
      return meta && typeof meta === 'object' && 'scriptId' in meta;
    });

    expect(scriptedActions.length).toBeGreaterThan(0);

    // All scripted actions should have system authority
    for (const action of scriptedActions) {
      const meta = (action as { meta?: Record<string, unknown> }).meta;
      expect(meta).toBeDefined();
      expect(meta?.authority).toBe('system');
    }
  });

  it('one-hand actions execute successfully regardless of session capabilities', () => {
    const config: GameConfig = {
      playerTypes: ['human', 'ai', 'ai', 'ai'],
      actionTransformers: [{ type: 'one-hand' }],
      shuffleSeed: 777777
    };

    // Create players with limited capabilities
    const limitedPlayers = config.playerTypes.map((type, i) => ({
      playerId: type === 'human' ? `player-${i}` : `ai-${i}`,
      playerIndex: i as 0 | 1 | 2 | 3,
      controlType: type,
      isConnected: true,
      name: `Player ${i + 1}`,
      capabilities: [
        { type: 'observe-hands' as const, playerIndices: [i] }
        // Intentionally omit 'act-as-player' capability
      ]
    }));

    const room = new Room('one-hand-limited-caps', config, limitedPlayers);
    const view = room.getView('player-0');

    // Should still reach playing phase despite limited capabilities
    // because system authority bypasses capability checks
    expect(view.state.phase).toBe('playing');
    expect(view.state.bids.length).toBeGreaterThanOrEqual(4);
  });
});
