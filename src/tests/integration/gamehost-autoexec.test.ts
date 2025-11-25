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
  it('runs introductory scripts for oneHand layer automatically', () => {
    const config: GameConfig = {
      playerTypes: ['human', 'ai', 'ai', 'ai'],
      layers: ['oneHand'],
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

  it('oneHand actions have system authority in their meta', () => {
    const config: GameConfig = {
      playerTypes: ['human', 'ai', 'ai', 'ai'],
      layers: ['oneHand'],
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

  it('oneHand actions execute successfully regardless of session capabilities', () => {
    const config: GameConfig = {
      playerTypes: ['human', 'ai', 'ai', 'ai'],
      layers: ['oneHand'],
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

  it('redeal action auto-executes after all players pass', () => {
    const config: GameConfig = {
      playerTypes: ['human', 'human', 'human', 'human'],
      shuffleSeed: 888888
    };

    const room = new Room('redeal-autoexec', config, createPlayers(config));

    // Get initial dealer
    const initialView = room.getView('player-0');
    const initialDealer = initialView.state.dealer;

    // Execute 4 passes to trigger redeal
    room.executeAction('player-0', { type: 'pass', player: 0 });
    room.executeAction('player-1', { type: 'pass', player: 1 });
    room.executeAction('player-2', { type: 'pass', player: 2 });
    room.executeAction('player-3', { type: 'pass', player: 3 });

    const view = room.getView('player-0');

    // After all passes, redeal should auto-execute and return to bidding phase
    expect(view.state.phase).toBe('bidding');

    // Bids should be cleared (new hand)
    expect(view.state.bids.length).toBe(0);

    // Dealer should advance from initial position
    const expectedNewDealer = (initialDealer + 1) % 4;
    expect(view.state.dealer).toBe(expectedNewDealer);

    // Current player should be dealer+1
    const expectedCurrentPlayer = (expectedNewDealer + 1) % 4;
    expect(view.state.currentPlayer).toBe(expectedCurrentPlayer);

    // Redeal should NOT be in available actions (already executed)
    const hasRedeal = view.validActions.some(a => a.action.type === 'redeal');
    expect(hasRedeal).toBe(false);

    // Should have new bidding actions available for current player
    const currentPlayerView = room.getView(`player-${expectedCurrentPlayer}`);
    const hasBiddingActions = currentPlayerView.validActions.some(a =>
      a.action.type === 'pass' || a.action.type === 'bid'
    );
    expect(hasBiddingActions).toBe(true);
  });

  it('redeal action has system authority in action history', () => {
    const config: GameConfig = {
      playerTypes: ['ai', 'ai', 'ai', 'ai'],
      shuffleSeed: 999999
    };

    const room = new Room('redeal-system-auth', config, createPlayers(config));

    // Execute 4 passes to trigger redeal
    room.executeAction('ai-0', { type: 'pass', player: 0 });
    room.executeAction('ai-1', { type: 'pass', player: 1 });
    room.executeAction('ai-2', { type: 'pass', player: 2 });
    room.executeAction('ai-3', { type: 'pass', player: 3 });

    const view = room.getView('ai-0');

    // Find redeal action in history
    const redealAction = view.state.actionHistory.find(a => a.type === 'redeal');

    expect(redealAction).toBeDefined();

    // Redeal should have system authority
    const meta = (redealAction as { meta?: Record<string, unknown> })?.meta;
    expect(meta).toBeDefined();
    expect(meta?.authority).toBe('system');
  });

  it('redeal executes successfully regardless of session capabilities', () => {
    const config: GameConfig = {
      playerTypes: ['ai', 'ai', 'ai', 'ai'],
      shuffleSeed: 111111
    };

    const room = new Room('redeal-limited-caps', config, createPlayers(config));

    // Get initial dealer
    const initialView = room.getView('ai-0');
    const initialDealer = initialView.state.dealer;

    // Execute 4 passes to trigger redeal
    room.executeAction('ai-0', { type: 'pass', player: 0 });
    room.executeAction('ai-1', { type: 'pass', player: 1 });
    room.executeAction('ai-2', { type: 'pass', player: 2 });
    // Mark redeal action with system authority before executing the last pass
    room.executeAction('ai-3', { type: 'pass', player: 3 });

    const view = room.getView('ai-0');

    // Verify redeal was in action history
    const redealAction = view.state.actionHistory.find(a => a.type === 'redeal');
    expect(redealAction).toBeDefined();

    // Redeal should execute despite being a system action
    // because system authority bypasses capability checks
    expect(view.state.phase).toBe('bidding');
    expect(view.state.bids.length).toBe(0);

    // Dealer should advance from initial position
    const expectedNewDealer = (initialDealer + 1) % 4;
    expect(view.state.dealer).toBe(expectedNewDealer);
  });
});
