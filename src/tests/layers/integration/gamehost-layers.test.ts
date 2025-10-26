import { describe, it, expect } from 'vitest';
import type { GameConfig } from '../../../game/types/config';
import { GameHost } from '../../../server/game/GameHost';
import { createHandWithDoubles } from '../../helpers/gameTestHelper';
import type { PlayerSession } from '../../../game/multiplayer/types';

function createPlayers(config: GameConfig): PlayerSession[] {
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

describe('GameHost Layers Integration', () => {
  describe('Layer Composition', () => {
    it('should compose rules from config.enabledLayers', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        enabledLayers: ['nello', 'plunge'],
        shuffleSeed: 123456
      };

      const host = new GameHost('layers-test', config, createPlayers(config));
      const state = host.getState();

      // Verify layers are stored in state
      expect(state.enabledLayers).toEqual(['nello', 'plunge']);
    });

    it('should thread rules through executeAction', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        enabledLayers: ['nello'],
        shuffleSeed: 789012
      };

      const host = new GameHost('nello-execute', config, createPlayers(config));

      // Bid marks
      let view = host.getView('player-0');
      const marksBid = view.validActions.find(a =>
        a.action.type === 'bid' &&
        a.action.bid === 'marks'
      );
      expect(marksBid).toBeDefined();

      host.executeAction('player-0', marksBid!.action, Date.now());

      // Others pass
      for (let i = 1; i <= 3; i++) {
        view = host.getView(`ai-${i}`);
        const pass = view.validActions.find(a => a.action.type === 'pass');
        host.executeAction(`ai-${i}`, pass!.action, Date.now());
      }

      // Verify nello is available in trump selection
      view = host.getView('player-0');
      expect(view.state.phase).toBe('trump_selection');

      const nelloOption = view.validActions.find(a =>
        a.action.type === 'select-trump' &&
        a.action.trump?.type === 'nello'
      );
      expect(nelloOption).toBeDefined();
    });

    it('should include enabledLayers in MultiplayerGameState', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        enabledLayers: ['splash', 'sevens'],
        shuffleSeed: 345678
      };

      const host = new GameHost('multi-layers', config, createPlayers(config));
      const mpState = host.getState();

      expect(mpState.enabledLayers).toBeDefined();
      expect(mpState.enabledLayers).toEqual(['splash', 'sevens']);
    });

    it('should execute actions with correct layer rules', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        enabledLayers: ['plunge'],
        shuffleSeed: 901234
      };

      const host = new GameHost('plunge-rules', config, createPlayers(config));

      // Give player 0 enough doubles for plunge
      const state = host.getState();
      state.coreState.players[0]!.hand = createHandWithDoubles(4);

      // Check plunge bid is available
      const view = host.getView('player-0');
      const plungeBid = view.validActions.find(a =>
        a.action.type === 'bid' &&
        a.action.bid === 'plunge'
      );

      // Plunge should be available with 4+ doubles
      expect(plungeBid).toBeDefined();
    });
  });

  describe('Layer Combinations', () => {
    it('should support multiple layers enabled simultaneously', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        enabledLayers: ['nello', 'plunge', 'splash', 'sevens'],
        shuffleSeed: 567890
      };

      const host = new GameHost('all-layers', config, createPlayers(config));
      const state = host.getState();

      expect(state.enabledLayers).toEqual(['nello', 'plunge', 'splash', 'sevens']);
      expect(state.coreState.phase).toBe('bidding');
    });

    it('should support base layer only (no special contracts)', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        enabledLayers: [], // Only base layer
        shuffleSeed: 111111
      };

      const host = new GameHost('base-only', config, createPlayers(config));
      const state = host.getState();

      expect(state.enabledLayers).toEqual([]);

      // Verify no special bids available
      const view = host.getView('player-0');
      const specialBids = view.validActions.filter(a =>
        a.action.type === 'bid' &&
        (a.action.bid === 'nello' || a.action.bid === 'plunge' || a.action.bid === 'splash')
      );

      expect(specialBids.length).toBe(0);
    });

    it('should support selective layer enabling', () => {
      // Only nello and sevens, not plunge or splash
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        enabledLayers: ['nello', 'sevens'],
        shuffleSeed: 222222
      };

      const host = new GameHost('selective-layers', config, createPlayers(config));
      const state = host.getState();

      expect(state.enabledLayers).toEqual(['nello', 'sevens']);
    });
  });

  describe('Rule Threading in Actions', () => {
    it('should apply nello rules when nello layer enabled', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        enabledLayers: ['nello'],
        shuffleSeed: 333333
      };

      const host = new GameHost('nello-threading', config, createPlayers(config));

      // Bid marks
      let view = host.getView('player-0');
      const marksBid = view.validActions.find(a =>
        a.action.type === 'bid' && a.action.bid === 'marks'
      );
      host.executeAction('player-0', marksBid!.action, Date.now());

      // Others pass
      for (let i = 1; i <= 3; i++) {
        view = host.getView(`ai-${i}`);
        const pass = view.validActions.find(a => a.action.type === 'pass');
        host.executeAction(`ai-${i}`, pass!.action, Date.now());
      }

      // Select nello
      view = host.getView('player-0');
      const nelloTrump = view.validActions.find(a =>
        a.action.type === 'select-trump' && a.action.trump?.type === 'nello'
      );
      host.executeAction('player-0', nelloTrump!.action, Date.now());

      // Verify playing phase
      view = host.getView('player-0');
      expect(view.state.phase).toBe('playing');
      expect(view.state.trump.type).toBe('nello');

      // Verify partner (player 2) is not current player
      expect(view.state.currentPlayer).not.toBe(2);
    });

    it('should apply plunge rules when plunge layer enabled', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        enabledLayers: ['plunge'],
        shuffleSeed: 444444
      };

      const host = new GameHost('plunge-threading', config, createPlayers(config));

      // Manually set up plunge scenario
      const state = host.getState();
      state.coreState.players[0]!.hand = createHandWithDoubles(4);

      // Bid plunge
      let view = host.getView('player-0');
      const plungeBid = view.validActions.find(a =>
        a.action.type === 'bid' && a.action.bid === 'plunge'
      );

      if (plungeBid) {
        host.executeAction('player-0', plungeBid.action, Date.now());

        // Others pass
        for (let i = 1; i <= 3; i++) {
          view = host.getView(`ai-${i}`);
          const pass = view.validActions.find(a => a.action.type === 'pass');
          if (pass) {
            host.executeAction(`ai-${i}`, pass.action, Date.now());
          }
        }

        // Verify partner (player 2) is trump selector
        view = host.getView('player-2');
        expect(view.state.phase).toBe('trump_selection');
        expect(view.state.currentPlayer).toBe(2);
      }
    });

    it('should apply sevens rules when sevens layer enabled', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        enabledLayers: ['sevens'],
        shuffleSeed: 555555
      };

      const host = new GameHost('sevens-threading', config, createPlayers(config));

      // Bid marks
      let view = host.getView('player-0');
      const marksBid = view.validActions.find(a =>
        a.action.type === 'bid' && a.action.bid === 'marks'
      );
      host.executeAction('player-0', marksBid!.action, Date.now());

      // Others pass
      for (let i = 1; i <= 3; i++) {
        view = host.getView(`ai-${i}`);
        const pass = view.validActions.find(a => a.action.type === 'pass');
        host.executeAction(`ai-${i}`, pass!.action, Date.now());
      }

      // Verify sevens is available
      view = host.getView('player-0');
      const sevensTrump = view.validActions.find(a =>
        a.action.type === 'select-trump' && a.action.trump?.type === 'sevens'
      );
      expect(sevensTrump).toBeDefined();
    });
  });

  describe('Layer Configuration Validation', () => {
    it('should handle empty enabledLayers array', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        enabledLayers: [],
        shuffleSeed: 666666
      };

      expect(() => {
        new GameHost('empty-layers', config, createPlayers(config));
      }).not.toThrow();
    });

    it('should handle undefined enabledLayers', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        // enabledLayers not specified
        shuffleSeed: 777777
      };

      expect(() => {
        new GameHost('undefined-layers', config, createPlayers(config));
      }).not.toThrow();

      const host = new GameHost('undefined-layers-2', config, createPlayers(config));
      const state = host.getState();

      // Should default to empty array
      expect(state.enabledLayers).toEqual([]);
    });

    it('should preserve layer order from config', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        enabledLayers: ['sevens', 'nello', 'plunge', 'splash'],
        shuffleSeed: 888888
      };

      const host = new GameHost('layer-order', config, createPlayers(config));
      const state = host.getState();

      expect(state.enabledLayers).toEqual(['sevens', 'nello', 'plunge', 'splash']);
    });
  });

  describe('Action Validation with Layers', () => {
    it('should validate actions through composed rules', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        enabledLayers: ['nello'],
        shuffleSeed: 999999
      };

      const host = new GameHost('action-validation', config, createPlayers(config));

      // Try to select nello without marks bid (should fail)
      const invalidAction = {
        type: 'select-trump' as const,
        player: 0,
        trump: { type: 'nello' as const }
      };

      const result = host.executeAction('player-0', invalidAction, Date.now());
      expect(result.success).toBe(false);
    });

    it('should allow valid actions through composed rules', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        enabledLayers: ['splash'],
        shuffleSeed: 101010
      };

      const host = new GameHost('valid-actions', config, createPlayers(config));

      // Pass should be valid
      const passAction = {
        type: 'pass' as const,
        player: 0
      };

      const result = host.executeAction('player-0', passAction, Date.now());
      expect(result.success).toBe(true);
    });
  });

  describe('View Generation with Layers', () => {
    it('should include layer-specific actions in validActions', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        enabledLayers: ['nello', 'sevens'],
        shuffleSeed: 121212
      };

      const host = new GameHost('view-actions', config, createPlayers(config));

      // Bid marks
      let view = host.getView('player-0');
      const marksBid = view.validActions.find(a =>
        a.action.type === 'bid' && a.action.bid === 'marks'
      );
      host.executeAction('player-0', marksBid!.action, Date.now());

      // Others pass
      for (let i = 1; i <= 3; i++) {
        view = host.getView(`ai-${i}`);
        const pass = view.validActions.find(a => a.action.type === 'pass');
        host.executeAction(`ai-${i}`, pass!.action, Date.now());
      }

      // Check trump options include both nello and sevens
      view = host.getView('player-0');
      const nelloOption = view.validActions.find(a =>
        a.action.type === 'select-trump' && a.action.trump?.type === 'nello'
      );
      const sevensOption = view.validActions.find(a =>
        a.action.type === 'select-trump' && a.action.trump?.type === 'sevens'
      );

      expect(nelloOption).toBeDefined();
      expect(sevensOption).toBeDefined();
    });

    it('should filter actions by player capabilities', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        enabledLayers: ['plunge'],
        shuffleSeed: 131313
      };

      const host = new GameHost('capability-filter', config, createPlayers(config));

      // Player 0's view should only show their actions
      const view0 = host.getView('player-0');
      const player0Actions = view0.validActions.filter(a =>
        'player' in a.action && a.action.player === 0
      );

      // All actions with a player field should be for player 0
      const allPlayerActions = view0.validActions.filter(a => 'player' in a.action);
      expect(player0Actions.length).toBe(allPlayerActions.length);
    });
  });
});
