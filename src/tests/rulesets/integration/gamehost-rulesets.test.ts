import { describe, it, expect } from 'vitest';
import type { GameConfig } from '../../../game/types/config';
import { GameKernel } from '../../../kernel/GameKernel';
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

describe('GameKernel Layers Integration', () => {
  describe('Layer Composition', () => {
    it('should compose rules from config.enabledRuleSets', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        enabledRuleSets: ['nello', 'plunge'],
        shuffleSeed: 123456
      };

      const kernel = new GameKernel('ruleSets-test', config, createPlayers(config));
      const state = kernel.getState();

      // Verify ruleSets are stored in state
      expect(state.enabledRuleSets).toEqual(['nello', 'plunge']);
    });

    it('should thread rules through executeAction', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        enabledRuleSets: ['nello'],
        shuffleSeed: 789012
      };

      const kernel = new GameKernel('nello-execute', config, createPlayers(config));

      // Bid marks
      let view = kernel.getView('player-0');
      const marksBid = view.validActions.find(a =>
        a.action.type === 'bid' &&
        a.action.bid === 'marks'
      );
      expect(marksBid).toBeDefined();

      kernel.executeAction('player-0', marksBid!.action, Date.now());

      // Others pass
      for (let i = 1; i <= 3; i++) {
        view = kernel.getView(`ai-${i}`);
        const pass = view.validActions.find(a => a.action.type === 'pass');
        kernel.executeAction(`ai-${i}`, pass!.action, Date.now());
      }

      // Verify nello is available in trump selection
      view = kernel.getView('player-0');
      expect(view.state.phase).toBe('trump_selection');

      const nelloOption = view.validActions.find(a =>
        a.action.type === 'select-trump' &&
        a.action.trump?.type === 'nello'
      );
      expect(nelloOption).toBeDefined();
    });

    it('should include enabledRuleSets in MultiplayerGameState', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        enabledRuleSets: ['splash', 'sevens'],
        shuffleSeed: 345678
      };

      const kernel = new GameKernel('multi-ruleSets', config, createPlayers(config));
      const mpState = kernel.getState();

      expect(mpState.enabledRuleSets).toBeDefined();
      expect(mpState.enabledRuleSets).toEqual(['splash', 'sevens']);
    });

    it('should execute actions with correct ruleset rules', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        enabledRuleSets: ['plunge'],
        shuffleSeed: 901234
      };

      const kernel = new GameKernel('plunge-rules', config, createPlayers(config));

      // Give player 0 enough doubles for plunge
      const state = kernel.getState();
      state.coreState.players[0]!.hand = createHandWithDoubles(4);

      // Check plunge bid is available
      const view = kernel.getView('player-0');
      const plungeBid = view.validActions.find(a =>
        a.action.type === 'bid' &&
        a.action.bid === 'plunge'
      );

      // Plunge should be available with 4+ doubles
      expect(plungeBid).toBeDefined();
    });
  });

  describe('Layer Combinations', () => {
    it('should support multiple ruleSets enabled simultaneously', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        enabledRuleSets: ['nello', 'plunge', 'splash', 'sevens'],
        shuffleSeed: 567890
      };

      const kernel = new GameKernel('all-ruleSets', config, createPlayers(config));
      const state = kernel.getState();

      expect(state.enabledRuleSets).toEqual(['nello', 'plunge', 'splash', 'sevens']);
      expect(state.coreState.phase).toBe('bidding');
    });

    it('should support base ruleset only (no special contracts)', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        enabledRuleSets: [], // Only base ruleSet
        shuffleSeed: 111111
      };

      const kernel = new GameKernel('base-only', config, createPlayers(config));
      const state = kernel.getState();

      expect(state.enabledRuleSets).toEqual([]);

      // Verify no special bids available
      const view = kernel.getView('player-0');
      const specialBids = view.validActions.filter(a =>
        a.action.type === 'bid' &&
        (a.action.bid === 'nello' || a.action.bid === 'plunge' || a.action.bid === 'splash')
      );

      expect(specialBids.length).toBe(0);
    });

    it('should support selective ruleset enabling', () => {
      // Only nello and sevens, not plunge or splash
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        enabledRuleSets: ['nello', 'sevens'],
        shuffleSeed: 222222
      };

      const kernel = new GameKernel('selective-ruleSets', config, createPlayers(config));
      const state = kernel.getState();

      expect(state.enabledRuleSets).toEqual(['nello', 'sevens']);
    });
  });

  describe('Rule Threading in Actions', () => {
    it('should apply nello rules when nello ruleset enabled', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        enabledRuleSets: ['nello'],
        shuffleSeed: 333333
      };

      const kernel = new GameKernel('nello-threading', config, createPlayers(config));

      // Bid marks
      let view = kernel.getView('player-0');
      const marksBid = view.validActions.find(a =>
        a.action.type === 'bid' && a.action.bid === 'marks'
      );
      kernel.executeAction('player-0', marksBid!.action, Date.now());

      // Others pass
      for (let i = 1; i <= 3; i++) {
        view = kernel.getView(`ai-${i}`);
        const pass = view.validActions.find(a => a.action.type === 'pass');
        kernel.executeAction(`ai-${i}`, pass!.action, Date.now());
      }

      // Select nello
      view = kernel.getView('player-0');
      const nelloTrump = view.validActions.find(a =>
        a.action.type === 'select-trump' && a.action.trump?.type === 'nello'
      );
      kernel.executeAction('player-0', nelloTrump!.action, Date.now());

      // Verify playing phase
      view = kernel.getView('player-0');
      expect(view.state.phase).toBe('playing');
      expect(view.state.trump.type).toBe('nello');

      // Verify partner (player 2) is not current player
      expect(view.state.currentPlayer).not.toBe(2);
    });

    it('should apply plunge rules when plunge ruleset enabled', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        enabledRuleSets: ['plunge'],
        shuffleSeed: 444444
      };

      const kernel = new GameKernel('plunge-threading', config, createPlayers(config));

      // Manually set up plunge scenario
      const state = kernel.getState();
      state.coreState.players[0]!.hand = createHandWithDoubles(4);

      // Bid plunge
      let view = kernel.getView('player-0');
      const plungeBid = view.validActions.find(a =>
        a.action.type === 'bid' && a.action.bid === 'plunge'
      );

      if (plungeBid) {
        kernel.executeAction('player-0', plungeBid.action, Date.now());

        // Others pass
        for (let i = 1; i <= 3; i++) {
          view = kernel.getView(`ai-${i}`);
          const pass = view.validActions.find(a => a.action.type === 'pass');
          if (pass) {
            kernel.executeAction(`ai-${i}`, pass.action, Date.now());
          }
        }

        // Verify partner (player 2) is trump selector
        view = kernel.getView('player-2');
        expect(view.state.phase).toBe('trump_selection');
        expect(view.state.currentPlayer).toBe(2);
      }
    });

    it('should apply sevens rules when sevens ruleset enabled', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        enabledRuleSets: ['sevens'],
        shuffleSeed: 555555
      };

      const kernel = new GameKernel('sevens-threading', config, createPlayers(config));

      // Bid marks
      let view = kernel.getView('player-0');
      const marksBid = view.validActions.find(a =>
        a.action.type === 'bid' && a.action.bid === 'marks'
      );
      kernel.executeAction('player-0', marksBid!.action, Date.now());

      // Others pass
      for (let i = 1; i <= 3; i++) {
        view = kernel.getView(`ai-${i}`);
        const pass = view.validActions.find(a => a.action.type === 'pass');
        kernel.executeAction(`ai-${i}`, pass!.action, Date.now());
      }

      // Verify sevens is available
      view = kernel.getView('player-0');
      const sevensTrump = view.validActions.find(a =>
        a.action.type === 'select-trump' && a.action.trump?.type === 'sevens'
      );
      expect(sevensTrump).toBeDefined();
    });
  });

  describe('Layer Configuration Validation', () => {
    it('should handle empty enabledRuleSets array', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        enabledRuleSets: [],
        shuffleSeed: 666666
      };

      expect(() => {
        new GameKernel('empty-ruleSets', config, createPlayers(config));
      }).not.toThrow();
    });

    it('should handle undefined enabledRuleSets', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        // enabledRuleSets not specified
        shuffleSeed: 777777
      };

      expect(() => {
        new GameKernel('undefined-ruleSets', config, createPlayers(config));
      }).not.toThrow();

      const kernel = new GameKernel('undefined-ruleSets-2', config, createPlayers(config));
      const state = kernel.getState();

      // Should default to empty array
      expect(state.enabledRuleSets).toEqual([]);
    });

    it('should preserve ruleset order from config', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        enabledRuleSets: ['sevens', 'nello', 'plunge', 'splash'],
        shuffleSeed: 888888
      };

      const kernel = new GameKernel('ruleSet-order', config, createPlayers(config));
      const state = kernel.getState();

      expect(state.enabledRuleSets).toEqual(['sevens', 'nello', 'plunge', 'splash']);
    });
  });

  describe('Action Validation with Layers', () => {
    it('should validate actions through composed rules', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        enabledRuleSets: ['nello'],
        shuffleSeed: 999999
      };

      const kernel = new GameKernel('action-validation', config, createPlayers(config));

      // Try to select nello without marks bid (should fail)
      const invalidAction = {
        type: 'select-trump' as const,
        player: 0,
        trump: { type: 'nello' as const }
      };

      const result = kernel.executeAction('player-0', invalidAction, Date.now());
      expect(result.success).toBe(false);
    });

    it('should allow valid actions through composed rules', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        enabledRuleSets: ['splash'],
        shuffleSeed: 101010
      };

      const kernel = new GameKernel('valid-actions', config, createPlayers(config));

      // Pass should be valid
      const passAction = {
        type: 'pass' as const,
        player: 0
      };

      const result = kernel.executeAction('player-0', passAction, Date.now());
      expect(result.success).toBe(true);
    });
  });

  describe('View Generation with Layers', () => {
    it('should include ruleSet-specific actions in validActions', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        enabledRuleSets: ['nello', 'sevens'],
        shuffleSeed: 121212
      };

      const kernel = new GameKernel('view-actions', config, createPlayers(config));

      // Bid marks
      let view = kernel.getView('player-0');
      const marksBid = view.validActions.find(a =>
        a.action.type === 'bid' && a.action.bid === 'marks'
      );
      kernel.executeAction('player-0', marksBid!.action, Date.now());

      // Others pass
      for (let i = 1; i <= 3; i++) {
        view = kernel.getView(`ai-${i}`);
        const pass = view.validActions.find(a => a.action.type === 'pass');
        kernel.executeAction(`ai-${i}`, pass!.action, Date.now());
      }

      // Check trump options include both nello and sevens
      view = kernel.getView('player-0');
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
        enabledRuleSets: ['plunge'],
        shuffleSeed: 131313
      };

      const kernel = new GameKernel('capability-filter', config, createPlayers(config));

      // Player 0's view should only show their actions
      const view0 = kernel.getView('player-0');
      const player0Actions = view0.validActions.filter(a =>
        'player' in a.action && a.action.player === 0
      );

      // All actions with a player field should be for player 0
      const allPlayerActions = view0.validActions.filter(a => 'player' in a.action);
      expect(player0Actions.length).toBe(allPlayerActions.length);
    });
  });
});
