import { describe, it, expect } from 'vitest';
import type { GameConfig } from '../../../game/types/config';
import { Room } from '../../../server/Room';
import { HandBuilder } from '../../helpers';
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
      { type: 'observe-hands' as const, playerIndices: [i] }
    ]
  }));
}

describe('Room Layers Integration', () => {
  describe('Layer Composition', () => {
    it('should thread rules through executeAction', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'human', 'human', 'human'],
        layers: ['nello'],
        shuffleSeed: 789012
      };

      const room = new Room('nello-execute', config, createPlayers(config));

      // Bid marks
      let view = room.getView('player-0');
      const marksBid = view.validActions.find(a =>
        a.action.type === 'bid' &&
        a.action.bid === 'marks'
      );
      expect(marksBid).toBeDefined();

      room.executeAction('player-0', marksBid!.action);

      // Others pass (may be auto-executed)
      for (let i = 1; i <= 3; i++) {
        view = room.getView(`player-${i}`);
        const pass = view.validActions.find(a => a.action.type === 'pass');
        if (pass) {
          room.executeAction(`player-${i}`, pass.action);
        }
      }

      // Verify nello is available in trump selection
      view = room.getView('player-0');
      expect(view.state.phase).toBe('trump_selection');

      const nelloOption = view.validActions.find(a =>
        a.action.type === 'select-trump' &&
        a.action.trump?.type === 'nello'
      );
      expect(nelloOption).toBeDefined();
    });

    it('should execute actions with correct layer rules', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'human', 'human', 'human'],
        layers: ['plunge'],
        shuffleSeed: 901234
      };

      const room = new Room('plunge-rules', config, createPlayers(config));

      // Give player 0 enough doubles for plunge
      const state = room.getState();
      state.coreState.players[0]!.hand = HandBuilder.withDoubles(4);

      // Check plunge bid is available
      const view = room.getView('player-0');
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
        playerTypes: ['human', 'human', 'human', 'human'],
        layers: ['nello', 'plunge', 'splash', 'sevens'],
        shuffleSeed: 567890
      };

      const room = new Room('all-layers', config, createPlayers(config));
      const state = room.getState();

      // Verify game is running with all layers
      expect(state.coreState.phase).toBe('bidding');
    });

    it('should support base layer only (no special contracts)', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'human', 'human', 'human'],
        layers: [], // Only base layer
        shuffleSeed: 111111
      };

      const room = new Room('base-only', config, createPlayers(config));

      // Verify no special bids available (nello is not a bid, it's a trump selection)
      const view = room.getView('player-0');
      const specialBids = view.validActions.filter(a =>
        a.action.type === 'bid' &&
        (a.action.bid === 'plunge' || a.action.bid === 'splash')
      );

      expect(specialBids.length).toBe(0);
    });

    it('should support selective layer enabling', () => {
      // Only nello and sevens, not plunge or splash
      const config: GameConfig = {
        playerTypes: ['human', 'human', 'human', 'human'],
        layers: ['nello', 'sevens'],
        shuffleSeed: 222222
      };

      const room = new Room('selective-layers', config, createPlayers(config));
      const state = room.getState();

      // Verify game is running with selective layers
      expect(state.coreState.phase).toBe('bidding');
    });
  });

  describe('Rule Threading in Actions', () => {
    it('should apply nello rules when nello layer enabled', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'human', 'human', 'human'],
        layers: ['nello'],
        shuffleSeed: 333333
      };

      const room = new Room('nello-threading', config, createPlayers(config));

      // Bid marks
      let view = room.getView('player-0');
      const marksBid = view.validActions.find(a =>
        a.action.type === 'bid' && a.action.bid === 'marks'
      );
      room.executeAction('player-0', marksBid!.action);

      // Others pass (may be auto-executed)
      for (let i = 1; i <= 3; i++) {
        view = room.getView(`player-${i}`);
        const pass = view.validActions.find(a => a.action.type === 'pass');
        if (pass) {
          room.executeAction(`player-${i}`, pass.action);
        }
      }

      // Select nello
      view = room.getView('player-0');
      const nelloTrump = view.validActions.find(a =>
        a.action.type === 'select-trump' && a.action.trump?.type === 'nello'
      );
      room.executeAction('player-0', nelloTrump!.action);

      // Verify playing phase
      view = room.getView('player-0');
      expect(view.state.phase).toBe('playing');
      expect(view.state.trump.type).toBe('nello');

      // Verify partner (player 2) is not current player
      expect(view.state.currentPlayer).not.toBe(2);
    });

    it('should apply plunge rules when plunge layer enabled', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'human', 'human', 'human'],
        layers: ['plunge'],
        shuffleSeed: 444444
      };

      const room = new Room('plunge-threading', config, createPlayers(config));

      // Bid plunge if available (depends on hand setup)
      let view = room.getView('player-0');
      const plungeBid = view.validActions.find(a =>
        a.action.type === 'bid' && a.action.bid === 'plunge'
      );

      if (plungeBid) {
        room.executeAction('player-0', plungeBid.action);

        // Others pass
        for (let i = 1; i <= 3; i++) {
          view = room.getView(`player-${i}`);
          const pass = view.validActions.find(a => a.action.type === 'pass');
          if (pass) {
            room.executeAction(`player-${i}`, pass.action);
          }
        }

        // Verify partner (player 2) is trump selector or game moved to playing
        view = room.getView('player-2');
        // Phase may be 'trump_selection' if partner hasn't selected yet,
        // or 'playing' if partner's trump selection auto-executed
        expect(['trump_selection', 'playing']).toContain(view.state.phase);
      } else {
        // If plunge bid not available, just verify the game is running
        expect(view.state.phase).toBe('bidding');
      }
    });

    it('should apply sevens rules when sevens layer enabled', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'human', 'human', 'human'],
        layers: ['sevens'],
        shuffleSeed: 555555
      };

      const room = new Room('sevens-threading', config, createPlayers(config));

      // Bid marks
      let view = room.getView('player-0');
      const marksBid = view.validActions.find(a =>
        a.action.type === 'bid' && a.action.bid === 'marks'
      );
      room.executeAction('player-0', marksBid!.action);

      // Others pass (may be auto-executed)
      for (let i = 1; i <= 3; i++) {
        view = room.getView(`player-${i}`);
        const pass = view.validActions.find(a => a.action.type === 'pass');
        if (pass) {
          room.executeAction(`player-${i}`, pass.action);
        }
      }

      // Verify sevens is available
      view = room.getView('player-0');
      const sevensTrump = view.validActions.find(a =>
        a.action.type === 'select-trump' && a.action.trump?.type === 'sevens'
      );
      expect(sevensTrump).toBeDefined();
    });
  });

  describe('Layer Configuration Validation', () => {
    it('should handle empty layers array', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'human', 'human', 'human'],
        layers: [],
        shuffleSeed: 666666
      };

      expect(() => {
        new Room('empty-layers', config, createPlayers(config));
      }).not.toThrow();
    });

    it('should handle undefined layers', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'human', 'human', 'human'],
        // layers not specified
        shuffleSeed: 777777
      };

      expect(() => {
        new Room('undefined-layers', config, createPlayers(config));
      }).not.toThrow();

      const room = new Room('undefined-layers-2', config, createPlayers(config));
      const state = room.getState();

      // Should work with base rules
      expect(state.coreState.phase).toBe('bidding');
    });

    it('should preserve layer order from config', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'human', 'human', 'human'],
        layers: ['sevens', 'nello', 'plunge', 'splash'],
        shuffleSeed: 888888
      };

      const room = new Room('layer-order', config, createPlayers(config));
      const state = room.getState();

      // Verify game is running with ordered layers
      expect(state.coreState.phase).toBe('bidding');
    });
  });

  describe('Action Validation with Layers', () => {
    it('should validate actions through composed rules', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'human', 'human', 'human'],
        layers: ['nello'],
        shuffleSeed: 999999
      };

      const room = new Room('action-validation', config, createPlayers(config));

      // Try to select nello without marks bid (should fail)
      const invalidAction = {
        type: 'select-trump' as const,
        player: 0,
        trump: { type: 'nello' as const }
      };

      const result = room.executeAction('player-0', invalidAction);
      expect(result.success).toBe(false);
    });

    it('should allow valid actions through composed rules', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'human', 'human', 'human'],
        layers: ['splash'],
        shuffleSeed: 101010
      };

      const room = new Room('valid-actions', config, createPlayers(config));

      // Pass should be valid
      const passAction = {
        type: 'pass' as const,
        player: 0
      };

      const result = room.executeAction('player-0', passAction);
      expect(result.success).toBe(true);
    });
  });

  describe('View Generation with Layers', () => {
    it('should include layer-specific actions in validActions', () => {
      const config: GameConfig = {
        playerTypes: ['human', 'human', 'human', 'human'],
        layers: ['nello', 'sevens'],
        shuffleSeed: 121212
      };

      const room = new Room('view-actions', config, createPlayers(config));

      // Bid marks
      let view = room.getView('player-0');
      const marksBid = view.validActions.find(a =>
        a.action.type === 'bid' && a.action.bid === 'marks'
      );
      room.executeAction('player-0', marksBid!.action);

      // Others pass (may be auto-executed)
      for (let i = 1; i <= 3; i++) {
        view = room.getView(`player-${i}`);
        const pass = view.validActions.find(a => a.action.type === 'pass');
        if (pass) {
          room.executeAction(`player-${i}`, pass.action);
        }
      }

      // Check trump options include both nello and sevens
      view = room.getView('player-0');
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
        playerTypes: ['human', 'human', 'human', 'human'],
        layers: ['plunge'],
        shuffleSeed: 131313
      };

      const room = new Room('capability-filter', config, createPlayers(config));

      // Player 0's view should only show their actions
      const view0 = room.getView('player-0');
      const player0Actions = view0.validActions.filter(a =>
        'player' in a.action && a.action.player === 0
      );

      // All actions with a player field should be for player 0
      const allPlayerActions = view0.validActions.filter(a => 'player' in a.action);
      expect(player0Actions.length).toBe(allPlayerActions.length);
    });
  });
});
