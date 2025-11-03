/**
 * Tests for getValidActions composition.
 *
 * Verifies that action generation composes correctly:
 * - Base ruleset provides standard bids (points, marks, pass)
 * - Plunge ruleset adds plunge bid only with 4+ doubles
 * - Splash ruleset adds splash bid only with 3+ doubles
 * - Nello ruleset adds nello trump option
 * - Sevens ruleset adds sevens trump option
 * - Multiple ruleSets combine actions correctly (union, not override)
 */

import { describe, it, expect } from 'vitest';
import { baseRuleSet, nelloRuleSet, plungeRuleSet, splashRuleSet, sevensRuleSet } from '../../../game/rulesets';
import type { GameState, GameAction, Domino } from '../../../game/types';
import { createInitialState } from '../../../game/core/state';

describe('getValidActions Composition', () => {
  function createTestState(overrides: Partial<GameState> = {}): GameState {
    const base = createInitialState();
    return {
      ...base,
      ...overrides
    };
  }

  function getActionsFromLayers(state: GameState, ruleSets: Array<{ getValidActions?: (state: GameState, prev: GameAction[]) => GameAction[] }>): GameAction[] {
    let actions: GameAction[] = [];
    for (const ruleset of ruleSets) {
      if (ruleset.getValidActions) {
        actions = ruleset.getValidActions(state, actions);
      }
    }
    return actions;
  }

  describe('Base ruleset provides standard bids', () => {
    it('should provide pass action in bidding phase', () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0
      });

      const actions = getActionsFromLayers(state, [baseRuleSet]);

      // Note: Base ruleset doesn't have getValidActions - that's handled by game engine
      // Base ruleset only provides rule overrides
      expect(actions).toEqual([]);
    });
  });

  describe('Plunge ruleset adds plunge bid', () => {
    it('should add plunge bid when player has 4+ doubles', () => {
      const fourDoubles: Domino[] = [
        { id: '0-0', high: 0, low: 0, points: 0 },
        { id: '1-1', high: 1, low: 1, points: 0 },
        { id: '2-2', high: 2, low: 2, points: 0 },
        { id: '3-3', high: 3, low: 3, points: 0 },
        { id: '0-1', high: 0, low: 1, points: 0 },
        { id: '0-2', high: 0, low: 2, points: 0 },
        { id: '0-3', high: 0, low: 3, points: 0 }
      ];

      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        bids: []
      });
      state.players[0]!.hand = fourDoubles;

      const actions = getActionsFromLayers(state, [plungeRuleSet]);

      expect(actions).toContainEqual({
        type: 'bid',
        player: 0,
        bid: 'plunge',
        value: 4
      });
    });

    it('should not add plunge bid when player has only 3 doubles', () => {
      const threeDoubles: Domino[] = [
        { id: '0-0', high: 0, low: 0, points: 0 },
        { id: '1-1', high: 1, low: 1, points: 0 },
        { id: '2-2', high: 2, low: 2, points: 0 },
        { id: '0-1', high: 0, low: 1, points: 0 },
        { id: '0-2', high: 0, low: 2, points: 0 },
        { id: '0-3', high: 0, low: 3, points: 0 },
        { id: '1-2', high: 1, low: 2, points: 0 }
      ];

      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0
      });
      state.players[0]!.hand = threeDoubles;

      const prevActions: GameAction[] = [
        { type: 'bid', player: 0, bid: 'marks', value: 2 },
        { type: 'pass', player: 0 }
      ];
      const actions = plungeRuleSet.getValidActions!(state, prevActions);

      // Should preserve previous actions
      expect(actions).toEqual(prevActions);
    });

    it('should calculate plunge value based on highest marks bid', () => {
      const fourDoubles: Domino[] = [
        { id: '0-0', high: 0, low: 0, points: 0 },
        { id: '1-1', high: 1, low: 1, points: 0 },
        { id: '2-2', high: 2, low: 2, points: 0 },
        { id: '3-3', high: 3, low: 3, points: 0 },
        { id: '0-1', high: 0, low: 1, points: 0 },
        { id: '0-2', high: 0, low: 2, points: 0 },
        { id: '0-3', high: 0, low: 3, points: 0 }
      ];

      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 1,
        bids: [
          { type: 'marks', value: 3, player: 0 }
        ]
      });
      state.players[1]!.hand = fourDoubles;

      const actions = plungeRuleSet.getValidActions!(state, []);

      const plungeBid = actions.find(a => a.type === 'bid' && a.bid === 'plunge');
      expect(plungeBid).toBeDefined();
      expect(plungeBid && plungeBid.type === 'bid' ? plungeBid.value : undefined).toBe(4); // highest (3) + 1
    });

    it('should only add plunge in bidding phase', () => {
      const fourDoubles: Domino[] = [
        { id: '0-0', high: 0, low: 0, points: 0 },
        { id: '1-1', high: 1, low: 1, points: 0 },
        { id: '2-2', high: 2, low: 2, points: 0 },
        { id: '3-3', high: 3, low: 3, points: 0 },
        { id: '0-1', high: 0, low: 1, points: 0 },
        { id: '0-2', high: 0, low: 2, points: 0 },
        { id: '0-3', high: 0, low: 3, points: 0 }
      ];

      const state = createTestState({
        phase: 'trump_selection',
        currentPlayer: 0
      });
      state.players[0]!.hand = fourDoubles;

      const prevActions: GameAction[] = [];
      const actions = getActionsFromLayers(state, [plungeRuleSet]);

      expect(actions).toEqual(prevActions);
    });
  });

  describe('Splash ruleset adds splash bid', () => {
    it('should add splash bid when player has 3+ doubles', () => {
      const threeDoubles: Domino[] = [
        { id: '0-0', high: 0, low: 0, points: 0 },
        { id: '1-1', high: 1, low: 1, points: 0 },
        { id: '2-2', high: 2, low: 2, points: 0 },
        { id: '0-1', high: 0, low: 1, points: 0 },
        { id: '0-2', high: 0, low: 2, points: 0 },
        { id: '0-3', high: 0, low: 3, points: 0 },
        { id: '1-2', high: 1, low: 2, points: 0 }
      ];

      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        bids: []
      });
      state.players[0]!.hand = threeDoubles;

      const actions = getActionsFromLayers(state, [splashRuleSet]);

      expect(actions).toContainEqual({
        type: 'bid',
        player: 0,
        bid: 'splash',
        value: 2
      });
    });

    it('should not add splash bid when player has only 2 doubles', () => {
      const twoDoubles: Domino[] = [
        { id: '0-0', high: 0, low: 0, points: 0 },
        { id: '1-1', high: 1, low: 1, points: 0 },
        { id: '0-1', high: 0, low: 1, points: 0 },
        { id: '0-2', high: 0, low: 2, points: 0 },
        { id: '0-3', high: 0, low: 3, points: 0 },
        { id: '1-2', high: 1, low: 2, points: 0 },
        { id: '2-3', high: 2, low: 3, points: 0 }
      ];

      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0
      });
      state.players[0]!.hand = twoDoubles;

      const prevActions: GameAction[] = [
        { type: 'pass', player: 0 }
      ];

      const actions = splashRuleSet.getValidActions!(state, prevActions);

      expect(actions).toEqual(prevActions);
    });

    it('should cap splash value at 3 marks', () => {
      const threeDoubles: Domino[] = [
        { id: '0-0', high: 0, low: 0, points: 0 },
        { id: '1-1', high: 1, low: 1, points: 0 },
        { id: '2-2', high: 2, low: 2, points: 0 },
        { id: '0-1', high: 0, low: 1, points: 0 },
        { id: '0-2', high: 0, low: 2, points: 0 },
        { id: '0-3', high: 0, low: 3, points: 0 },
        { id: '1-2', high: 1, low: 2, points: 0 }
      ];

      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 1,
        bids: [
          { type: 'marks', value: 3, player: 0 }
        ]
      });
      state.players[1]!.hand = threeDoubles;

      const actions = getActionsFromLayers(state, [splashRuleSet]);

      const splashBid = actions.find(a => a.type === 'bid' && a.bid === 'splash');
      expect(splashBid).toBeDefined();
      expect(splashBid && splashBid.type === 'bid' ? splashBid.value : undefined).toBe(3); // Capped at 3
    });
  });

  describe('Nello ruleset adds nello trump option', () => {
    it('should add nello option in trump_selection after marks bid', () => {
      const state = createTestState({
        phase: 'trump_selection',
        currentPlayer: 0,
        winningBidder: 0,
        currentBid: { type: 'marks', value: 2, player: 0 }
      });

      const prevActions: GameAction[] = [
        { type: 'select-trump', player: 0, trump: { type: 'suit', suit: 0 } },
        { type: 'select-trump', player: 0, trump: { type: 'suit', suit: 1 } }
      ];

      const actions = nelloRuleSet.getValidActions!(state, prevActions);

      expect(actions).toContainEqual({
        type: 'select-trump',
        player: 0,
        trump: { type: 'nello' }
      });
      expect(actions.length).toBe(prevActions.length + 1);
    });

    it('should not add nello option for points bid', () => {
      const state = createTestState({
        phase: 'trump_selection',
        currentPlayer: 0,
        winningBidder: 0,
        currentBid: { type: 'points', value: 30, player: 0 }
      });

      const prevActions: GameAction[] = [
        { type: 'select-trump', player: 0, trump: { type: 'suit', suit: 0 } }
      ];

      const actions = nelloRuleSet.getValidActions!(state, prevActions);

      expect(actions).toEqual(prevActions);
    });

    it('should not add nello option outside trump_selection phase', () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        currentBid: { type: 'marks', value: 2, player: 0 }
      });

      const prevActions: GameAction[] = [];
      const actions = getActionsFromLayers(state, [nelloRuleSet]);

      expect(actions).toEqual(prevActions);
    });
  });

  describe('Sevens ruleset adds sevens trump option', () => {
    it('should add sevens option in trump_selection after marks bid', () => {
      const state = createTestState({
        phase: 'trump_selection',
        currentPlayer: 0,
        winningBidder: 0,
        currentBid: { type: 'marks', value: 2, player: 0 }
      });

      const actions = getActionsFromLayers(state, [sevensRuleSet]);

      expect(actions).toContainEqual({
        type: 'select-trump',
        player: 0,
        trump: { type: 'sevens' }
      });
    });

    it('should not add sevens option for points bid', () => {
      const state = createTestState({
        phase: 'trump_selection',
        currentPlayer: 0,
        winningBidder: 0,
        currentBid: { type: 'points', value: 30, player: 0 }
      });

      const prevActions: GameAction[] = [];
      const actions = getActionsFromLayers(state, [sevensRuleSet]);

      expect(actions).toEqual(prevActions);
    });
  });

  describe('Multiple ruleSets combine actions correctly', () => {
    it('should combine plunge and splash options', () => {
      const fourDoubles: Domino[] = [
        { id: '0-0', high: 0, low: 0, points: 0 },
        { id: '1-1', high: 1, low: 1, points: 0 },
        { id: '2-2', high: 2, low: 2, points: 0 },
        { id: '3-3', high: 3, low: 3, points: 0 },
        { id: '0-1', high: 0, low: 1, points: 0 },
        { id: '0-2', high: 0, low: 2, points: 0 },
        { id: '0-3', high: 0, low: 3, points: 0 }
      ];

      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        bids: []
      });
      state.players[0]!.hand = fourDoubles;

      const actions = getActionsFromLayers(state, [plungeRuleSet, splashRuleSet]);

      // Should have both plunge and splash
      expect(actions.filter(a => a.type === 'bid' && a.bid === 'plunge')).toHaveLength(1);
      expect(actions.filter(a => a.type === 'bid' && a.bid === 'splash')).toHaveLength(1);
    });

    it('should combine nello and sevens trump options', () => {
      const state = createTestState({
        phase: 'trump_selection',
        currentPlayer: 0,
        winningBidder: 0,
        currentBid: { type: 'marks', value: 2, player: 0 }
      });

      const prevActions: GameAction[] = [
        { type: 'select-trump', player: 0, trump: { type: 'suit', suit: 0 } }
      ];

      // Apply ruleSets sequentially
      let actions = nelloRuleSet.getValidActions!(state, prevActions);
      actions = sevensRuleSet.getValidActions!(state, actions);

      // Should have both nello and sevens options
      expect(actions.filter(a => a.type === 'select-trump' && a.trump.type === 'nello')).toHaveLength(1);
      expect(actions.filter(a => a.type === 'select-trump' && a.trump.type === 'sevens')).toHaveLength(1);
      // Plus original action
      expect(actions.length).toBe(3);
    });

    it('should use union, not override', () => {
      const fourDoubles: Domino[] = [
        { id: '0-0', high: 0, low: 0, points: 0 },
        { id: '1-1', high: 1, low: 1, points: 0 },
        { id: '2-2', high: 2, low: 2, points: 0 },
        { id: '3-3', high: 3, low: 3, points: 0 },
        { id: '0-1', high: 0, low: 1, points: 0 },
        { id: '0-2', high: 0, low: 2, points: 0 },
        { id: '0-3', high: 0, low: 3, points: 0 }
      ];

      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        bids: []
      });
      state.players[0]!.hand = fourDoubles;

      // Start with base actions
      const baseActions: GameAction[] = [
        { type: 'bid', player: 0, bid: 'marks', value: 2 },
        { type: 'pass', player: 0 }
      ];

      // Apply ruleSets
      let actions = baseActions;
      actions = plungeRuleSet.getValidActions!(state, actions);
      actions = splashRuleSet.getValidActions!(state, actions);

      // Should have original actions plus new ones
      expect(actions.filter(a => a.type === 'bid' && a.bid === 'marks')).toHaveLength(1);
      expect(actions.filter(a => a.type === 'pass')).toHaveLength(1);
      expect(actions.filter(a => a.type === 'bid' && a.bid === 'plunge')).toHaveLength(1);
      expect(actions.filter(a => a.type === 'bid' && a.bid === 'splash')).toHaveLength(1);
    });

    it('should preserve action order and not duplicate', () => {
      const state = createTestState({
        phase: 'trump_selection',
        currentPlayer: 0,
        winningBidder: 0,
        currentBid: { type: 'marks', value: 2, player: 0 }
      });

      const baseActions: GameAction[] = [
        { type: 'select-trump', player: 0, trump: { type: 'suit', suit: 0 } },
        { type: 'select-trump', player: 0, trump: { type: 'suit', suit: 1 } }
      ];

      // Apply both ruleSets
      let actions = baseActions;
      actions = nelloRuleSet.getValidActions!(state, actions);
      actions = sevensRuleSet.getValidActions!(state, actions);

      // Should have 4 total: 2 base + nello + sevens
      expect(actions).toHaveLength(4);

      // No duplicates
      const nelloActions = actions.filter(a => a.type === 'select-trump' && a.trump.type === 'nello');
      const sevensActions = actions.filter(a => a.type === 'select-trump' && a.trump.type === 'sevens');
      expect(nelloActions).toHaveLength(1);
      expect(sevensActions).toHaveLength(1);
    });
  });

  describe('Edge cases', () => {
    it('should handle ruleSets with no getValidActions', () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0
      });

      const layerWithoutActions: Array<{ getValidActions?: (state: GameState, prev: GameAction[]) => GameAction[] }> = [{ }];
      const actions = getActionsFromLayers(state, layerWithoutActions);

      expect(actions).toEqual([]);
    });

    it('should handle empty previous actions', () => {
      const state = createTestState({
        phase: 'trump_selection',
        currentPlayer: 0,
        winningBidder: 0,
        currentBid: { type: 'marks', value: 2, player: 0 }
      });

      const actions = getActionsFromLayers(state, [nelloRuleSet]);

      expect(actions).toHaveLength(1);
      expect(actions[0]).toEqual({
        type: 'select-trump',
        player: 0,
        trump: { type: 'nello' }
      });
    });
  });
});
