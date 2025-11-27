/**
 * Tests for getValidActions composition.
 *
 * Tests the core composition mechanism: each layer's getValidActions receives
 * previous actions and can filter, add, or annotate them.
 */

import { describe, it, expect } from 'vitest';
import { baseLayer, nelloLayer, plungeLayer, splashLayer, sevensLayer } from '../../../game/layers';
import type { GameState, GameAction, Domino } from '../../../game/types';
import { createInitialState } from '../../../game/core/state';

describe('getValidActions Composition', () => {
  function createTestState(overrides: Partial<GameState> = {}): GameState {
    const base = createInitialState();
    return { ...base, ...overrides };
  }

  function getActionsFromLayers(
    state: GameState,
    layers: Array<{ getValidActions?: (state: GameState, prev: GameAction[]) => GameAction[] }>
  ): GameAction[] {
    let actions: GameAction[] = [];
    for (const layer of layers) {
      if (layer.getValidActions) {
        actions = layer.getValidActions(state, actions);
      }
    }
    return actions;
  }

  const threeDoubles: Domino[] = [
    { id: '0-0', high: 0, low: 0, points: 0 },
    { id: '1-1', high: 1, low: 1, points: 0 },
    { id: '2-2', high: 2, low: 2, points: 0 },
    { id: '0-1', high: 0, low: 1, points: 0 },
    { id: '0-2', high: 0, low: 2, points: 0 },
    { id: '0-3', high: 0, low: 3, points: 0 },
    { id: '1-2', high: 1, low: 2, points: 0 }
  ];

  const fourDoubles: Domino[] = [
    { id: '0-0', high: 0, low: 0, points: 0 },
    { id: '1-1', high: 1, low: 1, points: 0 },
    { id: '2-2', high: 2, low: 2, points: 0 },
    { id: '3-3', high: 3, low: 3, points: 0 },
    { id: '0-1', high: 0, low: 1, points: 0 },
    { id: '0-2', high: 0, low: 2, points: 0 },
    { id: '0-3', high: 0, low: 3, points: 0 }
  ];

  describe('Basic composition', () => {
    it('should thread actions through layers', () => {
      const state = createTestState({ phase: 'bidding', currentPlayer: 0, bids: [] });
      const actions = getActionsFromLayers(state, [baseLayer]);

      expect(actions.length).toBeGreaterThan(0);
      expect(actions).toContainEqual({ type: 'bid', player: 0, bid: 'points', value: 30 });
      expect(actions).toContainEqual({ type: 'bid', player: 0, bid: 'marks', value: 1 });
    });
  });

  describe('Filtering', () => {
    it('should preserve actions when conditions not met', () => {
      const state = createTestState({ phase: 'bidding', currentPlayer: 0 });
      state.players[0]!.hand = threeDoubles;

      const prevActions: GameAction[] = [
        { type: 'bid', player: 0, bid: 'marks', value: 2 },
        { type: 'pass', player: 0 }
      ];
      const actions = plungeLayer.getValidActions!(state, prevActions);

      expect(actions).toEqual(prevActions);
    });
  });

  describe('Adding actions', () => {
    it('should add plunge bid with 4+ doubles', () => {
      const state = createTestState({ phase: 'bidding', currentPlayer: 0, bids: [] });
      state.players[0]!.hand = fourDoubles;

      const actions = getActionsFromLayers(state, [plungeLayer]);

      expect(actions).toContainEqual({ type: 'bid', player: 0, bid: 'plunge', value: 4 });
    });

    it('should add splash bid with 3+ doubles', () => {
      const state = createTestState({ phase: 'bidding', currentPlayer: 0, bids: [] });
      state.players[0]!.hand = threeDoubles;

      const actions = getActionsFromLayers(state, [splashLayer]);

      expect(actions).toContainEqual({ type: 'bid', player: 0, bid: 'splash', value: 2 });
    });

    it('should add nello trump option for marks bid', () => {
      const state = createTestState({
        phase: 'trump_selection',
        currentPlayer: 0,
        winningBidder: 0,
        currentBid: { type: 'marks', value: 2, player: 0 }
      });

      const actions = nelloLayer.getValidActions!(state, [
        { type: 'select-trump', player: 0, trump: { type: 'suit', suit: 0 } }
      ]);

      expect(actions).toContainEqual({ type: 'select-trump', player: 0, trump: { type: 'nello' } });
      expect(actions.length).toBe(2);
    });

    it('should add sevens trump option for marks bid', () => {
      const state = createTestState({
        phase: 'trump_selection',
        currentPlayer: 0,
        winningBidder: 0,
        currentBid: { type: 'marks', value: 2, player: 0 }
      });

      const actions = getActionsFromLayers(state, [sevensLayer]);

      expect(actions).toContainEqual({ type: 'select-trump', player: 0, trump: { type: 'sevens' } });
    });
  });

  describe('Annotating', () => {
    it('should calculate plunge value based on highest marks bid', () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 1,
        bids: [{ type: 'marks', value: 3, player: 0 }]
      });
      state.players[1]!.hand = fourDoubles;

      const actions = plungeLayer.getValidActions!(state, []);

      const plungeBid = actions.find(a => a.type === 'bid' && a.bid === 'plunge');
      expect(plungeBid && plungeBid.type === 'bid' ? plungeBid.value : undefined).toBe(4);
    });
  });

  describe('Layer order', () => {
    it('should apply layers sequentially - later layers see earlier changes', () => {
      const state = createTestState({
        phase: 'trump_selection',
        currentPlayer: 0,
        winningBidder: 0,
        currentBid: { type: 'marks', value: 2, player: 0 }
      });

      let actions = nelloLayer.getValidActions!(state, [
        { type: 'select-trump', player: 0, trump: { type: 'suit', suit: 0 } }
      ]);
      actions = sevensLayer.getValidActions!(state, actions);

      expect(actions.length).toBe(3);
      expect(actions.filter(a => a.type === 'select-trump' && a.trump.type === 'nello')).toHaveLength(1);
      expect(actions.filter(a => a.type === 'select-trump' && a.trump.type === 'sevens')).toHaveLength(1);
    });
  });

  describe('Union behavior', () => {
    it('should preserve previous actions while adding new ones', () => {
      const state = createTestState({ phase: 'bidding', currentPlayer: 0, bids: [] });
      state.players[0]!.hand = fourDoubles;

      let actions = plungeLayer.getValidActions!(state, [
        { type: 'bid', player: 0, bid: 'marks', value: 2 },
        { type: 'pass', player: 0 }
      ]);
      actions = splashLayer.getValidActions!(state, actions);

      expect(actions.filter(a => a.type === 'bid' && a.bid === 'marks')).toHaveLength(1);
      expect(actions.filter(a => a.type === 'pass')).toHaveLength(1);
      expect(actions.filter(a => a.type === 'bid' && a.bid === 'plunge')).toHaveLength(1);
      expect(actions.filter(a => a.type === 'bid' && a.bid === 'splash')).toHaveLength(1);
    });

    it('should not duplicate actions', () => {
      const state = createTestState({
        phase: 'trump_selection',
        currentPlayer: 0,
        winningBidder: 0,
        currentBid: { type: 'marks', value: 2, player: 0 }
      });

      let actions = nelloLayer.getValidActions!(state, [
        { type: 'select-trump', player: 0, trump: { type: 'suit', suit: 0 } },
        { type: 'select-trump', player: 0, trump: { type: 'suit', suit: 1 } }
      ]);
      actions = sevensLayer.getValidActions!(state, actions);

      expect(actions).toHaveLength(4);
      expect(actions.filter(a => a.type === 'select-trump' && a.trump.type === 'nello')).toHaveLength(1);
      expect(actions.filter(a => a.type === 'select-trump' && a.trump.type === 'sevens')).toHaveLength(1);
    });
  });

  describe('Edge cases', () => {
    it('should handle layers without getValidActions', () => {
      const state = createTestState({ phase: 'bidding', currentPlayer: 0 });
      expect(getActionsFromLayers(state, [{}])).toEqual([]);
    });

    it('should handle empty previous actions', () => {
      const state = createTestState({
        phase: 'trump_selection',
        currentPlayer: 0,
        winningBidder: 0,
        currentBid: { type: 'marks', value: 2, player: 0 }
      });

      const actions = getActionsFromLayers(state, [nelloLayer]);

      expect(actions).toHaveLength(1);
      expect(actions[0]).toEqual({ type: 'select-trump', player: 0, trump: { type: 'nello' } });
    });
  });
});
