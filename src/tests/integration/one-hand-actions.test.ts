/**
 * Integration tests for one-hand mode retry/new-hand actions
 *
 * Verifies that retry-one-hand and new-one-hand actions work correctly:
 * - Reset game state to bidding phase
 * - Preserve/change seed appropriately
 * - Work only in one-hand-complete phase
 */

import { describe, it, expect } from 'vitest';
import { executeAction } from '../../game/core/actions';
import { createTestContext } from '../helpers/executionContext';
import type { GameState, GameAction } from '../../game/types';
import { dealDominoesWithSeed } from '../../game/core/dominoes';
import { NO_LEAD_SUIT } from '../../game/types';

describe('One-Hand Actions', () => {
  const ctx = createTestContext();

  // Helper to create a minimal one-hand-complete state
  function createOneHandCompleteState(seed: number): GameState {
    const hands = dealDominoesWithSeed(seed);

    return {
      phase: 'one-hand-complete',
      shuffleSeed: seed,
      dealer: 0,
      currentPlayer: 0,
      playerTypes: ['ai', 'ai', 'ai', 'ai'],
      players: [
        { id: 0, name: 'Player 1', hand: hands[0]!, teamId: 0, marks: 0 },
        { id: 1, name: 'Player 2', hand: hands[1]!, teamId: 1, marks: 0 },
        { id: 2, name: 'Player 3', hand: hands[2]!, teamId: 0, marks: 0 },
        { id: 3, name: 'Player 4', hand: hands[3]!, teamId: 1, marks: 0 }
      ],
      bids: [],
      currentBid: { type: 'pass', player: -1, value: 0 },
      winningBidder: -1,
      trump: { type: 'not-selected' },
      tricks: [],
      currentTrick: [],
      currentSuit: NO_LEAD_SUIT,
      teamScores: [5, 3],
      teamMarks: [1, 0],
      consensus: {
        completeTrick: new Set(),
        scoreHand: new Set()
      },
      actionHistory: [],
      gameTarget: 7,
      initialConfig: {
        playerTypes: ['ai', 'ai', 'ai', 'ai'],
        enabledLayers: ['oneHand']
      },
      theme: 'business',
      colorOverrides: {}
    };
  }

  describe('retry-one-hand action', () => {
    it('should reset to bidding phase with same seed', () => {
      const initialState = createOneHandCompleteState(12345);
      const action: GameAction = { type: 'retry-one-hand' };

      const newState = executeAction(initialState, action, ctx.rules);

      expect(newState.phase).toBe('bidding');
      expect(newState.shuffleSeed).toBe(12345); // Same seed
      expect(newState.bids).toEqual([]);
      expect(newState.teamScores).toEqual([0, 0]);
      expect(newState.tricks).toEqual([]);
    });

    it('should deal new hands with same seed', () => {
      const initialState = createOneHandCompleteState(12345);
      const originalHands = initialState.players.map(p => p.hand);

      const action: GameAction = { type: 'retry-one-hand' };
      const newState = executeAction(initialState, action, ctx.rules);

      // Hands should be dealt (not empty)
      newState.players.forEach(p => {
        expect(p.hand.length).toBe(7);
      });

      // Same seed means same hands
      newState.players.forEach((p, i) => {
        expect(p.hand).toEqual(originalHands[i]);
      });
    });

    it('should preserve dealer and teamMarks', () => {
      const initialState = createOneHandCompleteState(12345);
      initialState.dealer = 2;
      initialState.teamMarks = [3, 1];

      const action: GameAction = { type: 'retry-one-hand' };
      const newState = executeAction(initialState, action, ctx.rules);

      expect(newState.dealer).toBe(2);
      expect(newState.teamMarks).toEqual([3, 1]);
    });

    it('should throw error if not in one-hand-complete phase', () => {
      const initialState = createOneHandCompleteState(12345);
      initialState.phase = 'bidding';

      const action: GameAction = { type: 'retry-one-hand' };

      expect(() => executeAction(initialState, action, ctx.rules)).toThrow(
        'Can only retry in one-hand-complete phase'
      );
    });

    it('should clear consensus', () => {
      const initialState = createOneHandCompleteState(12345);
      initialState.consensus.scoreHand.add(0);
      initialState.consensus.scoreHand.add(1);

      const action: GameAction = { type: 'retry-one-hand' };
      const newState = executeAction(initialState, action, ctx.rules);

      expect(newState.consensus.scoreHand.size).toBe(0);
      expect(newState.consensus.completeTrick.size).toBe(0);
    });

    it('should work when dealer is not player 3', () => {
      const initialState = createOneHandCompleteState(12345);
      initialState.dealer = 1; // Player 1 is dealer
      // currentPlayer after reset should be player 2 (left of dealer)

      const action: GameAction = { type: 'retry-one-hand' };
      const newState = executeAction(initialState, action, ctx.rules);

      expect(newState.phase).toBe('bidding');
      expect(newState.dealer).toBe(1);
      expect(newState.currentPlayer).toBe(2); // Left of dealer
    });
  });

  describe('new-one-hand action', () => {
    it('should reset to bidding phase with new seed', () => {
      const initialState = createOneHandCompleteState(12345);
      const action: GameAction = { type: 'new-one-hand' };

      const newState = executeAction(initialState, action, ctx.rules);

      expect(newState.phase).toBe('bidding');
      expect(newState.shuffleSeed).toBe(12345 + 1000000); // New seed
      expect(newState.bids).toEqual([]);
      expect(newState.teamScores).toEqual([0, 0]);
      expect(newState.tricks).toEqual([]);
    });

    it('should deal different hands with new seed', () => {
      const initialState = createOneHandCompleteState(12345);
      const originalHands = initialState.players.map(p => p.hand);

      const action: GameAction = { type: 'new-one-hand' };
      const newState = executeAction(initialState, action, ctx.rules);

      // Hands should be dealt (not empty)
      newState.players.forEach(p => {
        expect(p.hand.length).toBe(7);
      });

      // Different seed means different hands
      let foundDifference = false;
      newState.players.forEach((p, i) => {
        if (JSON.stringify(p.hand) !== JSON.stringify(originalHands[i])) {
          foundDifference = true;
        }
      });
      expect(foundDifference).toBe(true);
    });

    it('should preserve dealer and teamMarks', () => {
      const initialState = createOneHandCompleteState(12345);
      initialState.dealer = 1;
      initialState.teamMarks = [2, 2];

      const action: GameAction = { type: 'new-one-hand' };
      const newState = executeAction(initialState, action, ctx.rules);

      expect(newState.dealer).toBe(1);
      expect(newState.teamMarks).toEqual([2, 2]);
    });

    it('should throw error if not in one-hand-complete phase', () => {
      const initialState = createOneHandCompleteState(12345);
      initialState.phase = 'playing';

      const action: GameAction = { type: 'new-one-hand' };

      expect(() => executeAction(initialState, action, ctx.rules)).toThrow(
        'Can only start new hand in one-hand-complete phase'
      );
    });
  });

  describe('action generation in one-hand-complete phase', () => {
    it('should generate retry-one-hand and new-one-hand actions', () => {
      const state = createOneHandCompleteState(12345);
      const actions = ctx.getValidActions(state);

      expect(actions).toContainEqual({ type: 'retry-one-hand' });
      expect(actions).toContainEqual({ type: 'new-one-hand' });
      expect(actions.length).toBe(2);
    });

    it('should not generate other actions in terminal phase', () => {
      const state = createOneHandCompleteState(12345);
      const actions = ctx.getValidActions(state);

      const nonOneHandActions = actions.filter(
        a => a.type !== 'retry-one-hand' && a.type !== 'new-one-hand'
      );
      expect(nonOneHandActions.length).toBe(0);
    });
  });
});
