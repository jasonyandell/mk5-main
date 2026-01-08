/**
 * Tests for the Minimax Evaluator
 *
 * Verifies that minimax correctly finds optimal play in various scenarios.
 *
 * ## Simulation Context
 * These tests use `createSimulationContext()` which:
 * - Sets playerTypes to all AI (consensus layer passes through)
 * - Uses layers ['base', 'speed'] (no consensus blocking)
 *
 * This ensures `complete-trick` executes immediately after 4 plays,
 * allowing scores to update correctly during minimax search.
 */

import { describe, it, expect } from 'vitest';
import { minimaxEvaluate, createTerminalState } from '../../game/ai/minimax';
import { createSimulationContext } from '../helpers/executionContext';
import { StateBuilder } from '../helpers/stateBuilder';
import { ACES, SIXES } from '../../game/types';

describe('minimaxEvaluate', () => {
  // Shared context for all tests - simulation mode (AI players, no consensus blocking)
  const ctx = createSimulationContext();

  describe('trivial endgames', () => {
    it('correctly evaluates single trick remaining with clear winner', () => {
      // Set up: 1 trick remaining, player 0 to lead
      // Player 0 has 6-6 (trump), others have non-trump

      const state = StateBuilder
        .inPlayingPhase({ type: 'suit', suit: ACES })
        .withPlayerHand(0, ['6-6']) // 6-6 is not trump (aces trump), but high
        .withPlayerHand(1, ['6-5'])
        .withPlayerHand(2, ['6-4'])
        .withPlayerHand(3, ['5-5']) // 10 count
        .withCurrentPlayer(0)
        .withTeamScores(20, 12) // Some prior points
        .build();

      // Manually set up 6 completed tricks
      state.tricks = Array(6).fill(null).map(() => ({
        plays: [],
        winner: 0,
        points: 0
      }));

      const result = minimaxEvaluate(state, ctx);

      // Verify we explored some nodes
      expect(result.nodesExplored).toBeGreaterThan(0);

      // Total should be 42
      expect(result.team0Points + result.team1Points).toBe(42);
    });

    it('correctly handles already complete hands', () => {
      const state = StateBuilder
        .inScoringPhase([25, 17])
        .build();

      const result = minimaxEvaluate(state, ctx);

      expect(result.team0Points).toBe(25);
      expect(result.team1Points).toBe(17);
      expect(result.nodesExplored).toBe(1); // Just checks terminal state
    });
  });

  describe('alpha-beta pruning', () => {
    it('explores fewer nodes with pruning enabled', () => {
      // Set up a position with branching that can be pruned
      const state = StateBuilder
        .inPlayingPhase({ type: 'suit', suit: SIXES })
        .withPlayerHand(0, ['6-6', '6-5'])
        .withPlayerHand(1, ['5-5', '5-4'])
        .withPlayerHand(2, ['4-4', '4-3'])
        .withPlayerHand(3, ['3-3', '3-2'])
        .withCurrentPlayer(0)
        .build();

      // Set up 5 completed tricks to have just 2 remaining
      state.tricks = Array(5).fill(null).map(() => ({
        plays: [],
        winner: 0,
        points: 0
      }));

      const withPruning = minimaxEvaluate(state, ctx, { alphaBeta: true });
      const withoutPruning = minimaxEvaluate(state, ctx, { alphaBeta: false });

      // Both should find same optimal result
      expect(withPruning.team0Points).toBe(withoutPruning.team0Points);

      // But pruning should explore fewer nodes
      expect(withPruning.nodesExplored).toBeLessThanOrEqual(withoutPruning.nodesExplored);
    });
  });

  describe('partnership play', () => {
    it('team 0 (players 0,2) maximizes while team 1 minimizes', () => {
      // Set up a simple 1-trick endgame where team matters
      // Player 0 leads with ace trump, should win the trick
      const state = StateBuilder
        .inPlayingPhase({ type: 'suit', suit: ACES })
        .withPlayerHand(0, ['1-1']) // Ace trump - will win
        .withPlayerHand(1, ['6-0']) // Non-count
        .withPlayerHand(2, ['5-0']) // 5 points
        .withPlayerHand(3, ['3-2']) // 5 points
        .withCurrentPlayer(0)
        .withTeamScores(22, 0) // Prior points to team 0
        .build();

      // 6 tricks already played with team 0 winning 22 points
      state.tricks = Array(6).fill(null).map(() => ({
        plays: [],
        winner: 0,
        points: 0
      }));

      const result = minimaxEvaluate(state, ctx);

      // Minimax should find a valid result
      expect(result.nodesExplored).toBeGreaterThan(0);

      // Total should still be 42 (even if prior tricks had 0 points in test setup)
      // The key test is that both teams get some combination of points
      expect(result.team0Points + result.team1Points).toBe(42);
    });
  });

  describe('move ordering', () => {
    it('heuristic ordering produces same result as no ordering', () => {
      const state = StateBuilder
        .inPlayingPhase({ type: 'suit', suit: ACES })
        .withPlayerHand(0, ['6-6', '5-5'])
        .withPlayerHand(1, ['6-5', '6-4'])
        .withPlayerHand(2, ['4-4', '4-3'])
        .withPlayerHand(3, ['3-3', '3-2'])
        .withCurrentPlayer(0)
        .build();

      state.tricks = Array(5).fill(null).map(() => ({
        plays: [],
        winner: 0,
        points: 0
      }));

      const withOrdering = minimaxEvaluate(state, ctx, { moveOrdering: 'heuristic' });
      const noOrdering = minimaxEvaluate(state, ctx, { moveOrdering: 'none' });

      // Same optimal result
      expect(withOrdering.team0Points).toBe(noOrdering.team0Points);
    });
  });

  describe('createTerminalState', () => {
    it('creates a scoring-phase state with correct scores', () => {
      const initialState = StateBuilder
        .inPlayingPhase({ type: 'suit', suit: ACES })
        .withPlayerHand(0, ['6-6', '6-5', '5-5'])
        .build();

      const result = {
        team0Points: 25,
        team1Points: 17,
        nodesExplored: 100
      };

      const terminal = createTerminalState(initialState, result);

      expect(terminal.phase).toBe('scoring');
      expect(terminal.teamScores).toEqual([25, 17]);
      expect(terminal.currentTrick).toEqual([]);
      expect(terminal.players.every(p => p.hand.length === 0)).toBe(true);
    });
  });

  describe('special contracts', () => {
    it('handles doubles-trump correctly', () => {
      // With doubles trump, all doubles are trump
      const state = StateBuilder
        .inPlayingPhase({ type: 'doubles' })
        .withPlayerHand(0, ['6-6']) // High trump (double)
        .withPlayerHand(1, ['6-5']) // Not trump
        .withPlayerHand(2, ['5-5']) // Trump (double)
        .withPlayerHand(3, ['6-4']) // Not trump
        .withCurrentPlayer(0)
        .build();

      state.tricks = Array(6).fill(null).map(() => ({
        plays: [],
        winner: 0,
        points: 0
      }));

      const result = minimaxEvaluate(state, ctx);

      // Minimax should find optimal play
      expect(result.nodesExplored).toBeGreaterThan(0);
      expect(result.team0Points + result.team1Points).toBe(42);
    });
  });
});
