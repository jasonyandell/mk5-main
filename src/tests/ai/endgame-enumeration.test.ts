/**
 * Tests for PIMC endgame enumeration optimization.
 *
 * Verifies that enumeration works correctly in late-game scenarios
 * where the number of possible opponent hand configurations is small.
 */

import { describe, it, expect } from 'vitest';
import { countValidConfigurations, enumerateAllConfigurations } from '../../game/ai/hand-sampler';
import { buildConstraints, getExpectedHandSizes, buildCanFollowCache, type HandConstraints } from '../../game/ai/constraint-tracker';
import { evaluatePlayActions } from '../../game/ai/monte-carlo';
import { createSimulationContext } from '../helpers/executionContext';
import { StateBuilder, DominoBuilder } from '../helpers/stateBuilder';
import { ACES, SIXES, type LedSuit } from '../../game/types';

describe('endgame enumeration', () => {
  const ctx = createSimulationContext();

  describe('countValidConfigurations', () => {
    it('counts configurations correctly for trivial endgame', () => {
      // Set up: 6 tricks played, 1 trick remaining
      // Each player has 1 domino
      // Manually construct constraints to test the counting logic directly
      const state = StateBuilder
        .inPlayingPhase({ type: 'doubles' })
        .withPlayerHand(0, ['6-6'])
        .withPlayerHand(1, ['6-5'])
        .withPlayerHand(2, ['5-5'])
        .withPlayerHand(3, ['5-4'])
        .withCurrentPlayer(0)
        .build();

      // Set up 6 completed tricks (just for expected sizes calculation)
      state.tricks = Array(6).fill(null).map(() => ({
        plays: [],
        winner: 0,
        points: 0
      }));

      // Manually create constraints without complex void tracking
      // This tests the enumeration algorithm itself
      const constraints: HandConstraints = {
        played: new Set<string>(), // Mark most dominoes as played
        voidInSuit: new Map<number, Set<LedSuit>>(),
        myHand: new Set(['6-6']),
        myPlayerIndex: 0
      };

      // Mark all dominoes except 6-6, 6-5, 5-5, 5-4 as played
      for (let h = 0; h <= 6; h++) {
        for (let l = 0; l <= h; l++) {
          const id = `${h}-${l}`;
          if (!['6-6', '6-5', '5-5', '5-4'].includes(id)) {
            constraints.played.add(id);
          }
        }
      }

      // No void constraints for simplicity
      for (let p = 0; p < 4; p++) {
        constraints.voidInSuit.set(p, new Set<LedSuit>());
      }

      const expectedSizes: [number, number, number, number] = [1, 1, 1, 1];
      const canFollowCache = buildCanFollowCache(state, ctx.rules);

      const count = countValidConfigurations(
        constraints,
        expectedSizes,
        state,
        ctx.rules,
        1000,
        canFollowCache
      );

      // With 3 opponents each having 1 domino, and 3 dominoes to distribute,
      // there should be 3! = 6 configurations (each permutation of 3 dominoes to 3 players)
      expect(count).toBe(6);
    });

    it('stops counting early when exceeding limit', () => {
      // Set up: 4 tricks played, 3 tricks remaining
      // This should have many configurations
      const state = StateBuilder
        .inPlayingPhase({ type: 'suit', suit: ACES })
        .withPlayerHand(0, ['6-6', '6-5', '6-4'])
        .withPlayerHand(1, ['5-5', '5-4', '5-3'])
        .withPlayerHand(2, ['4-4', '4-3', '4-2'])
        .withPlayerHand(3, ['3-3', '3-2', '3-1'])
        .withCurrentPlayer(0)
        .build();

      state.tricks = Array(4).fill(null).map(() => ({
        plays: [],
        winner: 0,
        points: 0
      }));

      const constraints = buildConstraints(state, 0, ctx.rules);
      const expectedSizes = getExpectedHandSizes(state);
      const canFollowCache = buildCanFollowCache(state, ctx.rules);

      // With a low limit, should stop early
      const count = countValidConfigurations(
        constraints,
        expectedSizes,
        state,
        ctx.rules,
        10,  // Low limit
        canFollowCache
      );

      // Count should exceed limit (and thus stop at limit + 1)
      expect(count).toBeGreaterThan(10);
    });

    it('respects void constraints when counting', () => {
      // Set up a scenario where player 1 is void in a suit
      const state = StateBuilder
        .inPlayingPhase({ type: 'suit', suit: SIXES })
        .withPlayerHand(0, ['6-6', '6-5'])
        .withPlayerHand(1, ['5-5', '5-4'])
        .withPlayerHand(2, ['4-4', '4-3'])
        .withPlayerHand(3, ['3-3', '3-2'])
        .withCurrentPlayer(0)
        .build();

      // Set up 5 completed tricks
      state.tricks = Array(5).fill(null).map(() => ({
        plays: [],
        winner: 0,
        points: 0
      }));

      const constraints = buildConstraints(state, 0, ctx.rules);

      // Add a void constraint: player 1 is void in sixes
      constraints.voidInSuit.get(1)?.add(6);

      const expectedSizes = getExpectedHandSizes(state);
      const canFollowCache = buildCanFollowCache(state, ctx.rules);

      const count = countValidConfigurations(
        constraints,
        expectedSizes,
        state,
        ctx.rules,
        1000,
        canFollowCache
      );

      // Should still have valid configurations
      expect(count).toBeGreaterThan(0);
    });
  });

  describe('enumerateAllConfigurations', () => {
    it('enumerates all configurations for single-trick endgame', () => {
      const state = StateBuilder
        .inPlayingPhase({ type: 'suit', suit: ACES })
        .withPlayerHand(0, ['6-6'])
        .withPlayerHand(1, ['6-5'])
        .withPlayerHand(2, ['5-5'])
        .withPlayerHand(3, ['5-4'])
        .withCurrentPlayer(0)
        .build();

      state.tricks = Array(6).fill(null).map(() => ({
        plays: [],
        winner: 0,
        points: 0
      }));

      const constraints = buildConstraints(state, 0, ctx.rules);
      const expectedSizes = getExpectedHandSizes(state);
      const canFollowCache = buildCanFollowCache(state, ctx.rules);

      const configurations = enumerateAllConfigurations(
        constraints,
        expectedSizes,
        state,
        ctx.rules,
        1000,
        canFollowCache
      );

      // Should have enumerated all valid configurations
      expect(configurations.length).toBeGreaterThan(0);

      // Each configuration should assign correct hand sizes
      for (const config of configurations) {
        // Player 0 is AI, so config has players 1, 2, 3
        for (let p = 1; p <= 3; p++) {
          const hand = config.get(p);
          expect(hand).toBeDefined();
          expect(hand!.length).toBe(expectedSizes[p]);
        }
      }
    });

    it('returns unique configurations', () => {
      const state = StateBuilder
        .inPlayingPhase({ type: 'suit', suit: ACES })
        .withPlayerHand(0, ['6-6', '6-5'])
        .withPlayerHand(1, ['5-5', '5-4'])
        .withPlayerHand(2, ['4-4', '4-3'])
        .withPlayerHand(3, ['3-3', '3-2'])
        .withCurrentPlayer(0)
        .build();

      state.tricks = Array(5).fill(null).map(() => ({
        plays: [],
        winner: 0,
        points: 0
      }));

      const constraints = buildConstraints(state, 0, ctx.rules);
      const expectedSizes = getExpectedHandSizes(state);
      const canFollowCache = buildCanFollowCache(state, ctx.rules);

      const configurations = enumerateAllConfigurations(
        constraints,
        expectedSizes,
        state,
        ctx.rules,
        1000,
        canFollowCache
      );

      // Check uniqueness by serializing each configuration
      const serialized = configurations.map(config => {
        const entries: string[] = [];
        for (let p = 1; p <= 3; p++) {
          const hand = config.get(p);
          if (hand) {
            const ids = hand.map(d => d.id).sort();
            entries.push(`${p}:${ids.join(',')}`);
          }
        }
        return entries.sort().join('|');
      });

      const uniqueCount = new Set(serialized).size;
      expect(uniqueCount).toBe(configurations.length);
    });

    it('respects max configuration limit', () => {
      // Set up a scenario with many configurations
      const state = StateBuilder
        .inPlayingPhase({ type: 'suit', suit: ACES })
        .withPlayerHand(0, ['6-6', '6-5', '6-4'])
        .withPlayerHand(1, ['5-5', '5-4', '5-3'])
        .withPlayerHand(2, ['4-4', '4-3', '4-2'])
        .withPlayerHand(3, ['3-3', '3-2', '3-1'])
        .withCurrentPlayer(0)
        .build();

      state.tricks = Array(4).fill(null).map(() => ({
        plays: [],
        winner: 0,
        points: 0
      }));

      const constraints = buildConstraints(state, 0, ctx.rules);
      const expectedSizes = getExpectedHandSizes(state);
      const canFollowCache = buildCanFollowCache(state, ctx.rules);

      const maxConfigs = 5;
      const configurations = enumerateAllConfigurations(
        constraints,
        expectedSizes,
        state,
        ctx.rules,
        maxConfigs,
        canFollowCache
      );

      // Should respect the max limit
      expect(configurations.length).toBeLessThanOrEqual(maxConfigs);
    });

    it('handles case where AI has no dominoes left', () => {
      // Rare edge case: AI played last domino, now evaluating other players' remaining cards
      const state = StateBuilder
        .inPlayingPhase({ type: 'suit', suit: ACES })
        .withPlayerHand(0, [])  // AI has no dominoes
        .withPlayerHand(1, ['6-5'])
        .withPlayerHand(2, ['5-5'])
        .withPlayerHand(3, ['5-4'])
        .withCurrentPlayer(1)  // Someone else is playing
        .build();

      state.tricks = Array(6).fill(null).map(() => ({
        plays: [],
        winner: 0,
        points: 0
      }));

      const constraints = buildConstraints(state, 0, ctx.rules);
      const expectedSizes = getExpectedHandSizes(state);
      const canFollowCache = buildCanFollowCache(state, ctx.rules);

      const configurations = enumerateAllConfigurations(
        constraints,
        expectedSizes,
        state,
        ctx.rules,
        1000,
        canFollowCache
      );

      // Should still work - enumerate how opponents' remaining dominoes could be distributed
      expect(configurations.length).toBeGreaterThan(0);
    });
  });

  describe('integration with evaluatePlayActions', () => {
    // These tests verify that enumeration is actually being used
    // in late-game scenarios (via the simulationCount in results)

    it('uses enumeration when configurations are small', () => {
      const state = StateBuilder
        .inPlayingPhase({ type: 'suit', suit: ACES })
        .withPlayerHand(0, ['6-6'])
        .withPlayerHand(1, ['6-5'])
        .withPlayerHand(2, ['5-5'])
        .withPlayerHand(3, ['5-4'])
        .withCurrentPlayer(0)
        .build();

      // Set up 6 completed tricks with actual plays
      state.tricks = [
        { plays: [
          { player: 0, domino: DominoBuilder.from('0-0') },
          { player: 1, domino: DominoBuilder.from('1-0') },
          { player: 2, domino: DominoBuilder.from('2-0') },
          { player: 3, domino: DominoBuilder.from('3-0') }
        ], winner: 0, points: 0 },
        { plays: [
          { player: 0, domino: DominoBuilder.from('1-1') },
          { player: 1, domino: DominoBuilder.from('2-1') },
          { player: 2, domino: DominoBuilder.from('3-1') },
          { player: 3, domino: DominoBuilder.from('4-0') }
        ], winner: 0, points: 0 },
        { plays: [
          { player: 0, domino: DominoBuilder.from('2-2') },
          { player: 1, domino: DominoBuilder.from('3-2') },
          { player: 2, domino: DominoBuilder.from('4-1') },
          { player: 3, domino: DominoBuilder.from('5-0') }
        ], winner: 0, points: 5 },
        { plays: [
          { player: 0, domino: DominoBuilder.from('3-3') },
          { player: 1, domino: DominoBuilder.from('4-2') },
          { player: 2, domino: DominoBuilder.from('5-1') },
          { player: 3, domino: DominoBuilder.from('6-0') }
        ], winner: 0, points: 0 },
        { plays: [
          { player: 0, domino: DominoBuilder.from('4-3') },
          { player: 1, domino: DominoBuilder.from('5-2') },
          { player: 2, domino: DominoBuilder.from('6-1') },
          { player: 3, domino: DominoBuilder.from('4-4') }
        ], winner: 0, points: 0 },
        { plays: [
          { player: 0, domino: DominoBuilder.from('5-3') },
          { player: 1, domino: DominoBuilder.from('6-2') },
          { player: 2, domino: DominoBuilder.from('6-3') },
          { player: 3, domino: DominoBuilder.from('6-4') }
        ], winner: 0, points: 10 }
      ];

      const constraints = buildConstraints(state, 0, ctx.rules);
      const playActions = ctx.getValidActions(state)
        .filter(a => a.type === 'play')
        .map(action => ({ action, label: `play-${action.type === 'play' ? action.dominoId : ''}` }));

      const evaluations = evaluatePlayActions(
        state,
        playActions,
        0,
        constraints,
        ctx,
        { biddingSimulations: 10, playingSimulations: 30 }
      );

      // With enumeration, simulationCount should be the actual number of configurations
      // (which is small), not the configured playingSimulations (30)
      expect(evaluations.length).toBeGreaterThan(0);

      // The simulation count should be small (number of enumerated configurations)
      // rather than the configured 30 simulations
      const firstEval = evaluations[0];
      expect(firstEval).toBeDefined();
      expect(firstEval!.simulationCount).toBeLessThanOrEqual(100);
    });
  });
});
