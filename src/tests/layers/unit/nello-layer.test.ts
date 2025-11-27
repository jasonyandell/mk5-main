/**
 * Unit tests for nello layer overrides.
 *
 * Nello rules (from docs/rules.md §8.A):
 * - Bidder must lose all tricks
 * - Partner sits out (3-player tricks)
 * - No trump suit
 * - Doubles form own suit (suit 7)
 * - Only available after marks bid
 * - Early termination when bidder wins any trick
 *
 * Tests focus on nello-specific overrides only.
 * Passthrough behavior tested in compose-rules.test.ts.
 * Integration scenarios tested in nello-three-player.test.ts.
 */

import { describe, it, expect } from 'vitest';
import { baseLayer } from '../../../game/layers/base';
import { nelloLayer } from '../../../game/layers/nello';
import { composeRules } from '../../../game/layers/compose';
import { StateBuilder } from '../../helpers';
import { BID_TYPES } from '../../../game/constants';
import { BLANKS, DOUBLES_AS_TRUMP } from '../../../game/types';

describe('Nello Layer Overrides', () => {
  const rules = composeRules([baseLayer, nelloLayer]);

  describe('getValidActions', () => {
    it('should add nello trump option only after marks bid in trump selection', () => {
      const stateMarks = StateBuilder.inTrumpSelection(0).withWinningBid(0, { type: BID_TYPES.MARKS, value: 2, player: 0 }).build();
      const actions = nelloLayer.getValidActions?.(stateMarks, []) ?? [];
      expect(actions).toHaveLength(1);
      expect(actions[0]).toEqual({ type: 'select-trump', player: 0, trump: { type: 'nello' } });

      // Not for points bid
      const statePoints = StateBuilder.inTrumpSelection(0, 30).build();
      expect(nelloLayer.getValidActions?.(statePoints, []) ?? []).toEqual([]);

      // Not during bidding phase
      const stateBidding = StateBuilder.inBiddingPhase().withCurrentPlayer(0).withBids([{ type: BID_TYPES.MARKS, value: 2, player: 0 }]).build();
      expect(nelloLayer.getValidActions?.(stateBidding, []) ?? []).toEqual([]);
    });

    it('should preserve previous actions when adding nello option', () => {
      const state = StateBuilder.inTrumpSelection(1).withWinningBid(1, { type: BID_TYPES.MARKS, value: 3, player: 1 }).build();
      const baseActions = [{ type: 'select-trump' as const, player: 1, trump: { type: 'suit' as const, suit: BLANKS } }];
      const actions = nelloLayer.getValidActions?.(state, baseActions) ?? [];

      expect(actions).toHaveLength(2);
      expect(actions[0]).toEqual(baseActions[0]);
      expect(actions[1]).toMatchObject({ type: 'select-trump', trump: { type: 'nello' } });
    });
  });

  describe('getNextPlayer', () => {
    it('should skip partner in three-player rotation', () => {
      // Bidder 0, partner 2: 0 -> 1 -> 3 -> 0
      const state0 = StateBuilder.nelloContract(0).withTrump({ type: 'nello' }).withHands([[], [], [], []]).build();
      expect(rules.getNextPlayer(state0, 0)).toBe(1);
      expect(rules.getNextPlayer(state0, 1)).toBe(3); // Skip 2
      expect(rules.getNextPlayer(state0, 3)).toBe(0);

      // Bidder 1, partner 3: 0 -> 1 -> 2 -> 0
      const state1 = StateBuilder.nelloContract(1).withTrump({ type: 'nello' }).withHands([[], [], [], []]).build();
      expect(rules.getNextPlayer(state1, 0)).toBe(1);
      expect(rules.getNextPlayer(state1, 1)).toBe(2);
      expect(rules.getNextPlayer(state1, 2)).toBe(0); // Skip 3
    });
  });

  describe('isTrickComplete', () => {
    it('should require exactly 3 plays (not 4)', () => {
      const state = StateBuilder.nelloContract(0).withTrump({ type: 'nello' });

      // 3 plays - complete
      expect(rules.isTrickComplete(
        state.withCurrentTrick([
          { player: 0, domino: '1-0' },
          { player: 1, domino: '2-0' },
          { player: 3, domino: '3-0' }
        ]).build()
      )).toBe(true);

      // Less than 3 - incomplete
      expect(rules.isTrickComplete(
        state.withCurrentTrick([
          { player: 0, domino: '1-0' },
          { player: 1, domino: '2-0' }
        ]).build()
      )).toBe(false);
    });
  });

  describe('checkHandOutcome', () => {
    it('should end immediately when bidding team wins any trick', () => {
      // Bidder wins trick - nello fails
      const stateBidderWins = StateBuilder
        .nelloContract(0)
        .withTrump({ type: 'nello' })
        .withHands([[], [], [], []])
        .addTrick(
          [
            { player: 0, domino: { id: '6-0', high: 6, low: 0 } },
            { player: 1, domino: { id: '2-0', high: 2, low: 0 } },
            { player: 3, domino: { id: '3-0', high: 3, low: 0 } }
          ],
          0, // Bidder wins
          0
        )
        .build();

      const outcomeBidder = rules.checkHandOutcome(stateBidderWins);
      expect(outcomeBidder.isDetermined).toBe(true);
      expect((outcomeBidder as { decidedAtTrick: number }).decidedAtTrick).toBe(1);

      // Partner wins trick - nello also fails
      const statePartnerWins = StateBuilder
        .nelloContract(1) // Partner is player 3
        .withTrump({ type: 'nello' })
        .withHands([[], [], [], []])
        .addTrick(
          [
            { player: 1, domino: { id: '1-0', high: 1, low: 0 } },
            { player: 0, domino: { id: '6-0', high: 6, low: 0 } },
            { player: 3, domino: { id: '2-0', high: 2, low: 0 } }
          ],
          3, // Partner wins
          0
        )
        .build();

      const outcomePartner = rules.checkHandOutcome(statePartnerWins);
      expect(outcomePartner.isDetermined).toBe(true);
      expect((outcomePartner as { decidedAtTrick: number }).decidedAtTrick).toBe(1);
    });
  });

  describe('getLedSuit', () => {
    it('should return suit 7 for doubles, higher pip for non-doubles', () => {
      const state = StateBuilder.nelloContract(0).withTrump({ type: 'nello' }).build();

      // All doubles become suit 7
      for (let i = 0; i <= 6; i++) {
        expect(rules.getLedSuit(state, { id: `${i}-${i}`, high: i, low: i })).toBe(DOUBLES_AS_TRUMP);
      }

      // Non-doubles use higher pip
      expect(rules.getLedSuit(state, { id: '6-2', high: 6, low: 2 })).toBe(6);
      expect(rules.getLedSuit(state, { id: '5-0', high: 5, low: 0 })).toBe(5);
    });
  });
});
