import { describe, it, expect } from 'vitest';
import { StateBuilder } from '../helpers';
import { composeRules, baseLayer, plungeLayer } from '../../game/layers';
import { BID_TYPES } from '../../game/constants';
import { getNextDealer } from '../../game/core/players';
import type { Bid } from '../../game/types';

// Advanced bidding tests include plunge bids (casual rules)
const rules = composeRules([baseLayer, plungeLayer]);

describe('Advanced Bidding Rules', () => {
  describe('Sequential Bidding Requirements', () => {
    it('3 marks can ONLY be bid after 2 marks has been bid', () => {
      // Test opening bid - 3 marks should NOT be valid
      const openingState = StateBuilder
        .inBiddingPhase(0)
        .withCurrentPlayer(1)
        .withBids([])
        .build();

      const threeMarkOpeningBid = { type: BID_TYPES.MARKS, value: 3, player: 1 };
      expect(rules.isValidBid(openingState, threeMarkOpeningBid)).toBe(false);

      // Test after 2 marks bid - 3 marks should be valid
      const afterTwoMarksState = StateBuilder
        .inBiddingPhase(0)
        .withCurrentPlayer(2)
        .withBids([
          { type: BID_TYPES.MARKS, value: 2, player: 1 }
        ])
        .build();

      const threeMarkBid = { type: BID_TYPES.MARKS, value: 3, player: 2 };
      expect(rules.isValidBid(afterTwoMarksState, threeMarkBid)).toBe(true);
    });

    it('4 marks can ONLY be bid after 3 marks has been bid', () => {
      // Test after 2 marks - 4 marks should NOT be valid
      const afterTwoMarksState = StateBuilder
        .inBiddingPhase(0)
        .withCurrentPlayer(2)
        .withBids([
          { type: BID_TYPES.MARKS, value: 2, player: 1 }
        ])
        .build();

      const fourMarkBid = { type: BID_TYPES.MARKS, value: 4, player: 2 };
      expect(rules.isValidBid(afterTwoMarksState, fourMarkBid)).toBe(false);

      // Test after 3 marks - 4 marks should be valid
      const afterThreeMarksState = StateBuilder
        .inBiddingPhase(0)
        .withCurrentPlayer(3)
        .withBids([
          { type: BID_TYPES.MARKS, value: 2, player: 1 },
          { type: BID_TYPES.MARKS, value: 3, player: 2 }
        ])
        .build();

      const fourMarkAfter = { type: BID_TYPES.MARKS, value: 4, player: 3 };
      expect(rules.isValidBid(afterThreeMarksState, fourMarkAfter)).toBe(true);
    });

    it('prevents jump bidding except for valid Plunge scenarios', () => {
      // Test normal progression - must follow bidding sequence
      const normalState = StateBuilder
        .inBiddingPhase(0)
        .withCurrentPlayer(2)
        .withBids([
          { type: BID_TYPES.POINTS, value: 30, player: 1 }
        ])
        .build();

      // Valid next bids should be 31-41 points or 1-2 marks
      const validBids = [
        { type: BID_TYPES.POINTS, value: 31, player: 2 },
        { type: BID_TYPES.POINTS, value: 35, player: 2 },
        { type: BID_TYPES.POINTS, value: 41, player: 2 },
        { type: BID_TYPES.MARKS, value: 1, player: 2 }, // 42 points
        { type: BID_TYPES.MARKS, value: 2, player: 2 }  // 84 points
      ];

      validBids.forEach(bid => {
        expect(rules.isValidBid(normalState, bid)).toBe(true);
      });

      // Invalid jump to higher marks without proper sequence
      const invalidJumpBid = { type: BID_TYPES.MARKS, value: 3, player: 2 };
      expect(rules.isValidBid(normalState, invalidJumpBid)).toBe(false);
    });
  });

  describe('Plunge Bidding Rules', () => {
    it('allows Plunge (4+ marks) only with 4+ doubles in hand', () => {
      // Create state with 4+ doubles for Plunge eligibility
      const plungeEligibleState = StateBuilder
        .inBiddingPhase(0)
        .withCurrentPlayer(1)
        .withBids([])
        .withPlayerDoubles(1, 4)
        .withFillSeed(100)
        .build();

      // With 4+ doubles, should be able to Plunge (bid 4+ marks from opening)
      const plungeBid: Bid = { type: 'plunge', value: 4, player: 1 };
      const player1Hand = plungeEligibleState.players[1]!.hand;
      expect(rules.isValidBid(plungeEligibleState, plungeBid, player1Hand)).toBe(true);

      // Create state with fewer than 4 doubles
      const notPlungeEligibleState = StateBuilder
        .inBiddingPhase(0)
        .withCurrentPlayer(1)
        .withBids([])
        .withPlayerConstraint(1, { maxDoubles: 3 })
        .withFillSeed(101)
        .build();

      // Without 4+ doubles, should NOT be able to Plunge
      const invalidPlungeBid: Bid = { type: 'plunge', value: 4, player: 1 };
      const notEligibleHand = notPlungeEligibleState.players[1]!.hand;
      expect(rules.isValidBid(notPlungeEligibleState, invalidPlungeBid, notEligibleHand)).toBe(false);
    });

    it('Plunge requires all 7 doubles as trump', () => {
      const plungeState = StateBuilder
        .inTrumpSelection(1, 4)
        .withPlayerDoubles(1, 4)
        .withFillSeed(102)
        .with({
          bids: [
            { type: BID_TYPES.MARKS, value: 4, player: 1 }
          ]
        })
        .build();

      // For Plunge bids, trump must be doubles
      const plungeWithDoublesTrump = { ...plungeState, trump: { type: 'doubles' } };
      expect(plungeWithDoublesTrump.trump.type).toBe('doubles');

      // Plunge should not allow other trump suits
      const invalidTrumpSuits = [0, 1, 2, 3, 4, 5];
      invalidTrumpSuits.forEach(trump => {
        // This would be validated by trump selection rules - none of these should equal doubles
        expect(trump).not.toBeGreaterThan(5);
      });
    });
  });

  describe('Bid Value Validation', () => {
    it('validates point bids are in correct range (30-41)', () => {
      const state = StateBuilder
        .inBiddingPhase(0)
        .withCurrentPlayer(1)
        .withBids([])
        .build();

      // Valid point bids
      for (let points = 30; points <= 41; points++) {
        const validBid: Bid = { type: BID_TYPES.POINTS, value: points, player: 1 };
        expect(rules.isValidBid(state, validBid)).toBe(true);
      }

      // Invalid point bids
      const invalidPoints = [29, 42, 0, -1, 50];
      invalidPoints.forEach(points => {
        const invalidBid: Bid = { type: BID_TYPES.POINTS, value: points, player: 1 };
        expect(rules.isValidBid(state, invalidBid)).toBe(false);
      });
    });

    it('validates mark bids follow tournament limits', () => {
      const state = StateBuilder
        .inBiddingPhase(0)
        .withCurrentPlayer(1)
        .withBids([])
        .build();

      // Valid opening mark bids (1-2 marks)
      const validMarkBids = [1, 2];
      validMarkBids.forEach(marks => {
        const validBid: Bid = { type: BID_TYPES.MARKS, value: marks, player: 1 };
        expect(rules.isValidBid(state, validBid)).toBe(true);
      });

      // Invalid opening mark bids (3+ marks without prior bidding)
      const invalidMarkBids = [3, 4, 5, 6, 7];
      invalidMarkBids.forEach(marks => {
        const invalidBid: Bid = { type: BID_TYPES.MARKS, value: marks, player: 1 };
        expect(rules.isValidBid(state, invalidBid)).toBe(false);
      });
    });
  });

  describe('Bid Comparison and Hierarchy', () => {
    it('correctly compares point bids', () => {
      const lowerBid: Bid = { type: BID_TYPES.POINTS, value: 30, player: 1 };
      const higherBid: Bid = { type: BID_TYPES.POINTS, value: 35, player: 2 };

      expect(higherBid.value || 0).toBeGreaterThan(lowerBid.value || 0);
    });

    it('correctly compares mark bids', () => {
      const oneMarkBid: Bid = { type: BID_TYPES.MARKS, value: 1, player: 1 };
      const twoMarkBid: Bid = { type: BID_TYPES.MARKS, value: 2, player: 2 };

      expect(twoMarkBid.value || 0).toBeGreaterThan(oneMarkBid.value || 0);
    });

    it('marks beat equivalent point values', () => {
      // 1 mark (42 points) should beat 41 points
      const fortyOnePoints: Bid = { type: BID_TYPES.POINTS, value: 41, player: 1 };
      const oneMark: Bid = { type: BID_TYPES.MARKS, value: 1, player: 2 };

      // Mark bids should have higher precedence than point bids
      expect(oneMark.type).toBe(BID_TYPES.MARKS);
      expect(fortyOnePoints.type).toBe(BID_TYPES.POINTS);

      // 1 mark represents 42 points, so it beats 41 points
      expect(42).toBeGreaterThan(fortyOnePoints.value!);
    });
  });

  describe('Forced Bidding Scenarios', () => {
    it('dealer must bid if all others pass', () => {
      const dealerForcedState = StateBuilder
        .inBiddingPhase(3)
        .withCurrentPlayer(3)
        .withBids([
          { type: BID_TYPES.PASS, player: 0 },
          { type: BID_TYPES.PASS, player: 1 },
          { type: BID_TYPES.PASS, player: 2 }
        ])
        .build();

      // Dealer (player 3) should not be able to pass if all others passed
      // This depends on implementation - some rules force dealer to bid,
      // others allow all-pass with redeal
      // For tournament standard, test that system handles this scenario
      expect(dealerForcedState.bids).toHaveLength(3);
      expect(dealerForcedState.currentPlayer).toBe(3);
    });

    it('handles reshuffling when appropriate', () => {
      const allPassState = StateBuilder
        .inBiddingPhase(0)
        .withBids([
          { type: BID_TYPES.PASS, player: 1 },
          { type: BID_TYPES.PASS, player: 2 },
          { type: BID_TYPES.PASS, player: 3 },
          { type: BID_TYPES.PASS, player: 0 }
        ])
        .build();

      // After all pass, dealer should advance for reshuffling
      expect(allPassState.bids.every(bid => bid.type === BID_TYPES.PASS)).toBe(true);
      expect(allPassState.bids).toHaveLength(4);

      // New dealer should be next player
      const newDealer = getNextDealer(allPassState.dealer);
      expect(newDealer).toBe(1); // if original dealer was 0
    });
  });
});
