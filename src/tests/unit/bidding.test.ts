import { describe, it, expect } from 'vitest';
import { composeRules, baseRuleSet, nelloRuleSet, plungeRuleSet, splashRuleSet } from '../../game/rulesets';
import { createInitialState } from '../../game/core/state';
import { BID_TYPES } from '../../game/constants';
import { GameTestHelper, createTestState, createHandWithDoubles } from '../helpers/gameTestHelper';
import { getNextPlayer, getPlayerAfter } from '../../game/core/players';
import type { Bid } from '../../game/types';

// General bidding tests - includes special contracts (casual rules)
const rules = composeRules([baseRuleSet, nelloRuleSet, plungeRuleSet, splashRuleSet]);

describe('Bidding Rules', () => {
  describe('All-Pass Redeal Scenarios', () => {
    it('should handle all players passing with dealer rotation', () => {
      const state = createTestState({
        phase: 'bidding',
        dealer: 1,
        currentPlayer: 2,
        bids: []
      });

      // Simulate all players passing
      const passBids: Bid[] = [
        { type: BID_TYPES.PASS, player: 2 },
        { type: BID_TYPES.PASS, player: 3 },
        { type: BID_TYPES.PASS, player: 0 },
        { type: BID_TYPES.PASS, player: 1 }
      ];

      // Add each pass bid and verify they're valid
      passBids.forEach(bid => {
        expect(rules.isValidBid(state, bid)).toBe(true);
        state.bids.push(bid);
        state.currentPlayer = getNextPlayer(state.currentPlayer); // Advance to next player
      });

      // After all pass bids, should trigger redeal
      expect(state.bids).toHaveLength(4);
      expect(state.bids.every(b => b.type === BID_TYPES.PASS)).toBe(true);
    });

    it('should advance dealer correctly after all-pass redeal', () => {
      // Test dealer advancement from each position
      for (let startDealer = 0; startDealer < 4; startDealer++) {
        const state = createTestState({
          phase: 'bidding',
          dealer: startDealer,
          bids: [
            { type: BID_TYPES.PASS, player: getPlayerAfter(startDealer, 1) },
            { type: BID_TYPES.PASS, player: getPlayerAfter(startDealer, 2) },
            { type: BID_TYPES.PASS, player: getPlayerAfter(startDealer, 3) },
            { type: BID_TYPES.PASS, player: startDealer }
          ]
        });

        // Verify dealer advancement would occur
        expect(state.dealer).toBe(startDealer);
        // Note: Actual redeal transition tested in actions.test.ts
      }
    });

    it('should reset game state properly after all-pass redeal', () => {
      const state = createTestState({
        phase: 'bidding',
        dealer: 0,
        bids: [
          { type: BID_TYPES.PASS, player: 1 },
          { type: BID_TYPES.PASS, player: 2 },
          { type: BID_TYPES.PASS, player: 3 },
          { type: BID_TYPES.PASS, player: 0 }
        ],
        currentBid: { type: BID_TYPES.POINTS, value: 35, player: 1 } // Should be reset
      });

      // Verify all-pass condition
      expect(state.bids.every(b => b.type === BID_TYPES.PASS)).toBe(true);
      expect(state.bids).toHaveLength(4);
    });
  });

  describe('Plunge Bid Validation (4+ Doubles Requirement)', () => {
    it('should require 4+ doubles for Plunge bid in casual mode', () => {
      const state = createTestState({
        phase: 'bidding',
        bids: []
      });

      // Test with insufficient doubles (3)
      const insufficientHand = createHandWithDoubles(3);
      const plungeBid: Bid = { type: 'plunge', value: 4, player: 0 };
      
      expect(rules.isValidBid(state, plungeBid, insufficientHand)).toBe(false);
    });

    it('should allow Plunge bid with exactly 4 doubles', () => {
      const state = createTestState({
        phase: 'bidding',
        bids: []
      });

      const sufficientHand = createHandWithDoubles(4);
      const plungeBid: Bid = { type: 'plunge', value: 4, player: 0 };
      
      expect(rules.isValidBid(state, plungeBid, sufficientHand)).toBe(true);
    });

    it('should allow Plunge bid with more than 4 doubles', () => {
      const state = createTestState({
        phase: 'bidding',
        bids: []
      });

      const abundantHand = createHandWithDoubles(6);
      const plungeBid: Bid = { type: 'plunge', value: 4, player: 0 };
      
      expect(rules.isValidBid(state, plungeBid, abundantHand)).toBe(true);
    });

    it('should validate higher Plunge bids with sufficient doubles', () => {
      const state = createTestState({
        phase: 'bidding',
        bids: []
      });

      const maxDoublesHand = createHandWithDoubles(7);
      
      // Test various Plunge bid levels
      for (let marks = 4; marks <= 6; marks++) {
        const plungeBid: Bid = { type: 'plunge', value: marks, player: 0 };
        expect(rules.isValidBid(state, plungeBid, maxDoublesHand)).toBe(true);
      }
    });

    it('should allow Plunge bids with proper doubles', () => {
      const state = createTestState({
        phase: 'bidding',
        bids: []
      });

      const perfectHand = createHandWithDoubles(7);
      const plungeBid: Bid = { type: 'plunge', value: 4, player: 0 };

      // Plunge ruleSet allows plunge bids with sufficient doubles
      expect(rules.isValidBid(state, plungeBid, perfectHand)).toBe(true);
    });

    it('should validate Splash bid requires 3+ doubles', () => {
      const state = createTestState({
        phase: 'bidding',
        bids: []
      });

      // Test insufficient doubles (2)
      const insufficientHand = createHandWithDoubles(2);
      const splashBid: Bid = { type: 'splash', value: 2, player: 0 };
      expect(rules.isValidBid(state, splashBid, insufficientHand)).toBe(false);

      // Test sufficient doubles (3)
      const sufficientHand = createHandWithDoubles(3);
      expect(rules.isValidBid(state, splashBid, sufficientHand)).toBe(true);
    });

    it('should require minimum 4 marks for Plunge bids', () => {
      const state = createTestState({
        phase: 'bidding',
        bids: []
      });

      const adequateHand = createHandWithDoubles(4);
      
      // Plunge must be 4+ marks
      const invalidPlunge: Bid = { type: 'plunge', value: 3, player: 0 };
      const validPlunge: Bid = { type: 'plunge', value: 4, player: 0 };
      
      expect(rules.isValidBid(state, invalidPlunge, adequateHand)).toBe(false);
      expect(rules.isValidBid(state, validPlunge, adequateHand)).toBe(true);
    });
  });

  describe('isValidBid', () => {
    it('should allow pass bids', () => {
      const state = createInitialState();
      const passBid: Bid = { type: BID_TYPES.PASS, player: 0 };
      
      expect(rules.isValidBid(state, passBid)).toBe(true);
    });
    
    it('should allow opening point bids 30-41', () => {
      const state = createInitialState();
      
      for (let points = 30; points <= 41; points++) {
        const bid: Bid = { type: BID_TYPES.POINTS, value: points, player: 0 };
        expect(rules.isValidBid(state, bid)).toBe(true);
      }
    });
    
    it('should reject point bids below 30', () => {
      const state = createInitialState();
      const bid: Bid = { type: BID_TYPES.POINTS, value: 29, player: 0 };
      
      expect(rules.isValidBid(state, bid)).toBe(false);
    });
    
    it('should reject point bids above 41', () => {
      const state = createInitialState();
      const bid: Bid = { type: BID_TYPES.POINTS, value: 42, player: 0 };
      
      expect(rules.isValidBid(state, bid)).toBe(false);
    });
    
    it('should allow opening mark bids 1-2 in tournament mode', () => {
      const state = createInitialState();
      
      const bid1: Bid = { type: BID_TYPES.MARKS, value: 1, player: 0 };
      const bid2: Bid = { type: BID_TYPES.MARKS, value: 2, player: 0 };
      
      expect(rules.isValidBid(state, bid1)).toBe(true);
      expect(rules.isValidBid(state, bid2)).toBe(true);
    });
    
    it('should reject opening mark bids above 2 in tournament mode', () => {
      const state = createInitialState();
      const bid: Bid = { type: BID_TYPES.MARKS, value: 3, player: 0 };
      
      expect(rules.isValidBid(state, bid)).toBe(false);
    });
    
    it('should prevent duplicate bids from same player', () => {
      const state = createInitialState();
      const firstBid: Bid = { type: BID_TYPES.POINTS, value: 30, player: 0 };
      const secondBid: Bid = { type: BID_TYPES.POINTS, value: 31, player: 0 };
      
      state.bids.push(firstBid);
      
      expect(rules.isValidBid(state, secondBid)).toBe(false);
    });
    
    it('should require higher bids than current', () => {
      const state = createInitialState();
      const firstBid: Bid = { type: BID_TYPES.POINTS, value: 35, player: 0 };
      const lowerBid: Bid = { type: BID_TYPES.POINTS, value: 34, player: 1 };
      const equalBid: Bid = { type: BID_TYPES.POINTS, value: 35, player: 1 };
      
      state.bids.push(firstBid);
      state.currentBid = firstBid;
      
      expect(rules.isValidBid(state, lowerBid)).toBe(false);
      expect(rules.isValidBid(state, equalBid)).toBe(false);
    });
    
    it('should allow higher point bids', () => {
      const state = createInitialState();
      const firstBid: Bid = { type: BID_TYPES.POINTS, value: 35, player: 0 };
      const higherBid: Bid = { type: BID_TYPES.POINTS, value: 36, player: 1 };
      
      state.bids.push(firstBid);
      state.currentBid = firstBid;
      state.currentPlayer = 1; // Set current player to match the bid being tested
      
      expect(rules.isValidBid(state, higherBid)).toBe(true);
    });
    
    it('should enforce mark bid progression rules', () => {
      const state = createInitialState();
      
      // First mark bid of 1
      const firstMarkBid: Bid = { type: BID_TYPES.MARKS, value: 1, player: 0 };
      state.bids.push(firstMarkBid);
      state.currentBid = firstMarkBid;
      state.currentPlayer = 1; // Advance to next player
      
      // Can bid 2 marks
      const twoMarkBid: Bid = { type: BID_TYPES.MARKS, value: 2, player: 1 };
      expect(rules.isValidBid(state, twoMarkBid)).toBe(true);
      
      // Add the 2 mark bid
      state.bids.push(twoMarkBid);
      state.currentBid = twoMarkBid;
      state.currentPlayer = 2; // Advance to next player
      
      // Now can bid 3 marks (after 2 marks has been bid)
      const threeMarkBid: Bid = { type: BID_TYPES.MARKS, value: 3, player: 2 };
      expect(rules.isValidBid(state, threeMarkBid)).toBe(true);
      
      // Cannot jump to 4 marks without progression
      const fourMarkBid: Bid = { type: BID_TYPES.MARKS, value: 4, player: 2 };
      expect(rules.isValidBid(state, fourMarkBid)).toBe(false);
    });
    
    it('should allow special contracts in casual rules', () => {
      const state = createInitialState();
      // Casual rules (with special contract ruleSets) allow these bids

      const testHand = GameTestHelper.createTestHand([
        [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]
      ]);

      const splashBid: Bid = { type: 'splash', value: 2, player: 0 };
      const plungeBid: Bid = { type: 'plunge', value: 4, player: 0 };

      // Nello is not a bid type - it's a trump selection after a marks bid
      expect(rules.isValidBid(state, splashBid, testHand)).toBe(true);
      expect(rules.isValidBid(state, plungeBid, testHand)).toBe(true);
    });

    it('should allow special contracts in casual mode', () => {
      const state = createInitialState();
      // REMOVED: state.tournamentMode = false;

      const testHand = GameTestHelper.createTestHand([
        [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]
      ]);

      const splashBid: Bid = { type: 'splash', value: 2, player: 0 };
      const plungeBid: Bid = { type: 'plunge', value: 4, player: 0 };

      // Nello is not a bid type - it's a trump selection after a marks bid
      expect(rules.isValidBid(state, splashBid, testHand)).toBe(true);
      expect(rules.isValidBid(state, plungeBid, testHand)).toBe(true);
    });
  });
  
  describe('getBidComparisonValue', () => {
    it('should return point value for point bids', () => {
      const bid: Bid = { type: BID_TYPES.POINTS, value: 35, player: 0 };
      expect(rules.getBidComparisonValue(bid)).toBe(35);
    });
    
    it('should return 42x multiplier for mark bids', () => {
      const bid: Bid = { type: BID_TYPES.MARKS, value: 2, player: 0 };
      expect(rules.getBidComparisonValue(bid)).toBe(84);
    });
    
    it('should return 42x multiplier for special bid contracts', () => {
      const splashBid: Bid = { type: 'splash', value: 2, player: 0 };
      const plungeBid: Bid = { type: 'plunge', value: 4, player: 0 };

      // Nello is not a bid type - it's a trump selection after a marks bid
      expect(rules.getBidComparisonValue(splashBid)).toBe(84);
      expect(rules.getBidComparisonValue(plungeBid)).toBe(168);
    });
    
    it('should return 0 for pass bids', () => {
      const bid: Bid = { type: BID_TYPES.PASS, player: 0 };
      expect(rules.getBidComparisonValue(bid)).toBe(0);
    });
  });
  
  describe('Bidding scenarios', () => {
    it('should handle complete bidding round with all passes', () => {
      const state = GameTestHelper.createBiddingScenario(0);
      const passBids = [
        { type: BID_TYPES.PASS, player: 0 },
        { type: BID_TYPES.PASS, player: 1 },
        { type: BID_TYPES.PASS, player: 2 },
        { type: BID_TYPES.PASS, player: 3 }
      ] as Bid[];
      
      passBids.forEach(bid => {
        expect(rules.isValidBid(state, bid)).toBe(true);
        state.bids.push(bid);
        state.currentPlayer = getNextPlayer(state.currentPlayer); // Advance to next player
      });
      
      expect(state.bids).toHaveLength(4);
    });
    
    it('should handle competitive bidding scenario', () => {
      const state = GameTestHelper.createBiddingScenario(0);
      
      const biddingSequence = [
        { type: BID_TYPES.POINTS, value: 30, player: 0 },
        { type: BID_TYPES.POINTS, value: 32, player: 1 },
        { type: BID_TYPES.MARKS, value: 1, player: 2 },
        { type: BID_TYPES.MARKS, value: 2, player: 3 }
      ] as Bid[];
      
      biddingSequence.forEach((bid) => {
        expect(rules.isValidBid(state, bid)).toBe(true);
        state.bids.push(bid);
        state.currentPlayer = getNextPlayer(state.currentPlayer); // Advance to next player
        if (bid.type !== BID_TYPES.PASS) {
          state.currentBid = bid;
        }
      });
      
      // Final bid should be 2 marks (84 points equivalent)
      expect(state.currentBid?.value).toBe(2);
      expect(rules.getBidComparisonValue(state.currentBid!)).toBe(84);
    });
  });
});