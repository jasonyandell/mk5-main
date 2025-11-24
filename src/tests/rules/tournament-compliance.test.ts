import { describe, it, expect } from 'vitest';
import { StateBuilder, HandBuilder } from '../helpers';
import { composeRules, baseRuleSet } from '../../game/layers';
import { getNextStates } from '../../game/core/state';
import { createTestContext } from '../helpers/executionContext';
import { BID_TYPES } from '../../game/constants';
import { getPlayerLeftOfDealer } from '../../game/core/players';
import type { Bid, BidType } from '../../game/types';

const rules = composeRules([baseRuleSet]);

describe('Tournament Rule Compliance', () => {
  const ctx = createTestContext();
  describe('Straight 42 Rules (No Special Contracts)', () => {
    it('rejects special contract bids in tournament rules', () => {
      const state = StateBuilder
        .inBiddingPhase()
        .withBids([])
        .build();

      const playerHand = HandBuilder.fromStrings([
        '1-1', '2-2', '3-3', '4-4'
      ]);

      // Nello is not a bid type - it's a trump selection (filtered separately)
      const splashBid: Bid = { type: 'splash', value: 2, player: 0 };
      const plungeBid: Bid = { type: 'plunge', value: 4, player: 0 };

      // Tournament rules (baseRuleSet only) reject special contract bids
      expect(rules.isValidBid(state, splashBid, playerHand)).toBe(false);
      expect(rules.isValidBid(state, plungeBid, playerHand)).toBe(false);
    });
  });

  describe('Sequential Mark Bidding Rules', () => {
    it('enforces maximum 2-mark opening bid', () => {
      const state = StateBuilder
        .inBiddingPhase()
        .withCurrentPlayer(0)
        .withBids([])
        .build();

      const validTwoMarks: Bid = { type: BID_TYPES.MARKS, value: 2, player: 0 };
      const invalidThreeMarks: Bid = { type: BID_TYPES.MARKS, value: 3, player: 0 };

      expect(rules.isValidBid(state, validTwoMarks)).toBe(true);
      expect(rules.isValidBid(state, invalidThreeMarks)).toBe(false);
    });

    it('requires 2-mark bid before allowing 3+ marks', () => {
      const state = StateBuilder
        .inBiddingPhase()
        .withCurrentPlayer(1)
        .withBids([
          { type: BID_TYPES.MARKS, value: 1, player: 0 }
        ])
        .build();

      // Cannot jump to 3 marks without 2 marks being bid
      const invalidJump: Bid = { type: BID_TYPES.MARKS, value: 3, player: 1 };
      expect(rules.isValidBid(state, invalidJump)).toBe(false);

      // Can bid 2 marks
      const validTwo: Bid = { type: BID_TYPES.MARKS, value: 2, player: 1 };
      expect(rules.isValidBid(state, validTwo)).toBe(true);
    });

    it('allows sequential mark progression after 2 marks', () => {
      const state = StateBuilder
        .inBiddingPhase()
        .withCurrentPlayer(2)
        .withBids([
          { type: BID_TYPES.MARKS, value: 1, player: 0 },
          { type: BID_TYPES.MARKS, value: 2, player: 1 }
        ])
        .build();

      // Can now bid 3 marks
      const validThree: Bid = { type: BID_TYPES.MARKS, value: 3, player: 2 };
      expect(rules.isValidBid(state, validThree)).toBe(true);

      // Cannot jump to 4 marks (only +1 allowed)
      const invalidJump: Bid = { type: BID_TYPES.MARKS, value: 4, player: 2 };
      expect(rules.isValidBid(state, invalidJump)).toBe(false);
    });

    it('prevents duplicate mark bids', () => {
      const state = StateBuilder
        .inBiddingPhase()
        .withBids([
          { type: BID_TYPES.MARKS, value: 1, player: 0 },
          { type: BID_TYPES.MARKS, value: 2, player: 1 }
        ])
        .build();

      // Cannot bid 2 marks again
      const duplicate: Bid = { type: BID_TYPES.MARKS, value: 2, player: 2 };
      expect(rules.isValidBid(state, duplicate)).toBe(false);
    });
  });

  describe('Minimum Opening Bid Rules', () => {
    it('enforces 30-point minimum opening bid', () => {
      const state = StateBuilder
        .inBiddingPhase()
        .withCurrentPlayer(0)
        .withBids([])
        .build();

      const tooLow: Bid = { type: BID_TYPES.POINTS, value: 29, player: 0 };
      const validMin: Bid = { type: BID_TYPES.POINTS, value: 30, player: 0 };

      expect(rules.isValidBid(state, tooLow)).toBe(false);
      expect(rules.isValidBid(state, validMin)).toBe(true);
    });

    it('enforces maximum 41-point opening bid', () => {
      const state = StateBuilder
        .inBiddingPhase()
        .withCurrentPlayer(0)
        .withBids([])
        .build();

      const validMax: Bid = { type: BID_TYPES.POINTS, value: 41, player: 0 };
      const tooHigh: Bid = { type: BID_TYPES.POINTS, value: 42, player: 0 };

      expect(rules.isValidBid(state, validMax)).toBe(true);
      expect(rules.isValidBid(state, tooHigh)).toBe(false);
    });
  });

  describe('Bid Progression and Jump Bidding', () => {
    it('requires higher bids than current high bid', () => {
      const state = StateBuilder
        .inBiddingPhase()
        .withCurrentPlayer(1)
        .withBids([
          { type: BID_TYPES.POINTS, value: 35, player: 0 }
        ])
        .build();

      const tooLow: Bid = { type: BID_TYPES.POINTS, value: 34, player: 1 };
      const equalBid: Bid = { type: BID_TYPES.POINTS, value: 35, player: 1 };
      const validHigher: Bid = { type: BID_TYPES.POINTS, value: 36, player: 1 };

      expect(rules.isValidBid(state, tooLow)).toBe(false);
      expect(rules.isValidBid(state, equalBid)).toBe(false);
      expect(rules.isValidBid(state, validHigher)).toBe(true);
    });

    it('allows jump from points to 1-2 marks', () => {
      const state = StateBuilder
        .inBiddingPhase()
        .withCurrentPlayer(1)
        .withBids([
          { type: BID_TYPES.POINTS, value: 38, player: 0 }
        ])
        .build();

      const oneMark: Bid = { type: BID_TYPES.MARKS, value: 1, player: 1 };
      const twoMarks: Bid = { type: BID_TYPES.MARKS, value: 2, player: 1 };

      expect(rules.isValidBid(state, oneMark)).toBe(true);
      expect(rules.isValidBid(state, twoMarks)).toBe(true);

      // Verify bid values (42 points = 1 mark)
      expect(rules.getBidComparisonValue(oneMark)).toBe(42);
      expect(rules.getBidComparisonValue(twoMarks)).toBe(84);
    });

    it('prevents jumping to 3+ marks from points', () => {
      const state = StateBuilder
        .inBiddingPhase()
        .withBids([
          { type: BID_TYPES.POINTS, value: 38, player: 0 }
        ])
        .build();

      const threeMarks: Bid = { type: BID_TYPES.MARKS, value: 3, player: 1 };
      expect(rules.isValidBid(state, threeMarks)).toBe(false);
    });
  });

  describe('Partnership and Communication Rules', () => {
    it('validates team assignments are correct', () => {
      const state = StateBuilder
        .inBiddingPhase()
        .build();

      // Verify standard partnership (0&2 vs 1&3)
      expect(state.players[0]!.teamId).toBe(0);
      expect(state.players[1]!.teamId).toBe(1);
      expect(state.players[2]!.teamId).toBe(0);
      expect(state.players[3]!.teamId).toBe(1);
    });

    it('prevents players from bidding multiple times', () => {
      const state = StateBuilder
        .inBiddingPhase()
        .withBids([
          { type: BID_TYPES.POINTS, value: 30, player: 0 }
        ])
        .build();

      // Player 0 already bid, cannot bid again
      const secondBid: Bid = { type: BID_TYPES.POINTS, value: 32, player: 0 };
      expect(rules.isValidBid(state, secondBid)).toBe(false);
    });
  });

  describe('Dealer Rotation and Setup Rules', () => {
    it('validates proper dealer advancement after all-pass', () => {
      const state = StateBuilder
        .inBiddingPhase(1)
        .withBids([
          { type: BID_TYPES.PASS, player: 2 },
          { type: BID_TYPES.PASS, player: 3 },
          { type: BID_TYPES.PASS, player: 0 },
          { type: BID_TYPES.PASS, player: 1 }
        ])
        .build();

      const transitions = getNextStates(state, ctx);
      const redealTransition = transitions.find(t => t.id === 'redeal');
      
      expect(redealTransition).toBeDefined();
      expect(redealTransition!.newState.dealer).toBe(2); // Advanced from 1 to 2
      expect(redealTransition!.newState.currentPlayer).toBe(3); // Dealer + 1
      expect(redealTransition!.newState.bids).toHaveLength(0); // Reset bids
    });

    it('validates correct bidding order after dealer change', () => {
      const state = StateBuilder
        .inBiddingPhase(2)
        .withCurrentPlayer(3)
        .build();

      expect(state.currentPlayer).toBe(getPlayerLeftOfDealer(state.dealer));
    });
  });

  describe('Tournament Target and Scoring', () => {
    it('uses 7 marks as tournament target', () => {
      const state = StateBuilder
        .inBiddingPhase()
        .build();

      expect(state.gameTarget).toBe(7);
    });

    it('validates mark scoring system', () => {
      const pointBid: Bid = { type: BID_TYPES.POINTS, value: 35, player: 0 };
      const markBid: Bid = { type: BID_TYPES.MARKS, value: 1, player: 0 };

      expect(rules.getBidComparisonValue(pointBid)).toBe(35);
      expect(rules.getBidComparisonValue(markBid)).toBe(42); // 1 mark = 42 points
    });
  });


  describe('Edge Cases and Error Handling', () => {
    it('handles empty bid arrays correctly', () => {
      const state = StateBuilder
        .inBiddingPhase()
        .withCurrentPlayer(0)
        .withBids([])
        .build();

      const validOpening: Bid = { type: BID_TYPES.POINTS, value: 30, player: 0 };
      expect(rules.isValidBid(state, validOpening)).toBe(true);
    });

    it('handles invalid bid types gracefully', () => {
      const state = StateBuilder
        .inBiddingPhase()
        .withBids([])
        .build();

      const invalidBid: Bid = { type: 'INVALID' as BidType, value: 30, player: 0 };
      expect(rules.isValidBid(state, invalidBid)).toBe(false);
    });

    it('validates game state consistency', () => {
      const tournamentState = StateBuilder.inBiddingPhase().build();
      const casualState = StateBuilder.inBiddingPhase().build();

      // REMOVED: tournamentMode validation (no longer in GameState)
      // Both states should be valid
      expect(tournamentState.players).toHaveLength(4);
      expect(casualState.players).toHaveLength(4);
    });
  });
});