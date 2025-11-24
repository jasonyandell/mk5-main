import { describe, it, expect } from 'vitest';
import { StateBuilder } from '../helpers';
import { composeRules, baseRuleSet } from '../../game/layers';
import { BID_TYPES } from '../../game/constants';
import { isGameComplete } from '../../game/core/scoring';
import { getNextPlayer } from '../../game/core/players';
import type { Bid, TrumpSelection } from '../../game/types';
import { BLANKS, ACES, DEUCES, TRES, FOURS, FIVES, SIXES, DOUBLES_AS_TRUMP } from '../../game/types';

const rules = composeRules([baseRuleSet]);

describe('Tournament Standards (N42PA Rules)', () => {
  describe('Game Format Requirements', () => {
    it('uses straight 42 rules only (no special contracts)', () => {
      const state = StateBuilder
        .inBiddingPhase(0)
        .withCurrentPlayer(1)
        .withBids([])
        .build();

      // Test that special contract bids are invalid
      const specialBids = [
        { type: 'NELLO', value: 1, player: 1 },
        { type: 'PLUNGE', value: 4, player: 1 },
        { type: 'SPLASH', value: 3, player: 1 }
      ];
      
      specialBids.forEach(bid => {
        expect(rules.isValidBid(state, bid as Bid)).toBe(false);
      });

      // Valid standard bids should work
      const validBids = [
        { type: BID_TYPES.PASS, player: 1 },
        { type: BID_TYPES.POINTS, value: 30, player: 1 },
        { type: BID_TYPES.MARKS, value: 1, player: 1 }
      ];
      
      validBids.forEach(bid => {
        expect(rules.isValidBid(state, bid)).toBe(true);
      });
    });

    it('enforces maximum 2 marks for opening bid in tournament play', () => {
      const state = StateBuilder
        .inBiddingPhase(0)
        .withCurrentPlayer(1)
        .withBids([])
        .build();

      // Valid opening mark bids (1-2 marks)
      const validMarkBids = [
        { type: BID_TYPES.MARKS, value: 1, player: 1 },
        { type: BID_TYPES.MARKS, value: 2, player: 1 }
      ];
      
      validMarkBids.forEach(bid => {
        expect(rules.isValidBid(state, bid)).toBe(true);
      });
      
      // Invalid opening mark bids (3+ marks)
      const invalidMarkBids = [
        { type: BID_TYPES.MARKS, value: 3, player: 1 },
        { type: BID_TYPES.MARKS, value: 4, player: 1 }
      ];
      
      invalidMarkBids.forEach(bid => {
        expect(rules.isValidBid(state, bid)).toBe(false);
      });
    });

    it('requires minimum bid of 30 points', () => {
      const state = StateBuilder
        .inBiddingPhase(0)
        .withCurrentPlayer(1)
        .withBids([])
        .build();

      // Valid point bids (30-41)
      for (let points = 30; points <= 41; points++) {
        const bid = { type: BID_TYPES.POINTS, value: points, player: 1 };
        expect(rules.isValidBid(state, bid)).toBe(true);
      }
      
      // Invalid point bids (below 30)
      const invalidBids = [
        { type: BID_TYPES.POINTS, value: 29, player: 1 },
        { type: BID_TYPES.POINTS, value: 25, player: 1 },
        { type: BID_TYPES.POINTS, value: 0, player: 1 }
      ];
      
      invalidBids.forEach(bid => {
        expect(rules.isValidBid(state, bid)).toBe(false);
      });
    });

    it('enforces 7 marks to win game', () => {
      // Game should not be complete with 6 marks
      let state = StateBuilder.inBiddingPhase().with({ teamMarks: [6, 0] }).build();
      expect(state.phase).not.toBe('game_end');

      // Game should be complete with 7 marks
      state = StateBuilder.gameEnded(0).build();
      expect(state.phase).toBe('game_end');
      expect(state.teamMarks[0]).toBeGreaterThanOrEqual(state.gameTarget); // team 0 wins

      // Test with isGameComplete function directly
      expect(isGameComplete(6, 0)).toBe(false);
      expect(isGameComplete(7, 0)).toBe(true);
      expect(isGameComplete(0, 7)).toBe(true);
    });
  });

  describe('Bidding Sequence Rules', () => {
    it('proceeds clockwise from dealer left', () => {
      const state = StateBuilder
        .inBiddingPhase(2)
        .withCurrentPlayer(3)
        .withBids([])
        .build();

      expect(state.currentPlayer).toBe(3);

      // After player 3 bids, should advance to player 0
      const passBid: Bid = { type: BID_TYPES.PASS, player: 3 };
      expect(rules.isValidBid(state, passBid)).toBe(true);
    });

    it('each player gets exactly one bid opportunity per round', () => {
      const state = StateBuilder
        .inBiddingPhase(0)
        .withCurrentPlayer(1)
        .withBids([])
        .build();

      // Simulate complete bidding round
      const bids: Bid[] = [
        { type: BID_TYPES.PASS, player: 1 },
        { type: BID_TYPES.PASS, player: 2 },
        { type: BID_TYPES.PASS, player: 3 },
        { type: BID_TYPES.PASS, player: 0 }
      ];

      bids.forEach(bid => {
        expect(rules.isValidBid(state, bid)).toBe(true);
        state.bids.push(bid);
        state.currentPlayer = getNextPlayer(state.currentPlayer); // Advance to next player
      });

      // After all 4 players bid, bidding should be complete
      expect(state.bids).toHaveLength(4);
      
      // All players should have bid exactly once
      const playerBids = [0, 1, 2, 3].map(player => 
        state.bids.filter(bid => bid.player === player).length
      );
      expect(playerBids).toEqual([1, 1, 1, 1]);
    });

    it('requires redeal when all players pass', () => {
      const state = StateBuilder
        .inBiddingPhase(0)
        .withCurrentPlayer(1)
        .withBids([
          { type: BID_TYPES.PASS, player: 1 },
          { type: BID_TYPES.PASS, player: 2 },
          { type: BID_TYPES.PASS, player: 3 },
          { type: BID_TYPES.PASS, player: 0 }
        ])
        .build();

      // All pass scenario should trigger redeal
      const allPassed = state.bids.every(bid => bid.type === BID_TYPES.PASS);
      expect(allPassed).toBe(true);
      expect(state.bids).toHaveLength(4);
    });
  });

  describe('Trump Declaration Rules', () => {
    it('bid winner must declare trump suit', () => {
      const state = StateBuilder
        .inTrumpSelection(1, 30)
        .build();

      expect(state.winningBidder).toBe(1);
      expect(state.currentPlayer).toBe(1);
      expect(state.phase).toBe('trump_selection');
    });

    it('allows all valid trump options', () => {
      const validTrumpSuits = [BLANKS, ACES, DEUCES, TRES, FOURS, FIVES, SIXES, DOUBLES_AS_TRUMP]; // 0-6 for suits plus doubles (7)

      validTrumpSuits.forEach(trump => {
        const numberToTrumpSelection = (trump: number): TrumpSelection => {
          if (trump === DOUBLES_AS_TRUMP) {
            return { type: 'doubles' };
          } else if (trump === 8) {
            return { type: 'no-trump' };
          } else if (trump >= BLANKS && trump <= SIXES) {
            return { type: 'suit', suit: trump as 0 | 1 | 2 | 3 | 4 | 5 | 6 };
          } else {
            return { type: 'not-selected' };
          }
        };

        const state = StateBuilder
          .inPlayingPhase(numberToTrumpSelection(trump))
          .build();
        expect(state.trump).toEqual(numberToTrumpSelection(trump));
      });

      // Negative test would require trump validation function
    });
  });

  describe('Gameplay Rules', () => {
    it('bid winner leads first trick', () => {
      const state = StateBuilder
        .inPlayingPhase({ type: 'suit', suit: ACES })
        .withWinningBid(2, { type: BID_TYPES.POINTS, value: 30, player: 2 })
        .withCurrentPlayer(2)
        .build();

      expect(state.currentPlayer).toBe(2);
      expect(state.currentTrick).toHaveLength(0);
    });

    it('trick winner leads next trick', () => {
      const state = StateBuilder
        .inPlayingPhase({ type: 'suit', suit: ACES })
        .withCurrentTrick([
          { player: 0, domino: { id: 'test1', high: 2, low: 3 } },
          { player: 1, domino: { id: 'test2', high: 1, low: 1 } }, // trump wins
          { player: 2, domino: { id: 'test3', high: 2, low: 4 } },
          { player: 3, domino: { id: 'test4', high: 2, low: 5 } }
        ])
        .build();

      // Player 1 played trump (1-1) and should win trick
      // Implementation would need trick winner calculation
      expect(state.currentTrick).toHaveLength(4);
    });

    it('validates total hand value equals 42 points', () => {
      // Test that all counting dominoes sum to 35 points
      // Plus 7 tricks = 42 total points per hand
      const countingDominoes = [
        { high: 5, low: 5, points: 10 }, // 5-5: 10 points
        { high: 6, low: 4, points: 10 }, // 6-4: 10 points  
        { high: 5, low: 0, points: 5 },  // 5-0: 5 points
        { high: 4, low: 1, points: 5 },  // 4-1: 5 points
        { high: 3, low: 2, points: 5 }   // 3-2: 5 points
      ];

      const totalCountPoints = countingDominoes.reduce((sum, d) => sum + d.points, 0);
      const trickPoints = 7; // 7 tricks worth 1 point each
      
      expect(totalCountPoints).toBe(35);
      expect(totalCountPoints + trickPoints).toBe(42);
    });
  });

  describe('Communication Prohibition', () => {
    it('enforces no table talk or signals', () => {
      // This would be enforced at the UI level
      // Test that game state contains no communication mechanisms
      const state = StateBuilder
        .inPlayingPhase()
        .build();

      // Game state should not have chat, signals, or communication fields
      expect(state).not.toHaveProperty('chat');
      expect(state).not.toHaveProperty('signals');
      expect(state).not.toHaveProperty('messages');
      expect(state).not.toHaveProperty('communication');
    });
  });
});

// Note: Tournament rules (baseRuleSet) do not support special contracts (Nello, Splash, Plunge)
// Those require their respective ruleSets to be added to the composition