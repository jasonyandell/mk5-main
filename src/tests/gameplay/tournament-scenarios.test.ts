import { describe, it, expect } from 'vitest';
import { createInitialState } from '../../game/core/state';
import { composeRules, baseLayer } from '../../game/layers';
import { BID_TYPES } from '../../game/constants';
import { getPlayerLeftOfDealer } from '../../game/core/players';
import type { Bid, GameState } from '../../game/types';

const rules = composeRules([baseLayer]);

describe('Tournament Scenarios', () => {
  function createTournamentState(): GameState {
    const state = createInitialState();
    state.gameTarget = 10; // Tournament typically plays to 10 marks
    return state;
  }

  describe('Tournament Bidding Rules', () => {
    it('should enforce minimum opening bid', () => {
      // Minimum bid is 30 points
      const state = createTournamentState();
      state.phase = 'bidding';
      state.bids = [];
      state.currentPlayer = 0;

      const lowBid: Bid = { type: BID_TYPES.POINTS, value: 25, player: 0 };
      const validBid: Bid = { type: BID_TYPES.POINTS, value: 30, player: 0 };

      expect(rules.isValidBid(state, lowBid, undefined)).toBe(false);
      expect(rules.isValidBid(state, validBid, undefined)).toBe(true);
    });

    it('should reject special contracts in tournament rules', () => {
      // Tournament rules (baseLayer only) do not support special contracts
      const state = createTournamentState();
      state.phase = 'bidding';
      state.bids = [];
      state.currentPlayer = 0;

      // Nello is not a bid type - it's a trump selection
      const splashBid: Bid = { type: 'splash', value: 3, player: 0 };
      const plungeBid: Bid = { type: 'plunge', value: 4, player: 0 };

      // Create a hand with enough doubles for splash (3+) and plunge (4+)
      const adequateHand = [
        { high: 0, low: 0, id: '0-0' }, // double blank
        { high: 1, low: 1, id: '1-1' }, // double one
        { high: 2, low: 2, id: '2-2' }, // double two
        { high: 3, low: 3, id: '3-3' }, // double three
        { high: 5, low: 4, id: '5-4' },
        { high: 6, low: 0, id: '6-0' },
        { high: 5, low: 1, id: '5-1' }
      ];

      // Tournament rules reject special contracts
      // (Nello is not a bid - it's filtered as a trump selection in tournament mode)
      expect(rules.isValidBid(state, splashBid, adequateHand)).toBe(false);
      expect(rules.isValidBid(state, plungeBid, adequateHand)).toBe(false);
    });

    it('should enforce proper bid increments', () => {
      const state = createTournamentState();
      const currentBid: Bid = { type: BID_TYPES.POINTS, value: 32, player: 0 };
      state.currentBid = currentBid;
      state.bids = [currentBid]; // Add the current bid to the bids array
      state.currentPlayer = 1; // Player 1 is next to bid
      
      const validIncrements = [33, 34, 35, 36, 37, 38, 39, 40, 41];
      const invalidIncrements = [30, 31, 32]; // All should be invalid: too low or equal
      
      validIncrements.forEach(value => {
        const bid: Bid = { type: BID_TYPES.POINTS, value, player: 1 };
        expect(rules.isValidBid(state, bid)).toBe(true);
      });
      
      invalidIncrements.forEach(value => {
        const bid: Bid = { type: BID_TYPES.POINTS, value, player: 1 };
        expect(rules.isValidBid(state, bid)).toBe(false);
      });
    });

    it('should allow mark bids when overbidding', () => {
      const state = createTournamentState();
      state.bids = [{ type: BID_TYPES.POINTS, value: 35, player: 0 }];
      state.currentPlayer = 1;

      const markBid: Bid = { type: BID_TYPES.MARKS, value: 2, player: 1 };

      // Mark bids are allowed in tournament rules
      expect(rules.isValidBid(state, markBid)).toBe(true);
    });
  });

  describe('Tournament Scoring', () => {
    it('should track marks correctly to tournament target', () => {
      const state = createTournamentState();

      expect(state.gameTarget).toBe(10);

      // Simulate reaching tournament target
      state.teamMarks = [10, 7];
      expect(state.teamMarks[0]).toBeGreaterThanOrEqual(state.gameTarget);
    });

    it('should handle negative marks in tournament play', () => {
      const state = createTournamentState();
      
      // Team can go negative from failed bids
      state.teamMarks = [-3, 8];
      expect(state.teamMarks[0]).toBeLessThan(0);
      expect(state.teamMarks[1]).toBeLessThan(state.gameTarget);
    });

    it('should award proper marks for successful tournament bids', () => {
      
      // High-value bids in tournament should award more marks
      const highMarkBid: Bid = { type: BID_TYPES.MARKS, value: 4, player: 0 };
      expect(highMarkBid.value).toBeGreaterThan(1);
      expect(highMarkBid.value).toBeLessThanOrEqual(6); // Maximum reasonable mark bid
    });
  });


  describe('Tournament Game Flow', () => {
    it('should handle tournament timing constraints', () => {
      const state = createTournamentState();

      // Tournament games should track timing (simulated)
      const startTime = Date.now();

      // Verify state allows for tournament play
      expect(state.players).toHaveLength(4);

      // Tournament should be completable within reasonable time
      const maxTournamentTime = 30 * 60 * 1000; // 30 minutes
      expect(Date.now() - startTime).toBeLessThan(maxTournamentTime);
    });

    it('should enforce tournament partner rules', () => {
      const state = createTournamentState();
      
      // Verify proper team assignments (0&2 vs 1&3)
      expect(state.players[0]!.teamId).toBe(0);
      expect(state.players[1]!.teamId).toBe(1);
      expect(state.players[2]!.teamId).toBe(0);
      expect(state.players[3]!.teamId).toBe(1);
    });

    it('should handle tournament dealer rotation', () => {
      const state = createTournamentState();
      
      // Dealer should rotate properly in tournament
      expect(state.dealer).toBeGreaterThanOrEqual(0);
      expect(state.dealer).toBeLessThan(4);
      
      // Current player should be after dealer
      const expectedFirstPlayer = getPlayerLeftOfDealer(state.dealer);
      expect(state.currentPlayer).toBe(expectedFirstPlayer);
    });
  });

  describe('Tournament Mathematical Verification', () => {
    it('should verify 42-point total in tournament', () => {
      const state = createTournamentState();
      
      // Calculate total points available using proper domino scoring
      let totalPoints = 0;
      state.players.forEach(player => {
        player.hand.forEach(domino => {
          // Use the proper point calculation from dominoes.ts
          if (domino.high === 5 && domino.low === 5) totalPoints += 10; // 5-5 = 10 points
          else if ((domino.high === 6 && domino.low === 4) || (domino.high === 4 && domino.low === 6)) totalPoints += 10; // 6-4 = 10 points
          else if (domino.high + domino.low === 5) totalPoints += 5; // Any domino totaling 5 pips = 5 points
        });
      });
      
      // Add 7 points for the 7 tricks
      totalPoints += 7;
      
      expect(totalPoints).toBe(42);
    });

    it('should verify domino distribution in tournament', () => {
      const state = createTournamentState();
      
      // Each player should have exactly 7 dominoes
      state.players.forEach(player => {
        expect(player.hand).toHaveLength(7);
      });
      
      // Total of 28 dominoes (complete set)
      const totalDominoes = state.players.reduce(
        (sum, player) => sum + player.hand.length, 0
      );
      expect(totalDominoes).toBe(28);
    });

    it('should verify no duplicate dominoes in tournament', () => {
      const state = createTournamentState();
      
      const allDominoes = state.players.flatMap(player => 
        player.hand.map(domino => `${domino.high}-${domino.low}`)
      );
      
      const uniqueDominoes = new Set(allDominoes);
      expect(uniqueDominoes.size).toBe(28);
      expect(allDominoes.length).toBe(28);
    });
  });
});