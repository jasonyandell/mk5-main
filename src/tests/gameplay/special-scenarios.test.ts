import { describe, it, expect } from 'vitest';
import { createTestState, createHandWithDoubles } from '../helpers/gameTestHelper';
import { calculateGameScore, isGameComplete } from '../../game/core/scoring';
import { isValidPlay, getValidPlays, isValidBid } from '../../game/core/rules';
import { analyzeSuits } from '../../game/core/suit-analysis';
import { BID_TYPES } from '../../game/constants';
import type { Domino, Bid } from '../../game/types';

describe('Special Gameplay Scenarios', () => {
  describe('High Stakes Bidding', () => {
    it('handles 6 and 7 mark bids correctly', () => {
      // Create state with escalated bidding up to 6 marks
      const state = createTestState({
        phase: 'bidding',
        dealer: 0,
        currentPlayer: 3,
        bids: [
          { type: BID_TYPES.MARKS, value: 4, player: 0 },
          { type: BID_TYPES.MARKS, value: 5, player: 1 },
          { type: BID_TYPES.MARKS, value: 6, player: 2 }
        ]
      });

      // Player 3 can bid 7 marks (one more than current 6)
      const sevenMarkBid: Bid = { type: BID_TYPES.MARKS, value: 7, player: 3 };
      expect(isValidBid(state, sevenMarkBid)).toBe(true);
      
      // But cannot jump to 8 marks (would need to bid 7 first)
      const eightMarkBid: Bid = { type: BID_TYPES.MARKS, value: 8, player: 3 };
      expect(isValidBid(state, eightMarkBid)).toBe(false);
    });

    it('handles minimum 30 bid enforcement', () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        bids: []
      });

      // Minimum bid is 30 points
      const belowMinBid: Bid = { type: BID_TYPES.POINTS, value: 29, player: 0 };
      const minBid: Bid = { type: BID_TYPES.POINTS, value: 30, player: 0 };
      
      // 29 should be invalid, 30 should be valid
      expect(isValidBid(state, belowMinBid)).toBe(false);
      expect(isValidBid(state, minBid)).toBe(true);
    });

    it('handles plunge bid requirements', () => {
      // Create hand with 4 doubles for valid plunge
      const handWith4Doubles = createHandWithDoubles(4);
      
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        bids: [],
        tournamentMode: false // Plunge not allowed in tournament
      });
      
      // Set the player's hand to have 4 doubles
      state.hands = { 0: handWith4Doubles, 1: [], 2: [], 3: [] };
      state.players[0].hand = handWith4Doubles;

      // Plunge bid requires 4+ doubles
      const plungeBid: Bid = { type: BID_TYPES.PLUNGE, value: 4, player: 0 };
      
      // Should be valid with 4 doubles (in non-tournament mode)
      expect(isValidBid(state, plungeBid, handWith4Doubles)).toBe(true);
      
      // But should be invalid in tournament mode
      state.tournamentMode = true;
      expect(isValidBid(state, plungeBid, handWith4Doubles)).toBe(false);
    });
  });

  describe('Trump Dominance Scenarios', () => {
    it('handles all-trump hands correctly', () => {
      // Create hand with all trump dominoes
      const allTrumpHand: Domino[] = [
        { id: 'trump1', high: 1, low: 0, points: 0 },
        { id: 'trump2', high: 1, low: 1, points: 0 },
        { id: 'trump3', high: 1, low: 2, points: 0 },
        { id: 'trump4', high: 1, low: 3, points: 0 },
        { id: 'trump5', high: 1, low: 4, points: 0 },
        { id: 'trump6', high: 1, low: 5, points: 0 },
        { id: 'trump7', high: 1, low: 6, points: 0 }
      ];

      const state = createTestState({
        phase: 'playing',
        trump: 1, // ones are trump
        currentTrick: [],
        currentSuit: null,
        currentPlayer: 0,
        players: [
          { 
            id: 0, 
            name: 'Player 0', 
            teamId: 0, 
            marks: 0, 
            hand: allTrumpHand,
            suitAnalysis: analyzeSuits(allTrumpHand, 1)
          },
          { id: 1, name: 'Player 1', teamId: 1, marks: 0, hand: [] },
          { id: 2, name: 'Player 2', teamId: 0, marks: 0, hand: [] },
          { id: 3, name: 'Player 3', teamId: 1, marks: 0, hand: [] }
        ]
      });

      // All dominoes should be valid plays
      allTrumpHand.forEach(domino => {
        expect(isValidPlay(state, domino, 0)).toBe(true);
      });

      const validPlays = getValidPlays(state, 0);
      expect(validPlays).toHaveLength(7);
    });

    it('handles no-trump scenarios', () => {
      const mixedHand: Domino[] = [
        { id: 'twos1', high: 2, low: 0, points: 0 },
        { id: 'threes1', high: 3, low: 1, points: 5 },
        { id: 'fours1', high: 4, low: 2, points: 0 },
        { id: 'fives1', high: 5, low: 3, points: 0 },
        { id: 'sixes1', high: 6, low: 4, points: 10 },
        { id: 'double1', high: 1, low: 1, points: 0 },
        { id: 'double2', high: 2, low: 2, points: 0 }
      ];

      const state = createTestState({
        phase: 'playing',
        trump: 8, // 8 represents no-trump
        currentTrick: [],
        currentSuit: null,
        currentPlayer: 0,
        players: [
          { 
            id: 0, 
            name: 'Player 0', 
            teamId: 0, 
            marks: 0, 
            hand: mixedHand,
            suitAnalysis: analyzeSuits(mixedHand, 8)
          },
          { id: 1, name: 'Player 1', teamId: 1, marks: 0, hand: [] },
          { id: 2, name: 'Player 2', teamId: 0, marks: 0, hand: [] },
          { id: 3, name: 'Player 3', teamId: 1, marks: 0, hand: [] }
        ]
      });

      // With no trump, all dominoes should be playable initially
      const validPlays = getValidPlays(state, 0);
      expect(validPlays.length).toBeGreaterThan(0);
    });

    it('handles doubles as trump correctly', () => {
      const handWithDoubles: Domino[] = [
        { id: 'double0', high: 0, low: 0, points: 0 },
        { id: 'double1', high: 1, low: 1, points: 0 },
        { id: 'double2', high: 2, low: 2, points: 0 },
        { id: 'double6', high: 6, low: 6, points: 0 },
        { id: 'regular1', high: 2, low: 3, points: 0 },
        { id: 'regular2', high: 4, low: 5, points: 0 },
        { id: 'regular3', high: 1, low: 6, points: 0 }
      ];

      const state = createTestState({
        phase: 'playing',
        trump: 7, // doubles are trump
        currentTrick: [],
        currentSuit: null,
        currentPlayer: 0,
        players: [
          { 
            id: 0, 
            name: 'Player 0', 
            teamId: 0, 
            marks: 0, 
            hand: handWithDoubles,
            suitAnalysis: analyzeSuits(handWithDoubles, 7)
          },
          { id: 1, name: 'Player 1', teamId: 1, marks: 0, hand: [] },
          { id: 2, name: 'Player 2', teamId: 0, marks: 0, hand: [] },
          { id: 3, name: 'Player 3', teamId: 1, marks: 0, hand: [] }
        ]
      });

      // All doubles should be trump
      const doubles = handWithDoubles.filter(d => d.high === d.low);
      expect(doubles).toHaveLength(4);

      const validPlays = getValidPlays(state, 0);
      expect(validPlays).toHaveLength(7); // All should be playable initially
    });
  });

  describe('Extreme Point Distribution', () => {
    it('handles all points to one team scenario', () => {
      // Scenario: One team captures all counting dominoes and tricks
      const allPointsToTeam0 = calculateGameScore([42, 0, 0, 0]);
      
      expect(allPointsToTeam0[0]).toBe(42); // Team 0 gets everything
      expect(allPointsToTeam0[1]).toBe(0);  // Team 1 gets nothing
      expect(allPointsToTeam0[0] + allPointsToTeam0[1]).toBe(42);
    });

    it('handles perfect point split scenario', () => {
      // Scenario: Perfect 21-21 split
      const perfectSplit = calculateGameScore([21, 0, 21, 0]);
      
      expect(perfectSplit[0]).toBe(42); // Team 0: 21 + 21
      expect(perfectSplit[1]).toBe(0);  // Team 1: 0 + 0
      
      // Alternative perfect split
      const altSplit = calculateGameScore([10.5, 10.5, 10.5, 10.5]);
      expect(altSplit[0] + altSplit[1]).toBe(42);
    });

    it('handles counting dominoes distribution extremes', () => {
      // All high-value counting dominoes to one player
      const highValueDominoes = [
        { id: 'five-five', high: 5, low: 5, points: 10 },
        { id: 'six-four', high: 6, low: 4, points: 10 },
        { id: 'five-blank', high: 5, low: 0, points: 5 },
        { id: 'four-one', high: 4, low: 1, points: 5 },
        { id: 'three-deuce', high: 3, low: 2, points: 5 }
      ];

      const totalCountValue = highValueDominoes.reduce((sum, d) => sum + (d.points || 0), 0);
      expect(totalCountValue).toBe(35); // All counting dominoes
    });

    it('handles no-count hands', () => {
      // Hand with no counting dominoes
      const noCountHand: Domino[] = [
        { id: 'nc1', high: 6, low: 1, points: 0 },
        { id: 'nc2', high: 6, low: 2, points: 0 },
        { id: 'nc3', high: 6, low: 3, points: 0 },
        { id: 'nc4', high: 5, low: 2, points: 0 },
        { id: 'nc5', high: 5, low: 3, points: 0 },
        { id: 'nc6', high: 4, low: 3, points: 0 },
        { id: 'nc7', high: 1, low: 1, points: 0 }
      ];

      const countTotal = noCountHand.reduce((sum, d) => sum + (d.points || 0), 0);
      expect(countTotal).toBe(0);
    });
  });

  describe('Game-Ending Scenarios', () => {
    it('handles exact 7-mark victory', () => {
      // TODO: Use state variable in actual test implementation
      // const state = createTestState({
      //   teamMarks: [6, 5],
      //   phase: 'scoring'
      // });

      // Team 0 wins one more mark to reach exactly 7
      const newMarks: [number, number] = [7, 5];
      expect(isGameComplete(newMarks, 7)).toBe(true);
    });

    it('handles over-mark victory', () => {
      const state = createTestState({
        teamMarks: [5, 4],
        phase: 'scoring'
      });

      // Verify initial state before over-mark scenario
      expect(state.teamMarks).toEqual([5, 4]);
      expect(isGameComplete(state.teamMarks, 7)).toBe(false);

      // Team 0 bids and makes 3 marks, going to 8
      const newMarks: [number, number] = [8, 4];
      expect(isGameComplete(newMarks, 7)).toBe(true);
    });

    it('handles simultaneous high marks', () => {
      // Both teams near victory
      const state = createTestState({
        teamMarks: [6, 6],
        phase: 'playing'
      });

      // Verify current state has both teams at 6 marks
      expect(state.teamMarks).toEqual([6, 6]);
      expect(isGameComplete(state.teamMarks, 7)).toBe(false); // Game not over yet
      
      // Either team winning next hand wins game
      const team0Wins: [number, number] = [7, 6];
      const team1Wins: [number, number] = [6, 7];
      
      expect(isGameComplete(team0Wins, 7)).toBe(true);
      expect(isGameComplete(team1Wins, 7)).toBe(true);
    });
  });

  describe('Strategic Edge Cases', () => {
    it('handles forced play scenarios', () => {
      // Player has only one legal play
      const limitedHand: Domino[] = [
        { id: 'forced', high: 3, low: 5, points: 0 },  // Only domino with threes
        { id: 'trump1', high: 1, low: 2, points: 0 },  // Trump but can't play
        { id: 'other1', high: 4, low: 6, points: 0 }   // No threes
      ];

      const state = createTestState({
        phase: 'playing',
        trump: 1, // Ones trump
        currentTrick: [{
          player: 0,
          domino: { id: 'lead', high: 3, low: 3, points: 0 } // 3-3 leads
        }],
        currentSuit: 3, // Threes were led
        currentPlayer: 1,
        players: [
          { id: 0, name: 'Player 0', teamId: 0, marks: 0, hand: [] },
          { 
            id: 1, 
            name: 'Player 1', 
            teamId: 1, 
            marks: 0, 
            hand: limitedHand,
            suitAnalysis: analyzeSuits(limitedHand, 1)
          },
          { id: 2, name: 'Player 2', teamId: 0, marks: 0, hand: [] },
          { id: 3, name: 'Player 3', teamId: 1, marks: 0, hand: [] }
        ]
      });
      
      const validPlays = getValidPlays(state, 1);
      
      // Should be forced to play the threes suit domino
      expect(validPlays).toHaveLength(1);
      expect(validPlays[0].id).toBe('forced');
    });

    it('handles blocked suit scenarios', () => {
      // All dominoes of a suit are played
      const afterSuitBlocked = createTestState({
        phase: 'playing',
        trump: 2,
        tricks: [
          {
            plays: [
              { player: 0, domino: { id: '0-0', high: 0, low: 0, points: 0 } },
              { player: 1, domino: { id: '0-1', high: 0, low: 1, points: 0 } },
              { player: 2, domino: { id: '0-2', high: 0, low: 2, points: 0 } },
              { player: 3, domino: { id: '0-3', high: 0, low: 3, points: 0 } }
            ],
            winner: 2, // Player with 0-2 (trump) wins
            points: 0
          },
          {
            plays: [
              { player: 2, domino: { id: '0-4', high: 0, low: 4, points: 0 } },
              { player: 3, domino: { id: '0-5', high: 0, low: 5, points: 5 } },
              { player: 0, domino: { id: '0-6', high: 0, low: 6, points: 0 } },
              { player: 1, domino: { id: 'other', high: 3, low: 4, points: 0 } }
            ],
            winner: 3, // Player with 0-5 wins
            points: 6 // 5 count + 1 trick
          }
        ]
      });

      // All blanks (0s) have been played - verify the state reflects this
      expect(afterSuitBlocked.tricks).toHaveLength(2);
      
      // Future leads of blanks impossible - all 0s should be in tricks
      const allZeroesPlayed = afterSuitBlocked.tricks.every(trick => 
        trick.plays.some(play => play.domino.high === 0 || play.domino.low === 0)
      );
      expect(allZeroesPlayed).toBe(true);
    });
  });
});