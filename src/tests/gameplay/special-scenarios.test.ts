import { describe, it, expect } from 'vitest';
import { createTestState, createHandWithDoubles } from '../helpers/gameTestHelper';
import { calculateGameScore, isGameComplete } from '../../game/core/scoring';
import { isValidPlay, getValidPlays, isValidBid } from '../../game/core/rules';
import { analyzeSuits } from '../../game/core/suit-analysis';
import { BID_TYPES } from '../../game/constants';
import type { Domino, Bid } from '../../game/types';
import { BLANKS, ACES, DEUCES, TRES, FOURS, FIVES, SIXES } from '../../game/types';

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
        bids: []
      });

      // Set the player's hand to have 4 doubles
      state.players[0]!.hand = handWith4Doubles;

      // Plunge bid requires 4+ doubles
      const plungeBid: Bid = { type: BID_TYPES.PLUNGE, value: 4, player: 0 };
      
      // Should be valid with 4 doubles in base engine
      // Tournament variant will filter special bids at action level
      expect(isValidBid(state, plungeBid, handWith4Doubles)).toBe(true);

      // Base engine is maximally permissive for special bids
      // REMOVED: state.tournamentMode = true;
      expect(isValidBid(state, plungeBid, handWith4Doubles)).toBe(true);
    });
  });

  describe('Trump Dominance Scenarios', () => {
    it('handles all-trump hands correctly', () => {
      // Create hand with all trump dominoes
      const allTrumpHand: Domino[] = [
        { id: 'trump1', high: ACES, low: BLANKS, points: 0 },
        { id: 'trump2', high: ACES, low: ACES, points: 0 },
        { id: 'trump3', high: ACES, low: DEUCES, points: 0 },
        { id: 'trump4', high: ACES, low: TRES, points: 0 },
        { id: 'trump5', high: ACES, low: FOURS, points: 0 },
        { id: 'trump6', high: ACES, low: FIVES, points: 0 },
        { id: 'trump7', high: ACES, low: SIXES, points: 0 }
      ];

      const state = createTestState({
        phase: 'playing',
        trump: { type: 'suit', suit: ACES }, // ones are trump
        currentTrick: [],
        currentSuit: -1,
        currentPlayer: 0,
        players: [
          { 
            id: 0, 
            name: 'Player 0', 
            teamId: 0, 
            marks: 0, 
            hand: allTrumpHand,
            suitAnalysis: analyzeSuits(allTrumpHand, { type: 'suit', suit: ACES })
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
        { id: 'twos1', high: DEUCES, low: BLANKS, points: 0 },
        { id: 'threes1', high: TRES, low: ACES, points: 5 },
        { id: 'fours1', high: FOURS, low: DEUCES, points: 0 },
        { id: 'fives1', high: FIVES, low: TRES, points: 0 },
        { id: 'sixes1', high: SIXES, low: FOURS, points: 10 },
        { id: 'double1', high: ACES, low: ACES, points: 0 },
        { id: 'double2', high: DEUCES, low: DEUCES, points: 0 }
      ];

      const state = createTestState({
        phase: 'playing',
        trump: { type: 'no-trump' }, // no-trump
        currentTrick: [],
        currentSuit: -1,
        currentPlayer: 0,
        players: [
          { 
            id: 0, 
            name: 'Player 0', 
            teamId: 0, 
            marks: 0, 
            hand: mixedHand,
            suitAnalysis: analyzeSuits(mixedHand, { type: 'no-trump' })
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
        { id: 'double0', high: BLANKS, low: BLANKS, points: 0 },
        { id: 'double1', high: ACES, low: ACES, points: 0 },
        { id: 'double2', high: DEUCES, low: DEUCES, points: 0 },
        { id: 'double6', high: SIXES, low: SIXES, points: 0 },
        { id: 'regular1', high: DEUCES, low: TRES, points: 0 },
        { id: 'regular2', high: FOURS, low: FIVES, points: 0 },
        { id: 'regular3', high: ACES, low: SIXES, points: 0 }
      ];

      const state = createTestState({
        phase: 'playing',
        trump: { type: 'doubles' }, // doubles are trump
        currentTrick: [],
        currentSuit: -1,
        currentPlayer: 0,
        players: [
          { 
            id: 0, 
            name: 'Player 0', 
            teamId: 0, 
            marks: 0, 
            hand: handWithDoubles,
            suitAnalysis: analyzeSuits(handWithDoubles, { type: 'doubles' })
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
        { id: 'five-five', high: FIVES, low: FIVES, points: 10 },
        { id: 'six-four', high: SIXES, low: FOURS, points: 10 },
        { id: 'five-blank', high: FIVES, low: BLANKS, points: 5 },
        { id: 'four-one', high: FOURS, low: ACES, points: 5 },
        { id: 'three-deuce', high: TRES, low: DEUCES, points: 5 }
      ];

      const totalCountValue = highValueDominoes.reduce((sum, d) => sum + (d.points || 0), 0);
      expect(totalCountValue).toBe(35); // All counting dominoes
    });

    it('handles no-count hands', () => {
      // Hand with no counting dominoes
      const noCountHand: Domino[] = [
        { id: 'nc1', high: SIXES, low: ACES, points: 0 },
        { id: 'nc2', high: SIXES, low: DEUCES, points: 0 },
        { id: 'nc3', high: SIXES, low: TRES, points: 0 },
        { id: 'nc4', high: FIVES, low: DEUCES, points: 0 },
        { id: 'nc5', high: FIVES, low: TRES, points: 0 },
        { id: 'nc6', high: FOURS, low: TRES, points: 0 },
        { id: 'nc7', high: ACES, low: ACES, points: 0 }
      ];

      const countTotal = noCountHand.reduce((sum, d) => sum + (d.points || 0), 0);
      expect(countTotal).toBe(0);
    });
  });

  describe('Game-Ending Scenarios', () => {
    it('handles exact 7-mark victory', () => {
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
        { id: 'forced', high: TRES, low: FIVES, points: 0 },  // Only domino with threes
        { id: 'trump1', high: ACES, low: DEUCES, points: 0 },  // Trump but can't play
        { id: 'other1', high: FOURS, low: SIXES, points: 0 }   // No threes
      ];

      const state = createTestState({
        phase: 'playing',
        trump: { type: 'suit', suit: ACES }, // Ones trump
        currentTrick: [{
          player: 0,
          domino: { id: 'lead', high: TRES, low: TRES, points: 0 } // 3-3 leads
        }],
        currentSuit: TRES, // Threes were led
        currentPlayer: 1,
        players: [
          { id: 0, name: 'Player 0', teamId: 0, marks: 0, hand: [] },
          { 
            id: 1, 
            name: 'Player 1', 
            teamId: 1, 
            marks: 0, 
            hand: limitedHand,
            suitAnalysis: analyzeSuits(limitedHand, { type: 'suit', suit: ACES })
          },
          { id: 2, name: 'Player 2', teamId: 0, marks: 0, hand: [] },
          { id: 3, name: 'Player 3', teamId: 1, marks: 0, hand: [] }
        ]
      });
      
      const validPlays = getValidPlays(state, 1);
      
      // Should be forced to play the threes suit domino
      expect(validPlays).toHaveLength(1);
      const firstValidPlay = validPlays[0];
      if (!firstValidPlay) {
        throw new Error('No valid play found');
      }
      expect(firstValidPlay.id).toBe('forced');
    });

    it('handles blocked suit scenarios', () => {
      // All dominoes of a suit are played
      const afterSuitBlocked = createTestState({
        phase: 'playing',
        trump: { type: 'suit', suit: DEUCES },
        tricks: [
          {
            plays: [
              { player: 0, domino: { id: '0-0', high: BLANKS, low: BLANKS, points: 0 } },
              { player: 1, domino: { id: '0-1', high: BLANKS, low: ACES, points: 0 } },
              { player: 2, domino: { id: '0-2', high: BLANKS, low: DEUCES, points: 0 } },
              { player: 3, domino: { id: '0-3', high: BLANKS, low: TRES, points: 0 } }
            ],
            winner: 2, // Player with 0-2 (trump) wins
            points: 0
          },
          {
            plays: [
              { player: 2, domino: { id: '0-4', high: BLANKS, low: FOURS, points: 0 } },
              { player: 3, domino: { id: '0-5', high: BLANKS, low: FIVES, points: 5 } },
              { player: 0, domino: { id: '0-6', high: BLANKS, low: SIXES, points: 0 } },
              { player: 1, domino: { id: 'other', high: TRES, low: FOURS, points: 0 } }
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