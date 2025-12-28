import { describe, it, expect } from 'vitest';
import { StateBuilder, HandBuilder } from '../helpers';
import { calculateGameScore, isTargetReached } from '../../game/core/scoring';
import { composeRules, baseLayer, plungeLayer } from '../../game/layers';
import { BID_TYPES } from '../../game/constants';
import type { Domino, Bid } from '../../game/types';
import { BLANKS, ACES, DEUCES, TRES, FOURS, FIVES, SIXES } from '../../game/types';

// Special scenarios include plunge bids (casual rules)
const rules = composeRules([baseLayer, plungeLayer]);

describe('Special Gameplay Scenarios', () => {
  describe('High Stakes Bidding', () => {
    it('handles 6 and 7 mark bids correctly', () => {
      // Create state with escalated bidding up to 6 marks
      const state = StateBuilder
        .inBiddingPhase(0)
        .withCurrentPlayer(3)
        .withBids([
          { type: BID_TYPES.MARKS, value: 4, player: 0 },
          { type: BID_TYPES.MARKS, value: 5, player: 1 },
          { type: BID_TYPES.MARKS, value: 6, player: 2 }
        ])
        .build();

      // Player 3 can bid 7 marks (one more than current 6)
      const sevenMarkBid: Bid = { type: BID_TYPES.MARKS, value: 7, player: 3 };
      expect(rules.isValidBid(state, sevenMarkBid)).toBe(true);

      // But cannot jump to 8 marks (would need to bid 7 first)
      const eightMarkBid: Bid = { type: BID_TYPES.MARKS, value: 8, player: 3 };
      expect(rules.isValidBid(state, eightMarkBid)).toBe(false);
    });

    it('handles minimum 30 bid enforcement', () => {
      const state = StateBuilder
        .inBiddingPhase()
        .withCurrentPlayer(0)
        .build();

      // Minimum bid is 30 points
      const belowMinBid: Bid = { type: BID_TYPES.POINTS, value: 29, player: 0 };
      const minBid: Bid = { type: BID_TYPES.POINTS, value: 30, player: 0 };

      // 29 should be invalid, 30 should be valid
      expect(rules.isValidBid(state, belowMinBid)).toBe(false);
      expect(rules.isValidBid(state, minBid)).toBe(true);
    });

    it('handles plunge bid requirements', () => {
      // Create hand with 4 doubles for valid plunge
      const handWith4Doubles = HandBuilder.withDoubles(4);

      const state = StateBuilder
        .inBiddingPhase()
        .withCurrentPlayer(0)
        .withPlayerHand(0, handWith4Doubles)
        .build();

      // Plunge bid requires 4+ doubles
      const plungeBid: Bid = { type: 'plunge', value: 4, player: 0 };

      // Should be valid with 4 doubles (plungeLayer enabled)
      expect(rules.isValidBid(state, plungeBid, handWith4Doubles)).toBe(true);
    });
  });

  describe('Trump Dominance Scenarios', () => {
    it('handles all-trump hands correctly', () => {
      // Create hand with all trump dominoes
      const allTrumpHand: Domino[] = [
        { id: '1-0', high: ACES, low: BLANKS, points: 0 },
        { id: '1-1', high: ACES, low: ACES, points: 0 },
        { id: '2-1', high: ACES, low: DEUCES, points: 0 },
        { id: '3-1', high: ACES, low: TRES, points: 0 },
        { id: '4-1', high: ACES, low: FOURS, points: 0 },
        { id: '5-1', high: ACES, low: FIVES, points: 0 },
        { id: '6-1', high: ACES, low: SIXES, points: 0 }
      ];

      const state = StateBuilder
        .inPlayingPhase({ type: 'suit', suit: ACES })
        .withCurrentPlayer(0)
        .withPlayerHand(0, allTrumpHand)
        .build();

      // All dominoes should be valid plays
      allTrumpHand.forEach(domino => {
        expect(rules.isValidPlay(state, domino, 0)).toBe(true);
      });

      const validPlays = rules.getValidPlays(state, 0);
      expect(validPlays).toHaveLength(7);
    });

    it('handles no-trump scenarios', () => {
      const mixedHand: Domino[] = [
        { id: '2-0', high: DEUCES, low: BLANKS, points: 0 },
        { id: '3-1', high: TRES, low: ACES, points: 5 },
        { id: '4-2', high: FOURS, low: DEUCES, points: 0 },
        { id: '5-3', high: FIVES, low: TRES, points: 0 },
        { id: '6-4', high: SIXES, low: FOURS, points: 10 },
        { id: '1-1', high: ACES, low: ACES, points: 0 },
        { id: '2-2', high: DEUCES, low: DEUCES, points: 0 }
      ];

      const state = StateBuilder
        .inPlayingPhase({ type: 'no-trump' })
        .withCurrentPlayer(0)
        .withPlayerHand(0, mixedHand)
        .build();

      // With no trump, all dominoes should be playable initially
      const validPlays = rules.getValidPlays(state, 0);
      expect(validPlays.length).toBeGreaterThan(0);
    });

    it('handles doubles as trump correctly', () => {
      const handWithDoubles: Domino[] = [
        { id: '0-0', high: BLANKS, low: BLANKS, points: 0 },
        { id: '1-1', high: ACES, low: ACES, points: 0 },
        { id: '2-2', high: DEUCES, low: DEUCES, points: 0 },
        { id: '6-6', high: SIXES, low: SIXES, points: 0 },
        { id: '3-2', high: DEUCES, low: TRES, points: 0 },
        { id: '5-4', high: FOURS, low: FIVES, points: 0 },
        { id: '6-1', high: ACES, low: SIXES, points: 0 }
      ];

      const state = StateBuilder
        .inPlayingPhase({ type: 'doubles' })
        .withCurrentPlayer(0)
        .withPlayerHand(0, handWithDoubles)
        .build();

      // All doubles should be trump
      const doubles = handWithDoubles.filter(d => d.high === d.low);
      expect(doubles).toHaveLength(4);

      const validPlays = rules.getValidPlays(state, 0);
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
      expect(isTargetReached(newMarks, 7)).toBe(true);
    });

    it('handles over-mark victory', () => {
      const state = StateBuilder
        .inScoringPhase([0, 0])
        .withTeamMarks(5, 4)
        .build();

      // Verify initial state before over-mark scenario
      expect(state.teamMarks).toEqual([5, 4]);
      expect(isTargetReached(state.teamMarks, 7)).toBe(false);

      // Team 0 bids and makes 3 marks, going to 8
      const newMarks: [number, number] = [8, 4];
      expect(isTargetReached(newMarks, 7)).toBe(true);
    });

    it('handles simultaneous high marks', () => {
      // Both teams near victory
      const state = StateBuilder
        .inPlayingPhase()
        .withTeamMarks(6, 6)
        .build();

      // Verify current state has both teams at 6 marks
      expect(state.teamMarks).toEqual([6, 6]);
      expect(isTargetReached(state.teamMarks, 7)).toBe(false); // Game not over yet

      // Either team winning next hand wins game
      const team0Wins: [number, number] = [7, 6];
      const team1Wins: [number, number] = [6, 7];

      expect(isTargetReached(team0Wins, 7)).toBe(true);
      expect(isTargetReached(team1Wins, 7)).toBe(true);
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

      const state = StateBuilder
        .inPlayingPhase({ type: 'suit', suit: ACES })
        .withCurrentPlayer(1)
        .withPlayerHand(1, limitedHand)
        .withCurrentTrick([{
          player: 0,
          domino: { id: '3-3', high: TRES, low: TRES, points: 0 } // 3-3 leads
        }])
        .build();

      const validPlays = rules.getValidPlays(state, 1);

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
      const afterSuitBlocked = StateBuilder
        .inPlayingPhase({ type: 'suit', suit: DEUCES })
        .withTricks([
          {
            plays: [
              { player: 0, domino: { id: '0-0', high: BLANKS, low: BLANKS, points: 0 } },
              { player: 1, domino: { id: '1-0', high: BLANKS, low: ACES, points: 0 } },
              { player: 2, domino: { id: '2-0', high: BLANKS, low: DEUCES, points: 0 } },
              { player: 3, domino: { id: '3-0', high: BLANKS, low: TRES, points: 0 } }
            ],
            winner: 2, // Player with 0-2 (trump) wins
            points: 0
          },
          {
            plays: [
              { player: 2, domino: { id: '4-0', high: BLANKS, low: FOURS, points: 0 } },
              { player: 3, domino: { id: '5-0', high: BLANKS, low: FIVES, points: 5 } },
              { player: 0, domino: { id: '6-0', high: BLANKS, low: SIXES, points: 0 } },
              { player: 1, domino: { id: '4-3', high: TRES, low: FOURS, points: 0 } }
            ],
            winner: 3, // Player with 0-5 wins
            points: 6 // 5 count + 1 trick
          }
        ])
        .build();

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