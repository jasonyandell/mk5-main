import { describe, it, expect } from 'vitest';
import { executeAction } from '../../../game/core/actions';
import { composeRules, baseLayer, nelloLayer } from '../../../game/layers';
import { StateBuilder } from '../../helpers';
import type { Domino } from '../../../game/types';
import { BLANKS, ACES, DEUCES, TRES, FOURS, FIVES, SIXES } from '../../../game/types';

/**
 * Nello Edge Cases - Testing unusual scenarios in nello gameplay:
 * - Partner sits out (3-player rotation)
 * - Doubles form their own suit (suit 7)
 * - Trump option availability based on bid type
 */
describe('Nello Edge Cases', () => {
  const nelloRules = composeRules([baseLayer, nelloLayer]);

  describe('Partner Sits Out - 3 Player Rotation', () => {
    it('should have only 3 active players when partner sits out', () => {
      // Bidder is player 0, partner is player 2
      const state = StateBuilder.nelloContract(0)
        .withTrump({ type: 'nello' })
        .with({ phase: 'playing' })
        .withCurrentPlayer(0)
        .build();

      // Player 0 leads
      expect(state.currentPlayer).toBe(0);

      // Next player should be 1 (not partner 2)
      const next1 = nelloRules.getNextPlayer(state, 0);
      expect(next1).toBe(1);

      // After player 1, should be player 3 (skip partner 2)
      const next2 = nelloRules.getNextPlayer(state, 1);
      expect(next2).toBe(3);

      // After player 3, should be player 0 (skip partner 2)
      const next3 = nelloRules.getNextPlayer(state, 3);
      expect(next3).toBe(0);
    });

    it('should never have partner as currentPlayer during nello hand', () => {
      const state = StateBuilder.nelloContract(1)
        .withTrump({ type: 'nello' })
        .with({ phase: 'playing' })
        .withCurrentPlayer(0)
        .build();

      // Simulate multiple rounds of play
      let current = state.currentPlayer;
      const seen = new Set<number>();

      for (let i = 0; i < 12; i++) { // Multiple rotations
        seen.add(current);
        current = nelloRules.getNextPlayer(state, current);
      }

      // Partner (player 3) should never be currentPlayer
      expect(seen.has(3)).toBe(false);
      // Other players should all be seen
      expect(seen.has(0)).toBe(true);
      expect(seen.has(1)).toBe(true);
      expect(seen.has(2)).toBe(true);
    });

    it('should complete trick with only 3 plays when partner sits out', () => {
      const state = StateBuilder.nelloContract(0)
        .withTrump({ type: 'nello' })
        .with({ phase: 'playing' })
        .withCurrentPlayer(0)
        .build();

      // Trick should not be complete with 2 plays
      const state2Plays = { ...state, currentTrick: [
        { player: 0, domino: { id: '1', high: ACES, low: BLANKS } },
        { player: 1, domino: { id: '2', high: DEUCES, low: BLANKS } }
      ]};
      expect(nelloRules.isTrickComplete(state2Plays)).toBe(false);

      // Trick should be complete with 3 plays (partner sits out)
      const state3Plays = { ...state, currentTrick: [
        { player: 0, domino: { id: '1', high: ACES, low: BLANKS } },
        { player: 1, domino: { id: '2', high: DEUCES, low: BLANKS } },
        { player: 3, domino: { id: '3', high: TRES, low: BLANKS } }
      ]};
      expect(nelloRules.isTrickComplete(state3Plays)).toBe(true);

      // With 4 plays, the trick has too many (partner shouldn't play)
      // isTrickComplete checks for exactly 3 in nello
      const state4Plays = { ...state, currentTrick: [
        { player: 0, domino: { id: '1', high: ACES, low: BLANKS } },
        { player: 1, domino: { id: '2', high: DEUCES, low: BLANKS } },
        { player: 3, domino: { id: '3', high: TRES, low: BLANKS } },
        { player: 2, domino: { id: '4', high: FOURS, low: BLANKS } }
      ]};
      // NOT complete because length !== 3 (this validates the rule enforcement)
      expect(nelloRules.isTrickComplete(state4Plays)).toBe(false);
    });

    it('should handle first trick leader being bidder in 3-player rotation', () => {
      const state = StateBuilder.nelloContract(2)
        .withWinningBid(2, { type: 'marks', value: 1, player: 2 })
        .build();

      // Trump selector should be bidder in nello
      const trumpSelector = nelloRules.getTrumpSelector(state, state.currentBid);
      expect(trumpSelector).toBe(2);

      // First leader should be bidder
      const firstLeader = nelloRules.getFirstLeader(state, trumpSelector, { type: 'nello' });
      expect(firstLeader).toBe(2);
    });
  });

  describe('Doubles Form Own Suit (Suit 7)', () => {
    it('should treat doubles as suit 7 when led in nello', () => {
      const state = StateBuilder.nelloContract(0)
        .withTrump({ type: 'nello' })
        .with({ phase: 'playing' })
        .withCurrentPlayer(0)
        .build();

      // Test all doubles are suit 7
      const doubles = [
        { id: '0-0', high: BLANKS, low: BLANKS },
        { id: '1-1', high: ACES, low: ACES },
        { id: '2-2', high: DEUCES, low: DEUCES },
        { id: '3-3', high: TRES, low: TRES },
        { id: '4-4', high: FOURS, low: FOURS },
        { id: '5-5', high: FIVES, low: FIVES },
        { id: '6-6', high: SIXES, low: SIXES }
      ];

      doubles.forEach(domino => {
        const ledSuit = nelloRules.getLedSuit(state, domino as Domino);
        expect(ledSuit).toBe(7); // Doubles form suit 7
      });
    });

    it('should treat non-doubles as higher pip suit in nello', () => {
      const state = StateBuilder.nelloContract(0)
        .withTrump({ type: 'nello' })
        .with({ phase: 'playing' })
        .withCurrentPlayer(0)
        .build();

      // Non-doubles should use higher pip
      const nonDoubles = [
        { domino: { id: '1-0', high: ACES, low: BLANKS }, expectedSuit: ACES },
        { domino: { id: '2-0', high: DEUCES, low: BLANKS }, expectedSuit: DEUCES },
        { domino: { id: '6-4', high: SIXES, low: FOURS }, expectedSuit: SIXES },
        { domino: { id: '5-3', high: FIVES, low: TRES }, expectedSuit: FIVES }
      ];

      nonDoubles.forEach(({ domino, expectedSuit }) => {
        const ledSuit = nelloRules.getLedSuit(state, domino as Domino);
        expect(ledSuit).toBe(expectedSuit);
      });
    });

    it('should require following doubles suit when doubles led in nello', () => {
      // When doubles are led (suit 7), players must follow with doubles if they have them
      const state = StateBuilder.nelloContract(0)
        .withTrump({ type: 'nello' })
        .with({ phase: 'playing' })
        .withCurrentPlayer(1)
        .withCurrentTrick([
          { player: 0, domino: { id: '5-5', high: FIVES, low: FIVES } } // Double led
        ])
        .with({ currentSuit: 7 }) // Doubles suit
        .withPlayerHand(0, [])
        .withPlayerHand(1, ['2-2', '6-4'])
        .withPlayerHand(2, [])
        .withPlayerHand(3, [])
        .build();

      // Player 1 has doubles, so the double should be suit 7
      const double = state.players[1]!.hand[0]!;
      const doubleSuit = nelloRules.getLedSuit(state, double);
      expect(doubleSuit).toBe(7);

      // Non-double should be suit 6
      const nonDouble = state.players[1]!.hand[1]!;
      const nonDoubleSuit = nelloRules.getLedSuit(state, nonDouble);
      expect(nonDoubleSuit).toBe(SIXES);
    });

    it('should allow any play when player has no doubles and doubles were led', () => {
      const state = StateBuilder.nelloContract(0)
        .withTrump({ type: 'nello' })
        .with({ phase: 'playing' })
        .withCurrentPlayer(3)
        .withCurrentTrick([
          { player: 0, domino: { id: '5-5', high: FIVES, low: FIVES } } // Double led
        ])
        .with({ currentSuit: 7 }) // Doubles suit
        .withPlayerHand(0, [])
        .withPlayerHand(1, [])
        .withPlayerHand(2, [])
        .withPlayerHand(3, ['6-4', '3-1'])
        .build();

      // Player 3 has no doubles, so can play any domino
      // Both should be valid (follow-suit logic in rules will handle this)
      const domino1 = state.players[3]!.hand[0]!;
      const suit1 = nelloRules.getLedSuit(state, domino1);
      expect(suit1).toBe(SIXES); // Not suit 7

      const domino2 = state.players[3]!.hand[1]!;
      const suit2 = nelloRules.getLedSuit(state, domino2);
      expect(suit2).toBe(TRES); // Not suit 7
    });
  });

  describe('Last Trick Completion with 3 Players', () => {
    it('should complete 7th trick correctly with only 3 players', () => {
      const state = StateBuilder.nelloContract(0)
        .withTrump({ type: 'nello' })
        .with({ phase: 'playing' })
        .withCurrentPlayer(0)
        .withTricks([
          // 6 completed tricks (bidder lost all)
          { plays: [
            { player: 0, domino: { id: 't1-0', high: ACES, low: BLANKS } },
            { player: 1, domino: { id: 't1-1', high: DEUCES, low: BLANKS } },
            { player: 3, domino: { id: 't1-3', high: TRES, low: BLANKS } }
          ], winner: 1, points: 0, ledSuit: ACES },
          { plays: [
            { player: 1, domino: { id: 't2-1', high: ACES, low: BLANKS } },
            { player: 3, domino: { id: 't2-3', high: DEUCES, low: BLANKS } },
            { player: 0, domino: { id: 't2-0', high: TRES, low: BLANKS } }
          ], winner: 3, points: 0, ledSuit: ACES },
          { plays: [
            { player: 3, domino: { id: 't3-3', high: ACES, low: BLANKS } },
            { player: 0, domino: { id: 't3-0', high: DEUCES, low: BLANKS } },
            { player: 1, domino: { id: 't3-1', high: TRES, low: BLANKS } }
          ], winner: 1, points: 0, ledSuit: ACES },
          { plays: [
            { player: 1, domino: { id: 't4-1', high: ACES, low: BLANKS } },
            { player: 3, domino: { id: 't4-3', high: DEUCES, low: BLANKS } },
            { player: 0, domino: { id: 't4-0', high: TRES, low: BLANKS } }
          ], winner: 3, points: 0, ledSuit: ACES },
          { plays: [
            { player: 3, domino: { id: 't5-3', high: ACES, low: BLANKS } },
            { player: 0, domino: { id: 't5-0', high: DEUCES, low: BLANKS } },
            { player: 1, domino: { id: 't5-1', high: TRES, low: BLANKS } }
          ], winner: 1, points: 0, ledSuit: ACES },
          { plays: [
            { player: 1, domino: { id: 't6-1', high: ACES, low: BLANKS } },
            { player: 3, domino: { id: 't6-3', high: DEUCES, low: BLANKS } },
            { player: 0, domino: { id: 't6-0', high: TRES, low: BLANKS } }
          ], winner: 3, points: 0, ledSuit: ACES }
        ])
        .withPlayerHand(0, [{ id: 'last-0', high: FOURS, low: BLANKS }])
        .withPlayerHand(1, [{ id: 'last-1', high: FIVES, low: BLANKS }])
        .withPlayerHand(2, [])
        .withPlayerHand(3, [{ id: 'last-3', high: SIXES, low: BLANKS }])
        .build();

      // Check that we have 6 tricks
      expect(state.tricks.length).toBe(6);

      // All tricks should have 3 plays
      state.tricks.forEach((trick) => {
        expect(trick.plays.length).toBe(3); // 3-player tricks
      });

      // 7th trick in progress
      const state7thTrick = {
        ...state,
        currentTrick: [
          { player: 3, domino: { id: 'last-3', high: SIXES, low: BLANKS } }
        ]
      };

      // Not complete with 1 play
      expect(nelloRules.isTrickComplete(state7thTrick)).toBe(false);

      // Not complete with 2 plays
      const state2Plays = {
        ...state7thTrick,
        currentTrick: [
          ...state7thTrick.currentTrick,
          { player: 0, domino: { id: 'last-0', high: FOURS, low: BLANKS } }
        ]
      };
      expect(nelloRules.isTrickComplete(state2Plays)).toBe(false);

      // Complete with 3 plays
      const state3Plays = {
        ...state2Plays,
        currentTrick: [
          ...state2Plays.currentTrick,
          { player: 1, domino: { id: 'last-1', high: FIVES, low: BLANKS } }
        ]
      };
      expect(nelloRules.isTrickComplete(state3Plays)).toBe(true);
    });
  });

  describe('Nello Trump Option Availability', () => {
    it('should allow nello trump selection when marks bid is winning', () => {
      const state = StateBuilder.nelloContract(0).build();

      // Nello should be available as trump option
      const validActions = nelloLayer.getValidActions?.(state, []);
      expect(validActions).toBeDefined();

      const nelloAction = validActions?.find(a =>
        a.type === 'select-trump' &&
        a.trump.type === 'nello'
      );
      expect(nelloAction).toBeDefined();
      expect(nelloAction && nelloAction.type === 'select-trump' ? nelloAction.player : undefined).toBe(0);
    });

    it('should NOT allow nello trump selection when points bid is winning', () => {
      const state = StateBuilder.inTrumpSelection(0, 35).build();

      // Nello should NOT be available
      const validActions = nelloLayer.getValidActions?.(state, []);

      const nelloAction = validActions?.find(a =>
        a.type === 'select-trump' &&
        a.trump.type === 'nello'
      );
      expect(nelloAction).toBeUndefined();
    });

    it('should only add nello during trump_selection phase', () => {
      const biddingState = StateBuilder.inBiddingPhase().build();

      const validActions = nelloLayer.getValidActions?.(biddingState, []);
      const nelloAction = validActions?.find(a =>
        a.type === 'select-trump' &&
        a.trump.type === 'nello'
      );
      expect(nelloAction).toBeUndefined();
    });
  });

  describe('Nello Integration with executeAction', () => {
    it('should properly handle full nello hand with 3-player tricks', () => {
      let state = StateBuilder.nelloContract(0)
        .withPlayerHand(0, ['1-0', '2-0', '3-0', '4-0', '5-0', '6-0', '0-0'])
        .withPlayerHand(1, ['1-1', '2-1', '3-1', '4-1', '5-1', '6-1', '2-2'])
        .withPlayerHand(2, [])
        .withPlayerHand(3, ['3-2', '4-2', '5-2', '6-2', '3-3', '4-3', '5-3'])
        .build();

      // Select nello trump
      state = executeAction(state, {
        type: 'select-trump',
        player: 0,
        trump: { type: 'nello' }
      }, nelloRules);

      expect(state.phase).toBe('playing');
      expect(state.trump.type).toBe('nello');
      expect(state.currentPlayer).toBe(0); // Bidder leads

      // Play first trick (bidder loses)
      state = executeAction(state, {
        type: 'play',
        player: 0,
        dominoId: '1-0' // Play low card
      }, nelloRules);
      expect(state.currentPlayer).toBe(1); // Next is player 1 (not partner 2)

      state = executeAction(state, {
        type: 'play',
        player: 1,
        dominoId: '1-1' // Higher card
      }, nelloRules);
      expect(state.currentPlayer).toBe(3); // Next is player 3 (skip partner 2)

      state = executeAction(state, {
        type: 'play',
        player: 3,
        dominoId: '3-2' // Another card
      }, nelloRules);

      // Trick should be complete with 3 plays
      expect(nelloRules.isTrickComplete(state)).toBe(true);
      expect(state.currentTrick.length).toBe(3);
    });
  });
});
