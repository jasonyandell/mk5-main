/**
 * Unit tests for base layer - the 14 core rules that define Texas 42 gameplay.
 * These tests document the foundation that all other layers build upon.
 */

import { describe, it, expect } from 'vitest';
import { baseLayer } from '../../../game/layers/base';
import { composeRules } from '../../../game/layers/compose';
import type { Play, Trick } from '../../../game/types';
import { StateBuilder } from '../../helpers';
import { BLANKS, ACES, DEUCES, TRES, SIXES, DOUBLES_AS_TRUMP } from '../../../game/types';
import { BID_TYPES } from '../../../game/constants';

describe('Base Layer Rules', () => {
  const rules = composeRules([baseLayer]);

  describe('getTrumpSelector', () => {
    it('returns bidding player for both points and marks bids', () => {
      const state = StateBuilder.inBiddingPhase().build();

      expect(rules.getTrumpSelector(state, { type: BID_TYPES.POINTS, value: 30, player: 2 })).toBe(2);
      expect(rules.getTrumpSelector(state, { type: BID_TYPES.MARKS, value: 2, player: 1 })).toBe(1);
      expect(rules.getTrumpSelector(state, { type: BID_TYPES.POINTS, value: 30, player: 3 })).toBe(3);
    });
  });

  describe('getFirstLeader', () => {
    it('returns trump selector as first leader for all trump types', () => {
      const state = StateBuilder.inBiddingPhase().build();

      expect(rules.getFirstLeader(state, 0, { type: 'suit', suit: ACES })).toBe(0);
      expect(rules.getFirstLeader(state, 2, { type: 'suit', suit: TRES })).toBe(2);
      expect(rules.getFirstLeader(state, 3, { type: 'doubles' })).toBe(3);
    });
  });

  describe('getNextPlayer', () => {
    it('rotates clockwise with wraparound (0 -> 1 -> 2 -> 3 -> 0)', () => {
      const state = StateBuilder.inBiddingPhase().build();

      expect(rules.getNextPlayer(state, 0)).toBe(1);
      expect(rules.getNextPlayer(state, 1)).toBe(2);
      expect(rules.getNextPlayer(state, 2)).toBe(3);
      expect(rules.getNextPlayer(state, 3)).toBe(0);
    });
  });

  describe('isTrickComplete', () => {
    it('returns true only when all 4 players have played', () => {
      const plays = [
        { player: 0, domino: { id: '1-0', high: 1, low: 0 } },
        { player: 1, domino: { id: '2-0', high: 2, low: 0 } },
        { player: 2, domino: { id: '3-0', high: 3, low: 0 } },
        { player: 3, domino: { id: '4-0', high: 4, low: 0 } }
      ];

      expect(rules.isTrickComplete(StateBuilder.inBiddingPhase().withCurrentTrick([]).build())).toBe(false);
      expect(rules.isTrickComplete(StateBuilder.inBiddingPhase().withCurrentTrick(plays.slice(0, 1)).build())).toBe(false);
      expect(rules.isTrickComplete(StateBuilder.inBiddingPhase().withCurrentTrick(plays.slice(0, 3)).build())).toBe(false);
      expect(rules.isTrickComplete(StateBuilder.inBiddingPhase().withCurrentTrick(plays).build())).toBe(true);
    });
  });

  describe('checkHandOutcome', () => {
    it('returns undetermined until all 7 tricks are played', () => {
      const makeTricks = (count: number): Trick[] =>
        Array.from({ length: count }, (_, i) => ({
          plays: [
            { player: 0, domino: { id: `${i}-0`, high: i % 7, low: 0 } },
            { player: 1, domino: { id: `${i}-1`, high: i % 7, low: 1 } },
            { player: 2, domino: { id: `${i}-2`, high: i % 7, low: 2 } },
            { player: 3, domino: { id: `${i}-3`, high: i % 7, low: 3 } }
          ],
          winner: 0,
          points: 0
        }));

      expect(rules.checkHandOutcome(StateBuilder.inBiddingPhase().withTricks([]).build()).isDetermined).toBe(false);
      expect(rules.checkHandOutcome(StateBuilder.inBiddingPhase().withTricks(makeTricks(3)).build()).isDetermined).toBe(false);
      expect(rules.checkHandOutcome(StateBuilder.inBiddingPhase().withTricks(makeTricks(6)).build()).isDetermined).toBe(false);

      const outcome = rules.checkHandOutcome(StateBuilder.inBiddingPhase().withTricks(makeTricks(7)).build());
      expect(outcome.isDetermined).toBe(true);
      expect((outcome as { isDetermined: true; reason: string }).reason).toBe('All tricks played');
    });
  });

  describe('getLedSuit', () => {
    it('returns higher pip for non-trump dominoes', () => {
      expect(rules.getLedSuit(
        StateBuilder.inBiddingPhase().withTrump({ type: 'suit', suit: ACES }).build(),
        { id: '6-2', high: 6, low: 2 }
      )).toBe(6);
    });

    it('returns trump suit when domino contains trump value', () => {
      expect(rules.getLedSuit(
        StateBuilder.inBiddingPhase().withTrump({ type: 'suit', suit: TRES }).build(),
        { id: '6-3', high: 6, low: 3 }
      )).toBe(TRES);

      expect(rules.getLedSuit(
        StateBuilder.inBiddingPhase().withTrump({ type: 'suit', suit: DEUCES }).build(),
        { id: '5-2', high: 5, low: 2 }
      )).toBe(DEUCES);
    });

    it('returns 7 for all doubles when doubles are trump', () => {
      const state = StateBuilder.inBiddingPhase().withTrump({ type: 'doubles' }).build();

      expect(rules.getLedSuit(state, { id: '0-0', high: 0, low: 0 })).toBe(DOUBLES_AS_TRUMP);
      expect(rules.getLedSuit(state, { id: '4-4', high: 4, low: 4 })).toBe(DOUBLES_AS_TRUMP);
      expect(rules.getLedSuit(state, { id: '6-2', high: 6, low: 2 })).toBe(6); // Non-double
    });
  });

  describe('calculateTrickWinner', () => {
    describe('trump hierarchy', () => {
      it('trump beats non-trump regardless of value', () => {
        const state = StateBuilder.inBiddingPhase()
          .withTrump({ type: 'suit', suit: TRES })
          .with({ currentSuit: BLANKS })
          .build();
        const trick: Play[] = [
          { player: 0, domino: { id: '6-0', high: 6, low: 0 } },
          { player: 1, domino: { id: '3-2', high: 3, low: 2 } }, // Trump wins
          { player: 2, domino: { id: '5-0', high: 5, low: 0 } },
          { player: 3, domino: { id: '4-0', high: 4, low: 0 } }
        ];

        expect(rules.calculateTrickWinner(state, trick)).toBe(1);
      });

      it('higher trump value wins when multiple trumps played', () => {
        const state = StateBuilder.inBiddingPhase()
          .withTrump({ type: 'suit', suit: SIXES })
          .with({ currentSuit: SIXES })
          .build();
        const trick: Play[] = [
          { player: 0, domino: { id: '6-0', high: 6, low: 0 } },
          { player: 1, domino: { id: '6-2', high: 6, low: 2 } },
          { player: 2, domino: { id: '6-4', high: 6, low: 4 } }, // Highest trump
          { player: 3, domino: { id: '5-0', high: 5, low: 0 } }
        ];

        expect(rules.calculateTrickWinner(state, trick)).toBe(2);
      });
    });

    describe('suit following', () => {
      it('highest led suit wins among followers; non-followers lose', () => {
        const state = StateBuilder.inBiddingPhase()
          .withTrump({ type: 'suit', suit: ACES })
          .with({ currentSuit: BLANKS })
          .build();
        const trick: Play[] = [
          { player: 0, domino: { id: '3-0', high: 3, low: 0 } },
          { player: 1, domino: { id: '6-0', high: 6, low: 0 } }, // Highest follower
          { player: 2, domino: { id: '2-0', high: 2, low: 0 } },
          { player: 3, domino: { id: '5-2', high: 5, low: 2 } }  // Doesn't follow
        ];

        expect(rules.calculateTrickWinner(state, trick)).toBe(1);
      });
    });

    describe('doubles trump', () => {
      it('any double beats non-doubles; highest double wins', () => {
        const state = StateBuilder.inBiddingPhase()
          .withTrump({ type: 'doubles' })
          .with({ currentSuit: DOUBLES_AS_TRUMP })
          .build();

        const trick1: Play[] = [
          { player: 0, domino: { id: '6-5', high: 6, low: 5 } },
          { player: 1, domino: { id: '2-2', high: 2, low: 2 } }, // Double wins
          { player: 2, domino: { id: '6-0', high: 6, low: 0 } },
          { player: 3, domino: { id: '5-3', high: 5, low: 3 } }
        ];
        expect(rules.calculateTrickWinner(state, trick1)).toBe(1);

        const trick2: Play[] = [
          { player: 0, domino: { id: '2-2', high: 2, low: 2 } },
          { player: 1, domino: { id: '5-5', high: 5, low: 5 } }, // Highest double
          { player: 2, domino: { id: '3-3', high: 3, low: 3 } },
          { player: 3, domino: { id: '6-0', high: 6, low: 0 } }
        ];
        expect(rules.calculateTrickWinner(state, trick2)).toBe(1);
      });
    });

    describe('edge cases', () => {
      it('throws on empty trick; works with single play; any position can win', () => {
        const state = StateBuilder.inBiddingPhase()
          .withTrump({ type: 'suit', suit: ACES })
          .with({ currentSuit: BLANKS })
          .build();

        expect(() => rules.calculateTrickWinner(state, [])).toThrow('Trick cannot be empty');

        expect(rules.calculateTrickWinner(state, [
          { player: 2, domino: { id: '6-0', high: 6, low: 0 } }
        ])).toBe(2);

        expect(rules.calculateTrickWinner(state, [
          { player: 0, domino: { id: '6-0', high: 6, low: 0 } },
          { player: 1, domino: { id: '2-0', high: 2, low: 0 } },
          { player: 2, domino: { id: '3-0', high: 3, low: 0 } },
          { player: 3, domino: { id: '4-0', high: 4, low: 0 } }
        ])).toBe(0);

        expect(rules.calculateTrickWinner(state, [
          { player: 0, domino: { id: '2-0', high: 2, low: 0 } },
          { player: 1, domino: { id: '3-0', high: 3, low: 0 } },
          { player: 2, domino: { id: '4-0', high: 4, low: 0 } },
          { player: 3, domino: { id: '6-0', high: 6, low: 0 } }
        ])).toBe(3);
      });
    });
  });
});
