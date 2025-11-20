/**
 * Unit tests for base ruleset game rules.
 *
 * Tests all 7 GameRules methods:
 * - getTrumpSelector: Bidder selects trump
 * - getFirstLeader: Trump selector leads first
 * - getNextPlayer: Clockwise turn order
 * - isTrickComplete: 4 plays = complete trick
 * - checkHandOutcome: All 7 tricks must be played
 * - getLedSuit: Correct led suit logic with doubles-trump
 * - calculateTrickWinner: Standard trick-taking hierarchy
 */

import { describe, it, expect } from 'vitest';
import { baseRuleSet } from '../../../game/rulesets/base';
import { composeRules } from '../../../game/rulesets/compose';
import type { Play, Trick } from '../../../game/types';
import { StateBuilder } from '../../helpers';
import { BLANKS, ACES, DEUCES, TRES, SIXES, DOUBLES_AS_TRUMP } from '../../../game/types';
import { BID_TYPES } from '../../../game/constants';

describe('Base RuleSet Rules', () => {
  // Create composed rules for testing
  const rules = composeRules([baseRuleSet]);

  describe('getTrumpSelector', () => {
    it('should return the bidding player as trump selector', () => {
      const state = StateBuilder.inBiddingPhase().build();
      const bid = { type: BID_TYPES.POINTS, value: 30, player: 2 };

      const selector = rules.getTrumpSelector(state, bid);

      expect(selector).toBe(2);
    });

    it('should work for different players', () => {
      const state = StateBuilder.inBiddingPhase().build();

      expect(rules.getTrumpSelector(state, { type: BID_TYPES.POINTS, value: 30, player: 0 })).toBe(0);
      expect(rules.getTrumpSelector(state, { type: BID_TYPES.POINTS, value: 30, player: 1 })).toBe(1);
      expect(rules.getTrumpSelector(state, { type: BID_TYPES.POINTS, value: 30, player: 3 })).toBe(3);
    });

    it('should work for marks bid', () => {
      const state = StateBuilder.inBiddingPhase().build();
      const bid = { type: BID_TYPES.MARKS, value: 2, player: 1 };

      expect(rules.getTrumpSelector(state, bid)).toBe(1);
    });
  });

  describe('getFirstLeader', () => {
    it('should return trump selector as first leader', () => {
      const state = StateBuilder.inBiddingPhase().build();
      const trump = { type: 'suit' as const, suit: ACES };

      const leader = rules.getFirstLeader(state, 2, trump);

      expect(leader).toBe(2);
    });

    it('should work for all player positions', () => {
      const state = StateBuilder.inBiddingPhase().build();
      const trump = { type: 'suit' as const, suit: TRES };

      expect(rules.getFirstLeader(state, 0, trump)).toBe(0);
      expect(rules.getFirstLeader(state, 1, trump)).toBe(1);
      expect(rules.getFirstLeader(state, 2, trump)).toBe(2);
      expect(rules.getFirstLeader(state, 3, trump)).toBe(3);
    });

    it('should work with doubles trump', () => {
      const state = StateBuilder.inBiddingPhase().build();
      const trump = { type: 'doubles' as const };

      expect(rules.getFirstLeader(state, 3, trump)).toBe(3);
    });
  });

  describe('getNextPlayer', () => {
    it('should rotate clockwise (0 -> 1 -> 2 -> 3 -> 0)', () => {
      const state = StateBuilder.inBiddingPhase().build();

      expect(rules.getNextPlayer(state, 0)).toBe(1);
      expect(rules.getNextPlayer(state, 1)).toBe(2);
      expect(rules.getNextPlayer(state, 2)).toBe(3);
      expect(rules.getNextPlayer(state, 3)).toBe(0);
    });

    it('should wrap around correctly', () => {
      const state = StateBuilder.inBiddingPhase().build();

      // Starting from player 3, should wrap to 0
      const next = rules.getNextPlayer(state, 3);
      expect(next).toBe(0);

      // Chain: 3 -> 0 -> 1 -> 2
      const next2 = rules.getNextPlayer(state, next);
      expect(next2).toBe(1);
    });
  });

  describe('isTrickComplete', () => {
    it('should return false for empty trick', () => {
      const state = StateBuilder.inBiddingPhase()
        .with({ currentTrick: [] })
        .build();

      expect(rules.isTrickComplete(state)).toBe(false);
    });

    it('should return false for 1 play', () => {
      const state = StateBuilder.inBiddingPhase()
        .withCurrentTrick([
          { player: 0, domino: { id: '1-0', high: 1, low: 0 } }
        ])
        .build();

      expect(rules.isTrickComplete(state)).toBe(false);
    });

    it('should return false for 2 plays', () => {
      const state = StateBuilder.inBiddingPhase()
        .withCurrentTrick([
          { player: 0, domino: { id: '1-0', high: 1, low: 0 } },
          { player: 1, domino: { id: '2-0', high: 2, low: 0 } }
        ])
        .build();

      expect(rules.isTrickComplete(state)).toBe(false);
    });

    it('should return false for 3 plays', () => {
      const state = StateBuilder.inBiddingPhase()
        .withCurrentTrick([
          { player: 0, domino: { id: '1-0', high: 1, low: 0 } },
          { player: 1, domino: { id: '2-0', high: 2, low: 0 } },
          { player: 2, domino: { id: '3-0', high: 3, low: 0 } }
        ])
        .build();

      expect(rules.isTrickComplete(state)).toBe(false);
    });

    it('should return true for 4 plays', () => {
      const state = StateBuilder.inBiddingPhase()
        .withCurrentTrick([
          { player: 0, domino: { id: '1-0', high: 1, low: 0 } },
          { player: 1, domino: { id: '2-0', high: 2, low: 0 } },
          { player: 2, domino: { id: '3-0', high: 3, low: 0 } },
          { player: 3, domino: { id: '4-0', high: 4, low: 0 } }
        ])
        .build();

      expect(rules.isTrickComplete(state)).toBe(true);
    });
  });

  describe('checkHandOutcome', () => {
    it('should return undetermined when no tricks played', () => {
      const state = StateBuilder.inBiddingPhase()
        .withTricks([])
        .build();

      const outcome = rules.checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(false);
    });

    it('should return undetermined when 1-6 tricks played', () => {
      for (let trickCount = 1; trickCount <= 6; trickCount++) {
        const tricks: Trick[] = Array.from({ length: trickCount }, (_, i) => ({
          plays: [
            { player: 0, domino: { id: `${i}-0`, high: i, low: 0 } },
            { player: 1, domino: { id: `${i}-1`, high: i, low: 1 } },
            { player: 2, domino: { id: `${i}-2`, high: i, low: 2 } },
            { player: 3, domino: { id: `${i}-3`, high: i, low: 3 } }
          ],
          winner: 0,
          points: 0
        }));

        const state = StateBuilder.inBiddingPhase().withTricks(tricks).build();
        const outcome = rules.checkHandOutcome(state);

        expect(outcome.isDetermined).toBe(false);
      }
    });

    it('should return determined outcome after 7 tricks', () => {
      const tricks: Trick[] = Array.from({ length: 7 }, (_, i) => ({
        plays: [
          { player: 0, domino: { id: `${i}-0`, high: i % 7, low: 0 } },
          { player: 1, domino: { id: `${i}-1`, high: i % 7, low: 1 } },
          { player: 2, domino: { id: `${i}-2`, high: i % 7, low: 2 } },
          { player: 3, domino: { id: `${i}-3`, high: i % 7, low: 3 } }
        ],
        winner: 0,
        points: 0
      }));

      const state = StateBuilder.inBiddingPhase().withTricks(tricks).build();
      const outcome = rules.checkHandOutcome(state);

      expect(outcome.isDetermined).toBe(true);
      expect((outcome as { isDetermined: true; reason: string }).reason).toBe('All tricks played');
    });
  });

  describe('getLedSuit', () => {
    it('should return higher pip for non-doubles with no trump', () => {
      const state = StateBuilder.inBiddingPhase()
        .withTrump({ type: 'not-selected' })
        .build();
      const domino = { id: '6-2', high: 6, low: 2 };

      expect(rules.getLedSuit(state, domino)).toBe(6);
    });

    it('should return higher pip for non-doubles with regular trump', () => {
      const state = StateBuilder.inBiddingPhase()
        .withTrump({ type: 'suit', suit: ACES })
        .build();
      const domino = { id: '6-2', high: 6, low: 2 };

      expect(rules.getLedSuit(state, domino)).toBe(6);
    });

    it('should return trump suit for trump dominoes', () => {
      const state = StateBuilder.inBiddingPhase()
        .withTrump({ type: 'suit', suit: TRES })
        .build();
      const domino = { id: '6-3', high: 6, low: 3 };

      expect(rules.getLedSuit(state, domino)).toBe(TRES);
    });

    it('should return trump suit when low value is trump', () => {
      const state = StateBuilder.inBiddingPhase()
        .withTrump({ type: 'suit', suit: DEUCES })
        .build();
      const domino = { id: '5-2', high: 5, low: 2 };

      expect(rules.getLedSuit(state, domino)).toBe(DEUCES);
    });

    it('should return 7 for doubles when doubles are trump', () => {
      const state = StateBuilder.inBiddingPhase()
        .withTrump({ type: 'doubles' })
        .build();
      const double = { id: '4-4', high: 4, low: 4 };

      expect(rules.getLedSuit(state, double)).toBe(DOUBLES_AS_TRUMP);
    });

    it('should return higher pip for non-doubles when doubles are trump', () => {
      const state = StateBuilder.inBiddingPhase()
        .withTrump({ type: 'doubles' })
        .build();
      const domino = { id: '6-2', high: 6, low: 2 };

      expect(rules.getLedSuit(state, domino)).toBe(6);
    });

    it('should handle all doubles correctly with doubles trump', () => {
      const state = StateBuilder.inBiddingPhase()
        .withTrump({ type: 'doubles' })
        .build();

      for (let i = 0; i <= 6; i++) {
        const double = { id: `${i}-${i}`, high: i, low: i };
        expect(rules.getLedSuit(state, double)).toBe(DOUBLES_AS_TRUMP);
      }
    });
  });

  describe('calculateTrickWinner', () => {
    describe('basic trump hierarchy', () => {
      it('should select first play when all play same value non-trump', () => {
        const state = StateBuilder.inBiddingPhase()
          .withTrump({ type: 'suit', suit: ACES })
          .with({ currentSuit: BLANKS })
          .build();
        const trick: Play[] = [
          { player: 0, domino: { id: '6-0', high: 6, low: 0 } },
          { player: 1, domino: { id: '5-0', high: 5, low: 0 } },
          { player: 2, domino: { id: '4-0', high: 4, low: 0 } },
          { player: 3, domino: { id: '3-0', high: 3, low: 0 } }
        ];

        const winner = rules.calculateTrickWinner(state, trick);
        expect(winner).toBe(0); // 6-0 is highest
      });

      it('should award trump over non-trump', () => {
        const state = StateBuilder.inBiddingPhase()
          .withTrump({ type: 'suit', suit: TRES })
          .with({ currentSuit: BLANKS })
          .build();
        const trick: Play[] = [
          { player: 0, domino: { id: '6-0', high: 6, low: 0 } }, // Non-trump
          { player: 1, domino: { id: '3-2', high: 3, low: 2 } }, // Trump (contains 3)
          { player: 2, domino: { id: '5-0', high: 5, low: 0 } }, // Non-trump
          { player: 3, domino: { id: '4-0', high: 4, low: 0 } }  // Non-trump
        ];

        const winner = rules.calculateTrickWinner(state, trick);
        expect(winner).toBe(1); // Player 1 played trump
      });

      it('should select higher trump when multiple trumps played', () => {
        const state = StateBuilder.inBiddingPhase()
          .withTrump({ type: 'suit', suit: SIXES })
          .with({ currentSuit: SIXES })
          .build();
        const trick: Play[] = [
          { player: 0, domino: { id: '6-0', high: 6, low: 0 } }, // Trump
          { player: 1, domino: { id: '6-2', high: 6, low: 2 } }, // Trump
          { player: 2, domino: { id: '6-4', high: 6, low: 4 } }, // Trump (highest value)
          { player: 3, domino: { id: '5-0', high: 5, low: 0 } }  // Non-trump
        ];

        const winner = rules.calculateTrickWinner(state, trick);
        expect(winner).toBe(2); // 6-4 has highest trump value
      });
    });

    describe('following suit', () => {
      it('should award higher value when following suit', () => {
        const state = StateBuilder.inBiddingPhase()
          .withTrump({ type: 'suit', suit: ACES })
          .with({ currentSuit: BLANKS })
          .build();
        const trick: Play[] = [
          { player: 0, domino: { id: '3-0', high: 3, low: 0 } }, // Led suit BLANKS
          { player: 1, domino: { id: '6-0', high: 6, low: 0 } }, // Follows suit, higher
          { player: 2, domino: { id: '2-0', high: 2, low: 0 } }, // Follows suit, lower
          { player: 3, domino: { id: '5-2', high: 5, low: 2 } }  // Doesn't follow
        ];

        const winner = rules.calculateTrickWinner(state, trick);
        expect(winner).toBe(1); // 6-0 is highest following suit
      });

      it('should ignore non-followers even with higher values', () => {
        const state = StateBuilder.inBiddingPhase()
          .withTrump({ type: 'suit', suit: ACES })
          .with({ currentSuit: DEUCES })
          .build();
        const trick: Play[] = [
          { player: 0, domino: { id: '3-2', high: 3, low: 2 } }, // Led suit DEUCES
          { player: 1, domino: { id: '6-0', high: 6, low: 0 } }, // Doesn't follow
          { player: 2, domino: { id: '4-2', high: 4, low: 2 } }, // Follows suit
          { player: 3, domino: { id: '5-0', high: 5, low: 0 } }  // Doesn't follow
        ];

        const winner = rules.calculateTrickWinner(state, trick);
        expect(winner).toBe(2); // 4-2 follows suit, beats non-followers
      });
    });

    describe('doubles trump', () => {
      it('should award double over non-doubles when doubles are trump', () => {
        const state = StateBuilder.inBiddingPhase()
          .withTrump({ type: 'doubles' })
          .with({ currentSuit: DOUBLES_AS_TRUMP })
          .build();
        const trick: Play[] = [
          { player: 0, domino: { id: '6-5', high: 6, low: 5 } }, // Non-double
          { player: 1, domino: { id: '2-2', high: 2, low: 2 } }, // Double (trump)
          { player: 2, domino: { id: '6-0', high: 6, low: 0 } }, // Non-double
          { player: 3, domino: { id: '5-3', high: 5, low: 3 } }  // Non-double
        ];

        const winner = rules.calculateTrickWinner(state, trick);
        expect(winner).toBe(1); // Double wins
      });

      it('should select higher double when multiple doubles played', () => {
        const state = StateBuilder.inBiddingPhase()
          .withTrump({ type: 'doubles' })
          .with({ currentSuit: DOUBLES_AS_TRUMP })
          .build();
        const trick: Play[] = [
          { player: 0, domino: { id: '2-2', high: 2, low: 2 } }, // Double
          { player: 1, domino: { id: '5-5', high: 5, low: 5 } }, // Higher double
          { player: 2, domino: { id: '3-3', high: 3, low: 3 } }, // Lower double
          { player: 3, domino: { id: '6-0', high: 6, low: 0 } }  // Non-double
        ];

        const winner = rules.calculateTrickWinner(state, trick);
        expect(winner).toBe(1); // 5-5 is highest double
      });
    });

    describe('edge cases', () => {
      it('should handle single play trick', () => {
        const state = StateBuilder.inBiddingPhase()
          .withTrump({ type: 'suit', suit: ACES })
          .build();
        const trick: Play[] = [
          { player: 2, domino: { id: '6-0', high: 6, low: 0 } }
        ];

        const winner = rules.calculateTrickWinner(state, trick);
        expect(winner).toBe(2);
      });

      it('should throw error for empty trick', () => {
        const state = StateBuilder.inBiddingPhase()
          .withTrump({ type: 'suit', suit: ACES })
          .build();

        expect(() => rules.calculateTrickWinner(state, [])).toThrow('Trick cannot be empty');
      });

      it('should handle first player winning', () => {
        const state = StateBuilder.inBiddingPhase()
          .withTrump({ type: 'suit', suit: ACES })
          .with({ currentSuit: BLANKS })
          .build();
        const trick: Play[] = [
          { player: 0, domino: { id: '6-0', high: 6, low: 0 } }, // Highest
          { player: 1, domino: { id: '2-0', high: 2, low: 0 } },
          { player: 2, domino: { id: '3-0', high: 3, low: 0 } },
          { player: 3, domino: { id: '4-0', high: 4, low: 0 } }
        ];

        expect(rules.calculateTrickWinner(state, trick)).toBe(0);
      });

      it('should handle last player winning', () => {
        const state = StateBuilder.inBiddingPhase()
          .withTrump({ type: 'suit', suit: ACES })
          .with({ currentSuit: BLANKS })
          .build();
        const trick: Play[] = [
          { player: 0, domino: { id: '2-0', high: 2, low: 0 } },
          { player: 1, domino: { id: '3-0', high: 3, low: 0 } },
          { player: 2, domino: { id: '4-0', high: 4, low: 0 } },
          { player: 3, domino: { id: '6-0', high: 6, low: 0 } } // Highest
        ];

        expect(rules.calculateTrickWinner(state, trick)).toBe(3);
      });
    });
  });
});
