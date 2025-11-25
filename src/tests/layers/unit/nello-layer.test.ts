/**
 * Unit tests for nello layer game rules.
 *
 * Nello rules (from docs/rules.md §8.A):
 * - Bidder must lose all tricks
 * - Partner sits out (3-player tricks)
 * - No trump suit
 * - Doubles form own suit (suit 7)
 * - Only available after marks bid
 * - Early termination when bidder wins any trick
 */

import { describe, it, expect } from 'vitest';
import { baseLayer } from '../../../game/layers/base';
import { nelloLayer } from '../../../game/layers/nello';
import { composeRules } from '../../../game/layers/compose';
import type { Play } from '../../../game/types';
import { StateBuilder } from '../../helpers';
import { BID_TYPES } from '../../../game/constants';
import { BLANKS, DEUCES, DOUBLES_AS_TRUMP } from '../../../game/types';

describe('Nello Layer Rules', () => {
  const rules = composeRules([baseLayer, nelloLayer]);

  describe('getValidActions', () => {
    it('should add nello trump option after marks bid', () => {
      const state = StateBuilder
        .inTrumpSelection(0)
        .withWinningBid(0, { type: BID_TYPES.MARKS, value: 2, player: 0 })
        .build();

      const baseActions: never[] = [];
      const actions = nelloLayer.getValidActions?.(state, baseActions) ?? [];

      expect(actions).toHaveLength(1);
      expect(actions[0]).toEqual({
        type: 'select-trump',
        player: 0,
        trump: { type: 'nello' }
      });
    });

    it('should not add nello option for points bid', () => {
      const state = StateBuilder
        .inTrumpSelection(0, 30)
        .build();

      const baseActions: never[] = [];
      const actions = nelloLayer.getValidActions?.(state, baseActions) ?? [];

      expect(actions).toEqual([]);
    });

    it('should not add actions during bidding phase', () => {
      const state = StateBuilder
        .inBiddingPhase()
        .withCurrentPlayer(0)
        .withBids([{ type: BID_TYPES.MARKS, value: 2, player: 0 }])
        .build();

      const baseActions: never[] = [];
      const actions = nelloLayer.getValidActions?.(state, baseActions) ?? [];

      // Nello is not a bid type - it's a trump selection
      expect(actions).toEqual([]);
    });

    it('should preserve previous actions', () => {
      const state = StateBuilder
        .inTrumpSelection(1)
        .withWinningBid(1, { type: BID_TYPES.MARKS, value: 3, player: 1 })
        .build();

      const baseActions = [
        { type: 'select-trump' as const, player: 1, trump: { type: 'suit' as const, suit: BLANKS } }
      ];
      const actions = nelloLayer.getValidActions?.(state, baseActions) ?? [];

      expect(actions).toHaveLength(2);
      expect(actions[0]).toEqual(baseActions[0]);
      const action = actions[1];
      if (!action || action.type !== 'select-trump') throw new Error('Expected select-trump action');
      expect(action.trump).toEqual({ type: 'nello' });
    });
  });

  describe('getTrumpSelector', () => {
    it('should pass through to base (bidder selects trump)', () => {
      const state = StateBuilder
        .nelloContract(2)
        .withTrump({ type: 'nello' })
        .build();
      const bid = { type: BID_TYPES.MARKS, value: 2, player: 2 };

      const selector = rules.getTrumpSelector(state, bid);

      expect(selector).toBe(2);
    });
  });

  describe('getFirstLeader', () => {
    it('should pass through to base (bidder leads)', () => {
      const state = StateBuilder
        .nelloContract(1)
        .withTrump({ type: 'nello' })
        .build();

      const leader = rules.getFirstLeader(state, 1, { type: 'nello' });

      expect(leader).toBe(1);
    });
  });

  describe('getNextPlayer', () => {
    it('should skip partner in turn order', () => {
      const state = StateBuilder
        .nelloContract(0) // Partner is player 2
        .withTrump({ type: 'nello' })
        .withCurrentPlayer(0)
        .withHands([[], [], [], []])
        .build();

      // 0 -> 1 (skip 2) -> 3 -> 0
      expect(rules.getNextPlayer(state, 0)).toBe(1);
      expect(rules.getNextPlayer(state, 1)).toBe(3); // Skip player 2
      expect(rules.getNextPlayer(state, 3)).toBe(0);
    });

    it('should skip partner when bidder is player 1', () => {
      const state = StateBuilder
        .nelloContract(1) // Partner is player 3
        .withTrump({ type: 'nello' })
        .withCurrentPlayer(0)
        .withHands([[], [], [], []])
        .build();

      expect(rules.getNextPlayer(state, 0)).toBe(1);
      expect(rules.getNextPlayer(state, 1)).toBe(2);
      expect(rules.getNextPlayer(state, 2)).toBe(0); // Skip player 3
    });

    it('should not skip partner when not nello', () => {
      const state = StateBuilder
        .inPlayingPhase({ type: 'suit', suit: BLANKS })
        .withCurrentPlayer(0)
        .withHands([[], [], [], []])
        .build();

      // Normal rotation
      expect(rules.getNextPlayer(state, 0)).toBe(1);
      expect(rules.getNextPlayer(state, 1)).toBe(2); // Don't skip
      expect(rules.getNextPlayer(state, 2)).toBe(3);
      expect(rules.getNextPlayer(state, 3)).toBe(0);
    });
  });

  describe('isTrickComplete', () => {
    it('should return true after 3 plays in nello', () => {
      const state = StateBuilder
        .nelloContract(0)
        .withTrump({ type: 'nello' })
        .withCurrentTrick([
          { player: 0, domino: '1-0' },
          { player: 1, domino: '2-0' },
          { player: 3, domino: '3-0' }
        ])
        .build();

      expect(rules.isTrickComplete(state)).toBe(true);
    });

    it('should return false for 2 plays in nello', () => {
      const state = StateBuilder
        .nelloContract(0)
        .withTrump({ type: 'nello' })
        .withCurrentTrick([
          { player: 0, domino: '1-0' },
          { player: 1, domino: '2-0' }
        ])
        .build();

      expect(rules.isTrickComplete(state)).toBe(false);
    });

    it('should return false for 1 play in nello', () => {
      const state = StateBuilder
        .nelloContract(0)
        .withTrump({ type: 'nello' })
        .withCurrentTrick([
          { player: 0, domino: '1-0' }
        ])
        .build();

      expect(rules.isTrickComplete(state)).toBe(false);
    });

    it('should use base rules when not nello (4 plays)', () => {
      const state = StateBuilder
        .inPlayingPhase({ type: 'suit', suit: DEUCES })
        .withCurrentTrick([
          { player: 0, domino: '1-0' },
          { player: 1, domino: '2-0' },
          { player: 2, domino: '3-0' }
        ])
        .build();

      expect(rules.isTrickComplete(state)).toBe(false);
    });
  });

  describe('checkHandOutcome', () => {
    it('should return null when bidding team has not won any tricks', () => {
      const plays1 = [
        { player: 0, domino: { id: '1-0', high: 1, low: 0 } },
        { player: 1, domino: { id: '6-0', high: 6, low: 0 } },
        { player: 3, domino: { id: '2-0', high: 2, low: 0 } }
      ];
      const plays2 = [
        { player: 1, domino: { id: '5-0', high: 5, low: 0 } },
        { player: 3, domino: { id: '6-1', high: 6, low: 1 } },
        { player: 0, domino: { id: '2-1', high: 2, low: 1 } }
      ];

      const state = StateBuilder
        .nelloContract(0)
        .withTrump({ type: 'nello' })
        .withHands([[], [], [], []])
        .addTrick(plays1, 1, 0) // Team 1 wins
        .addTrick(plays2, 3, 0) // Team 1 wins
        .build();

      const outcome = rules.checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(false);
    });

    it('should return determined when bidder wins a trick', () => {
      const plays1 = [
        { player: 0, domino: { id: '1-0', high: 1, low: 0 } },
        { player: 1, domino: { id: '2-0', high: 2, low: 0 } },
        { player: 3, domino: { id: '3-0', high: 3, low: 0 } }
      ];
      const plays2 = [
        { player: 1, domino: { id: '5-0', high: 5, low: 0 } },
        { player: 3, domino: { id: '6-0', high: 6, low: 0 } },
        { player: 0, domino: { id: '6-1', high: 6, low: 1 } }
      ];

      const state = StateBuilder
        .nelloContract(0)
        .withTrump({ type: 'nello' })
        .withHands([[], [], [], []])
        .addTrick(plays1, 1, 0) // Team 1 wins - good
        .addTrick(plays2, 0, 0) // Team 0 wins - nello fails!
        .build();

      const outcome = rules.checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(true);
      expect((outcome as { isDetermined: true; reason: string }).reason).toContain('Bidding team won trick');
      expect((outcome as { isDetermined: true; decidedAtTrick: number }).decidedAtTrick).toBe(2);
    });

    it('should return determined when partner wins a trick', () => {
      const plays1 = [
        { player: 1, domino: { id: '1-0', high: 1, low: 0 } },
        { player: 0, domino: { id: '6-0', high: 6, low: 0 } },
        { player: 3, domino: { id: '2-0', high: 2, low: 0 } }
      ];

      const state = StateBuilder
        .nelloContract(1) // Partner is player 3
        .withTrump({ type: 'nello' })
        .withHands([[], [], [], []])
        .addTrick(plays1, 3, 0) // Partner wins - nello fails!
        .build();

      const outcome = rules.checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(true);
      expect((outcome as { isDetermined: true; decidedAtTrick: number }).decidedAtTrick).toBe(1);
    });

    it('should not trigger early termination when not nello', () => {
      const plays1 = [
        { player: 0, domino: { id: '6-0', high: 6, low: 0 } },
        { player: 1, domino: { id: '2-0', high: 2, low: 0 } },
        { player: 2, domino: { id: '3-0', high: 3, low: 0 } },
        { player: 3, domino: { id: '4-0', high: 4, low: 0 } }
      ];

      const state = StateBuilder
        .inPlayingPhase({ type: 'suit', suit: BLANKS })
        .withHands([[], [], [], []])
        .addTrick(plays1, 0, 0) // Bidder wins
        .build();

      const outcome = rules.checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(false); // Should play all tricks
    });
  });

  describe('getLedSuit', () => {
    it('should return 7 for doubles in nello', () => {
      const state = StateBuilder
        .nelloContract(0)
        .withTrump({ type: 'nello' })
        .build();

      for (let i = 0; i <= 6; i++) {
        const double = { id: `${i}-${i}`, high: i, low: i };
        expect(rules.getLedSuit(state, double)).toBe(DOUBLES_AS_TRUMP);
      }
    });

    it('should return higher pip for non-doubles in nello', () => {
      const state = StateBuilder
        .nelloContract(0)
        .withTrump({ type: 'nello' })
        .build();

      const domino1 = { id: '6-2', high: 6, low: 2 };
      expect(rules.getLedSuit(state, domino1)).toBe(6);

      const domino2 = { id: '5-0', high: 5, low: 0 };
      expect(rules.getLedSuit(state, domino2)).toBe(5);

      const domino3 = { id: '4-1', high: 4, low: 1 };
      expect(rules.getLedSuit(state, domino3)).toBe(4);
    });

    it('should not affect led suit when not nello', () => {
      const state = StateBuilder
        .inPlayingPhase({ type: 'suit', suit: BLANKS })
        .build();

      const domino = { id: '6-2', high: 6, low: 2 };
      expect(rules.getLedSuit(state, domino)).toBe(6);
    });
  });

  describe('calculateTrickWinner', () => {
    it('should use base trick-taking rules (no trump hierarchy)', () => {
      const state = StateBuilder
        .nelloContract(0)
        .withTrump({ type: 'nello' })
        .with({ currentSuit: BLANKS })
        .build();
      const trick: Play[] = [
        { player: 0, domino: { id: '3-0', high: 3, low: 0 } },
        { player: 1, domino: { id: '6-0', high: 6, low: 0 } }, // Highest following suit
        { player: 3, domino: { id: '2-0', high: 2, low: 0 } }
      ];

      const winner = rules.calculateTrickWinner(state, trick);
      expect(winner).toBe(1); // 6-0 is highest
    });

    it('should handle doubles leading (suit 7)', () => {
      const state = StateBuilder
        .nelloContract(0)
        .withTrump({ type: 'nello' })
        .with({ currentSuit: DOUBLES_AS_TRUMP })
        .build();
      const trick: Play[] = [
        { player: 0, domino: { id: '2-2', high: 2, low: 2 } }, // Double leads suit 7
        { player: 1, domino: { id: '5-5', high: 5, low: 5 } }, // Double follows suit 7, higher value
        { player: 3, domino: { id: '6-0', high: 6, low: 0 } }  // Non-double, doesn't follow
      ];

      const winner = rules.calculateTrickWinner(state, trick);
      // In nello: doubles lead suit 7, other doubles follow suit 7
      // Player 1 (5-5) follows suit and has higher value than player 0 (2-2)
      expect(winner).toBe(1);
    });
  });

  describe('integration: complete nello hand', () => {
    it('should handle a complete successful nello hand', () => {
      let builder = StateBuilder
        .nelloContract(0)
        .withTrump({ type: 'nello' })
        .withHands([[], [], [], []]);

      // Simulate 7 tricks where opponents always win
      for (let i = 0; i < 7; i++) {
        const plays: Play[] = [
          { player: 0, domino: { id: `${i}-0`, high: i % 7, low: 0 } },
          { player: 1, domino: { id: `${i}-1`, high: (i + 1) % 7, low: 1 } },
          { player: 3, domino: { id: `${i}-3`, high: (i + 2) % 7, low: 3 } }
        ];
        const winner = i % 2 === 0 ? 1 : 3; // Alternate between opponents
        builder = builder.addTrick(plays, winner, 0) as StateBuilder;
      }

      const finalState = builder.build();
      const outcome = rules.checkHandOutcome(finalState);

      // Should be determined by base layer after all 7 tricks
      expect(outcome.isDetermined).toBe(true);
    });

    it('should end early when bidder wins first trick', () => {
      const plays1 = [
        { player: 0, domino: { id: '6-0', high: 6, low: 0 } },
        { player: 1, domino: { id: '2-0', high: 2, low: 0 } },
        { player: 3, domino: { id: '3-0', high: 3, low: 0 } }
      ];

      const state = StateBuilder
        .nelloContract(0)
        .withTrump({ type: 'nello' })
        .withHands([[], [], [], []])
        .addTrick(plays1, 0, 0) // Bidder wins first trick - nello fails immediately
        .build();

      const outcome = rules.checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(true);
      expect((outcome as { isDetermined: true; decidedAtTrick: number }).decidedAtTrick).toBe(1);
    });
  });
});
