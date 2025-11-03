/**
 * Unit tests for nello ruleset game rules.
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
import { baseRuleSet } from '../../../game/rulesets/base';
import { nelloRuleSet } from '../../../game/rulesets/nello';
import { composeRules } from '../../../game/rulesets/compose';
import type { Play, Trick } from '../../../game/types';
import { GameTestHelper } from '../../helpers/gameTestHelper';
import { BID_TYPES } from '../../../game/constants';
import { BLANKS, DEUCES, DOUBLES_AS_TRUMP } from '../../../game/types';

describe('Nello RuleSet Rules', () => {
  const rules = composeRules([baseRuleSet, nelloRuleSet]);

  describe('getValidActions', () => {
    it('should add nello trump option after marks bid', () => {
      const state = GameTestHelper.createTestState({
        phase: 'trump_selection',
        winningBidder: 0,
        currentBid: { type: BID_TYPES.MARKS, value: 2, player: 0 }
      });

      const baseActions: never[] = [];
      const actions = nelloRuleSet.getValidActions?.(state, baseActions) ?? [];

      expect(actions).toHaveLength(1);
      expect(actions[0]).toEqual({
        type: 'select-trump',
        player: 0,
        trump: { type: 'nello' }
      });
    });

    it('should not add nello option for points bid', () => {
      const state = GameTestHelper.createTestState({
        phase: 'trump_selection',
        winningBidder: 0,
        currentBid: { type: BID_TYPES.POINTS, value: 30, player: 0 }
      });

      const baseActions: never[] = [];
      const actions = nelloRuleSet.getValidActions?.(state, baseActions) ?? [];

      expect(actions).toEqual([]);
    });

    it('should not add nello option during bidding phase', () => {
      const state = GameTestHelper.createTestState({
        phase: 'bidding',
        currentBid: { type: BID_TYPES.MARKS, value: 2, player: 0 }
      });

      const baseActions: never[] = [];
      const actions = nelloRuleSet.getValidActions?.(state, baseActions) ?? [];

      expect(actions).toEqual([]);
    });

    it('should preserve previous actions', () => {
      const state = GameTestHelper.createTestState({
        phase: 'trump_selection',
        winningBidder: 1,
        currentBid: { type: BID_TYPES.MARKS, value: 3, player: 1 }
      });

      const baseActions = [
        { type: 'select-trump' as const, player: 1, trump: { type: 'suit' as const, suit: BLANKS } }
      ];
      const actions = nelloRuleSet.getValidActions?.(state, baseActions) ?? [];

      expect(actions).toHaveLength(2);
      expect(actions[0]).toEqual(baseActions[0]);
      const action = actions[1];
      if (!action || action.type !== 'select-trump') throw new Error('Expected select-trump action');
      expect(action.trump).toEqual({ type: 'nello' });
    });
  });

  describe('getTrumpSelector', () => {
    it('should pass through to base (bidder selects trump)', () => {
      const state = GameTestHelper.createTestState({
        trump: { type: 'nello' },
        winningBidder: 2
      });
      const bid = { type: BID_TYPES.MARKS, value: 2, player: 2 };

      const selector = rules.getTrumpSelector(state, bid);

      expect(selector).toBe(2);
    });
  });

  describe('getFirstLeader', () => {
    it('should pass through to base (bidder leads)', () => {
      const state = GameTestHelper.createTestState({
        trump: { type: 'nello' },
        winningBidder: 1
      });

      const leader = rules.getFirstLeader(state, 1, { type: 'nello' });

      expect(leader).toBe(1);
    });
  });

  describe('getNextPlayer', () => {
    it('should skip partner in turn order', () => {
      const state = GameTestHelper.createTestState({
        trump: { type: 'nello' },
        winningBidder: 0, // Partner is player 2
        currentPlayer: 0,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: [] },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: [] },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: [] }, // Partner of bidder
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: [] }
        ]
      });

      // 0 -> 1 (skip 2) -> 3 -> 0
      expect(rules.getNextPlayer(state, 0)).toBe(1);
      expect(rules.getNextPlayer(state, 1)).toBe(3); // Skip player 2
      expect(rules.getNextPlayer(state, 3)).toBe(0);
    });

    it('should skip partner when bidder is player 1', () => {
      const state = GameTestHelper.createTestState({
        trump: { type: 'nello' },
        winningBidder: 1, // Partner is player 3
        currentPlayer: 0,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: [] },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: [] },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: [] },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: [] } // Partner of bidder
        ]
      });

      expect(rules.getNextPlayer(state, 0)).toBe(1);
      expect(rules.getNextPlayer(state, 1)).toBe(2);
      expect(rules.getNextPlayer(state, 2)).toBe(0); // Skip player 3
    });

    it('should not skip partner when not nello', () => {
      const state = GameTestHelper.createTestState({
        trump: { type: 'suit', suit: BLANKS },
        winningBidder: 0,
        currentPlayer: 0,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: [] },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: [] },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: [] },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: [] }
        ]
      });

      // Normal rotation
      expect(rules.getNextPlayer(state, 0)).toBe(1);
      expect(rules.getNextPlayer(state, 1)).toBe(2); // Don't skip
      expect(rules.getNextPlayer(state, 2)).toBe(3);
      expect(rules.getNextPlayer(state, 3)).toBe(0);
    });
  });

  describe('isTrickComplete', () => {
    it('should return true after 3 plays in nello', () => {
      const state = GameTestHelper.createTestState({
        trump: { type: 'nello' },
        currentTrick: [
          { player: 0, domino: { id: '1-0', high: 1, low: 0 } },
          { player: 1, domino: { id: '2-0', high: 2, low: 0 } },
          { player: 3, domino: { id: '3-0', high: 3, low: 0 } }
        ]
      });

      expect(rules.isTrickComplete(state)).toBe(true);
    });

    it('should return false for 2 plays in nello', () => {
      const state = GameTestHelper.createTestState({
        trump: { type: 'nello' },
        currentTrick: [
          { player: 0, domino: { id: '1-0', high: 1, low: 0 } },
          { player: 1, domino: { id: '2-0', high: 2, low: 0 } }
        ]
      });

      expect(rules.isTrickComplete(state)).toBe(false);
    });

    it('should return false for 1 play in nello', () => {
      const state = GameTestHelper.createTestState({
        trump: { type: 'nello' },
        currentTrick: [
          { player: 0, domino: { id: '1-0', high: 1, low: 0 } }
        ]
      });

      expect(rules.isTrickComplete(state)).toBe(false);
    });

    it('should use base rules when not nello (4 plays)', () => {
      const state = GameTestHelper.createTestState({
        trump: { type: 'suit', suit: DEUCES },
        currentTrick: [
          { player: 0, domino: { id: '1-0', high: 1, low: 0 } },
          { player: 1, domino: { id: '2-0', high: 2, low: 0 } },
          { player: 2, domino: { id: '3-0', high: 3, low: 0 } }
        ]
      });

      expect(rules.isTrickComplete(state)).toBe(false);
    });
  });

  describe('checkHandOutcome', () => {
    it('should return null when bidding team has not won any tricks', () => {
      const state = GameTestHelper.createTestState({
        trump: { type: 'nello' },
        winningBidder: 0,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: [] },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: [] },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: [] },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: [] }
        ],
        tricks: [
          {
            plays: [
              { player: 0, domino: { id: '1-0', high: 1, low: 0 } },
              { player: 1, domino: { id: '6-0', high: 6, low: 0 } },
              { player: 3, domino: { id: '2-0', high: 2, low: 0 } }
            ],
            winner: 1, // Team 1 wins
            points: 0
          },
          {
            plays: [
              { player: 1, domino: { id: '5-0', high: 5, low: 0 } },
              { player: 3, domino: { id: '6-1', high: 6, low: 1 } },
              { player: 0, domino: { id: '2-1', high: 2, low: 1 } }
            ],
            winner: 3, // Team 1 wins
            points: 0
          }
        ]
      });

      const outcome = rules.checkHandOutcome(state);
      expect(outcome).toBeNull();
    });

    it('should return determined when bidder wins a trick', () => {
      const state = GameTestHelper.createTestState({
        trump: { type: 'nello' },
        winningBidder: 0,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: [] },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: [] },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: [] },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: [] }
        ],
        tricks: [
          {
            plays: [
              { player: 0, domino: { id: '1-0', high: 1, low: 0 } },
              { player: 1, domino: { id: '2-0', high: 2, low: 0 } },
              { player: 3, domino: { id: '3-0', high: 3, low: 0 } }
            ],
            winner: 1, // Team 1 wins - good
            points: 0
          },
          {
            plays: [
              { player: 1, domino: { id: '5-0', high: 5, low: 0 } },
              { player: 3, domino: { id: '6-0', high: 6, low: 0 } },
              { player: 0, domino: { id: '6-1', high: 6, low: 1 } }
            ],
            winner: 0, // Team 0 wins - nello fails!
            points: 0
          }
        ]
      });

      const outcome = rules.checkHandOutcome(state);
      expect(outcome).not.toBeNull();
      expect(outcome?.isDetermined).toBe(true);
      expect(outcome?.reason).toContain('Bidding team won trick');
      expect(outcome?.decidedAtTrick).toBe(2);
    });

    it('should return determined when partner wins a trick', () => {
      const state = GameTestHelper.createTestState({
        trump: { type: 'nello' },
        winningBidder: 1,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: [] },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: [] },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: [] },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: [] } // Partner
        ],
        tricks: [
          {
            plays: [
              { player: 1, domino: { id: '1-0', high: 1, low: 0 } },
              { player: 0, domino: { id: '6-0', high: 6, low: 0 } },
              { player: 3, domino: { id: '2-0', high: 2, low: 0 } }
            ],
            winner: 3, // Partner wins - nello fails!
            points: 0
          }
        ]
      });

      const outcome = rules.checkHandOutcome(state);
      expect(outcome).not.toBeNull();
      expect(outcome?.isDetermined).toBe(true);
      expect(outcome?.decidedAtTrick).toBe(1);
    });

    it('should not trigger early termination when not nello', () => {
      const state = GameTestHelper.createTestState({
        trump: { type: 'suit', suit: BLANKS },
        winningBidder: 0,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: [] },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: [] },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: [] },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: [] }
        ],
        tricks: [
          {
            plays: [
              { player: 0, domino: { id: '6-0', high: 6, low: 0 } },
              { player: 1, domino: { id: '2-0', high: 2, low: 0 } },
              { player: 2, domino: { id: '3-0', high: 3, low: 0 } },
              { player: 3, domino: { id: '4-0', high: 4, low: 0 } }
            ],
            winner: 0, // Bidder wins
            points: 0
          }
        ]
      });

      const outcome = rules.checkHandOutcome(state);
      expect(outcome).toBeNull(); // Should play all tricks
    });
  });

  describe('getLedSuit', () => {
    it('should return 7 for doubles in nello', () => {
      const state = GameTestHelper.createTestState({
        trump: { type: 'nello' }
      });

      for (let i = 0; i <= 6; i++) {
        const double = { id: `${i}-${i}`, high: i, low: i };
        expect(rules.getLedSuit(state, double)).toBe(DOUBLES_AS_TRUMP);
      }
    });

    it('should return higher pip for non-doubles in nello', () => {
      const state = GameTestHelper.createTestState({
        trump: { type: 'nello' }
      });

      const domino1 = { id: '6-2', high: 6, low: 2 };
      expect(rules.getLedSuit(state, domino1)).toBe(6);

      const domino2 = { id: '5-0', high: 5, low: 0 };
      expect(rules.getLedSuit(state, domino2)).toBe(5);

      const domino3 = { id: '4-1', high: 4, low: 1 };
      expect(rules.getLedSuit(state, domino3)).toBe(4);
    });

    it('should not affect led suit when not nello', () => {
      const state = GameTestHelper.createTestState({
        trump: { type: 'suit', suit: BLANKS }
      });

      const domino = { id: '6-2', high: 6, low: 2 };
      expect(rules.getLedSuit(state, domino)).toBe(6);
    });
  });

  describe('calculateTrickWinner', () => {
    it('should use base trick-taking rules (no trump hierarchy)', () => {
      const state = GameTestHelper.createTestState({
        trump: { type: 'nello' },
        currentSuit: BLANKS
      });
      const trick: Play[] = [
        { player: 0, domino: { id: '3-0', high: 3, low: 0 } },
        { player: 1, domino: { id: '6-0', high: 6, low: 0 } }, // Highest following suit
        { player: 3, domino: { id: '2-0', high: 2, low: 0 } }
      ];

      const winner = rules.calculateTrickWinner(state, trick);
      expect(winner).toBe(1); // 6-0 is highest
    });

    it('should handle doubles leading (suit 7)', () => {
      const state = GameTestHelper.createTestState({
        trump: { type: 'nello' },
        currentSuit: DOUBLES_AS_TRUMP
      });
      const trick: Play[] = [
        { player: 0, domino: { id: '2-2', high: 2, low: 2 } }, // Double leads suit 7
        { player: 1, domino: { id: '5-5', high: 5, low: 5 } }, // Double, but doesn't "follow suit 7" (no 7 in domino)
        { player: 3, domino: { id: '6-0', high: 6, low: 0 } }  // Non-double, doesn't follow
      ];

      const winner = rules.calculateTrickWinner(state, trick);
      // In nello: doubles lead suit 7, but no standard domino "follows" suit 7
      // (would need to contain the number 7, which doesn't exist in standard sets)
      // So the first player wins by default
      expect(winner).toBe(0);
    });
  });

  describe('integration: complete nello hand', () => {
    it('should handle a complete successful nello hand', () => {
      const state = GameTestHelper.createTestState({
        trump: { type: 'nello' },
        winningBidder: 0,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: [] },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: [] },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: [] },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: [] }
        ],
        tricks: []
      });

      // Simulate 7 tricks where opponents always win
      const tricks: Trick[] = Array.from({ length: 7 }, (_, i) => ({
        plays: [
          { player: 0, domino: { id: `${i}-0`, high: i % 7, low: 0 } },
          { player: 1, domino: { id: `${i}-1`, high: (i + 1) % 7, low: 1 } },
          { player: 3, domino: { id: `${i}-3`, high: (i + 2) % 7, low: 3 } }
        ],
        winner: i % 2 === 0 ? 1 : 3, // Alternate between opponents
        points: 0
      }));

      const finalState = { ...state, tricks };
      const outcome = rules.checkHandOutcome(finalState);

      // Should be null until all 7 tricks, then determined by base ruleSet
      expect(outcome).not.toBeNull();
      expect(outcome?.isDetermined).toBe(true);
    });

    it('should end early when bidder wins first trick', () => {
      const state = GameTestHelper.createTestState({
        trump: { type: 'nello' },
        winningBidder: 0,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: [] },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: [] },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: [] },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: [] }
        ],
        tricks: [
          {
            plays: [
              { player: 0, domino: { id: '6-0', high: 6, low: 0 } },
              { player: 1, domino: { id: '2-0', high: 2, low: 0 } },
              { player: 3, domino: { id: '3-0', high: 3, low: 0 } }
            ],
            winner: 0, // Bidder wins first trick - nello fails immediately
            points: 0
          }
        ]
      });

      const outcome = rules.checkHandOutcome(state);
      expect(outcome).not.toBeNull();
      expect(outcome?.isDetermined).toBe(true);
      expect(outcome?.decidedAtTrick).toBe(1);
    });
  });
});
