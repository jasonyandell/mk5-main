/**
 * Unit tests for plunge ruleset game rules.
 *
 * Plunge rules (from docs/rules.md §8.A):
 * - Requires 4+ doubles in hand
 * - Bid value: Automatic based on highest marks bid + 1, minimum 4
 * - Partner selects trump and leads
 * - Must win all 7 tricks (early termination if opponents win any trick)
 */

import { describe, it, expect } from 'vitest';
import { baseRuleSet } from '../../../game/rulesets/base';
import { plungeRuleSet } from '../../../game/rulesets/plunge';
import { composeRules } from '../../../game/rulesets/compose';
import type { Trick, Bid } from '../../../game/types';
import { GameTestHelper, createHandWithDoubles } from '../../helpers/gameTestHelper';
import { BID_TYPES } from '../../../game/constants';
import { BLANKS, ACES } from '../../../game/types';

describe('Plunge RuleSet Rules', () => {
  const rules = composeRules([baseRuleSet, plungeRuleSet]);

  describe('getValidActions', () => {
    it('should add plunge bid when player has 4+ doubles', () => {
      const hand = createHandWithDoubles(4);
      const state = GameTestHelper.createTestState({
        phase: 'bidding',
        currentPlayer: 1,
        bids: [],
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: [] },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand }
        ]
      });

      const baseActions = [
        { type: 'pass' as const, player: 1 }
      ];
      const actions = plungeRuleSet.getValidActions?.(state, baseActions) ?? [];

      expect(actions).toHaveLength(2);
      expect(actions[1]).toEqual({
        type: 'bid',
        player: 1,
        bid: 'plunge',
        value: 4 // Minimum plunge value
      });
    });

    it('should not add plunge bid when player has 3 doubles', () => {
      const hand = createHandWithDoubles(3);
      const state = GameTestHelper.createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand }
        ]
      });

      const baseActions: never[] = [];
      const actions = plungeRuleSet.getValidActions?.(state, baseActions) ?? [];

      expect(actions).toEqual([]);
    });

    it('should calculate plunge value as highest marks bid + 1', () => {
      const hand = createHandWithDoubles(5);
      const state = GameTestHelper.createTestState({
        phase: 'bidding',
        currentPlayer: 2,
        bids: [
          { type: BID_TYPES.MARKS, value: 2, player: 0 },
          { type: BID_TYPES.MARKS, value: 3, player: 1 }
        ],
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: [] },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: [] },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand }
        ]
      });

      const actions = plungeRuleSet.getValidActions?.(state, []) ?? [];

      expect(actions[0] && actions[0].type === 'bid' ? actions[0].value : undefined).toBe(4); // 3 + 1 = 4
    });

    it('should use minimum value of 4 when no marks bids exist', () => {
      const hand = createHandWithDoubles(4);
      const state = GameTestHelper.createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        bids: [
          { type: BID_TYPES.POINTS, value: 30, player: 1 }
        ],
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand }
        ]
      });

      const actions = plungeRuleSet.getValidActions?.(state, []) ?? [];

      expect(actions[0] && actions[0].type === 'bid' ? actions[0].value : undefined).toBe(4);
    });

    it('should handle plunge bid value jumping over existing marks bid', () => {
      const hand = createHandWithDoubles(6);
      const state = GameTestHelper.createTestState({
        phase: 'bidding',
        currentPlayer: 3,
        bids: [
          { type: BID_TYPES.MARKS, value: 5, player: 1 }
        ],
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: [] },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: [] },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: [] },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand }
        ]
      });

      const actions = plungeRuleSet.getValidActions?.(state, []) ?? [];

      expect(actions[0] && actions[0].type === 'bid' ? actions[0].value : undefined).toBe(6); // 5 + 1 = 6
    });

    it('should not add plunge bid when not in bidding phase', () => {
      const hand = createHandWithDoubles(5);
      const state = GameTestHelper.createTestState({
        phase: 'trump_selection',
        currentPlayer: 0,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand }
        ]
      });

      const actions = plungeRuleSet.getValidActions?.(state, []) ?? [];

      expect(actions).toEqual([]);
    });

    it('should handle exactly 4 doubles', () => {
      const hand = createHandWithDoubles(4);
      const state = GameTestHelper.createTestState({
        phase: 'bidding',
        currentPlayer: 2,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: [] },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: [] },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand }
        ]
      });

      const actions = plungeRuleSet.getValidActions?.(state, []) ?? [];

      expect(actions).toHaveLength(1);
      expect(actions[0] && actions[0].type === 'bid' ? actions[0].bid : undefined).toBe('plunge');
    });

    it('should handle all 7 doubles', () => {
      const hand = createHandWithDoubles(7);
      const state = GameTestHelper.createTestState({
        phase: 'bidding',
        currentPlayer: 1,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: [] },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand }
        ]
      });

      const actions = plungeRuleSet.getValidActions?.(state, []) ?? [];

      expect(actions).toHaveLength(1);
      expect(actions[0] && actions[0].type === 'bid' ? actions[0].bid : undefined).toBe('plunge');
    });
  });

  describe('getTrumpSelector', () => {
    it('should return partner for plunge bid', () => {
      const state = GameTestHelper.createTestState({
        currentBid: { type: 'plunge', value: 4, player: 0 }
      });
      const bid: Bid = { type: 'plunge', value: 4, player: 0 };

      const selector = rules.getTrumpSelector(state, bid);

      expect(selector).toBe(2); // Partner of player 0 is player 2
    });

    it('should return partner for plunge bid from player 1', () => {
      const bid: Bid = { type: 'plunge', value: 5, player: 1 };
      const state = GameTestHelper.createTestState();

      expect(rules.getTrumpSelector(state, bid)).toBe(3); // Partner of player 1 is player 3
    });

    it('should return partner for plunge bid from player 2', () => {
      const bid: Bid = { type: 'plunge', value: 4, player: 2 };
      const state = GameTestHelper.createTestState();

      expect(rules.getTrumpSelector(state, bid)).toBe(0); // Partner of player 2 is player 0
    });

    it('should return partner for plunge bid from player 3', () => {
      const bid: Bid = { type: 'plunge', value: 6, player: 3 };
      const state = GameTestHelper.createTestState();

      expect(rules.getTrumpSelector(state, bid)).toBe(1); // Partner of player 3 is player 1
    });

    it('should pass through to base for non-plunge bids', () => {
      const bid = { type: BID_TYPES.MARKS, value: 2, player: 1 };
      const state = GameTestHelper.createTestState();

      expect(rules.getTrumpSelector(state, bid)).toBe(1); // Bidder selects trump
    });
  });

  describe('getFirstLeader', () => {
    it('should pass through to base (partner leads since they selected trump)', () => {
      const state = GameTestHelper.createTestState({
        currentBid: { type: 'plunge', value: 4, player: 0 },
        winningBidder: 0
      });

      // Partner (player 2) is trump selector
      const leader = rules.getFirstLeader(state, 2, { type: 'suit', suit: ACES });

      expect(leader).toBe(2); // Partner leads
    });
  });

  describe('getNextPlayer', () => {
    it('should use standard rotation (no skipping)', () => {
      const state = GameTestHelper.createTestState({
        currentBid: { type: 'plunge', value: 4, player: 0 }
      });

      expect(rules.getNextPlayer(state, 0)).toBe(1);
      expect(rules.getNextPlayer(state, 1)).toBe(2);
      expect(rules.getNextPlayer(state, 2)).toBe(3);
      expect(rules.getNextPlayer(state, 3)).toBe(0);
    });
  });

  describe('isTrickComplete', () => {
    it('should use base rule (4 plays)', () => {
      const state = GameTestHelper.createTestState({
        currentBid: { type: 'plunge', value: 4, player: 0 },
        currentTrick: [
          { player: 0, domino: { id: '1-0', high: 1, low: 0 } },
          { player: 1, domino: { id: '2-0', high: 2, low: 0 } },
          { player: 2, domino: { id: '3-0', high: 3, low: 0 } },
          { player: 3, domino: { id: '4-0', high: 4, low: 0 } }
        ]
      });

      expect(rules.isTrickComplete(state)).toBe(true);
    });

    it('should return false for 3 plays', () => {
      const state = GameTestHelper.createTestState({
        currentBid: { type: 'plunge', value: 4, player: 0 },
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
    it('should return null when bidding team wins all tricks so far', () => {
      const state = GameTestHelper.createTestState({
        currentBid: { type: 'plunge', value: 4, player: 0 },
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
              { player: 2, domino: { id: '3-0', high: 3, low: 0 } },
              { player: 3, domino: { id: '4-0', high: 4, low: 0 } }
            ],
            winner: 0, // Team 0 wins
            points: 10
          },
          {
            plays: [
              { player: 0, domino: { id: '5-0', high: 5, low: 0 } },
              { player: 1, domino: { id: '6-0', high: 6, low: 0 } },
              { player: 2, domino: { id: '1-1', high: 1, low: 1 } },
              { player: 3, domino: { id: '2-1', high: 2, low: 1 } }
            ],
            winner: 2, // Team 0 wins (partner)
            points: 0
          }
        ]
      });

      const outcome = rules.checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(false);
    });

    it('should return determined when opponents win any trick', () => {
      const state = GameTestHelper.createTestState({
        currentBid: { type: 'plunge', value: 4, player: 0 },
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
              { player: 2, domino: { id: '3-0', high: 3, low: 0 } },
              { player: 3, domino: { id: '4-0', high: 4, low: 0 } }
            ],
            winner: 0, // Team 0 wins
            points: 10
          },
          {
            plays: [
              { player: 0, domino: { id: '5-0', high: 5, low: 0 } },
              { player: 1, domino: { id: '6-0', high: 6, low: 0 } },
              { player: 2, domino: { id: '1-1', high: 1, low: 1 } },
              { player: 3, domino: { id: '2-1', high: 2, low: 1 } }
            ],
            winner: 1, // Team 1 wins - plunge fails!
            points: 5
          }
        ]
      });

      const outcome = rules.checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(true);
      expect((outcome as { isDetermined: true; reason: string }).reason).toContain('Defending team won trick');
      expect((outcome as { isDetermined: true; decidedAtTrick: number }).decidedAtTrick).toBe(2);
    });

    it('should end on first trick if opponents win', () => {
      const state = GameTestHelper.createTestState({
        currentBid: { type: 'plunge', value: 4, player: 1 },
        winningBidder: 1,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: [] },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: [] },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: [] },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: [] }
        ],
        tricks: [
          {
            plays: [
              { player: 2, domino: { id: '6-0', high: 6, low: 0 } },
              { player: 3, domino: { id: '1-0', high: 1, low: 0 } },
              { player: 0, domino: { id: '2-0', high: 2, low: 0 } },
              { player: 1, domino: { id: '3-0', high: 3, low: 0 } }
            ],
            winner: 2, // Team 0 wins first trick - plunge fails immediately
            points: 0
          }
        ]
      });

      const outcome = rules.checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(true);
      expect((outcome as { isDetermined: true; decidedAtTrick: number }).decidedAtTrick).toBe(1);
    });

    it('should not trigger early termination for non-plunge bids', () => {
      const state = GameTestHelper.createTestState({
        currentBid: { type: BID_TYPES.MARKS, value: 2, player: 0 },
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
              { player: 2, domino: { id: '3-0', high: 3, low: 0 } },
              { player: 3, domino: { id: '4-0', high: 4, low: 0 } }
            ],
            winner: 1, // Opponents win
            points: 0
          }
        ]
      });

      const outcome = rules.checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(false); // Should play all tricks for marks bid
    });

    it('should respect already determined outcome from previous ruleSet', () => {
      const state = GameTestHelper.createTestState({
        currentBid: { type: 'plunge', value: 4, player: 0 },
        winningBidder: 0,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: [] },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: [] },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: [] },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: [] }
        ],
        tricks: Array.from({ length: 7 }, () => ({
          plays: [
            { player: 0, domino: { id: '1-0', high: 1, low: 0 } },
            { player: 1, domino: { id: '2-0', high: 2, low: 0 } },
            { player: 2, domino: { id: '3-0', high: 3, low: 0 } },
            { player: 3, domino: { id: '4-0', high: 4, low: 0 } }
          ],
          winner: 0,
          points: 0
        }))
      });

      const outcome = rules.checkHandOutcome(state);
      // Base ruleset says "all tricks played"
      expect(outcome.isDetermined).toBe(true);
    });
  });

  describe('getLedSuit', () => {
    it('should use base rules (no override)', () => {
      const state = GameTestHelper.createTestState({
        currentBid: { type: 'plunge', value: 4, player: 0 },
        trump: { type: 'suit', suit: BLANKS }
      });

      const domino = { id: '6-2', high: 6, low: 2 };
      expect(rules.getLedSuit(state, domino)).toBe(6);
    });
  });

  describe('calculateTrickWinner', () => {
    it('should use base rules (standard trick-taking)', () => {
      const state = GameTestHelper.createTestState({
        currentBid: { type: 'plunge', value: 4, player: 0 },
        trump: { type: 'suit', suit: ACES },
        currentSuit: BLANKS
      });
      const trick = [
        { player: 0, domino: { id: '3-0', high: 3, low: 0 } },
        { player: 1, domino: { id: '6-0', high: 6, low: 0 } },
        { player: 2, domino: { id: '2-0', high: 2, low: 0 } },
        { player: 3, domino: { id: '4-0', high: 4, low: 0 } }
      ];

      const winner = rules.calculateTrickWinner(state, trick);
      expect(winner).toBe(1); // 6-0 is highest
    });
  });

  describe('integration: complete plunge hand', () => {
    it('should succeed when bidding team wins all 7 tricks', () => {
      const state = GameTestHelper.createTestState({
        currentBid: { type: 'plunge', value: 4, player: 0 },
        winningBidder: 0,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: [] },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: [] },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: [] },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: [] }
        ],
        tricks: []
      });

      // Simulate 7 tricks, alternating winners between bidder and partner
      const tricks: Trick[] = Array.from({ length: 7 }, (_, i) => ({
        plays: [
          { player: 0, domino: { id: `${i}-0`, high: i % 7, low: 0 } },
          { player: 1, domino: { id: `${i}-1`, high: (i + 1) % 7, low: 1 } },
          { player: 2, domino: { id: `${i}-2`, high: (i + 2) % 7, low: 2 } },
          { player: 3, domino: { id: `${i}-3`, high: (i + 3) % 7, low: 3 } }
        ],
        winner: i % 2 === 0 ? 0 : 2, // Alternate between team 0 members
        points: 5
      }));

      const finalState = { ...state, tricks };
      const outcome = rules.checkHandOutcome(finalState);

      // Should be determined by base ruleset after all 7 tricks
      expect(outcome.isDetermined).toBe(true);
    });

    it('should fail immediately on first lost trick', () => {
      const state = GameTestHelper.createTestState({
        currentBid: { type: 'plunge', value: 5, player: 3 },
        winningBidder: 3,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: [] },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: [] },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: [] },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: [] }
        ],
        tricks: [
          {
            plays: [
              { player: 1, domino: { id: '1-0', high: 1, low: 0 } },
              { player: 2, domino: { id: '6-0', high: 6, low: 0 } }, // Opponent wins
              { player: 3, domino: { id: '2-0', high: 2, low: 0 } },
              { player: 0, domino: { id: '3-0', high: 3, low: 0 } }
            ],
            winner: 2, // Team 0 wins
            points: 0
          }
        ]
      });

      const outcome = rules.checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(true);
      expect((outcome as { isDetermined: true; decidedAtTrick: number }).decidedAtTrick).toBe(1);
      expect((outcome as { isDetermined: true; reason: string }).reason).toContain('Defending team won trick 1');
    });
  });
});
