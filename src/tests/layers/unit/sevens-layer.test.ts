/**
 * Unit tests for sevens layer game rules.
 *
 * Sevens rules:
 * - Only available when marks bid won (trump selection option, not bid type)
 * - Domino closest to 7 total pips wins trick
 * - Ties won by first played
 * - No trump suit (no follow-suit requirement)
 * - Must win all tricks (early termination if opponents win any trick)
 * - Bidder selects trump and leads normally
 */

import { describe, it, expect } from 'vitest';
import { baseLayer } from '../../../game/layers/base';
import { sevensLayer } from '../../../game/layers/sevens';
import { composeRules } from '../../../game/layers/compose';
import type { Play, Trick } from '../../../game/types';
import { GameTestHelper } from '../../helpers/gameTestHelper';
import { BID_TYPES } from '../../../game/constants';
import { BLANKS } from '../../../game/types';

describe('Sevens Layer Rules', () => {
  const rules = composeRules([baseLayer, sevensLayer]);

  describe('getValidActions', () => {
    it('should add sevens trump option after marks bid', () => {
      const state = GameTestHelper.createTestState({
        phase: 'trump_selection',
        winningBidder: 0,
        currentBid: { type: BID_TYPES.MARKS, value: 2, player: 0 }
      });

      const baseActions: never[] = [];
      const actions = sevensLayer.getValidActions?.(state, baseActions) ?? [];

      expect(actions).toHaveLength(1);
      expect(actions[0]).toEqual({
        type: 'select-trump',
        player: 0,
        trump: { type: 'sevens' }
      });
    });

    it('should not add sevens option for points bid', () => {
      const state = GameTestHelper.createTestState({
        phase: 'trump_selection',
        winningBidder: 0,
        currentBid: { type: BID_TYPES.POINTS, value: 30, player: 0 }
      });

      const baseActions: never[] = [];
      const actions = sevensLayer.getValidActions?.(state, baseActions) ?? [];

      expect(actions).toEqual([]);
    });

    it('should not add sevens option during bidding phase', () => {
      const state = GameTestHelper.createTestState({
        phase: 'bidding',
        currentBid: { type: BID_TYPES.MARKS, value: 2, player: 0 }
      });

      const baseActions: never[] = [];
      const actions = sevensLayer.getValidActions?.(state, baseActions) ?? [];

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
      const actions = sevensLayer.getValidActions?.(state, baseActions) ?? [];

      expect(actions).toHaveLength(2);
      expect(actions[0]).toEqual(baseActions[0]);
      const action = actions[1];
      if (!action || action.type !== 'select-trump') throw new Error('Expected select-trump action');
      expect(action.trump).toEqual({ type: 'sevens' });
    });
  });

  describe('getTrumpSelector', () => {
    it('should pass through to base (bidder selects trump)', () => {
      const state = GameTestHelper.createTestState({
        trump: { type: 'sevens' },
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
        trump: { type: 'sevens' },
        winningBidder: 1
      });

      const leader = rules.getFirstLeader(state, 1, { type: 'sevens' });

      expect(leader).toBe(1);
    });
  });

  describe('getNextPlayer', () => {
    it('should use standard rotation (no skipping)', () => {
      const state = GameTestHelper.createTestState({
        trump: { type: 'sevens' }
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
        trump: { type: 'sevens' },
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
        trump: { type: 'sevens' },
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
        trump: { type: 'sevens' },
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
              { player: 0, domino: { id: '4-3', high: 4, low: 3 } },
              { player: 1, domino: { id: '6-0', high: 6, low: 0 } },
              { player: 2, domino: { id: '2-1', high: 2, low: 1 } },
              { player: 3, domino: { id: '5-0', high: 5, low: 0 } }
            ],
            winner: 0, // Team 0 wins (4+3=7, perfect)
            points: 5
          }
        ]
      });

      const outcome = rules.checkHandOutcome(state);
      expect(outcome).toBeNull();
    });

    it('should return determined when opponents win any trick', () => {
      const state = GameTestHelper.createTestState({
        trump: { type: 'sevens' },
        currentBid: { type: BID_TYPES.MARKS, value: 3, player: 0 },
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
              { player: 0, domino: { id: '4-3', high: 4, low: 3 } },
              { player: 1, domino: { id: '6-0', high: 6, low: 0 } },
              { player: 2, domino: { id: '2-1', high: 2, low: 1 } },
              { player: 3, domino: { id: '5-0', high: 5, low: 0 } }
            ],
            winner: 0, // Team 0 wins
            points: 5
          },
          {
            plays: [
              { player: 0, domino: { id: '5-2', high: 5, low: 2 } },
              { player: 1, domino: { id: '3-4', high: 3, low: 4 } }, // 3+4=7, perfect
              { player: 2, domino: { id: '6-1', high: 6, low: 1 } },
              { player: 3, domino: { id: '2-2', high: 2, low: 2 } }
            ],
            winner: 1, // Team 1 wins - sevens fails!
            points: 0
          }
        ]
      });

      const outcome = rules.checkHandOutcome(state);
      expect(outcome).not.toBeNull();
      expect(outcome?.isDetermined).toBe(true);
      expect(outcome?.reason).toContain('Defending team won trick');
      expect(outcome?.decidedAtTrick).toBe(2);
    });

    it('should end on first trick if opponents win', () => {
      const state = GameTestHelper.createTestState({
        trump: { type: 'sevens' },
        currentBid: { type: BID_TYPES.MARKS, value: 2, player: 1 },
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
              { player: 1, domino: { id: '6-0', high: 6, low: 0 } },
              { player: 2, domino: { id: '4-3', high: 4, low: 3 } }, // 4+3=7, perfect - wins!
              { player: 3, domino: { id: '5-1', high: 5, low: 1 } },
              { player: 0, domino: { id: '2-0', high: 2, low: 0 } }
            ],
            winner: 2, // Team 0 wins first trick - sevens fails immediately
            points: 0
          }
        ]
      });

      const outcome = rules.checkHandOutcome(state);
      expect(outcome).not.toBeNull();
      expect(outcome?.decidedAtTrick).toBe(1);
    });

    it('should not trigger early termination for non-sevens trump', () => {
      const state = GameTestHelper.createTestState({
        trump: { type: 'suit', suit: BLANKS },
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
      expect(outcome).toBeNull(); // Should play all tricks for regular marks bid
    });
  });

  describe('getLedSuit', () => {
    it('should use base rules (no override)', () => {
      const state = GameTestHelper.createTestState({
        trump: { type: 'sevens' }
      });

      const domino = { id: '6-2', high: 6, low: 2 };
      expect(rules.getLedSuit(state, domino)).toBe(6);
    });
  });

  describe('calculateTrickWinner', () => {
    describe('distance from 7', () => {
      it('should award exact 7 (distance 0)', () => {
        const state = GameTestHelper.createTestState({
          trump: { type: 'sevens' }
        });
        const trick: Play[] = [
          { player: 0, domino: { id: '6-0', high: 6, low: 0 } }, // 6+0=6, distance 1
          { player: 1, domino: { id: '4-3', high: 4, low: 3 } }, // 4+3=7, distance 0 - WINS!
          { player: 2, domino: { id: '5-1', high: 5, low: 1 } }, // 5+1=6, distance 1
          { player: 3, domino: { id: '6-6', high: 6, low: 6 } }  // 6+6=12, distance 5
        ];

        const winner = rules.calculateTrickWinner(state, trick);
        expect(winner).toBe(1);
      });

      it('should award closer distance when no exact 7', () => {
        const state = GameTestHelper.createTestState({
          trump: { type: 'sevens' }
        });
        const trick: Play[] = [
          { player: 0, domino: { id: '6-0', high: 6, low: 0 } }, // 6, distance 1
          { player: 1, domino: { id: '2-2', high: 2, low: 2 } }, // 4, distance 3
          { player: 2, domino: { id: '5-3', high: 5, low: 3 } }, // 8, distance 1
          { player: 3, domino: { id: '6-6', high: 6, low: 6 } }  // 12, distance 5
        ];

        // Both 0 and 2 have distance 1, but player 0 played first
        const winner = rules.calculateTrickWinner(state, trick);
        expect(winner).toBe(0);
      });

      it('should handle ties by awarding first player', () => {
        const state = GameTestHelper.createTestState({
          trump: { type: 'sevens' }
        });
        const trick: Play[] = [
          { player: 0, domino: { id: '5-1', high: 5, low: 1 } }, // 6, distance 1
          { player: 1, domino: { id: '5-2', high: 5, low: 2 } }, // 7, distance 0
          { player: 2, domino: { id: '4-3', high: 4, low: 3 } }, // 7, distance 0 (tie!)
          { player: 3, domino: { id: '3-4', high: 3, low: 4 } }  // 7, distance 0 (tie!)
        ];

        // Player 1 played first 7
        const winner = rules.calculateTrickWinner(state, trick);
        expect(winner).toBe(1);
      });

      it('should handle low totals correctly', () => {
        const state = GameTestHelper.createTestState({
          trump: { type: 'sevens' }
        });
        const trick: Play[] = [
          { player: 0, domino: { id: '0-0', high: 0, low: 0 } }, // 0, distance 7
          { player: 1, domino: { id: '1-0', high: 1, low: 0 } }, // 1, distance 6
          { player: 2, domino: { id: '2-0', high: 2, low: 0 } }, // 2, distance 5
          { player: 3, domino: { id: '3-0', high: 3, low: 0 } }  // 3, distance 4 - WINS!
        ];

        const winner = rules.calculateTrickWinner(state, trick);
        expect(winner).toBe(3);
      });

      it('should handle high totals correctly', () => {
        const state = GameTestHelper.createTestState({
          trump: { type: 'sevens' }
        });
        const trick: Play[] = [
          { player: 0, domino: { id: '6-6', high: 6, low: 6 } }, // 12, distance 5
          { player: 1, domino: { id: '6-5', high: 6, low: 5 } }, // 11, distance 4
          { player: 2, domino: { id: '5-5', high: 5, low: 5 } }, // 10, distance 3 - WINS!
          { player: 3, domino: { id: '6-4', high: 6, low: 4 } }  // 10, distance 3 (tie, but player 2 first)
        ];

        const winner = rules.calculateTrickWinner(state, trick);
        expect(winner).toBe(2);
      });
    });

    describe('all possible totals', () => {
      it('should handle total of 0 (distance 7)', () => {
        const state = GameTestHelper.createTestState({ trump: { type: 'sevens' } });
        const trick: Play[] = [
          { player: 0, domino: { id: '0-0', high: 0, low: 0 } },
          { player: 1, domino: { id: '6-6', high: 6, low: 6 } }
        ];
        expect(rules.calculateTrickWinner(state, trick)).toBe(1); // 12 is closer to 7 than 0
      });

      it('should handle total of 7 multiple ways', () => {
        const state = GameTestHelper.createTestState({ trump: { type: 'sevens' } });

        // Different dominoes that total 7
        const combinations = [
          { player: 0, domino: { id: '4-3', high: 4, low: 3 } },
          { player: 1, domino: { id: '5-2', high: 5, low: 2 } },
          { player: 2, domino: { id: '6-1', high: 6, low: 1 } },
          { player: 3, domino: { id: '0-7', high: 0, low: 7 } } // Hypothetical
        ];

        const winner = rules.calculateTrickWinner(state, combinations.slice(0, 3));
        expect(winner).toBe(0); // First to play 7
      });

      it('should handle total of 12 (distance 5)', () => {
        const state = GameTestHelper.createTestState({ trump: { type: 'sevens' } });
        const trick: Play[] = [
          { player: 0, domino: { id: '6-6', high: 6, low: 6 } }, // 12, distance 5
          { player: 1, domino: { id: '1-0', high: 1, low: 0 } }  // 1, distance 6
        ];
        expect(rules.calculateTrickWinner(state, trick)).toBe(0); // 12 is closer to 7 than 1
      });
    });

    describe('integration with base layer', () => {
      it('should not use sevens logic when not sevens trump', () => {
        const state = GameTestHelper.createTestState({
          trump: { type: 'suit', suit: BLANKS },
          currentSuit: BLANKS
        });
        const trick: Play[] = [
          { player: 0, domino: { id: '6-0', high: 6, low: 0 } }, // Highest
          { player: 1, domino: { id: '4-3', high: 4, low: 3 } }, // Would win in sevens (7 total)
          { player: 2, domino: { id: '2-0', high: 2, low: 0 } },
          { player: 3, domino: { id: '3-0', high: 3, low: 0 } }
        ];

        const winner = rules.calculateTrickWinner(state, trick);
        expect(winner).toBe(0); // Base rules: 6-0 is highest
      });
    });
  });

  describe('integration: complete sevens hand', () => {
    it('should succeed when bidding team wins all 7 tricks', () => {
      const state = GameTestHelper.createTestState({
        trump: { type: 'sevens' },
        currentBid: { type: BID_TYPES.MARKS, value: 2, player: 0 },
        winningBidder: 0,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: [] },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: [] },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: [] },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: [] }
        ],
        tricks: []
      });

      // Simulate 7 tricks where team 0 always plays closest to 7
      const tricks: Trick[] = Array.from({ length: 7 }, (_, i) => ({
        plays: [
          { player: 0, domino: { id: `${i}-0`, high: 4, low: 3 } }, // 7 - perfect
          { player: 1, domino: { id: `${i}-1`, high: 6, low: 6 } }, // 12 - far
          { player: 2, domino: { id: `${i}-2`, high: 5, low: 1 } }, // 6 - close
          { player: 3, domino: { id: `${i}-3`, high: 0, low: 0 } }  // 0 - far
        ],
        winner: 0, // Player 0 always has 7
        points: 0
      }));

      const finalState = { ...state, tricks };
      const outcome = rules.checkHandOutcome(finalState);

      // Should be determined by base layer after all 7 tricks
      expect(outcome).not.toBeNull();
      expect(outcome?.isDetermined).toBe(true);
    });

    it('should fail immediately when opponents get closer to 7', () => {
      const state = GameTestHelper.createTestState({
        trump: { type: 'sevens' },
        currentBid: { type: BID_TYPES.MARKS, value: 3, player: 2 },
        winningBidder: 2,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: [] },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: [] },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: [] },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: [] }
        ],
        tricks: [
          {
            plays: [
              { player: 2, domino: { id: '6-0', high: 6, low: 0 } }, // 6, distance 1
              { player: 3, domino: { id: '4-3', high: 4, low: 3 } }, // 7, distance 0 - WINS!
              { player: 0, domino: { id: '5-1', high: 5, low: 1 } }, // 6, distance 1
              { player: 1, domino: { id: '6-6', high: 6, low: 6 } }  // 12, distance 5
            ],
            winner: 3, // Team 1 wins
            points: 0
          }
        ]
      });

      const outcome = rules.checkHandOutcome(state);
      expect(outcome).not.toBeNull();
      expect(outcome?.decidedAtTrick).toBe(1);
      expect(outcome?.reason).toContain('Defending team won trick 1');
    });
  });

  describe('edge cases', () => {
    it('should handle single play trick', () => {
      const state = GameTestHelper.createTestState({
        trump: { type: 'sevens' }
      });
      const trick: Play[] = [
        { player: 2, domino: { id: '6-0', high: 6, low: 0 } }
      ];

      const winner = rules.calculateTrickWinner(state, trick);
      expect(winner).toBe(2);
    });

    it('should handle all players equidistant from 7', () => {
      const state = GameTestHelper.createTestState({
        trump: { type: 'sevens' }
      });
      const trick: Play[] = [
        { player: 0, domino: { id: '5-1', high: 5, low: 1 } }, // 6, distance 1
        { player: 1, domino: { id: '6-0', high: 6, low: 0 } }, // 6, distance 1
        { player: 2, domino: { id: '4-2', high: 4, low: 2 } }, // 6, distance 1
        { player: 3, domino: { id: '3-3', high: 3, low: 3 } }  // 6, distance 1
      ];

      // All tied, first player wins
      const winner = rules.calculateTrickWinner(state, trick);
      expect(winner).toBe(0);
    });
  });

  describe('getValidPlays', () => {
    it('enforces must play closest to 7 rule', () => {
      const state = GameTestHelper.createTestState({
        phase: 'playing',
        trump: { type: 'sevens' },
        currentPlayer: 0,
        players: [
          {
            id: 0,
            name: 'P0',
            teamId: 0,
            marks: 0,
            hand: [
              { id: '6-1', high: 6, low: 1 }, // 7 total - distance 0 ✓
              { id: '5-2', high: 5, low: 2 }, // 7 total - distance 0 ✓
              { id: '4-0', high: 4, low: 0 }, // 4 total - distance 3 ✗
              { id: '3-0', high: 3, low: 0 }, // 3 total - distance 4 ✗
            ]
          },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: [] },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: [] },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: [] },
        ]
      });

      const validPlays = rules.getValidPlays(state, 0);

      // Should only return dominoes with distance 0 (6-1 and 5-2)
      expect(validPlays.length).toBe(2);
      expect(validPlays.map(d => d.id).sort()).toEqual(['5-2', '6-1']);

      // Verify isValidPlay matches
      expect(rules.isValidPlay(state, { id: '6-1', high: 6, low: 1 }, 0)).toBe(true);
      expect(rules.isValidPlay(state, { id: '5-2', high: 5, low: 2 }, 0)).toBe(true);
      expect(rules.isValidPlay(state, { id: '4-0', high: 4, low: 0 }, 0)).toBe(false);
      expect(rules.isValidPlay(state, { id: '3-0', high: 3, low: 0 }, 0)).toBe(false);
    });

    it('winner of trick leads next trick', () => {
      const state = GameTestHelper.createTestState({
        phase: 'playing',
        trump: { type: 'sevens' },
        currentPlayer: 0,
        currentTrick: [],
        tricks: [
          {
            plays: [
              { player: 0, domino: { id: '5-0', high: 5, low: 0 } }, // distance 2
              { player: 1, domino: { id: '4-3', high: 4, low: 3 } }, // distance 0 ✓ WINNER
              { player: 2, domino: { id: '6-2', high: 6, low: 2 } }, // distance 1
              { player: 3, domino: { id: '3-1', high: 3, low: 1 } }, // distance 3
            ],
            winner: 1, // Player 1 won with 4-3 (7 total)
            points: 0,
            ledSuit: 5
          }
        ]
      });

      // Winner (player 1) should lead next trick
      const nextPlayer = rules.getNextPlayer(state, state.currentPlayer);
      expect(nextPlayer).toBe(1);
    });

    it('user scenario: partner wins with 7, not set', () => {
      // I (player 0) lead 5-0 (distance 2)
      // Opponent (player 1) plays 6-0 (distance 1)
      // My partner (player 2) plays 6-1 (distance 0 = 7 total) ✓ WINS
      // Bidding team = 0 (players 0 and 2)

      const state = GameTestHelper.createTestState({
        phase: 'playing',
        trump: { type: 'sevens' },
        winningBidder: 0,
        currentBid: { type: BID_TYPES.MARKS, value: 2, player: 0 },
        tricks: [
          {
            plays: [
              { player: 0, domino: { id: '5-0', high: 5, low: 0 } }, // distance 2
              { player: 1, domino: { id: '6-0', high: 6, low: 0 } }, // distance 1
              { player: 2, domino: { id: '6-1', high: 6, low: 1 } }, // distance 0 ✓ WINS
              { player: 3, domino: { id: '4-0', high: 4, low: 0 } }, // distance 3
            ],
            winner: 2, // Partner won
            points: 0,
            ledSuit: 5
          }
        ],
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: [] },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: [] },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: [] }, // Partner
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: [] },
        ]
      });

      // Check that bidding team is NOT set (partner won)
      const outcome = rules.checkHandOutcome(state);
      expect(outcome).toBeNull(); // Not determined, continue playing

      // Partner (player 2) should lead next trick
      expect(rules.getNextPlayer(state, 0)).toBe(2);
    });
  });
});
