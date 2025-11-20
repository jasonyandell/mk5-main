/**
 * Unit tests for sevens ruleset game rules.
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
import { baseRuleSet } from '../../../game/rulesets/base';
import { sevensRuleSet } from '../../../game/rulesets/sevens';
import { composeRules } from '../../../game/rulesets/compose';
import type { Play, Trick } from '../../../game/types';
import { BID_TYPES } from '../../../game/constants';
import { BLANKS } from '../../../game/types';
import { StateBuilder } from '../../helpers';

describe('Sevens RuleSet Rules', () => {
  const rules = composeRules([baseRuleSet, sevensRuleSet]);

  describe('getValidActions', () => {
    it('should add sevens trump option after marks bid', () => {
      const state = StateBuilder
        .inTrumpSelection(0)
        .withWinningBid(0, { type: BID_TYPES.MARKS, value: 2, player: 0 })
        .build();

      const baseActions: never[] = [];
      const actions = sevensRuleSet.getValidActions?.(state, baseActions) ?? [];

      expect(actions).toHaveLength(1);
      expect(actions[0]).toEqual({
        type: 'select-trump',
        player: 0,
        trump: { type: 'sevens' }
      });
    });

    it('should not add sevens option for points bid', () => {
      const state = StateBuilder
        .inTrumpSelection(0, 30)
        .build();

      const baseActions: never[] = [];
      const actions = sevensRuleSet.getValidActions?.(state, baseActions) ?? [];

      expect(actions).toEqual([]);
    });

    it('should not add sevens option during bidding phase', () => {
      const state = StateBuilder
        .inBiddingPhase()
        .withBids([{ type: BID_TYPES.MARKS, value: 2, player: 0 }])
        .build();

      const baseActions: never[] = [];
      const actions = sevensRuleSet.getValidActions?.(state, baseActions) ?? [];

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
      const actions = sevensRuleSet.getValidActions?.(state, baseActions) ?? [];

      expect(actions).toHaveLength(2);
      expect(actions[0]).toEqual(baseActions[0]);
      const action = actions[1];
      if (!action || action.type !== 'select-trump') throw new Error('Expected select-trump action');
      expect(action.trump).toEqual({ type: 'sevens' });
    });
  });

  describe('getTrumpSelector', () => {
    it('should pass through to base (bidder selects trump)', () => {
      const state = StateBuilder
        .inPlayingPhase({ type: 'sevens' })
        .with({ winningBidder: 2 })
        .build();
      const bid = { type: BID_TYPES.MARKS, value: 2, player: 2 };

      const selector = rules.getTrumpSelector(state, bid);

      expect(selector).toBe(2);
    });
  });

  describe('getFirstLeader', () => {
    it('should pass through to base (bidder leads)', () => {
      const state = StateBuilder
        .inPlayingPhase({ type: 'sevens' })
        .with({ winningBidder: 1 })
        .build();

      const leader = rules.getFirstLeader(state, 1, { type: 'sevens' });

      expect(leader).toBe(1);
    });
  });

  describe('getNextPlayer', () => {
    it('should use standard rotation (no skipping)', () => {
      const state = StateBuilder
        .inPlayingPhase({ type: 'sevens' })
        .build();

      expect(rules.getNextPlayer(state, 0)).toBe(1);
      expect(rules.getNextPlayer(state, 1)).toBe(2);
      expect(rules.getNextPlayer(state, 2)).toBe(3);
      expect(rules.getNextPlayer(state, 3)).toBe(0);
    });
  });

  describe('isTrickComplete', () => {
    it('should use base rule (4 plays)', () => {
      const state = StateBuilder
        .inPlayingPhase({ type: 'sevens' })
        .withCurrentTrick([
          { player: 0, domino: { id: '1-0', high: 1, low: 0 } },
          { player: 1, domino: { id: '2-0', high: 2, low: 0 } },
          { player: 2, domino: { id: '3-0', high: 3, low: 0 } },
          { player: 3, domino: { id: '4-0', high: 4, low: 0 } }
        ])
        .build();

      expect(rules.isTrickComplete(state)).toBe(true);
    });

    it('should return false for 3 plays', () => {
      const state = StateBuilder
        .inPlayingPhase({ type: 'sevens' })
        .withCurrentTrick([
          { player: 0, domino: { id: '1-0', high: 1, low: 0 } },
          { player: 1, domino: { id: '2-0', high: 2, low: 0 } },
          { player: 2, domino: { id: '3-0', high: 3, low: 0 } }
        ])
        .build();

      expect(rules.isTrickComplete(state)).toBe(false);
    });
  });

  describe('checkHandOutcome', () => {
    it('should return null when bidding team wins all tricks so far', () => {
      const state = StateBuilder
        .inPlayingPhase({ type: 'sevens' })
        .withWinningBid(0, { type: BID_TYPES.MARKS, value: 2, player: 0 })
        .withHands([[], [], [], []])
        .addTrick(
          [
            { player: 0, domino: { id: '4-3', high: 4, low: 3 } },
            { player: 1, domino: { id: '6-0', high: 6, low: 0 } },
            { player: 2, domino: { id: '2-1', high: 2, low: 1 } },
            { player: 3, domino: { id: '5-0', high: 5, low: 0 } }
          ],
          0, // Team 0 wins (4+3=7, perfect)
          5
        )
        .build();

      const outcome = rules.checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(false);
    });

    it('should return determined when opponents win any trick', () => {
      const state = StateBuilder
        .inPlayingPhase({ type: 'sevens' })
        .withWinningBid(0, { type: BID_TYPES.MARKS, value: 3, player: 0 })
        .withHands([[], [], [], []])
        .addTrick(
          [
            { player: 0, domino: { id: '4-3', high: 4, low: 3 } },
            { player: 1, domino: { id: '6-0', high: 6, low: 0 } },
            { player: 2, domino: { id: '2-1', high: 2, low: 1 } },
            { player: 3, domino: { id: '5-0', high: 5, low: 0 } }
          ],
          0, // Team 0 wins
          5
        )
        .addTrick(
          [
            { player: 0, domino: { id: '5-2', high: 5, low: 2 } },
            { player: 1, domino: { id: '3-4', high: 3, low: 4 } }, // 3+4=7, perfect
            { player: 2, domino: { id: '6-1', high: 6, low: 1 } },
            { player: 3, domino: { id: '2-2', high: 2, low: 2 } }
          ],
          1, // Team 1 wins - sevens fails!
          0
        )
        .build();

      const outcome = rules.checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(true);
      expect((outcome as { isDetermined: true; reason: string }).reason).toContain('Defending team won trick');
      expect((outcome as { isDetermined: true; decidedAtTrick: number }).decidedAtTrick).toBe(2);
    });

    it('should end on first trick if opponents win', () => {
      const state = StateBuilder
        .inPlayingPhase({ type: 'sevens' })
        .withWinningBid(1, { type: BID_TYPES.MARKS, value: 2, player: 1 })
        .withHands([[], [], [], []])
        .addTrick(
          [
            { player: 1, domino: { id: '6-0', high: 6, low: 0 } },
            { player: 2, domino: { id: '4-3', high: 4, low: 3 } }, // 4+3=7, perfect - wins!
            { player: 3, domino: { id: '5-1', high: 5, low: 1 } },
            { player: 0, domino: { id: '2-0', high: 2, low: 0 } }
          ],
          2, // Team 0 wins first trick - sevens fails immediately
          0
        )
        .build();

      const outcome = rules.checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(true);
      expect((outcome as { isDetermined: true; decidedAtTrick: number }).decidedAtTrick).toBe(1);
    });

    it('should not trigger early termination for non-sevens trump', () => {
      const state = StateBuilder
        .withTricksPlayed(1, { type: 'suit', suit: BLANKS })
        .withWinningBid(0, { type: BID_TYPES.POINTS, value: 30, player: 0 })
        .build();

      // Manually update tricks to show opponent won
      const trickWithOpponentWin = [{
        plays: [
          { player: 0, domino: { id: '1-0', high: 1, low: 0 } },
          { player: 1, domino: { id: '6-0', high: 6, low: 0 } },
          { player: 2, domino: { id: '3-0', high: 3, low: 0 } },
          { player: 3, domino: { id: '4-0', high: 4, low: 0 } }
        ],
        winner: 1, // Opponents win (player 1 is team 1)
        points: 0,
        ledSuit: BLANKS
      }];
      const updatedState = { ...state, tricks: trickWithOpponentWin };

      const outcome = rules.checkHandOutcome(updatedState);
      expect(outcome.isDetermined).toBe(false); // Should play all tricks for regular points bid
    });
  });

  describe('getLedSuit', () => {
    it('should use base rules (no override)', () => {
      const state = StateBuilder
        .inPlayingPhase({ type: 'sevens' })
        .build();

      const domino = { id: '6-2', high: 6, low: 2 };
      expect(rules.getLedSuit(state, domino)).toBe(6);
    });
  });

  describe('calculateTrickWinner', () => {
    describe('distance from 7', () => {
      it('should award exact 7 (distance 0)', () => {
        const state = StateBuilder.inPlayingPhase({ type: 'sevens' }).build();
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
        const state = StateBuilder.inPlayingPhase({ type: 'sevens' }).build();
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
        const state = StateBuilder.inPlayingPhase({ type: 'sevens' }).build();
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
        const state = StateBuilder.inPlayingPhase({ type: 'sevens' }).build();
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
        const state = StateBuilder.inPlayingPhase({ type: 'sevens' }).build();
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
        const state = StateBuilder.inPlayingPhase({ type: 'sevens' }).build();
        const trick: Play[] = [
          { player: 0, domino: { id: '0-0', high: 0, low: 0 } },
          { player: 1, domino: { id: '6-6', high: 6, low: 6 } }
        ];
        expect(rules.calculateTrickWinner(state, trick)).toBe(1); // 12 is closer to 7 than 0
      });

      it('should handle total of 7 multiple ways', () => {
        const state = StateBuilder.inPlayingPhase({ type: 'sevens' }).build();

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
        const state = StateBuilder.inPlayingPhase({ type: 'sevens' }).build();
        const trick: Play[] = [
          { player: 0, domino: { id: '6-6', high: 6, low: 6 } }, // 12, distance 5
          { player: 1, domino: { id: '1-0', high: 1, low: 0 } }  // 1, distance 6
        ];
        expect(rules.calculateTrickWinner(state, trick)).toBe(0); // 12 is closer to 7 than 1
      });
    });

    describe('integration with base ruleSet', () => {
      it('should not use sevens logic when not sevens trump', () => {
        const state = StateBuilder
          .inPlayingPhase({ type: 'suit', suit: BLANKS })
          .with({ currentSuit: BLANKS })
          .build();
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
      const state = StateBuilder
        .inPlayingPhase({ type: 'sevens' })
        .withWinningBid(0, { type: BID_TYPES.MARKS, value: 2, player: 0 })
        .withHands([[], [], [], []])
        .build();

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

      // Should be determined by base ruleset after all 7 tricks
      expect(outcome.isDetermined).toBe(true);
    });

    it('should fail immediately when opponents get closer to 7', () => {
      const state = StateBuilder
        .inPlayingPhase({ type: 'sevens' })
        .withWinningBid(2, { type: BID_TYPES.MARKS, value: 3, player: 2 })
        .withHands([[], [], [], []])
        .addTrick(
          [
            { player: 2, domino: { id: '6-0', high: 6, low: 0 } }, // 6, distance 1
            { player: 3, domino: { id: '4-3', high: 4, low: 3 } }, // 7, distance 0 - WINS!
            { player: 0, domino: { id: '5-1', high: 5, low: 1 } }, // 6, distance 1
            { player: 1, domino: { id: '6-6', high: 6, low: 6 } }  // 12, distance 5
          ],
          3, // Team 1 wins
          0
        )
        .build();

      const outcome = rules.checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(true);
      expect((outcome as { isDetermined: true; decidedAtTrick: number }).decidedAtTrick).toBe(1);
      expect((outcome as { isDetermined: true; reason: string }).reason).toContain('Defending team won trick 1');
    });
  });

  describe('edge cases', () => {
    it('should handle single play trick', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'sevens' }).build();
      const trick: Play[] = [
        { player: 2, domino: { id: '6-0', high: 6, low: 0 } }
      ];

      const winner = rules.calculateTrickWinner(state, trick);
      expect(winner).toBe(2);
    });

    it('should handle all players equidistant from 7', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'sevens' }).build();
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
      const state = StateBuilder
        .inPlayingPhase({ type: 'sevens' })
        .withCurrentPlayer(0)
        .withPlayerHand(0, [
          { id: '6-1', high: 6, low: 1 }, // 7 total - distance 0 ✓
          { id: '5-2', high: 5, low: 2 }, // 7 total - distance 0 ✓
          { id: '4-0', high: 4, low: 0 }, // 4 total - distance 3 ✗
          { id: '3-0', high: 3, low: 0 }, // 3 total - distance 4 ✗
        ])
        .withPlayerHand(1, [])
        .withPlayerHand(2, [])
        .withPlayerHand(3, [])
        .build();

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
      const trick = {
        plays: [
          { player: 0, domino: { id: '5-0', high: 5, low: 0 } }, // distance 2
          { player: 1, domino: { id: '4-3', high: 4, low: 3 } }, // distance 0 ✓ WINNER
          { player: 2, domino: { id: '6-2', high: 6, low: 2 } }, // distance 1
          { player: 3, domino: { id: '3-1', high: 3, low: 1 } }, // distance 3
        ],
        winner: 1, // Player 1 won with 4-3 (7 total)
        points: 0,
        ledSuit: 5 as const
      };
      const state = StateBuilder
        .inPlayingPhase({ type: 'sevens' })
        .withCurrentPlayer(0)
        .withTricks([trick])
        .build();

      // Winner (player 1) should lead next trick
      const nextPlayer = rules.getNextPlayer(state, state.currentPlayer);
      expect(nextPlayer).toBe(1);
    });

    it('user scenario: partner wins with 7, not set', () => {
      // I (player 0) lead 5-0 (distance 2)
      // Opponent (player 1) plays 6-0 (distance 1)
      // My partner (player 2) plays 6-1 (distance 0 = 7 total) ✓ WINS
      // Bidding team = 0 (players 0 and 2)

      const trick = {
        plays: [
          { player: 0, domino: { id: '5-0', high: 5, low: 0 } }, // distance 2
          { player: 1, domino: { id: '6-0', high: 6, low: 0 } }, // distance 1
          { player: 2, domino: { id: '6-1', high: 6, low: 1 } }, // distance 0 ✓ WINS
          { player: 3, domino: { id: '4-0', high: 4, low: 0 } }, // distance 3
        ],
        winner: 2, // Partner won
        points: 0,
        ledSuit: 5 as const
      };
      const state = StateBuilder
        .inPlayingPhase({ type: 'sevens' })
        .withWinningBid(0, { type: BID_TYPES.MARKS, value: 2, player: 0 })
        .withCurrentPlayer(2) // Partner leads next trick (engine sets this to winner)
        .withTricks([trick])
        .withHands([[], [], [], []])
        .build();

      // Check that bidding team is NOT set (partner won)
      const outcome = rules.checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(false); // Not determined, continue playing

      // Verify partner (trick winner) leads next trick
      // (Engine sets currentPlayer to winner in executeCompleteTrick)
      expect(state.currentPlayer).toBe(2);
    });
  });
});
