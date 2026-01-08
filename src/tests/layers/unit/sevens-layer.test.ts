/**
 * Unit tests for sevens layer game rules.
 *
 * Sevens-specific overrides:
 * - getValidActions: adds 'sevens' trump option after marks bid
 * - calculateTrickWinner: domino closest to 7 total pips wins
 * - isValidPlay/getValidPlays: must play closest to 7
 * - checkHandOutcome: early termination if opponents win any trick
 * - calculateScore: sevens-specific scoring
 */

import { describe, it, expect } from 'vitest';
import { baseLayer } from '../../../game/layers/base';
import { sevensLayer } from '../../../game/layers/sevens';
import { composeRules } from '../../../game/layers/compose';
import { BID_TYPES } from '../../../game/constants';
import { BLANKS } from '../../../game/types';
import { StateBuilder } from '../../helpers';

describe('Sevens Layer Rules', () => {
  const rules = composeRules([baseLayer, sevensLayer]);

  describe('getValidActions', () => {
    it('should add sevens trump option after marks bid only', () => {
      // Marks bid - adds sevens option
      const marksState = StateBuilder
        .inTrumpSelection(0)
        .withWinningBid(0, { type: BID_TYPES.MARKS, value: 2, player: 0 })
        .build();
      const marksActions = sevensLayer.getValidActions?.(marksState, []) ?? [];
      expect(marksActions).toHaveLength(1);
      expect(marksActions[0]).toEqual({
        type: 'select-trump',
        player: 0,
        trump: { type: 'sevens' }
      });

      // Points bid - no sevens option
      const pointsState = StateBuilder.inTrumpSelection(0, 30).build();
      expect(sevensLayer.getValidActions?.(pointsState, []) ?? []).toEqual([]);
    });

    it('should preserve previous actions', () => {
      const state = StateBuilder
        .inTrumpSelection(1)
        .withWinningBid(1, { type: BID_TYPES.MARKS, value: 3, player: 1 })
        .build();

      const baseActions = [
        { type: 'select-trump' as const, player: 1, trump: { type: 'suit' as const, suit: BLANKS } }
      ];
      const actions = sevensLayer.getValidActions?.(state, baseActions) ?? [];

      expect(actions).toHaveLength(2);
      expect(actions[0]).toEqual(baseActions[0]);
      expect(actions[1] && 'trump' in actions[1] && actions[1].trump).toEqual({ type: 'sevens' });
    });
  });

  describe('checkHandOutcome', () => {
    it('should continue when bidding team wins, terminate when opponents win', () => {
      // Bidding team wins - continue
      const winningState = StateBuilder
        .inPlayingPhase({ type: 'sevens' })
        .withWinningBid(0, { type: BID_TYPES.MARKS, value: 2, player: 0 })
        .withHands([[], [], [], []])
        .addTrick([
          { player: 0, domino: { id: '4-3', high: 4, low: 3 } },
          { player: 1, domino: { id: '6-0', high: 6, low: 0 } },
          { player: 2, domino: { id: '2-1', high: 2, low: 1 } },
          { player: 3, domino: { id: '5-0', high: 5, low: 0 } }
        ], 0, 5)
        .build();
      expect(rules.checkHandOutcome(winningState).isDetermined).toBe(false);

      // Opponents win - terminate immediately
      const losingState = StateBuilder
        .inPlayingPhase({ type: 'sevens' })
        .withWinningBid(1, { type: BID_TYPES.MARKS, value: 2, player: 1 })
        .withHands([[], [], [], []])
        .addTrick([
          { player: 1, domino: { id: '6-0', high: 6, low: 0 } },
          { player: 2, domino: { id: '4-3', high: 4, low: 3 } },
          { player: 3, domino: { id: '5-1', high: 5, low: 1 } },
          { player: 0, domino: { id: '2-0', high: 2, low: 0 } }
        ], 2, 0)
        .build();

      const outcome = rules.checkHandOutcome(losingState);
      expect(outcome.isDetermined).toBe(true);
      expect(outcome.isDetermined && 'decidedAtTrick' in outcome && outcome.decidedAtTrick).toBe(1);
    });
  });

  describe('calculateTrickWinner', () => {
    const state = StateBuilder.inPlayingPhase({ type: 'sevens' }).build();

    it('should award domino closest to 7, breaking ties by first played', () => {
      // Exact 7 wins
      expect(rules.calculateTrickWinner(state, [
        { player: 0, domino: { id: '6-0', high: 6, low: 0 } }, // distance 1
        { player: 1, domino: { id: '4-3', high: 4, low: 3 } }, // distance 0 ✓
        { player: 2, domino: { id: '5-1', high: 5, low: 1 } }, // distance 1
        { player: 3, domino: { id: '6-6', high: 6, low: 6 } }  // distance 5
      ])).toBe(1);

      // Tie broken by first played
      expect(rules.calculateTrickWinner(state, [
        { player: 0, domino: { id: '6-0', high: 6, low: 0 } }, // distance 1 ✓
        { player: 1, domino: { id: '2-2', high: 2, low: 2 } }, // distance 3
        { player: 2, domino: { id: '5-3', high: 5, low: 3 } }, // distance 1
        { player: 3, domino: { id: '6-6', high: 6, low: 6 } }  // distance 5
      ])).toBe(0);

      // Extreme distances (12 closer to 7 than 0)
      expect(rules.calculateTrickWinner(state, [
        { player: 0, domino: { id: '0-0', high: 0, low: 0 } }, // distance 7
        { player: 1, domino: { id: '6-6', high: 6, low: 6 } }  // distance 5 ✓
      ])).toBe(1);
    });

    it('should not apply to non-sevens trump', () => {
      const nonSevensState = StateBuilder
        .inPlayingPhase({ type: 'suit', suit: BLANKS })
        .with({ currentSuit: BLANKS })
        .build();

      expect(rules.calculateTrickWinner(nonSevensState, [
        { player: 0, domino: { id: '6-0', high: 6, low: 0 } }, // Highest ✓
        { player: 1, domino: { id: '4-3', high: 4, low: 3 } }, // Would win in sevens
        { player: 2, domino: { id: '2-0', high: 2, low: 0 } },
        { player: 3, domino: { id: '3-0', high: 3, low: 0 } }
      ])).toBe(0);
    });
  });

  describe('isValidPlay and getValidPlays', () => {
    it('should constrain plays to dominoes closest to 7', () => {
      // Distance 0 (exact 7) only
      const exactState = StateBuilder
        .inPlayingPhase({ type: 'sevens' })
        .withPlayerHand(0, [
          { id: '6-1', high: 6, low: 1 }, // distance 0 ✓
          { id: '5-2', high: 5, low: 2 }, // distance 0 ✓
          { id: '4-0', high: 4, low: 0 }, // distance 3 ✗
        ])
        .withPlayerHand(1, []).withPlayerHand(2, []).withPlayerHand(3, [])
        .build();
      expect(rules.getValidPlays(exactState, 0).map(d => d.id).sort()).toEqual(['5-2', '6-1']);
      expect(rules.isValidPlay(exactState, { id: '6-1', high: 6, low: 1 }, 0)).toBe(true);
      expect(rules.isValidPlay(exactState, { id: '4-0', high: 4, low: 0 }, 0)).toBe(false);

      // Multiple dominoes tied for closest distance
      const tiedState = StateBuilder
        .inPlayingPhase({ type: 'sevens' })
        .withPlayerHand(0, [
          { id: '5-1', high: 5, low: 1 }, // distance 1 ✓
          { id: '5-3', high: 5, low: 3 }, // distance 1 ✓
          { id: '4-0', high: 4, low: 0 }, // distance 3 ✗
        ])
        .withPlayerHand(1, []).withPlayerHand(2, []).withPlayerHand(3, [])
        .build();
      expect(rules.getValidPlays(tiedState, 0).map(d => d.id).sort()).toEqual(['5-1', '5-3']);
    });
  });
});
