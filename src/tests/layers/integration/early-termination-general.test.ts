import { describe, it, expect } from 'vitest';
import { composeRules, baseRuleSet, nelloRuleSet, plungeRuleSet, splashRuleSet, sevensRuleSet } from '../../../game/layers';
import { StateBuilder } from '../../helpers';
import { BLANKS, ACES, DEUCES, TRES, FOURS, FIVES, SIXES } from '../../../game/types';

/**
 * General Early Termination Tests - Cross-contract concerns
 *
 * These tests verify that early termination works correctly across all contract types
 * for general concerns like:
 * - Remaining dominoes stay in hands after early termination
 * - Phase transitions work correctly
 * - Winner/loser determination is accurate
 */
describe('Early Termination - General Cross-Contract Tests', () => {
  const nelloRules = composeRules([baseRuleSet, nelloRuleSet]);
  const plungeRules = composeRules([baseRuleSet, plungeRuleSet]);
  const splashRules = composeRules([baseRuleSet, splashRuleSet]);
  const sevensRules = composeRules([baseRuleSet, sevensRuleSet]);

  describe('Remaining Dominoes Stay in Hands', () => {
    it('should keep unplayed dominoes in player hands after nello early termination', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'nello' })
        .withWinningBid(0, { type: 'marks', value: 2, player: 0 })
        .withCurrentPlayer(0)
        .withTricks([
          {
            plays: [
              { player: 0, domino: { id: 't1-0', high: SIXES, low: BLANKS } },
              { player: 1, domino: { id: 't1-1', high: ACES, low: BLANKS } },
              { player: 3, domino: { id: 't1-3', high: DEUCES, low: BLANKS } }
            ],
            winner: 0, // Bidder won - nello failed on trick 1
            points: 0,
            ledSuit: SIXES
          }
        ])
        .withPlayerHand(0, [
          // Bidder still has 6 dominoes
          { id: 'h0-1', high: ACES, low: BLANKS },
          { id: 'h0-2', high: DEUCES, low: BLANKS },
          { id: 'h0-3', high: TRES, low: BLANKS },
          { id: 'h0-4', high: FOURS, low: BLANKS },
          { id: 'h0-5', high: FIVES, low: BLANKS },
          { id: 'h0-6', high: BLANKS, low: BLANKS }
        ])
        .withPlayerHand(1, [
          // Player 1 has 6 dominoes
          { id: 'h1-1', high: DEUCES, low: ACES },
          { id: 'h1-2', high: TRES, low: ACES },
          { id: 'h1-3', high: FOURS, low: ACES },
          { id: 'h1-4', high: FIVES, low: ACES },
          { id: 'h1-5', high: SIXES, low: ACES },
          { id: 'h1-6', high: ACES, low: ACES }
        ])
        .withPlayerHand(2, []) // Partner (sits out in nello)
        .withPlayerHand(3, [
          // Player 3 has 6 dominoes
          { id: 'h3-1', high: TRES, low: DEUCES },
          { id: 'h3-2', high: FOURS, low: DEUCES },
          { id: 'h3-3', high: FIVES, low: DEUCES },
          { id: 'h3-4', high: SIXES, low: DEUCES },
          { id: 'h3-5', high: DEUCES, low: DEUCES },
          { id: 'h3-6', high: TRES, low: TRES }
        ])
        .build();

      // Verify early termination
      const outcome = nelloRules.checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(true);
      expect((outcome as { isDetermined: true; decidedAtTrick: number }).decidedAtTrick).toBe(1);

      // Verify players still have dominoes
      expect(state.players[0]!.hand.length).toBe(6);
      expect(state.players[1]!.hand.length).toBe(6);
      expect(state.players[2]!.hand.length).toBe(0); // Partner
      expect(state.players[3]!.hand.length).toBe(6);

      // Total dominoes: 1 trick (3 plays) + 18 in hands = 21 dominoes accounted for
      const totalPlayed = state.tricks.reduce((sum, trick) => sum + trick.plays.length, 0);
      const totalInHands = state.players.reduce((sum, p) => sum + p.hand.length, 0);
      expect(totalPlayed + totalInHands).toBe(21); // 3 played + 18 in hands
    });

    it('should not play remaining tricks after plunge fails', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'suit', suit: ACES })
        .withWinningBid(0, { type: 'plunge', value: 4, player: 0 })
        .withCurrentPlayer(2)
        .withTricks([
          { plays: [], winner: 0, points: 0, ledSuit: ACES },
          {
            plays: [
              { player: 0, domino: { id: 't2-0', high: DEUCES, low: BLANKS } },
              { player: 1, domino: { id: 't2-1', high: DEUCES, low: DEUCES } },
              { player: 2, domino: { id: 't2-2', high: TRES, low: BLANKS } },
              { player: 3, domino: { id: 't2-3', high: SIXES, low: BLANKS } }
            ],
            winner: 3, // Opponent won - plunge failed
            points: 0,
            ledSuit: DEUCES
          }
        ])
        .withPlayerHand(0, [{ id: 'r0', high: FOURS, low: BLANKS }])
        .withPlayerHand(1, [{ id: 'r1', high: FIVES, low: BLANKS }])
        .withPlayerHand(2, [{ id: 'r2', high: SIXES, low: FIVES }])
        .withPlayerHand(3, [{ id: 'r3', high: ACES, low: ACES }])
        .build();

      const outcome = plungeRules.checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(true);
      expect((outcome as { isDetermined: true; decidedAtTrick: number }).decidedAtTrick).toBe(2);

      // Each player still has 1 domino unplayed
      state.players.forEach(player => {
        expect(player.hand.length).toBe(1);
      });

      // Only 2 tricks played (8 dominoes), 4 remain in hands
      expect(state.tricks.length).toBe(2);
      const totalInHands = state.players.reduce((sum, p) => sum + p.hand.length, 0);
      expect(totalInHands).toBe(4);
    });
  });

  describe('Phase Transitions on Early Termination', () => {
    it('should indicate hand is complete when nello fails early', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'nello' })
        .withWinningBid(0, { type: 'marks', value: 2, player: 0 })
        .withCurrentPlayer(0)
        .withTricks([
          {
            plays: [
              { player: 0, domino: { id: 't1-0', high: SIXES, low: BLANKS } },
              { player: 1, domino: { id: 't1-1', high: ACES, low: BLANKS } },
              { player: 3, domino: { id: 't1-3', high: DEUCES, low: BLANKS } }
            ],
            winner: 0, // Bidder won
            points: 0,
            ledSuit: SIXES
          }
        ])
        .build();

      const outcome = nelloRules.checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(true);

      // Phase should still be 'playing' - scoring logic will transition
      expect(state.phase).toBe('playing');

      // But checkHandOutcome indicates it's ready to score
      expect((outcome as { isDetermined: true; decidedAtTrick: number }).decidedAtTrick).toBe(1);
    });

    it('should indicate hand is complete when splash fails on trick 2', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'doubles' })
        .withWinningBid(1, { type: 'splash', value: 2, player: 1 })
        .withCurrentPlayer(3)
        .withTricks([
          { plays: [], winner: 1, points: 10, ledSuit: ACES },
          {
            plays: [
              { player: 1, domino: { id: 't2-1', high: DEUCES, low: BLANKS } },
              { player: 2, domino: { id: 't2-2', high: TRES, low: BLANKS } },
              { player: 3, domino: { id: 't2-3', high: FOURS, low: FOURS } }, // Partner plays trump
              { player: 0, domino: { id: 't2-0', high: SIXES, low: SIXES } }  // Opponent plays higher trump
            ],
            winner: 0, // Opponent won
            points: 10,
            ledSuit: DEUCES
          }
        ])
        .build();

      const outcome = splashRules.checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(true);
      expect((outcome as { isDetermined: true; decidedAtTrick: number }).decidedAtTrick).toBe(2);
      expect(state.phase).toBe('playing'); // Still in playing phase
    });
  });

  describe('Correct Winner/Loser Determination', () => {
    it('should identify bidding team lost when nello fails', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'nello' })
        .withWinningBid(0, { type: 'marks', value: 2, player: 0 }) // Team 0
        .withCurrentPlayer(0)
        .withTricks([
          {
            plays: [
              { player: 0, domino: { id: 't1-0', high: SIXES, low: BLANKS } },
              { player: 1, domino: { id: 't1-1', high: ACES, low: BLANKS } },
              { player: 3, domino: { id: 't1-3', high: DEUCES, low: BLANKS } }
            ],
            winner: 0, // Bidder (team 0) won - nello failed
            points: 0,
            ledSuit: SIXES
          }
        ])
        .build();

      const outcome = nelloRules.checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(true);
      expect((outcome as { isDetermined: true; reason: string }).reason).toContain('Bidding team won trick'); // Team 0 failed
    });

    it('should identify bidding team lost when plunge fails', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'suit', suit: ACES })
        .withWinningBid(2, { type: 'plunge', value: 4, player: 2 }) // Team 0
        .withCurrentPlayer(0)
        .withTricks([
          { plays: [], winner: 2, points: 5, ledSuit: ACES },
          {
            plays: [
              { player: 2, domino: { id: 't2-2', high: DEUCES, low: BLANKS } },
              { player: 3, domino: { id: 't2-3', high: SIXES, low: BLANKS } }, // Opponent wins
              { player: 0, domino: { id: 't2-0', high: TRES, low: BLANKS } },
              { player: 1, domino: { id: 't2-1', high: FOURS, low: BLANKS } }
            ],
            winner: 3, // Opponent (team 1) won
            points: 0,
            ledSuit: DEUCES
          }
        ])
        .build();

      const outcome = plungeRules.checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(true);
      expect((outcome as { isDetermined: true; reason: string }).reason).toContain('Defending team won trick'); // Opponents won
    });

    it('should identify bidding team lost when sevens fails', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'sevens' })
        .withWinningBid(1, { type: 'marks', value: 3, player: 1 }) // Team 1 (players 1 and 3)
        .withCurrentPlayer(1)
        .withTricks([
          {
            plays: [
              { player: 1, domino: { id: 't1-1', high: ACES, low: BLANKS } }, // 1 pip (distance 6)
              { player: 2, domino: { id: 't1-2', high: TRES, low: FOURS } }, // 7 pips (distance 0) - WINS
              { player: 3, domino: { id: 't1-3', high: DEUCES, low: BLANKS } }, // 2 pips (distance 5)
              { player: 0, domino: { id: 't1-0', high: FIVES, low: BLANKS } } // 5 pips (distance 2)
            ],
            winner: 2, // Player 2 (team 0) won - opponent!
            points: 5,
            ledSuit: ACES
          }
        ])
        .build();

      const outcome = sevensRules.checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(true);
      expect((outcome as { isDetermined: true; decidedAtTrick: number }).decidedAtTrick).toBe(1); // Lost on trick 1
      expect((outcome as { isDetermined: true; reason: string }).reason).toContain('Defending team won trick');
    });
  });
});
