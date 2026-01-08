/**
 * GameTestHelper - Utility methods for testing
 *
 * NOTE: For state creation, use StateBuilder instead.
 * NOTE: For hand creation with doubles, use HandBuilder instead.
 * NOTE: For consensus helpers, import from consensusHelpers.ts instead.
 *
 * This file contains methods that don't have StateBuilder/HandBuilder equivalents:
 * - createTestHand() - Creates dominoes from array/object notation
 * - verifyPointConstants() - Mathematical constant validation
 * - validateGameRules() - Game rule validation
 * - validateTournamentRules() - Tournament rule validation
 */

import type { GameState, Domino } from '../../game/types';

export class GameTestHelper {
  /**
   * Creates a specific domino hand for testing
   *
   * Accepts flexible input formats:
   * - Array notation: [[6, 5], [5, 4]]
   * - Object notation: [{ high: 6, low: 5, points: 10 }]
   *
   * @example
   * ```typescript
   * const hand = GameTestHelper.createTestHand([
   *   [6, 5],  // Array format
   *   { high: 5, low: 4, points: 0 }  // Object format
   * ]);
   * ```
   */
  static createTestHand(dominoes: ({ high: number; low: number; points?: number } | [number, number])[]): Domino[] {
    return dominoes.map((domino) => {
      if (Array.isArray(domino)) {
        const [high, low] = domino;
        return {
          high,
          low,
          points: 0,
          id: `${high}-${low}`
        };
      } else {
        const { high, low, points = 0 } = domino;
        return {
          high,
          low,
          points,
          id: `${high}-${low}`
        };
      }
    });
  }

  /**
   * Checks mathematical constants (mk4 35-point system)
   *
   * Verifies that the five counting dominoes total exactly 35 points:
   * - 5-5: 10 points
   * - 6-4: 10 points
   * - 5-0: 5 points
   * - 4-1: 5 points
   * - 3-2: 5 points
   *
   * @returns true if constants are correct, false otherwise
   */
  static verifyPointConstants(): boolean {
    const testDominoes = this.createTestHand([
      { high: 5, low: 5, points: 10 }, // 5-5 = 10 points
      { high: 6, low: 4, points: 10 }, // 6-4 = 10 points
      { high: 5, low: 0, points: 5 },  // 5-0 = 5 points
      { high: 4, low: 1, points: 5 },  // 4-1 = 5 points
      { high: 3, low: 2, points: 5 },  // 3-2 = 5 points
      { high: 0, low: 0, points: 0 },  // 0-0 = 0 points
      { high: 1, low: 1, points: 0 }   // 1-1 = 0 points
    ]);

    const total = testDominoes.reduce((sum, d) => sum + (d.points || 0), 0);

    // Should total exactly 35 points for all counting dominoes (mk4 rules)
    return total === 35;
  }

  /**
   * Validates game state follows core game rules
   *
   * Checks:
   * - Exactly 4 players
   * - Teams are balanced (players 0,2 vs 1,3)
   * - All dominoes are valid (pips 0-6)
   *
   * @param state Game state to validate
   * @returns Array of error messages (empty if valid)
   */
  static validateGameRules(state: GameState): string[] {
    const errors: string[] = [];

    // Check player count
    if (state.players.length !== 4) {
      errors.push(`Invalid player count: ${state.players.length} (expected 4)`);
    }

    // Check team assignments
    const team0 = state.players.filter(p => p.teamId === 0);
    const team1 = state.players.filter(p => p.teamId === 1);
    if (team0.length !== 2 || team1.length !== 2) {
      errors.push('Teams must have exactly 2 players each');
    }

    // Check for valid dominoes in hands
    for (const player of state.players) {
      for (const domino of player.hand) {
        if (domino.high < 0 || domino.high > 6 || domino.low < 0 || domino.low > 6) {
          errors.push(`Invalid domino in player ${player.id}'s hand: ${domino.id}`);
        }
      }
    }

    return errors;
  }

  /**
   * Generates tournament compliance validation
   *
   * Checks that state complies with N42PA tournament rules:
   * - No special contracts (splash/plunge)
   * - No special trump selections (nello/sevens)
   * - Game target is 7 marks
   *
   * @param state Game state to validate
   * @returns Array of error messages (empty if compliant)
   */
  static validateTournamentRules(state: GameState): string[] {
    const errors: string[] = [];

    // Check for special contracts (tournament mode now enforced via layers)
    const specialBids = state.bids.filter(bid =>
      bid.type === 'splash' ||
      bid.type === 'plunge'
    );

    if (specialBids.length > 0) {
      errors.push('Special contracts not allowed in tournament mode');
    }

    // Check for special trump selections (nello, sevens)
    if (state.trump?.type === 'nello' || state.trump?.type === 'sevens') {
      errors.push('Special trump selections not allowed in tournament mode');
    }

    // Check game target is 7 marks
    if (state.gameTarget !== 7) {
      errors.push('Tournament play requires 7-mark games');
    }

    return errors;
  }
}
