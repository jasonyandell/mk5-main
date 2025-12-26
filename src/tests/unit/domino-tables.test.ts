/**
 * Verification tests for domino-tables.ts
 *
 * These tests verify that the table-driven implementation produces
 * the same results as the current rules-base.ts implementation.
 */

import { describe, test, expect } from 'vitest';
import {
  dominoToId,
  getAbsorptionId,
  getPowerId,
  EFFECTIVE_SUIT,
  SUIT_MASK,
  RANK,
  DOMINO_PIPS,
  CALLED_SUIT,
  canFollowFromTable,
  isTrumpFromTable,
} from '../../game/core/domino-tables';
import {
  getLedSuitBase,
  canFollowBase,
  isTrumpBase,
  rankInTrickBase,
} from '../../game/layers/rules-base';
import type { Domino, GameState, TrumpSelection, LedSuit, RegularSuit } from '../../game/types';
import { CALLED } from '../../game/types';

// Helper to create a minimal domino
function dom(high: number, low: number): Domino {
  return { high, low, id: `${high}-${low}` };
}

// Helper to create minimal game state with trump
function stateWithTrump(trump: TrumpSelection): GameState {
  return {
    trump,
    phase: 'playing',
    players: [],
    currentPlayer: 0,
    dealer: 0,
    bids: [],
    currentBid: { type: 'pass', player: -1 },
    winningBidder: 0,
    tricks: [],
    currentTrick: [],
    currentSuit: -1,
    teamScores: [0, 0],
    teamMarks: [0, 0],
    gameTarget: 7,
    shuffleSeed: 0,
    playerTypes: ['human', 'human', 'human', 'human'],
    actionHistory: [],
    initialConfig: {} as GameState['initialConfig'],
    theme: 'business',
    colorOverrides: {},
  };
}

describe('domino-tables', () => {
  describe('dominoToId', () => {
    test('maps all 28 dominoes to unique indices 0-27', () => {
      const seen = new Set<number>();
      for (let hi = 0; hi <= 6; hi++) {
        for (let lo = 0; lo <= hi; lo++) {
          const id = dominoToId(dom(hi, lo));
          expect(id).toBeGreaterThanOrEqual(0);
          expect(id).toBeLessThan(28);
          expect(seen.has(id)).toBe(false);
          seen.add(id);
        }
      }
      expect(seen.size).toBe(28);
    });

    test('DOMINO_PIPS is inverse of dominoToId', () => {
      for (let hi = 0; hi <= 6; hi++) {
        for (let lo = 0; lo <= hi; lo++) {
          const id = dominoToId(dom(hi, lo));
          const [pipLo, pipHi] = DOMINO_PIPS[id]!;
          expect(pipLo).toBe(lo);
          expect(pipHi).toBe(hi);
        }
      }
    });
  });

  describe('getAbsorptionId / getPowerId', () => {
    test('suit trump: absorption = power = suit', () => {
      for (let suit = 0; suit <= 6; suit++) {
        const trump: TrumpSelection = { type: 'suit', suit: suit as RegularSuit };
        expect(getAbsorptionId(trump)).toBe(suit);
        expect(getPowerId(trump)).toBe(suit);
      }
    });

    test('doubles trump: absorption = power = 7', () => {
      const trump: TrumpSelection = { type: 'doubles' };
      expect(getAbsorptionId(trump)).toBe(7);
      expect(getPowerId(trump)).toBe(7);
    });

    test('nello: absorption = 7, power = 8', () => {
      const trump: TrumpSelection = { type: 'nello' };
      expect(getAbsorptionId(trump)).toBe(7);
      expect(getPowerId(trump)).toBe(8);
    });

    test('no-trump: absorption = power = 8', () => {
      const trump: TrumpSelection = { type: 'no-trump' };
      expect(getAbsorptionId(trump)).toBe(8);
      expect(getPowerId(trump)).toBe(8);
    });
  });

  describe('EFFECTIVE_SUIT matches getLedSuitBase', () => {
    const trumpConfigs: TrumpSelection[] = [
      { type: 'suit', suit: 0 },
      { type: 'suit', suit: 3 },
      { type: 'suit', suit: 5 },
      { type: 'suit', suit: 6 },
      { type: 'doubles' },
      { type: 'no-trump' },
    ];

    test.each(trumpConfigs)('trump config: %o', (trump) => {
      const state = stateWithTrump(trump);
      const absorptionId = getAbsorptionId(trump);

      for (let hi = 0; hi <= 6; hi++) {
        for (let lo = 0; lo <= hi; lo++) {
          const domino = dom(hi, lo);
          const id = dominoToId(domino);

          const expected = getLedSuitBase(state, domino);
          const actual = EFFECTIVE_SUIT[id]![absorptionId]!;

          expect(actual).toBe(expected);
        }
      }
    });
  });

  describe('canFollowFromTable matches canFollowBase', () => {
    const trumpConfigs: TrumpSelection[] = [
      { type: 'suit', suit: 5 },
      { type: 'doubles' },
      { type: 'no-trump' },
    ];

    const ledSuits: LedSuit[] = [0, 1, 2, 3, 4, 5, 6, CALLED];

    test.each(trumpConfigs)('trump config: %o', (trump) => {
      const state = stateWithTrump(trump);
      const absorptionId = getAbsorptionId(trump);

      for (const led of ledSuits) {
        for (let hi = 0; hi <= 6; hi++) {
          for (let lo = 0; lo <= hi; lo++) {
            const domino = dom(hi, lo);
            const id = dominoToId(domino);

            const expected = canFollowBase(state, led, domino);
            const actual = canFollowFromTable(id, absorptionId, led);

            expect(actual).toBe(expected);
          }
        }
      }
    });
  });

  describe('isTrumpFromTable matches isTrumpBase', () => {
    const trumpConfigs: TrumpSelection[] = [
      { type: 'suit', suit: 0 },
      { type: 'suit', suit: 5 },
      { type: 'doubles' },
      { type: 'no-trump' },
      { type: 'not-selected' },
    ];

    test.each(trumpConfigs)('trump config: %o', (trump) => {
      const state = stateWithTrump(trump);
      const powerId = getPowerId(trump);

      for (let hi = 0; hi <= 6; hi++) {
        for (let lo = 0; lo <= hi; lo++) {
          const domino = dom(hi, lo);
          const id = dominoToId(domino);

          const expected = isTrumpBase(state, domino);
          const actual = isTrumpFromTable(id, powerId);

          expect(actual).toBe(expected);
        }
      }
    });
  });

  describe('RANK ordering matches rankInTrickBase', () => {
    // We verify relative ordering, not absolute values
    const trumpConfigs: TrumpSelection[] = [
      { type: 'suit', suit: 5 },
      { type: 'doubles' },
      { type: 'no-trump' },
    ];

    const ledSuits: LedSuit[] = [0, 3, 5, 6, CALLED];

    test.each(trumpConfigs)('trump config: %o', (trump) => {
      const state = stateWithTrump(trump);
      const powerId = getPowerId(trump);
      const absorptionId = getAbsorptionId(trump);

      for (const led of ledSuits) {
        // Collect all dominoes that can participate (follow or trump)
        const participants: { domino: Domino; id: number }[] = [];

        for (let hi = 0; hi <= 6; hi++) {
          for (let lo = 0; lo <= hi; lo++) {
            const domino = dom(hi, lo);
            const id = dominoToId(domino);

            // Check if this domino can participate
            const canFollow = canFollowFromTable(id, absorptionId, led);
            const isTrump = isTrumpFromTable(id, powerId);

            if (canFollow || isTrump) {
              participants.push({ domino, id });
            }
          }
        }

        // For each pair, verify relative ordering matches
        for (let i = 0; i < participants.length; i++) {
          for (let j = i + 1; j < participants.length; j++) {
            const a = participants[i]!;
            const b = participants[j]!;

            const rankBaseA = rankInTrickBase(state, led, a.domino);
            const rankBaseB = rankInTrickBase(state, led, b.domino);
            const rankTableA = RANK[a.id]![powerId]!;
            const rankTableB = RANK[b.id]![powerId]!;

            // Compare relative ordering
            const baseOrder = Math.sign(rankBaseA - rankBaseB);
            const tableOrder = Math.sign(rankTableA - rankTableB);

            expect(tableOrder).toBe(baseOrder);
          }
        }
      }
    });
  });

  describe('edge cases', () => {
    test('double of trump pip is highest trump', () => {
      // 5s trump: 5-5 should beat all other trump
      const powerId = 5;
      const doubleOfTrump = dominoToId(dom(5, 5));
      const highestNonDouble = dominoToId(dom(6, 5)); // 6-5 is trump

      expect(RANK[doubleOfTrump]![powerId]!).toBeGreaterThan(RANK[highestNonDouble]![powerId]!);
    });

    test('doubles beat non-doubles in same suit (following)', () => {
      // No trump: 3-3 should beat 6-3 when following threes
      const powerId = 8; // no power
      const double = dominoToId(dom(3, 3));
      const nonDouble = dominoToId(dom(6, 3));

      expect(RANK[double]![powerId]!).toBeGreaterThan(RANK[nonDouble]![powerId]!);
    });

    test('trump beats non-trump even when following suit', () => {
      // 5s trump, sixes led: 5-0 (trump) beats 6-6 (following)
      const powerId = 5;
      const trump = dominoToId(dom(5, 0));
      const follower = dominoToId(dom(6, 6));

      expect(RANK[trump]![powerId]!).toBeGreaterThan(RANK[follower]![powerId]!);
    });

    test('absorbed dominoes cannot follow non-absorbed suits', () => {
      // 5s trump: 6-5 is absorbed, cannot follow 6s
      const absorptionId = 5;
      const absorbed = dominoToId(dom(6, 5));

      expect(canFollowFromTable(absorbed, absorptionId, 6)).toBe(false);
      expect(canFollowFromTable(absorbed, absorptionId, CALLED_SUIT)).toBe(true);
    });

    test('SUIT_MASK correctly filters absorbed vs non-absorbed', () => {
      // 5s trump: suit 6 mask should NOT include 6-5 (absorbed)
      const absorptionId = 5;
      const dominoId65 = dominoToId(dom(6, 5));
      const dominoId64 = dominoToId(dom(6, 4));

      const suitMask6 = SUIT_MASK[absorptionId]![6]!;
      expect(suitMask6 & (1 << dominoId65)).toBe(0); // 6-5 not in sixes
      expect(suitMask6 & (1 << dominoId64)).not.toBe(0); // 6-4 in sixes
    });
  });
});
