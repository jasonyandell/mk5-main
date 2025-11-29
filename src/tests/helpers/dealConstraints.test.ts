/**
 * Tests for dealConstraints.ts - Pure constraint satisfaction for domino deals
 */

import { describe, it, expect } from 'vitest';
import {
  generateDealFromConstraints,
  dealWithDoubles,
  dealWithExactHand,
  dealWithVoid,
  parseDominoId,
  type DealResult,
  type ConstraintDomino
} from './dealConstraints';
import { StateBuilder } from './stateBuilder';

// ============================================================================
// Test Utilities
// ============================================================================

function isDouble(d: ConstraintDomino): boolean {
  return d.high === d.low;
}

function hasSuit(d: ConstraintDomino, suit: number): boolean {
  return d.high === suit || d.low === suit;
}

function countDoubles(hand: ConstraintDomino[]): number {
  return hand.filter(isDouble).length;
}

function getHandPoints(hand: ConstraintDomino[]): number {
  return hand.reduce((sum, d) => {
    if (d.high === 5 && d.low === 5) return sum + 10;
    if (d.high === 6 && d.low === 4) return sum + 10;
    if (d.high + d.low === 5) return sum + 5;
    return sum;
  }, 0);
}

function getAllDominoIds(result: DealResult): Set<string> {
  return new Set(result.flatMap(hand => hand.map(d => d.id)));
}

// ============================================================================
// Basic Functionality Tests
// ============================================================================

describe('generateDealFromConstraints', () => {
  describe('basic functionality', () => {
    it('generates a valid 28-domino deal with no constraints', () => {
      const result = generateDealFromConstraints({});

      // 4 players with 7 dominoes each
      expect(result).toHaveLength(4);
      for (const hand of result) {
        expect(hand).toHaveLength(7);
      }

      // All 28 unique dominoes
      const allIds = getAllDominoIds(result);
      expect(allIds.size).toBe(28);
    });

    it('is deterministic with same fillSeed', () => {
      const result1 = generateDealFromConstraints({ fillSeed: 12345 });
      const result2 = generateDealFromConstraints({ fillSeed: 12345 });

      for (let p = 0; p < 4; p++) {
        expect(result1[p]!.map(d => d.id)).toEqual(result2[p]!.map(d => d.id));
      }
    });

    it('produces different results with different fillSeeds', () => {
      const result1 = generateDealFromConstraints({ fillSeed: 1 });
      const result2 = generateDealFromConstraints({ fillSeed: 2 });

      // At least one hand should be different
      const hands1 = result1.map(h => h.map(d => d.id).sort().join(','));
      const hands2 = result2.map(h => h.map(d => d.id).sort().join(','));

      expect(hands1).not.toEqual(hands2);
    });
  });

  // ============================================================================
  // exactDominoes Constraint Tests
  // ============================================================================

  describe('exactDominoes constraint', () => {
    it('assigns exact dominoes to specified player', () => {
      const exact = ['6-6', '5-5', '4-4'];
      const result = generateDealFromConstraints({
        players: { 0: { exactDominoes: exact } },
        fillSeed: 42
      });

      const handIds = new Set(result[0]!.map(d => d.id));
      for (const id of exact) {
        expect(handIds.has(id)).toBe(true);
      }
    });

    it('assigns exact dominoes to multiple players', () => {
      const result = generateDealFromConstraints({
        players: {
          0: { exactDominoes: ['6-6', '5-5'] },
          1: { exactDominoes: ['4-4', '3-3'] },
          2: { exactDominoes: ['2-2'] }
        },
        fillSeed: 42
      });

      expect(result[0]!.some(d => d.id === '6-6')).toBe(true);
      expect(result[0]!.some(d => d.id === '5-5')).toBe(true);
      expect(result[1]!.some(d => d.id === '4-4')).toBe(true);
      expect(result[1]!.some(d => d.id === '3-3')).toBe(true);
      expect(result[2]!.some(d => d.id === '2-2')).toBe(true);
    });

    it('accepts non-normalized domino IDs', () => {
      // "5-6" should be normalized to "6-5"
      const result = generateDealFromConstraints({
        players: { 0: { exactDominoes: ['5-6', '4-6'] } },
        fillSeed: 42
      });

      const handIds = new Set(result[0]!.map(d => d.id));
      expect(handIds.has('6-5')).toBe(true);
      expect(handIds.has('6-4')).toBe(true);
    });

    it('throws on duplicate exactDominoes across players', () => {
      expect(() => generateDealFromConstraints({
        players: {
          0: { exactDominoes: ['6-6'] },
          1: { exactDominoes: ['6-6'] }
        }
      })).toThrow(/assigned to multiple players/);
    });

    it('throws on invalid domino ID', () => {
      expect(() => generateDealFromConstraints({
        players: { 0: { exactDominoes: ['7-7'] } }
      })).toThrow(/Invalid domino ID/);

      expect(() => generateDealFromConstraints({
        players: { 0: { exactDominoes: ['abc'] } }
      })).toThrow(/Invalid domino ID/);
    });

    it('throws on too many exactDominoes', () => {
      expect(() => generateDealFromConstraints({
        players: {
          0: { exactDominoes: ['0-0', '1-1', '2-2', '3-3', '4-4', '5-5', '6-6', '6-5'] }
        }
      })).toThrow(/Cannot have 8 exact dominoes/);
    });
  });

  // ============================================================================
  // minDoubles/maxDoubles Constraint Tests
  // ============================================================================

  describe('minDoubles constraint', () => {
    it('ensures player has minimum doubles', () => {
      const result = generateDealFromConstraints({
        players: { 0: { minDoubles: 4 } },
        fillSeed: 42
      });

      const doubles = countDoubles(result[0]!);
      expect(doubles).toBeGreaterThanOrEqual(4);
    });

    it('works for all 7 doubles (extreme case)', () => {
      const result = generateDealFromConstraints({
        players: { 0: { minDoubles: 7 } },
        fillSeed: 42
      });

      const doubles = countDoubles(result[0]!);
      expect(doubles).toBe(7);
    });

    it('combines with exactDominoes', () => {
      const result = generateDealFromConstraints({
        players: {
          0: {
            exactDominoes: ['6-6'],
            minDoubles: 3
          }
        },
        fillSeed: 42
      });

      const hand = result[0]!;
      expect(hand.some(d => d.id === '6-6')).toBe(true);
      expect(countDoubles(hand)).toBeGreaterThanOrEqual(3);
    });

    it('throws when not enough doubles available', () => {
      // Player 0 wants 5 doubles, but player 1 already claimed 3
      expect(() => generateDealFromConstraints({
        players: {
          0: { exactDominoes: ['0-0', '1-1', '2-2'] },
          1: { minDoubles: 5 }
        }
      })).toThrow(/Cannot satisfy minDoubles/);
    });
  });

  describe('maxDoubles constraint', () => {
    it('limits number of doubles in hand', () => {
      const result = generateDealFromConstraints({
        players: { 0: { maxDoubles: 1 } },
        fillSeed: 42
      });

      const doubles = countDoubles(result[0]!);
      expect(doubles).toBeLessThanOrEqual(1);
    });

    it('allows zero doubles', () => {
      const result = generateDealFromConstraints({
        players: { 0: { maxDoubles: 0 } },
        fillSeed: 42
      });

      const doubles = countDoubles(result[0]!);
      expect(doubles).toBe(0);
    });

    it('combines with minDoubles for exact count', () => {
      const result = generateDealFromConstraints({
        players: { 0: { minDoubles: 2, maxDoubles: 2 } },
        fillSeed: 42
      });

      const doubles = countDoubles(result[0]!);
      expect(doubles).toBe(2);
    });

    it('throws when minDoubles > maxDoubles', () => {
      expect(() => generateDealFromConstraints({
        players: { 0: { minDoubles: 3, maxDoubles: 2 } }
      })).toThrow(/minDoubles.*> maxDoubles/);
    });
  });

  // ============================================================================
  // Suit Constraint Tests
  // ============================================================================

  describe('mustHaveSuit constraint', () => {
    it('ensures player has at least one domino in suit', () => {
      const result = generateDealFromConstraints({
        players: { 0: { mustHaveSuit: [6] } },
        fillSeed: 42
      });

      const hasSixes = result[0]!.some(d => hasSuit(d, 6));
      expect(hasSixes).toBe(true);
    });

    it('satisfies multiple required suits', () => {
      const result = generateDealFromConstraints({
        players: { 0: { mustHaveSuit: [0, 1, 2] } },
        fillSeed: 42
      });

      expect(result[0]!.some(d => hasSuit(d, 0))).toBe(true);
      expect(result[0]!.some(d => hasSuit(d, 1))).toBe(true);
      expect(result[0]!.some(d => hasSuit(d, 2))).toBe(true);
    });
  });

  describe('voidInSuit constraint', () => {
    it('ensures player has no dominoes in void suit', () => {
      const result = generateDealFromConstraints({
        players: { 0: { voidInSuit: [6] } },
        fillSeed: 42
      });

      const hasSixes = result[0]!.some(d => hasSuit(d, 6));
      expect(hasSixes).toBe(false);
    });

    it('handles multiple void suits', () => {
      const result = generateDealFromConstraints({
        players: { 0: { voidInSuit: [5, 6] } },
        fillSeed: 42
      });

      expect(result[0]!.some(d => hasSuit(d, 5))).toBe(false);
      expect(result[0]!.some(d => hasSuit(d, 6))).toBe(false);
    });

    it('throws on mustHaveSuit/voidInSuit conflict', () => {
      expect(() => generateDealFromConstraints({
        players: { 0: { mustHaveSuit: [6], voidInSuit: [6] } }
      })).toThrow(/both mustHaveSuit and voidInSuit/);
    });

    it('respects voidInSuit when filling', () => {
      // Even with many void suits, should still fill hand
      const result = generateDealFromConstraints({
        players: { 0: { voidInSuit: [0, 1, 2] } },
        fillSeed: 42
      });

      expect(result[0]).toHaveLength(7);
      for (const d of result[0]!) {
        expect(hasSuit(d, 0)).toBe(false);
        expect(hasSuit(d, 1)).toBe(false);
        expect(hasSuit(d, 2)).toBe(false);
      }
    });
  });

  describe('minSuitCount constraint', () => {
    it('ensures minimum dominoes in suit', () => {
      const result = generateDealFromConstraints({
        players: { 0: { minSuitCount: { 6: 3 } } },
        fillSeed: 42
      });

      const sixes = result[0]!.filter(d => hasSuit(d, 6));
      expect(sixes.length).toBeGreaterThanOrEqual(3);
    });

    it('handles multiple suit requirements', () => {
      const result = generateDealFromConstraints({
        players: { 0: { minSuitCount: { 5: 2, 6: 2 } } },
        fillSeed: 42
      });

      const fives = result[0]!.filter(d => hasSuit(d, 5));
      const sixes = result[0]!.filter(d => hasSuit(d, 6));
      expect(fives.length).toBeGreaterThanOrEqual(2);
      expect(sixes.length).toBeGreaterThanOrEqual(2);
    });
  });

  // ============================================================================
  // minPoints Constraint Tests
  // ============================================================================

  describe('minPoints constraint', () => {
    it('ensures minimum point value in hand', () => {
      const result = generateDealFromConstraints({
        players: { 0: { minPoints: 20 } },
        fillSeed: 42
      });

      const points = getHandPoints(result[0]!);
      expect(points).toBeGreaterThanOrEqual(20);
    });

    it('throws on impossible point requirement', () => {
      expect(() => generateDealFromConstraints({
        players: { 0: { minPoints: 100 } }
      })).toThrow(/exceeds maximum possible/);
    });
  });

  // ============================================================================
  // Complex Multi-Constraint Tests
  // ============================================================================

  describe('complex constraint combinations', () => {
    it('handles plunge-eligible hand scenario', () => {
      // Player 0 needs 4 doubles for plunge
      const result = generateDealFromConstraints({
        players: { 0: { minDoubles: 4 } },
        fillSeed: 42
      });

      expect(countDoubles(result[0]!)).toBeGreaterThanOrEqual(4);
      expect(result[0]).toHaveLength(7);
    });

    it('handles nello-like scenario (void in high suits)', () => {
      const result = generateDealFromConstraints({
        players: { 0: { voidInSuit: [5, 6], maxDoubles: 1 } },
        fillSeed: 42
      });

      const hand = result[0]!;
      expect(hand.some(d => hasSuit(d, 5))).toBe(false);
      expect(hand.some(d => hasSuit(d, 6))).toBe(false);
      expect(countDoubles(hand)).toBeLessThanOrEqual(1);
    });

    it('handles constrained opponent scenario', () => {
      // Player 0 bids sixes trump, player 1 void in sixes
      const result = generateDealFromConstraints({
        players: {
          0: { mustHaveSuit: [6], minSuitCount: { 6: 3 } },
          1: { voidInSuit: [6] }
        },
        fillSeed: 42
      });

      expect(result[0]!.filter(d => hasSuit(d, 6)).length).toBeGreaterThanOrEqual(3);
      expect(result[1]!.some(d => hasSuit(d, 6))).toBe(false);
    });

    it('handles all four players constrained', () => {
      // Use less restrictive constraints that are always satisfiable
      const result = generateDealFromConstraints({
        players: {
          0: { minDoubles: 2 },
          1: { maxDoubles: 2 },
          2: { mustHaveSuit: [6] },
          3: { mustHaveSuit: [5] }
        },
        fillSeed: 42
      });

      expect(countDoubles(result[0]!)).toBeGreaterThanOrEqual(2);
      expect(countDoubles(result[1]!)).toBeLessThanOrEqual(2);
      expect(result[2]!.some(d => hasSuit(d, 6))).toBe(true);
      expect(result[3]!.some(d => hasSuit(d, 5))).toBe(true);
    });

    it('handles exactDominoes with other constraints', () => {
      const result = generateDealFromConstraints({
        players: {
          0: {
            exactDominoes: ['6-6', '5-5'],
            minDoubles: 3,
            mustHaveSuit: [4]
          }
        },
        fillSeed: 42
      });

      const hand = result[0]!;
      expect(hand.some(d => d.id === '6-6')).toBe(true);
      expect(hand.some(d => d.id === '5-5')).toBe(true);
      expect(countDoubles(hand)).toBeGreaterThanOrEqual(3);
      expect(hand.some(d => hasSuit(d, 4))).toBe(true);
    });
  });

  // ============================================================================
  // Convenience Function Tests
  // ============================================================================

  describe('dealWithDoubles', () => {
    it('creates deal with specified doubles', () => {
      const result = dealWithDoubles(0, 4, 42);

      expect(countDoubles(result[0]!)).toBeGreaterThanOrEqual(4);
      expect(getAllDominoIds(result).size).toBe(28);
    });
  });

  describe('dealWithExactHand', () => {
    it('creates deal with exact hand', () => {
      const dominoes = ['6-6', '6-5', '6-4', '6-3', '6-2', '6-1', '6-0'];
      const result = dealWithExactHand(0, dominoes, 42);

      const handIds = new Set(result[0]!.map(d => d.id));
      for (const id of dominoes) {
        expect(handIds.has(id)).toBe(true);
      }
    });
  });

  describe('dealWithVoid', () => {
    it('creates deal with void in specified suits', () => {
      const result = dealWithVoid(0, [5, 6], 42);

      for (const d of result[0]!) {
        expect(hasSuit(d, 5)).toBe(false);
        expect(hasSuit(d, 6)).toBe(false);
      }
    });
  });
});

// ============================================================================
// parseDominoId Tests
// ============================================================================

// ============================================================================
// StateBuilder Integration Tests
// ============================================================================

describe('StateBuilder with constraints', () => {
  it('builds state with player doubles constraint', () => {
    const state = StateBuilder.inBiddingPhase()
      .withPlayerDoubles(0, 4)
      .withFillSeed(42)
      .build();

    const player0 = state.players[0]!;
    const doubles = player0.hand.filter(d => d.high === d.low);
    expect(doubles.length).toBeGreaterThanOrEqual(4);
    expect(player0.hand.length).toBe(7);
  });

  it('builds state with player constraint', () => {
    const state = StateBuilder.inBiddingPhase()
      .withPlayerConstraint(0, {
        exactDominoes: ['6-6', '5-5'],
        minDoubles: 3
      })
      .withFillSeed(123)
      .build();

    const player0 = state.players[0]!;
    const handIds = new Set(player0.hand.map(d => String(d.id)));
    expect(handIds.has('6-6')).toBe(true);
    expect(handIds.has('5-5')).toBe(true);

    const doubles = player0.hand.filter(d => d.high === d.low);
    expect(doubles.length).toBeGreaterThanOrEqual(3);
  });

  it('builds state with full deal constraints', () => {
    const state = StateBuilder.inBiddingPhase()
      .withDealConstraints({
        players: {
          0: { minDoubles: 3 },
          1: { maxDoubles: 1 }
        },
        fillSeed: 999
      })
      .build();

    const player0 = state.players[0]!;
    const player1 = state.players[1]!;
    const p0doubles = player0.hand.filter(d => d.high === d.low);
    const p1doubles = player1.hand.filter(d => d.high === d.low);

    expect(p0doubles.length).toBeGreaterThanOrEqual(3);
    expect(p1doubles.length).toBeLessThanOrEqual(1);
  });

  it('is deterministic with same fillSeed', () => {
    const state1 = StateBuilder.inBiddingPhase()
      .withPlayerDoubles(0, 4)
      .withFillSeed(42)
      .build();

    const state2 = StateBuilder.inBiddingPhase()
      .withPlayerDoubles(0, 4)
      .withFillSeed(42)
      .build();

    const ids1 = state1.players[0]!.hand.map(d => String(d.id)).sort();
    const ids2 = state2.players[0]!.hand.map(d => String(d.id)).sort();

    expect(ids1).toEqual(ids2);
  });

  it('all 28 dominoes are distributed across players', () => {
    const state = StateBuilder.inBiddingPhase()
      .withPlayerDoubles(0, 4)
      .withFillSeed(42)
      .build();

    const allIds = new Set<string>();
    for (const player of state.players) {
      expect(player.hand.length).toBe(7);
      for (const d of player.hand) {
        const id = String(d.id);
        expect(allIds.has(id)).toBe(false); // No duplicates
        allIds.add(id);
      }
    }
    expect(allIds.size).toBe(28);
  });
});

describe('parseDominoId', () => {
  it('parses normalized IDs correctly', () => {
    const d = parseDominoId('6-5');
    expect(d.high).toBe(6);
    expect(d.low).toBe(5);
    expect(d.id).toBe('6-5');
  });

  it('normalizes reversed IDs', () => {
    const d = parseDominoId('5-6');
    expect(d.high).toBe(6);
    expect(d.low).toBe(5);
    expect(d.id).toBe('6-5');
  });

  it('handles doubles', () => {
    const d = parseDominoId('4-4');
    expect(d.high).toBe(4);
    expect(d.low).toBe(4);
    expect(d.id).toBe('4-4');
  });

  it('throws on invalid format', () => {
    expect(() => parseDominoId('abc')).toThrow();
    expect(() => parseDominoId('6')).toThrow();
    expect(() => parseDominoId('')).toThrow();
  });
});
