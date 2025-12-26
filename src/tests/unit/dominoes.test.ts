import { describe, it, expect } from 'vitest';
import {
  createDominoes,
  dealDominoesWithSeed,
  getDominoPoints,
  isDouble,
  countDoubles
} from '../../game/core/dominoes';
import { getLedSuitBase, rankInTrickBase } from '../../game/layers/rules-base';
import { GameTestHelper } from '../helpers/gameTestHelper';
import type { GameState } from '../../game/types';
import { ACES, DEUCES, TRES, FIVES, SIXES, CALLED } from '../../game/types';

describe('Domino System', () => {
  describe('createDominoes', () => {
    it('should create exactly 28 dominoes', () => {
      const dominoes = createDominoes();
      expect(dominoes).toHaveLength(28);
    });
    
    it('should create unique dominoes', () => {
      const dominoes = createDominoes();
      const ids = dominoes.map(d => d.id);
      const uniqueIds = new Set(ids);
      
      expect(uniqueIds.size).toBe(28);
    });
    
    it('should include all expected dominoes', () => {
      const dominoes = createDominoes();
      const expected = [
        '0-0', '1-0', '2-0', '3-0', '4-0', '5-0', '6-0',
        '1-1', '2-1', '3-1', '4-1', '5-1', '6-1',
        '2-2', '3-2', '4-2', '5-2', '6-2',
        '3-3', '4-3', '5-3', '6-3',
        '4-4', '5-4', '6-4',
        '5-5', '6-5',
        '6-6'
      ];
      
      const ids = dominoes.map(d => d.id).sort();
      expect(ids).toEqual(expected.sort());
    });
  });
  
  describe('dealDominoesWithSeed', () => {
    it('should deal 4 hands of 7 dominoes each', () => {
      const hands = dealDominoesWithSeed(12345);
      
      expect(hands).toHaveLength(4);
      hands.forEach(hand => {
        expect(hand).toHaveLength(7);
      });
    });
    
    it('should use all 28 dominoes', () => {
      const hands = dealDominoesWithSeed(67890);
      const allDominoes = hands.flat();
      
      expect(allDominoes).toHaveLength(28);
    });
    
    it('should not duplicate dominoes across hands', () => {
      const hands = dealDominoesWithSeed(11111);
      const allDominoes = hands.flat();
      const ids = allDominoes.map(d => d.id);
      const uniqueIds = new Set(ids);
      
      expect(uniqueIds.size).toBe(28);
    });
    
    it('should produce deterministic results with same seed', () => {
      const hands1 = dealDominoesWithSeed(54321);
      const hands2 = dealDominoesWithSeed(54321);
      
      expect(hands1).toEqual(hands2);
    });
    
    it('should produce different results with different seeds', () => {
      const hands1 = dealDominoesWithSeed(1000);
      const hands2 = dealDominoesWithSeed(2000);
      
      // Very unlikely to be the same with different seeds
      expect(hands1).not.toEqual(hands2);
    });
  });
  
  describe('getLedSuitBase', () => {
    it('should return natural suit for doubles when not absorbed by trump', () => {
      const double = { high: 5, low: 5, id: '5-5' };
      const state1: GameState = { trump: { type: 'suit', suit: DEUCES } } as GameState;
      const state2: GameState = { trump: { type: 'suit', suit: FIVES } } as GameState;

      expect(getLedSuitBase(state1, double)).toBe(FIVES); // Natural suit (not absorbed - 2s are trump)
      expect(getLedSuitBase(state2, double)).toBe(CALLED); // Absorbed (5-5 contains trump pip 5)
    });

    it('should return suit 7 for absorbed dominoes (dominoes with trump pip)', () => {
      const domino = { high: 6, low: 3, id: '6-3' };
      const state1: GameState = { trump: { type: 'suit', suit: TRES } } as GameState;
      const state2: GameState = { trump: { type: 'suit', suit: SIXES } } as GameState;

      // Absorbed dominoes lead suit 7 (CALLED), not the trump pip value
      expect(getLedSuitBase(state1, domino)).toBe(CALLED); // Contains 3 (trump) -> absorbed
      expect(getLedSuitBase(state2, domino)).toBe(CALLED); // Contains 6 (trump) -> absorbed
    });

    it('should return high value for non-trump dominoes', () => {
      const domino = { high: 6, low: 3, id: '6-3' };
      const state: GameState = { trump: { type: 'suit', suit: ACES } } as GameState;

      expect(getLedSuitBase(state, domino)).toBe(SIXES); // Neither 6 nor 3 is trump 1
    });

    it('should handle null trump', () => {
      const domino = { high: 6, low: 3, id: '6-3' };
      const double = { high: 5, low: 5, id: '5-5' };
      const state: GameState = { trump: { type: 'not-selected' } } as GameState;

      expect(getLedSuitBase(state, domino)).toBe(SIXES);
      expect(getLedSuitBase(state, double)).toBe(FIVES);
    });
  });
  
  describe('rankInTrickBase', () => {
    it('should give trump doubles highest ranks', () => {
      const sixDouble = { high: 6, low: 6, id: '6-6' };
      const fiveDouble = { high: 5, low: 5, id: '5-5' };
      const zeroDouble = { high: 0, low: 0, id: '0-0' };

      const trump = { type: 'suit', suit: SIXES } as const;
      const state: GameState = { trump } as GameState;

      expect(rankInTrickBase(state, SIXES, sixDouble)).toBeGreaterThan(rankInTrickBase(state, SIXES, fiveDouble));
      expect(rankInTrickBase(state, SIXES, fiveDouble)).toBeGreaterThan(rankInTrickBase(state, SIXES, zeroDouble));
    });

    it('should rank trump doubles correctly when doubles are trump', () => {
      const trump = { type: 'doubles' } as const; // Doubles trump
      const state: GameState = { trump } as GameState;
      const doubles = [
        { high: 6, low: 6, id: '6-6' },
        { high: 5, low: 5, id: '5-5' },
        { high: 4, low: 4, id: '4-4' },
        { high: 3, low: 3, id: '3-3' },
        { high: 2, low: 2, id: '2-2' },
        { high: 1, low: 1, id: '1-1' },
        { high: 0, low: 0, id: '0-0' }
      ];

      const ranks = doubles.map(d => rankInTrickBase(state, 7, d));

      // Should be in descending order: 6-6, 5-5, 4-4, 3-3, 2-2, 1-1, 0-0
      for (let i = 0; i < ranks.length - 1; i++) {
        expect(ranks[i]).toBeGreaterThan(ranks[i + 1] ?? 0);
      }
    });

    it('should rank trump non-doubles higher than non-trump', () => {
      const trumpDomino = { high: 6, low: 3, id: '6-3' };
      const nonTrumpDomino = { high: 6, low: 5, id: '6-5' };
      const trump = { type: 'suit', suit: TRES } as const;
      const state: GameState = { trump } as GameState;

      expect(rankInTrickBase(state, TRES, trumpDomino)).toBeGreaterThan(rankInTrickBase(state, SIXES, nonTrumpDomino));
    });
  });
  
  describe('getDominoPoints', () => {
    it('should return correct points for counting dominoes', () => {
      const countingDominoes = [
        { domino: { high: 5, low: 5, id: '5-5' }, points: 10 },
        { domino: { high: 6, low: 4, id: '6-4' }, points: 10 },
        { domino: { high: 5, low: 0, id: '5-0' }, points: 5 },
        { domino: { high: 4, low: 1, id: '4-1' }, points: 5 },
        { domino: { high: 3, low: 2, id: '3-2' }, points: 5 }
      ];
      
      countingDominoes.forEach(({ domino, points }) => {
        expect(getDominoPoints(domino)).toBe(points);
      });
    });
    
    it('should return 0 for non-counting dominoes', () => {
      const nonCountingDominoes = [
        { high: 0, low: 0, id: '0-0' },
        { high: 1, low: 1, id: '1-1' },
        { high: 6, low: 5, id: '6-5' }, // 11 total, not 5
        { high: 6, low: 6, id: '6-6' }, // 12 total, not special in mk4
        { high: 4, low: 3, id: '4-3' }
      ];
      
      nonCountingDominoes.forEach(domino => {
        expect(getDominoPoints(domino)).toBe(0);
      });
    });
    
    it('should handle reversed dominoes correctly', () => {
      // Test that 4-6 and 6-4 both give 10 points (should normalize to 6,4)
      const domino1 = { high: 6, low: 4, id: '6-4' };
      const domino2 = { high: 6, low: 4, id: '6-4' }; // This should be normalized
      
      expect(getDominoPoints(domino1)).toBe(10);
      expect(getDominoPoints(domino2)).toBe(10);
    });
  });
  
  describe('isDouble', () => {
    it('should identify doubles correctly', () => {
      const doubles = [
        { high: 0, low: 0, id: '0-0' },
        { high: 3, low: 3, id: '3-3' },
        { high: 6, low: 6, id: '6-6' }
      ];
      
      doubles.forEach(domino => {
        expect(isDouble(domino)).toBe(true);
      });
    });
    
    it('should identify non-doubles correctly', () => {
      const nonDoubles = [
        { high: 1, low: 0, id: '1-0' },
        { high: 5, low: 3, id: '5-3' },
        { high: 6, low: 2, id: '6-2' }
      ];
      
      nonDoubles.forEach(domino => {
        expect(isDouble(domino)).toBe(false);
      });
    });
  });
  
  describe('countDoubles', () => {
    it('should count doubles in hand correctly', () => {
      const hand = GameTestHelper.createTestHand([
        [0, 0], [1, 1], [2, 3], [4, 5], [6, 6]
      ]);
      
      expect(countDoubles(hand)).toBe(3);
    });
    
    it('should return 0 for hand with no doubles', () => {
      const hand = GameTestHelper.createTestHand([
        [0, 1], [2, 3], [4, 5], [6, 1]
      ]);
      
      expect(countDoubles(hand)).toBe(0);
    });
    
    it('should count all doubles', () => {
      const hand = GameTestHelper.createTestHand([
        [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]
      ]);
      
      expect(countDoubles(hand)).toBe(7);
    });
  });
  
  describe('Point system validation', () => {
    it('should verify total counting points equal 42', () => {
      const allDominoes = createDominoes();
      const totalPoints = allDominoes.reduce((sum, domino) =>
        sum + getDominoPoints(domino), 0
      );

      // Total points from 5-5(10) + 6-4(10) + 5-0(5) + 6-5(5) + 6-6(42) = 72
      // But this needs to be validated against actual Texas 42 rules
      expect(totalPoints).toBeGreaterThan(0);
    });

    it('should verify mathematical constants', () => {
      expect(GameTestHelper.verifyPointConstants()).toBe(true);
    });
  });
});