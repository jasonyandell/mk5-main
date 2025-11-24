import { describe, it, expect } from 'vitest';
import { composeRules, baseRuleSet } from '../../game/layers';
import { getTrumpValue } from '../../game/core/rules';
import type { TrumpSelection } from '../../game/types';

const rules = composeRules([baseRuleSet]);

describe('Trump Validation', () => {
  describe('isValidTrump', () => {
    it('should accept valid suit trumps', () => {
      for (let suit = 0; suit <= 6; suit++) {
        const trump: TrumpSelection = { type: 'suit', suit: suit as 0 | 1 | 2 | 3 | 4 | 5 | 6 };
        expect(rules.isValidTrump(trump)).toBe(true);
      }
    });

    it('should accept doubles trump', () => {
      const trump: TrumpSelection = { type: 'doubles' };
      expect(rules.isValidTrump(trump)).toBe(true);
    });

    it('should accept no-trump', () => {
      const trump: TrumpSelection = { type: 'no-trump' };
      expect(rules.isValidTrump(trump)).toBe(true);
    });

    it('should reject invalid suit values', () => {
      const invalidTrump: TrumpSelection = { type: 'suit', suit: -1 as 0 | 1 | 2 | 3 | 4 | 5 | 6 };
      expect(rules.isValidTrump(invalidTrump)).toBe(false);
      
      const invalidTrump2: TrumpSelection = { type: 'suit', suit: 7 as 0 | 1 | 2 | 3 | 4 | 5 | 6 };
      expect(rules.isValidTrump(invalidTrump2)).toBe(false);
    });
  });

  describe('getTrumpValue', () => {
    it('should return correct trump priority values for suit trumps', () => {
      for (let suit = 0; suit <= 6; suit++) {
        const trump: TrumpSelection = { type: 'suit', suit: suit as 0 | 1 | 2 | 3 | 4 | 5 | 6 };
        const value = getTrumpValue(trump);
        expect(value).toBe(suit);
      }
    });

    it('should return correct value for doubles trump', () => {
      const trump: TrumpSelection = { type: 'doubles' };
      const value = getTrumpValue(trump);
      expect(value).toBe(7);
    });

    it('should return correct value for no-trump', () => {
      const trump: TrumpSelection = { type: 'no-trump' };
      const value = getTrumpValue(trump);
      expect(value).toBe(8);
    });

    it('should return correct value for none trump', () => {
      const trump: TrumpSelection = { type: 'not-selected' };
      const value = getTrumpValue(trump);
      expect(value).toBe(-1);
    });
  });
});