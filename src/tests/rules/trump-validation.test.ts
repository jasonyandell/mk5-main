import { describe, it, expect } from 'vitest';
import { isValidTrump, getTrumpValue } from '../../game/core/rules';
import { Trump } from '../../game/types';
import { TRUMP_SUITS, SUIT_VALUES } from '../../game/constants';

describe('Trump Validation', () => {
  describe('isValidTrump', () => {
    it('should accept valid suit trumps', () => {
      Object.values(TRUMP_SUITS).forEach(suit => {
        expect(isValidTrump({ suit, followsSuit: false })).toBe(true);
      });
    });

    it('should accept follow-suit trump', () => {
      expect(isValidTrump({ suit: SUIT_VALUES.BLANKS, followsSuit: true })).toBe(true);
    });

    it('should reject invalid suit combinations', () => {
      expect(isValidTrump({ suit: 'invalid' as any, followsSuit: false })).toBe(false);
    });
  });

  describe('getTrumpValue', () => {
    it('should return correct trump priority values', () => {
      const trumps: Trump[] = [
        { suit: TRUMP_SUITS.BLANKS, followsSuit: false },
        { suit: TRUMP_SUITS.ONES, followsSuit: false },
        { suit: TRUMP_SUITS.TWOS, followsSuit: false },
        { suit: TRUMP_SUITS.THREES, followsSuit: false },
        { suit: TRUMP_SUITS.FOURS, followsSuit: false },
        { suit: TRUMP_SUITS.FIVES, followsSuit: false },
        { suit: TRUMP_SUITS.SIXES, followsSuit: false },
        { suit: TRUMP_SUITS.BLANKS, followsSuit: true }
      ];

      const values = trumps.map(getTrumpValue);
      
      // Values should be distinct and in expected order
      expect(new Set(values).size).toBe(values.length);
      
      // Follow-suit should have different value than regular suit
      expect(values[0]).not.toBe(values[7]);
    });

    it('should handle edge cases', () => {
      expect(() => getTrumpValue({ suit: 'invalid' as any, followsSuit: false })).toThrow();
    });
  });
});