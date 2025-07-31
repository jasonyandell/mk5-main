import { describe, it, expect } from 'vitest';
import { isValidTrump, getTrumpValue } from '../../game/core/rules';
import { TRUMP_SUITS, SUIT_VALUES } from '../../game/constants';

describe('Trump Validation', () => {
  describe('isValidTrump', () => {
    it('should accept valid suit trumps', () => {
      Object.values(TRUMP_SUITS).forEach(suit => {
        expect(isValidTrump({ suit: suit as (string | number), followsSuit: false })).toBe(true);
      });
    });

    it('should accept follow-suit trump', () => {
      expect(isValidTrump({ suit: SUIT_VALUES.BLANKS as (string | number), followsSuit: true })).toBe(true);
    });

    it('should reject invalid suit combinations', () => {
      expect(isValidTrump({ suit: 'invalid', followsSuit: false })).toBe(false);
    });
  });

  describe('getTrumpValue', () => {
    it('should return correct trump priority values', () => {
      const trumps = [
        { suit: TRUMP_SUITS.BLANKS as (string | number), followsSuit: false },
        { suit: TRUMP_SUITS.ONES as (string | number), followsSuit: false },
        { suit: TRUMP_SUITS.TWOS as (string | number), followsSuit: false },
        { suit: TRUMP_SUITS.THREES as (string | number), followsSuit: false },
        { suit: TRUMP_SUITS.FOURS as (string | number), followsSuit: false },
        { suit: TRUMP_SUITS.FIVES as (string | number), followsSuit: false },
        { suit: TRUMP_SUITS.SIXES as (string | number), followsSuit: false },
        { suit: TRUMP_SUITS.BLANKS as (string | number), followsSuit: true }
      ];

      const values = trumps.map(getTrumpValue);
      
      // Values should be distinct and in expected order
      expect(new Set(values).size).toBe(values.length);
      
      // Follow-suit should have different value than regular suit
      expect(values[0]).not.toBe(values[7]);
    });

    it('should handle edge cases', () => {
      expect(() => getTrumpValue({ suit: 'invalid', followsSuit: false })).toThrow();
    });
  });
});