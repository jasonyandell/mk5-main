import { describe, test, expect } from 'vitest';
import { getCurrentSuit } from '../../game/core/rules';
import type { Domino } from '../../game/types';

describe('Current Suit Display', () => {
  const doublesAreTrump = 7;
  const fivesAreTrump = 5;
  const sixesAreTrump = 6;
  const noTrump = 8;

  test('should return "None" when no domino is led', () => {
    const emptyTrick: { player: number; domino: Domino }[] = [];
    expect(getCurrentSuit(emptyTrick, doublesAreTrump)).toBe('None (no domino led)');
  });

  test('should return "None" when trump is not set', () => {
    const trick = [{ player: 0, domino: { high: 6, low: 5, id: "6-5" } }];
    expect(getCurrentSuit(trick, null)).toBe('None (no trump set)');
  });

  describe('When doubles are trump', () => {
    test('6-6 leads -> Doubles (Trump)', () => {
      const trick = [{ player: 0, domino: { high: 6, low: 6, id: "6-6" } }];
      expect(getCurrentSuit(trick, doublesAreTrump)).toBe('Doubles (Trump)');
    });

    test('3-3 leads -> Doubles (Trump)', () => {
      const trick = [{ player: 0, domino: { high: 3, low: 3, id: "3-3" } }];
      expect(getCurrentSuit(trick, doublesAreTrump)).toBe('Doubles (Trump)');
    });

    test('6-5 leads -> Sixes (higher end)', () => {
      const trick = [{ player: 0, domino: { high: 6, low: 5, id: "6-5" } }];
      expect(getCurrentSuit(trick, doublesAreTrump)).toBe('Sixes');
    });

    test('4-2 leads -> Fours (higher end)', () => {
      const trick = [{ player: 0, domino: { high: 4, low: 2, id: "4-2" } }];
      expect(getCurrentSuit(trick, doublesAreTrump)).toBe('Fours');
    });
  });

  describe('When specific suit is trump (not doubles)', () => {
    test('6-6 leads with 6s trump -> Sixes (Trump)', () => {
      const trick = [{ player: 0, domino: { high: 6, low: 6, id: "6-6" } }];
      expect(getCurrentSuit(trick, sixesAreTrump)).toBe('Sixes (Trump)');
    });

    test('5-5 leads with 6s trump -> Fives', () => {
      const trick = [{ player: 0, domino: { high: 5, low: 5, id: "5-5" } }];
      expect(getCurrentSuit(trick, sixesAreTrump)).toBe('Fives');
    });

    test('6-5 leads with 6s trump -> Sixes (Trump)', () => {
      const trick = [{ player: 0, domino: { high: 6, low: 5, id: "6-5" } }];
      expect(getCurrentSuit(trick, sixesAreTrump)).toBe('Sixes (Trump)');
    });

    test('6-5 leads with 5s trump -> Fives (Trump)', () => {
      const trick = [{ player: 0, domino: { high: 6, low: 5, id: "6-5" } }];
      expect(getCurrentSuit(trick, fivesAreTrump)).toBe('Fives (Trump)');
    });

    test('4-2 leads with 5s trump -> Fours', () => {
      const trick = [{ player: 0, domino: { high: 4, low: 2, id: "4-2" } }];
      expect(getCurrentSuit(trick, fivesAreTrump)).toBe('Fours');
    });
  });

  describe('No trump (follow-me)', () => {
    test('6-6 leads -> Sixes', () => {
      const trick = [{ player: 0, domino: { high: 6, low: 6, id: "6-6" } }];
      expect(getCurrentSuit(trick, noTrump)).toBe('Sixes');
    });

    test('6-5 leads -> Sixes (higher end)', () => {
      const trick = [{ player: 0, domino: { high: 6, low: 5, id: "6-5" } }];
      expect(getCurrentSuit(trick, noTrump)).toBe('Sixes');
    });

    test('0-0 leads -> Blanks', () => {
      const trick = [{ player: 0, domino: { high: 0, low: 0, id: "0-0" } }];
      expect(getCurrentSuit(trick, noTrump)).toBe('Blanks');
    });
  });

  describe('All suit names', () => {
    const testCases = [
      { domino: { high: 0, low: 0, id: "0-0" }, expected: 'Blanks' },
      { domino: { high: 1, low: 1, id: "1-1" }, expected: 'Ones' },
      { domino: { high: 2, low: 2, id: "2-2" }, expected: 'Twos' },
      { domino: { high: 3, low: 3, id: "3-3" }, expected: 'Threes' },
      { domino: { high: 4, low: 4, id: "4-4" }, expected: 'Fours' },
      { domino: { high: 5, low: 5, id: "5-5" }, expected: 'Fives' },
      { domino: { high: 6, low: 6, id: "6-6" }, expected: 'Sixes' }
    ];

    testCases.forEach(({ domino, expected }) => {
      test(`${domino.id} with no trump -> ${expected}`, () => {
        const trick = [{ player: 0, domino }];
        expect(getCurrentSuit(trick, noTrump)).toBe(expected);
      });
    });
  });

  describe('Edge cases', () => {
    test('Complex scenario - 6-4 led with 4s trump should show Fours (Trump)', () => {
      const trick = [{ player: 0, domino: { high: 6, low: 4, id: "6-4" } }];
      expect(getCurrentSuit(trick, 4)).toBe('Fours (Trump)');
    });

    test('Domino with trump number led should show trump', () => {
      const trick = [{ player: 0, domino: { high: 5, low: 2, id: "5-2" } }];
      expect(getCurrentSuit(trick, 5)).toBe('Fives (Trump)');
    });
  });
});