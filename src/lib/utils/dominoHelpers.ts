import type { Domino } from '../../game/types';

export function parseDomino(dominoStr: string): Domino {
  const parts = dominoStr.split('-').map(Number);
  if (parts.length !== 2 || parts.some(p => isNaN(p))) {
    throw new Error(`Invalid domino string: ${dominoStr}`);
  }
  const [high, low] = parts as [number, number];
  const sortedHigh = Math.max(high, low);
  const sortedLow = Math.min(high, low);

  let points = 0;
  const sum = sortedHigh + sortedLow;
  if (sum === 10 || sum === 5) {
    points = sum === 10 ? 10 : 5;
  }

  return {
    high: sortedHigh,
    low: sortedLow,
    id: `${sortedHigh}-${sortedLow}`,
    points
  };
}

export function parseTrumpFromString(trumpStr: string): number {
  const trumpMap: Record<string, number> = {
    'blanks': 0,
    'aces': 1,
    'deuces': 2,
    'tres': 3,
    'fours': 4,
    'fives': 5,
    'sixes': 6,
    'doubles': 7,
    'no-trump': 8
  };

  return trumpMap[trumpStr] ?? 8;
}