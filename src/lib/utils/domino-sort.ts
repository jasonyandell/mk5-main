import type { Domino } from '../../game/types';
import { dominoHasSuit } from '../../game/core/dominoes';

/**
 * Convert trump string to suit number for comparison.
 * Returns null for 'doubles' or 'no-trump'.
 */
function trumpToSuit(trumpStr: string): number | null {
  const suitMap: Record<string, number> = {
    'blanks': 0,
    'aces': 1,
    'deuces': 2,
    'tres': 3,
    'fours': 4,
    'fives': 5,
    'sixes': 6
  };
  return suitMap[trumpStr] ?? null;
}

/**
 * Check if a domino is trump given a trump string.
 */
function isTrump(domino: Domino, trumpStr: string): boolean {
  if (trumpStr === 'no-trump') return false;
  if (trumpStr === 'doubles') return domino.high === domino.low;

  const suit = trumpToSuit(trumpStr);
  if (suit !== null) {
    return dominoHasSuit(domino, suit);
  }
  return false;
}

/**
 * Sort dominoes for display: trumps first, then by high pip, then low pip.
 *
 * Used by PerfectHandDisplay and PerfectsApp to consistently display
 * dominoes with trump dominoes grouped first.
 */
export function sortDominoesForDisplay(dominoes: Domino[], trumpStr: string): Domino[] {
  return dominoes.slice().sort((a, b) => {
    const aIsTrump = isTrump(a, trumpStr);
    const bIsTrump = isTrump(b, trumpStr);

    if (aIsTrump && !bIsTrump) return -1;
    if (!aIsTrump && bIsTrump) return 1;

    // Within trumps or non-trumps, sort by high pip then low pip
    if (a.high !== b.high) return b.high - a.high;
    return b.low - a.low;
  });
}
