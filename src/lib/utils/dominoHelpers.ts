import type { Domino, TrumpSelection, LedSuitOrNone } from '../../game/types';
import { PLAYED_AS_TRUMP } from '../../game/types';
import { getDominoStrength } from '../../game/ai/strength-table.generated';
import { dominoHasSuit } from '../../game/core/dominoes';
import { parseTrumpString, TRUMP_NAME_TO_NUMBER } from '../../game/game-terms';

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
  return TRUMP_NAME_TO_NUMBER[trumpStr.toLowerCase()] ?? 8;
}

function createTrumpSelection(trumpStr: string): TrumpSelection {
  return parseTrumpString(trumpStr);
}

function getPlayContexts(domino: Domino, trump: TrumpSelection): LedSuitOrNone[] {
  const contexts: LedSuitOrNone[] = [];

  // Check if it's trump
  const isTrumpDomino = isTrump(domino, trump);

  if (isTrumpDomino) {
    contexts.push(PLAYED_AS_TRUMP);
  }

  // When we LEAD (which is what we care about for analysis)
  if (domino.high === domino.low) {
    // Double leads as its suit or as doubles if doubles are trump
    if (trump.type === 'doubles') {
      contexts.push(7 as LedSuitOrNone);
    } else {
      contexts.push(domino.high as LedSuitOrNone);
    }
  } else {
    // Non-double ALWAYS leads as its HIGH pip
    contexts.push(domino.high as LedSuitOrNone);
  }

  return contexts;
}

function isTrump(domino: Domino, trump: TrumpSelection): boolean {
  if (trump.type === 'no-trump') return false;
  if (trump.type === 'doubles') return domino.high === domino.low;
  return dominoHasSuit(domino, trump.suit!);
}

export function computeExternalBeaters(leftoverDominoes: string[], bestTrumpStr: string): string[] {
  const trump = createTrumpSelection(bestTrumpStr);
  const leftoverSet = new Set(leftoverDominoes);
  const allExternalBeaters = new Set<string>();

  for (const dominoStr of leftoverDominoes) {
    const domino = parseDomino(dominoStr);
    const contexts = getPlayContexts(domino, trump);

    for (const context of contexts) {
      const strength = getDominoStrength(domino, trump, context);

      if (strength) {
        // Add external beaters (not in leftover hand)
        for (const beaterId of strength.beatenBy) {
          if (!leftoverSet.has(beaterId)) {
            allExternalBeaters.add(beaterId);
          }
        }
      }
    }
  }

  // Convert to array and sort for consistent display
  return Array.from(allExternalBeaters).sort((a, b) => {
    const [aHigh, aLow] = a.split('-').map(Number);
    const [bHigh, bLow] = b.split('-').map(Number);
    if (aHigh !== bHigh) return (bHigh ?? 0) - (aHigh ?? 0);
    return (bLow ?? 0) - (aLow ?? 0);
  });
}