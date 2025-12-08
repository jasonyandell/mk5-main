/**
 * Hand Strength Calculation Module
 *
 * Utilities for trump selection in bidding decisions.
 */

import type { Domino, TrumpSelection, SuitAnalysis } from '../types';
import { countDoubles, dominoHasSuit } from '../core/dominoes';

/**
 * Score a suit for trump selection.
 *
 * Factors considered:
 * 1. Having the double (worth +3 count equivalent - it's the highest card)
 * 2. Number of dominoes in the suit
 * 3. Quality bonus for high pip cards
 */
function scoreSuitForTrump(hand: Domino[], suit: number): number {
  let score = 0;
  let hasDouble = false;

  for (const domino of hand) {
    if (!dominoHasSuit(domino, suit)) continue;

    // Check if this is the double of the suit
    if (domino.high === suit && domino.low === suit) {
      hasDouble = true;
      score += 1; // Count the double once
    } else {
      score += 1; // Count non-doubles
    }
  }

  // Bonus for having the double (worth ~2 extra dominoes)
  // Having the highest card in trump is critical
  if (hasDouble) {
    score += 2;
  }

  return score;
}

/**
 * Determine the best trump selection for a hand.
 *
 * Considers:
 * - Number of dominoes in each suit
 * - Whether we have the double (highest card) in that suit
 * - 3+ doubles triggers doubles trump
 */
export function determineBestTrump(hand: Domino[], _suitAnalysis?: SuitAnalysis): TrumpSelection {
  // Check if doubles trump would be better (3+ doubles)
  const doubleCount = countDoubles(hand);
  if (doubleCount >= 3) {
    return { type: 'doubles' as const };
  }

  // Score each suit
  let bestSuit = 6;
  let bestScore = -1;

  for (let suit = 6; suit >= 0; suit--) {
    const score = scoreSuitForTrump(hand, suit);
    if (score > bestScore) {
      bestScore = score;
      bestSuit = suit;
    }
  }

  return { type: 'suit' as const, suit: bestSuit as 0 | 1 | 2 | 3 | 4 | 5 | 6 };
}
