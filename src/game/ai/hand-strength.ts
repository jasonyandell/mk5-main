/**
 * Hand Strength Calculation Module
 * 
 * Shared utilities for evaluating hand strength for bidding decisions.
 * Used by both AI strategies and testing scripts.
 */

import type { Domino, TrumpSelection, SuitAnalysis } from '../types';
import { getStrongestSuits } from '../core/suit-analysis';
import { countDoubles, dominoHasSuit } from '../core/dominoes';

/**
 * Special score for laydown hands (guaranteed win)
 */
export const LAYDOWN_SCORE = 999;

/**
 * Bidding thresholds for different bid levels
 */
export const BID_THRESHOLDS = {
  PASS: 55,
  BID_30: 55,
  BID_31: 998,
  BID_32: 998,  // Adjusted to get better hands bidding 32
  BID_33: 998,  // Raised to be more selective (target ~60%)
  BID_34: 998,  // Raised to be more selective (target ~60%)
  BID_35: 998   // Raised slightly for consistency
};

/**
 * Determine the best trump selection for a hand
 */
export function determineBestTrump(hand: Domino[], suitAnalysis?: SuitAnalysis): TrumpSelection {
  let bestSuit: number;
  
  if (suitAnalysis) {
    const strongestSuits = getStrongestSuits(suitAnalysis);
    bestSuit = strongestSuits[0] || 6;
  } else {
    // When no analysis provided, count occurrences of each suit
    const suitCounts: number[] = new Array(7).fill(0);
    
    for (const domino of hand) {
      // Count each suit (0-6) - note doubles count for their suit once
      for (let suit = 0; suit <= 6; suit++) {
        if (dominoHasSuit(domino, suit)) {
          suitCounts[suit] = (suitCounts[suit] || 0) + 1;
        }
      }
    }
    
    // Find the suit with the most dominoes
    let maxCount = 0;
    bestSuit = 6; // Default to sixes if somehow all counts are 0
    
    for (let suit = 6; suit >= 0; suit--) { // Start from 6 for tie-breaking
      const count = suitCounts[suit] || 0;
      if (count > maxCount) {
        maxCount = count;
        bestSuit = suit;
      }
    }
  }
  
  // Check if doubles trump would be better (3+ doubles)
  const doubleCount = countDoubles(hand);
  
  return doubleCount >= 3 
    ? { type: 'doubles' as const }
    : { type: 'suit' as const, suit: bestSuit as 0 | 1 | 2 | 3 | 4 | 5 | 6 };
}

