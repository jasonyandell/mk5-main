/**
 * Hand Strength Calculation Module
 * 
 * Shared utilities for evaluating hand strength for bidding decisions.
 * Used by both AI strategies and testing scripts.
 */

import type { Domino, TrumpSelection, SuitAnalysis } from '../types';
import { analyzeHand, createMinimalAnalysisState } from './utilities';
import { getStrongestSuits } from '../core/suit-analysis';
import { countDoubles, dominoHasSuit } from '../core/dominoes';
import {
  calculateControlFactor,
  calculateTrumpFactor,
  calculateCountSafety,
  calculateDefensiveFactor,
  calculateSynergyMultiplier,
  hasTopTrumpDomino,
  getOffSuitDoubles
} from './hand-strength-components';

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

/**
 * Calculate the strength of a hand for bidding purposes
 * 
 * @param hand The dominoes in the player's hand
 * @param suitAnalysis Optional suit analysis data
 * @param customTrump Optional specific trump to evaluate with (otherwise auto-detects best)
 * @returns Strength score (0-150 for normal hands, 999 for laydowns)
 */
export function calculateHandStrengthWithTrump(
  hand: Domino[],
  suitAnalysis?: SuitAnalysis,
  customTrump?: TrumpSelection
): number {
  // Determine trump
  const trump = customTrump || determineBestTrump(hand, suitAnalysis);

  // Create minimal state for analysis
  const minimalState = createMinimalAnalysisState(hand, trump, 0);
  const analysis = analyzeHand(minimalState, 0);

  // NON-LINEAR HAND STRENGTH CALCULATION

  // 1. CONTROL FACTOR
  const unbeatables = analysis.dominoes.filter(d => d.beatenBy && d.beatenBy.length === 0);

  // LAYDOWN DETECTION - if all 7 dominoes are unbeatable, it's a guaranteed win
  if (unbeatables.length === 7) {
    return LAYDOWN_SCORE;
  }

  const controlFactor = calculateControlFactor(unbeatables.length);

  // 2. TRUMP QUALITY
  const trumpDominoes = analysis.dominoes.filter(d => d.isTrump);
  const trumpCount = trumpDominoes.length;
  const hasTopTrump = hasTopTrumpDomino(hand, trump);
  const trumpFactor = calculateTrumpFactor(trumpCount, hasTopTrump);

  // 3. COUNT SAFETY
  const countingDominoes = analysis.dominoes.filter(d => d.points > 0);
  const countSafety = calculateCountSafety(countingDominoes);

  // 4. DEFENSIVE STRENGTH
  const offSuitDoubles = getOffSuitDoubles(hand, trump);
  const defensiveFactor = calculateDefensiveFactor(offSuitDoubles);

  // 5. SYNERGY MULTIPLIERS
  const synergyMultiplier = calculateSynergyMultiplier({
    controlFactor,
    trumpFactor,
    unbeatableCount: unbeatables.length,
    hasTopTrump,
    trumpCount
  });

  // Calculate final strength
  const baseStrength = controlFactor + trumpFactor + countSafety + defensiveFactor;
  const finalStrength = baseStrength * synergyMultiplier;

  return Math.floor(finalStrength);
}

/**
 * Get detailed strength breakdown for debugging/analysis
 */
export function getHandStrengthBreakdown(
  hand: Domino[],
  suitAnalysis?: SuitAnalysis,
  customTrump?: TrumpSelection
): {
  totalStrength: number;
  trump: TrumpSelection;
  unbeatables: number;
  controlFactor: number;
  trumpFactor: number;
  countSafety: number;
  defensiveFactor: number;
  synergyMultiplier: number;
  isLaydown: boolean;
} {
  const trump = customTrump || determineBestTrump(hand, suitAnalysis);

  // Create minimal state for analysis
  const minimalState = createMinimalAnalysisState(hand, trump, 0);
  const analysis = analyzeHand(minimalState, 0);
  const unbeatables = analysis.dominoes.filter(d => d.beatenBy && d.beatenBy.length === 0);

  if (unbeatables.length === 7) {
    return {
      totalStrength: LAYDOWN_SCORE,
      trump,
      unbeatables: 7,
      controlFactor: 999,
      trumpFactor: 999,
      countSafety: 999,
      defensiveFactor: 999,
      synergyMultiplier: 1,
      isLaydown: true
    };
  }

  // Calculate all components using extracted utilities
  const controlFactor = calculateControlFactor(unbeatables.length);

  const trumpDominoes = analysis.dominoes.filter(d => d.isTrump);
  const trumpCount = trumpDominoes.length;
  const hasTopTrump = hasTopTrumpDomino(hand, trump);
  const trumpFactor = calculateTrumpFactor(trumpCount, hasTopTrump);

  const countingDominoes = analysis.dominoes.filter(d => d.points > 0);
  const countSafety = calculateCountSafety(countingDominoes);

  const offSuitDoubles = getOffSuitDoubles(hand, trump);
  const defensiveFactor = calculateDefensiveFactor(offSuitDoubles);

  const synergyMultiplier = calculateSynergyMultiplier({
    controlFactor,
    trumpFactor,
    unbeatableCount: unbeatables.length,
    hasTopTrump,
    trumpCount
  });

  const baseStrength = controlFactor + trumpFactor + countSafety + defensiveFactor;

  return {
    totalStrength: Math.floor(baseStrength * synergyMultiplier),
    trump,
    unbeatables: unbeatables.length,
    controlFactor,
    trumpFactor: Math.floor(trumpFactor),
    countSafety: Math.floor(countSafety),
    defensiveFactor,
    synergyMultiplier,
    isLaydown: false
  };
}