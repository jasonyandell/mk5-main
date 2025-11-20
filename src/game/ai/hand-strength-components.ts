/**
 * Hand Strength Component Calculators
 *
 * Pure calculation functions for hand strength evaluation.
 * Extracted from hand-strength.ts to eliminate duplication.
 */

import type { Domino, TrumpSelection } from '../types';
import type { DominoAnalysis } from './utilities';

/**
 * Parameters for synergy multiplier calculation
 */
export interface SynergyParams {
  controlFactor: number;
  trumpFactor: number;
  unbeatableCount: number;
  hasTopTrump: boolean;
  trumpCount: number;
}

/**
 * Calculate control factor based on number of unbeatable dominoes
 *
 * @param unbeatableCount Number of dominoes that cannot be beaten (0-7)
 * @returns Control factor score
 */
export function calculateControlFactor(unbeatableCount: number): number {
  return unbeatableCount === 0 ? 0 :
         unbeatableCount === 1 ? 5 :
         unbeatableCount === 2 ? 12 :
         unbeatableCount === 3 ? 20 :
         unbeatableCount === 4 ? 30 :
         unbeatableCount === 5 ? 42 :
         55; // 6 unbeatables (near-laydown)
}

/**
 * Calculate trump factor based on trump count and top trump ownership
 *
 * @param trumpCount Number of trump dominoes in hand
 * @param hasTopTrump Whether hand contains the top trump
 * @returns Trump factor score
 */
export function calculateTrumpFactor(trumpCount: number, hasTopTrump: boolean): number {
  let trumpFactor = 0;
  if (trumpCount > 0) {
    trumpFactor = trumpCount * 2;
    if (trumpCount >= 3) trumpFactor += 3;
    if (trumpCount >= 4) trumpFactor += 5;
    if (trumpCount >= 5) trumpFactor += 8;

    if (hasTopTrump) {
      trumpFactor *= 1.3;
      if (trumpCount >= 3) trumpFactor += 5;
    }
  }
  return trumpFactor;
}

/**
 * Calculate count safety based on counting dominoes and their vulnerability
 *
 * @param countingDominoes Dominoes with counting points and their analysis
 * @returns Count safety score
 */
export function calculateCountSafety(countingDominoes: DominoAnalysis[]): number {
  let countSafety = 0;
  for (const counter of countingDominoes) {
    if (counter.isTrump) {
      countSafety += counter.points * 0.8;
    } else if (counter.beatenBy !== undefined && counter.beatenBy.length <= 2) {
      countSafety += counter.points * 0.5;
    } else {
      countSafety += counter.points * 0.2;
    }
  }
  return countSafety;
}

/**
 * Calculate defensive factor based on off-suit doubles
 *
 * @param offSuitDoubles Doubles that are not trump
 * @returns Defensive factor score
 */
export function calculateDefensiveFactor(offSuitDoubles: Domino[]): number {
  return offSuitDoubles.length * 4 +
         offSuitDoubles.filter(d => d.high >= 5).length * 3;
}

/**
 * Calculate synergy multiplier based on hand characteristics
 *
 * @param params Hand strength components
 * @returns Synergy multiplier (typically 1.0 to ~1.4)
 */
export function calculateSynergyMultiplier(params: SynergyParams): number {
  let synergyMultiplier = 1.0;
  if (params.controlFactor > 10 && params.trumpFactor > 10) {
    synergyMultiplier *= 1.15;
  }
  if (params.unbeatableCount >= 2) {
    synergyMultiplier *= 1.1;
  }
  if (params.hasTopTrump && params.trumpCount >= 4) {
    synergyMultiplier *= 1.15;
  }
  return synergyMultiplier;
}

/**
 * Check if hand contains the top trump domino
 *
 * @param hand Dominoes in hand
 * @param trump Trump selection
 * @returns true if hand contains top trump
 */
export function hasTopTrumpDomino(hand: Domino[], trump: TrumpSelection): boolean {
  return trump.type === 'doubles'
    ? hand.some(d => d.high === 6 && d.low === 6)
    : trump.type === 'suit' && hand.some(d =>
        (d.high === trump.suit && d.low === trump.suit));
}

/**
 * Get all off-suit doubles from a hand
 *
 * @param hand Dominoes in hand
 * @param trump Trump selection
 * @returns Array of doubles that are not trump
 */
export function getOffSuitDoubles(hand: Domino[], trump: TrumpSelection): Domino[] {
  return hand.filter(d =>
    d.high === d.low &&
    (trump.type !== 'doubles' &&
     (trump.type !== 'suit' || (d.high !== trump.suit)))
  );
}
