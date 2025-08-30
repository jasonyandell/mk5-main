/**
 * Hand Strength Calculation Module
 * 
 * Shared utilities for evaluating hand strength for bidding decisions.
 * Used by both AI strategies and testing scripts.
 */

import type { Domino, GameState, TrumpSelection, SuitAnalysis } from '../types';
import { NO_BIDDER, NO_LEAD_SUIT } from '../types';
import { analyzeHand } from './utilities';
import { getStrongestSuits } from '../core/suit-analysis';
import { countDoubles } from '../core/dominoes';

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
  BID_31: 75,
  BID_32: 100,  // Adjusted to get better hands bidding 32
  BID_33: 125,  // Raised to be more selective (target ~60%)
  BID_34: 145,  // Raised to be more selective (target ~60%)
  BID_35: 160   // Raised slightly for consistency
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
        if (domino.high === suit || domino.low === suit) {
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
  const analyzingPlayerId = 0;
  const minimalState: GameState = {
    phase: 'playing',
    players: [
      { id: 0, name: 'Analyzer', hand, teamId: 0, marks: 0 },
      { id: 1, name: 'Opponent1', hand: [], teamId: 1, marks: 0 },
      { id: 2, name: 'Partner', hand: [], teamId: 0, marks: 0 },
      { id: 3, name: 'Opponent2', hand: [], teamId: 1, marks: 0 }
    ],
    currentPlayer: analyzingPlayerId,
    dealer: 0,
    bids: [],
    currentBid: { type: 'pass', player: NO_BIDDER },
    winningBidder: NO_BIDDER,
    trump,
    tricks: [],
    currentTrick: [],
    currentSuit: NO_LEAD_SUIT,
    teamScores: [0, 0],
    teamMarks: [0, 0],
    gameTarget: 250,
    tournamentMode: false,
    shuffleSeed: 0,
    playerTypes: ['human', 'ai', 'ai', 'ai'],
    consensus: { completeTrick: new Set(), scoreHand: new Set() },
    actionHistory: [],
    aiSchedule: {},
    currentTick: 0
  };
  
  const analysis = analyzeHand(minimalState, analyzingPlayerId);
  
  // NON-LINEAR HAND STRENGTH CALCULATION
  
  // 1. CONTROL FACTOR
  const unbeatables = analysis.dominoes.filter(d => d.beatenBy && d.beatenBy.length === 0);
  
  // LAYDOWN DETECTION - if all 7 dominoes are unbeatable, it's a guaranteed win
  if (unbeatables.length === 7) {
    return LAYDOWN_SCORE;
  }
  
  const controlFactor = unbeatables.length === 0 ? 0 :
                        unbeatables.length === 1 ? 5 :
                        unbeatables.length === 2 ? 12 :
                        unbeatables.length === 3 ? 20 :
                        unbeatables.length === 4 ? 30 :
                        unbeatables.length === 5 ? 42 :
                        55; // 6 unbeatables (near-laydown)
  
  // 2. TRUMP QUALITY
  const trumpDominoes = analysis.dominoes.filter(d => d.isTrump);
  const trumpCount = trumpDominoes.length;
  
  const hasTopTrump = trump.type === 'doubles' 
    ? hand.some(d => d.high === 6 && d.low === 6)
    : trump.type === 'suit' && hand.some(d => 
        (d.high === trump.suit && d.low === trump.suit));
  
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
  
  // 3. COUNT SAFETY
  let countSafety = 0;
  const countingDominoes = analysis.dominoes.filter(d => d.points > 0);
  for (const counter of countingDominoes) {
    if (counter.isTrump) {
      countSafety += counter.points * 0.8;
    } else if (counter.beatenBy !== undefined && counter.beatenBy.length <= 2) {
      countSafety += counter.points * 0.5;
    } else {
      countSafety += counter.points * 0.2;
    }
  }
  
  // 4. DEFENSIVE STRENGTH
  const offSuitDoubles = hand.filter(d => 
    d.high === d.low && 
    (trump.type !== 'doubles' && 
     (trump.type !== 'suit' || (d.high !== trump.suit)))
  );
  const defensiveFactor = offSuitDoubles.length * 4 + 
                         offSuitDoubles.filter(d => d.high >= 5).length * 3;
  
  // 5. SYNERGY MULTIPLIERS
  let synergyMultiplier = 1.0;
  if (controlFactor > 10 && trumpFactor > 10) {
    synergyMultiplier *= 1.15;
  }
  if (unbeatables.length >= 2) {
    synergyMultiplier *= 1.1;
  }
  if (hasTopTrump && trumpCount >= 4) {
    synergyMultiplier *= 1.15;
  }
  
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
  const minimalState: GameState = {
    phase: 'playing',
    players: [
      { id: 0, name: 'Analyzer', hand, teamId: 0, marks: 0 },
      { id: 1, name: 'Opponent1', hand: [], teamId: 1, marks: 0 },
      { id: 2, name: 'Partner', hand: [], teamId: 0, marks: 0 },
      { id: 3, name: 'Opponent2', hand: [], teamId: 1, marks: 0 }
    ],
    currentPlayer: 0,
    dealer: 0,
    bids: [],
    currentBid: { type: 'pass', player: NO_BIDDER },
    winningBidder: NO_BIDDER,
    trump,
    tricks: [],
    currentTrick: [],
    currentSuit: NO_LEAD_SUIT,
    teamScores: [0, 0],
    teamMarks: [0, 0],
    gameTarget: 250,
    tournamentMode: false,
    shuffleSeed: 0,
    playerTypes: ['human', 'ai', 'ai', 'ai'],
    consensus: { completeTrick: new Set(), scoreHand: new Set() },
    actionHistory: [],
    aiSchedule: {},
    currentTick: 0
  };
  
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
  
  // Calculate all components (same logic as main function)
  const controlFactor = unbeatables.length === 0 ? 0 :
                        unbeatables.length === 1 ? 5 :
                        unbeatables.length === 2 ? 12 :
                        unbeatables.length === 3 ? 20 :
                        unbeatables.length === 4 ? 30 :
                        unbeatables.length === 5 ? 42 :
                        55;
  
  const trumpDominoes = analysis.dominoes.filter(d => d.isTrump);
  const trumpCount = trumpDominoes.length;
  const hasTopTrump = trump.type === 'doubles' 
    ? hand.some(d => d.high === 6 && d.low === 6)
    : trump.type === 'suit' && hand.some(d => 
        (d.high === trump.suit && d.low === trump.suit));
  
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
  
  let countSafety = 0;
  const countingDominoes = analysis.dominoes.filter(d => d.points > 0);
  for (const counter of countingDominoes) {
    if (counter.isTrump) {
      countSafety += counter.points * 0.8;
    } else if (counter.beatenBy !== undefined && counter.beatenBy.length <= 2) {
      countSafety += counter.points * 0.5;
    } else {
      countSafety += counter.points * 0.2;
    }
  }
  
  const offSuitDoubles = hand.filter(d => 
    d.high === d.low && 
    (trump.type !== 'doubles' && 
     (trump.type !== 'suit' || (d.high !== trump.suit)))
  );
  const defensiveFactor = offSuitDoubles.length * 4 + 
                         offSuitDoubles.filter(d => d.high >= 5).length * 3;
  
  let synergyMultiplier = 1.0;
  if (controlFactor > 10 && trumpFactor > 10) {
    synergyMultiplier *= 1.15;
  }
  if (unbeatables.length >= 2) {
    synergyMultiplier *= 1.1;
  }
  if (hasTopTrump && trumpCount >= 4) {
    synergyMultiplier *= 1.15;
  }
  
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