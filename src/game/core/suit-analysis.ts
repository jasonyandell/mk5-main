import type { Domino, TrumpSelection, SuitCount, SuitRanking, SuitAnalysis } from '../types';
import { DOUBLES_AS_TRUMP } from '../types';
import { getTrumpSuit, isRegularSuitTrump, isDoublesTrump, dominoHasSuit } from './dominoes';

// Types are defined in types.ts - re-export for module consumers
export type { SuitCount, SuitRanking, SuitAnalysis } from '../types';

/**
 * Calculates suit count for a hand - how many dominoes contain each suit number
 */
export function calculateSuitCount(hand: Domino[], trump: TrumpSelection = { type: 'not-selected' }): SuitCount {
  const count: SuitCount = {
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, doubles: 0, trump: 0
  };

  hand.forEach(domino => {
    // Count doubles separately
    if (domino.high === domino.low) {
      count.doubles++;
      // Doubles also count for their natural suit
      (count[domino.high as keyof Omit<SuitCount, 'doubles' | 'trump'>])++;
    } else {
      // Non-double dominoes count for both suits they contain
      (count[domino.high as keyof Omit<SuitCount, 'doubles' | 'trump'>])++;
      (count[domino.low as keyof Omit<SuitCount, 'doubles' | 'trump'>])++;
    }
  });

  // Calculate trump count based on trump suit
  if (trump.type !== 'not-selected') {
    count.trump = calculateTrumpCount(hand, trump);
  }

  return count;
}

/**
 * Calculates how many trump dominoes are in a hand
 */
function calculateTrumpCount(hand: Domino[], trump: TrumpSelection): number {

  const trumpSuit = getTrumpSuit(trump);

  if (isDoublesTrump(trumpSuit)) {
    // Doubles trump - count all doubles
    return hand.filter(d => d.high === d.low).length;
  } else if (isRegularSuitTrump(trumpSuit)) {
    // Regular suit trump - count dominoes containing that number
    return hand.filter(d => dominoHasSuit(d, trumpSuit)).length;
  }

  return 0;
}


/**
 * Calculates suit ranking for a hand - organizes dominoes by suit with highest first
 */
export function calculateSuitRanking(hand: Domino[], trump: TrumpSelection = { type: 'not-selected' }): SuitRanking {
  const rank: SuitRanking = {
    0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], doubles: [], trump: []
  };

  // Separate doubles from non-doubles
  const doubles = hand.filter(d => d.high === d.low);
  const nonDoubles = hand.filter(d => d.high !== d.low);

  // Sort doubles by value (highest first)
  doubles.sort((a, b) => b.high - a.high);
  rank.doubles = doubles;

  // Get trump suit for checking
  const trumpSuit = getTrumpSuit(trump);
  const isDoublesTrump = trumpSuit === DOUBLES_AS_TRUMP;

  // Add doubles to their natural suit rankings
  doubles.forEach(domino => {
    // If doubles are trump, don't add any double to natural suits
    if (isDoublesTrump) {
      return; // Skip adding to natural suit
    }
    rank[domino.high as keyof Omit<SuitRanking, 'doubles' | 'trump'>].push(domino);
  });

  // Process non-doubles - each goes into both suits it contains
  nonDoubles.forEach(domino => {
    rank[domino.high as keyof Omit<SuitRanking, 'doubles' | 'trump'>].push(domino);
    rank[domino.low as keyof Omit<SuitRanking, 'doubles' | 'trump'>].push(domino);
  });

  // Sort each suit's dominoes: doubles first, then by pip total (highest first), then by high value
  for (let suit = 0; suit <= 6; suit++) {
    const suitKey = suit as keyof Omit<SuitRanking, 'doubles' | 'trump'>;
    rank[suitKey].sort((a, b) => {
      const aIsDouble = a.high === a.low;
      const bIsDouble = b.high === b.low;
      
      // Doubles come first
      if (aIsDouble && !bIsDouble) return -1;
      if (!aIsDouble && bIsDouble) return 1;
      
      // If both are doubles or both are non-doubles, sort by pip total
      const totalA = a.high + a.low;
      const totalB = b.high + b.low;
      if (totalA !== totalB) return totalB - totalA;
      return b.high - a.high;
    });
  }

  // Calculate trump ranking
  if (trump.type !== 'not-selected') {
    rank.trump = calculateTrumpRanking(hand, trump);
  }

  return rank;
}

/**
 * Calculates trump ranking for a hand - organizes trump dominoes by value
 */
function calculateTrumpRanking(hand: Domino[], trump: TrumpSelection): Domino[] {
  const trumpSuit = getTrumpSuit(trump);
  let trumpDominoes: Domino[] = [];

  if (isDoublesTrump(trumpSuit)) {
    // Doubles trump - all doubles are trump, sorted highest to lowest
    trumpDominoes = hand.filter(d => d.high === d.low);
    trumpDominoes.sort((a, b) => b.high - a.high);
  } else if (isRegularSuitTrump(trumpSuit)) {
    // Regular suit trump - dominoes containing that number
    trumpDominoes = hand.filter(d => dominoHasSuit(d, trumpSuit));
    
    // Sort trump dominoes by trump value priority:
    // 1. Doubles of trump suit first
    // 2. Then by total pip count (highest first)
    // 3. Then by high value
    trumpDominoes.sort((a, b) => {
      const aIsDouble = a.high === a.low;
      const bIsDouble = b.high === b.low;
      
      // Doubles of trump suit rank highest
      if (aIsDouble && bIsDouble) return b.high - a.high;
      if (aIsDouble && !bIsDouble) return -1;
      if (!aIsDouble && bIsDouble) return 1;
      
      // For non-doubles, sort by total pips then high value
      const totalA = a.high + a.low;
      const totalB = b.high + b.low;
      if (totalA !== totalB) return totalB - totalA;
      return b.high - a.high;
    });
  }
  
  return trumpDominoes;
}

/**
 * Performs complete suit analysis for a hand
 */
export function analyzeSuits(hand: Domino[], trump: TrumpSelection = { type: 'not-selected' }): SuitAnalysis {
  return {
    count: calculateSuitCount(hand, trump),
    rank: calculateSuitRanking(hand, trump)
  };
}

/**
 * Gets the strongest suits for a hand (by count, then by highest domino)
 */
export function getStrongestSuits(analysis: SuitAnalysis): number[] {
  const suits = [0, 1, 2, 3, 4, 5, 6];
  
  return suits.sort((a, b) => {
    const countA = analysis.count[a as keyof Omit<SuitCount, 'doubles' | 'trump'>];
    const countB = analysis.count[b as keyof Omit<SuitCount, 'doubles' | 'trump'>];
    
    // Sort by count first
    if (countA !== countB) return countB - countA;
    
    // If counts equal, sort by highest domino in suit
    const suitA = analysis.rank[a as keyof Omit<SuitRanking, 'doubles' | 'trump'>];
    const suitB = analysis.rank[b as keyof Omit<SuitRanking, 'doubles' | 'trump'>];
    
    if (suitA.length === 0 && suitB.length === 0) return 0;
    if (suitA.length === 0) return 1;
    if (suitB.length === 0) return -1;
    
    const highestA = suitA[0]!;
    const highestB = suitB[0]!;
    
    const totalA = highestA.high + highestA.low;
    const totalB = highestB.high + highestB.low;
    
    if (totalA !== totalB) return totalB - totalA;
    return highestB.high - highestA.high;
  });
}