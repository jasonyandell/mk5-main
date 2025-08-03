import type { Domino, TrumpSelection } from '../types';

/**
 * Suit count for a player's hand - counts dominoes by suit number
 */
export interface SuitCount {
  0: number; // blanks
  1: number; // ones  
  2: number; // twos
  3: number; // threes
  4: number; // fours
  5: number; // fives
  6: number; // sixes
  doubles: number; // count of doubles
  trump: number; // count of trump dominoes (identical to trump suit when trump is declared)
}

/**
 * Suit ranking for a player's hand - lists dominoes by suit, highest to lowest
 */
export interface SuitRanking {
  0: Domino[]; // blanks
  1: Domino[]; // ones
  2: Domino[]; // twos
  3: Domino[]; // threes
  4: Domino[]; // fours
  5: Domino[]; // fives
  6: Domino[]; // sixes
  doubles: Domino[]; // all doubles
  trump: Domino[]; // all trump dominoes (identical to trump suit when trump is declared)
}

/**
 * Complete suit analysis for a player's hand
 */
export interface SuitAnalysis {
  count: SuitCount;
  rank: SuitRanking;
}

/**
 * Calculates suit count for a hand - how many dominoes contain each suit number
 */
export function calculateSuitCount(hand: Domino[], trump: TrumpSelection = { type: 'none' }): SuitCount {
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
  if (trump.type !== 'none') {
    count.trump = calculateTrumpCount(hand, trump);
  }

  return count;
}

/**
 * Calculates how many trump dominoes are in a hand
 */
function calculateTrumpCount(hand: Domino[], trump: TrumpSelection): number {
  
  const numericTrump = trumpToNumber(trump);
  
  if (numericTrump === 7) {
    // Doubles trump - count all doubles
    return hand.filter(d => d.high === d.low).length;
  } else if (numericTrump !== null && numericTrump >= 0 && numericTrump <= 6) {
    // Regular suit trump - count dominoes containing that number
    return hand.filter(d => d.high === numericTrump || d.low === numericTrump).length;
  }
  
  return 0;
}

/**
 * Converts TrumpSelection to numeric value
 */
function trumpToNumber(trump: TrumpSelection): number | null {
  switch (trump.type) {
    case 'none': return null;
    case 'suit': return trump.suit!;
    case 'doubles': return 7;
    case 'no-trump': return 8;
  }
}

/**
 * Calculates suit ranking for a hand - organizes dominoes by suit with highest first
 */
export function calculateSuitRanking(hand: Domino[], trump: TrumpSelection = { type: 'none' }): SuitRanking {
  const rank: SuitRanking = {
    0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], doubles: [], trump: []
  };

  // Separate doubles from non-doubles
  const doubles = hand.filter(d => d.high === d.low);
  const nonDoubles = hand.filter(d => d.high !== d.low);

  // Sort doubles by value (highest first)
  doubles.sort((a, b) => b.high - a.high);
  rank.doubles = doubles;

  // Add doubles to their natural suit rankings
  doubles.forEach(domino => {
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
  if (trump.type !== 'none') {
    rank.trump = calculateTrumpRanking(hand, trump);
  }

  return rank;
}

/**
 * Calculates trump ranking for a hand - organizes trump dominoes by value
 */
function calculateTrumpRanking(hand: Domino[], trump: TrumpSelection): Domino[] {
  const numericTrump = trumpToNumber(trump);
  let trumpDominoes: Domino[] = [];
  
  if (numericTrump === 7) {
    // Doubles trump - all doubles are trump, sorted highest to lowest
    trumpDominoes = hand.filter(d => d.high === d.low);
    trumpDominoes.sort((a, b) => b.high - a.high);
  } else if (numericTrump !== null && numericTrump >= 0 && numericTrump <= 6) {
    // Regular suit trump - dominoes containing that number
    trumpDominoes = hand.filter(d => d.high === numericTrump || d.low === numericTrump);
    
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
export function analyzeSuits(hand: Domino[], trump: TrumpSelection = { type: 'none' }): SuitAnalysis {
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
    
    const highestA = suitA[0];
    const highestB = suitB[0];
    
    const totalA = highestA.high + highestA.low;
    const totalB = highestB.high + highestB.low;
    
    if (totalA !== totalB) return totalB - totalA;
    return highestB.high - highestA.high;
  });
}