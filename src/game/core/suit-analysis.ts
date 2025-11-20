import type { Domino, TrumpSelection } from '../types';
import { DOUBLES_AS_TRUMP, TRUMP_NOT_SELECTED } from '../types';
import { getTrumpSuit } from './dominoes';
import { getPlayedDominoesFromTricks } from './domino-tracking';

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
  
  if (trumpSuit === DOUBLES_AS_TRUMP) {
    // Doubles trump - count all doubles
    return hand.filter(d => d.high === d.low).length;
  } else if (trumpSuit >= 0 && trumpSuit <= 6) {
    // Regular suit trump - count dominoes containing that number
    return hand.filter(d => d.high === trumpSuit || d.low === trumpSuit).length;
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
  
  if (trumpSuit === DOUBLES_AS_TRUMP) {
    // Doubles trump - all doubles are trump, sorted highest to lowest
    trumpDominoes = hand.filter(d => d.high === d.low);
    trumpDominoes.sort((a, b) => b.high - a.high);
  } else if (trumpSuit >= 0 && trumpSuit <= 6) {
    // Regular suit trump - dominoes containing that number
    trumpDominoes = hand.filter(d => d.high === trumpSuit || d.low === trumpSuit);
    
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

/**
 * Lead analysis result - categorizes dominoes for leading decisions
 */
export interface LeadAnalysis {
  goodLeads: Domino[];  // Guaranteed to win if led
  ranked: Domino[];     // All dominoes ranked by strength
}

// Removed: duplicated logic now in domino-tracking.ts
// Use getPlayedDominoesFromTricks() from './domino-tracking' instead

/**
 * Checks if a domino is the highest unplayed in its suit
 */
function isHighestUnplayed(domino: Domino, suit: number, played: Set<string>): boolean {
  // Check all possible dominoes in this suit, from highest to lowest
  // We need to check in order of actual domino values
  
  // First check doubles (they're highest in their suit)
  if (suit >= 0 && suit <= 6) {
    const doubleDomino = `${suit}-${suit}`;
    if (!played.has(doubleDomino)) {
      // This double is unplayed
      return domino.id.toString() === doubleDomino;
    }
  }
  
  // Then check non-doubles containing this suit, from highest total to lowest
  for (let total = 12; total >= 0; total--) {
    for (let high = 6; high >= 0; high--) {
      const low = total - high;
      if (low < 0 || low > 6 || low > high) continue;
      
      // Check if this domino contains our suit
      if (high !== suit && low !== suit) continue;
      
      const testId = `${high}-${low}`;
      
      // Skip if already played
      if (played.has(testId)) continue;
      
      // If we found our domino, it's the highest unplayed
      if (testId === domino.id.toString()) {
        return true;
      }
      
      // Found a higher unplayed domino in this suit
      return false;
    }
  }
  
  return false;
}

/**
 * Analyzes a hand to identify good leads (guaranteed winners) vs ranked dominoes
 * 
 * Good leads are dominoes that are guaranteed to win if led:
 * - The high trump (always wins)
 * - Additional trump if you have enough (e.g., with 6-6, 6-5, 6-4, the 6-4 is also good)
 * - When no trump remains, the highest unplayed in any suit
 */
export function analyzeLeads(
  hand: Domino[], 
  state: { 
    trump: TrumpSelection;
    tricks: Array<{ plays: Array<{ domino: Domino }> }>;
  },
  trumpCount?: number  // Optional: pre-calculated trump count in hand
): LeadAnalysis {
  const goodLeads: Domino[] = [];
  const played = getPlayedDominoesFromTricks(state.tricks);
  
  const trumpSuit = getTrumpSuit(state.trump);
  
  // 1. Identify trump good leads
  if (trumpSuit !== TRUMP_NOT_SELECTED) {
    // Find all trump in our hand
    const trumpInHand = hand.filter(d => {
      if (trumpSuit === DOUBLES_AS_TRUMP) {
        // Doubles trump
        return d.high === d.low;
      } else if (trumpSuit >= 0 && trumpSuit <= 6) {
        // Regular suit trump
        return d.high === trumpSuit || d.low === trumpSuit;
      }
      return false;
    });
    
    // Sort trump by value (highest first)
    trumpInHand.sort((a, b) => {
      if (trumpSuit === DOUBLES_AS_TRUMP) {
        // Doubles trump - sort by pip value
        return b.high - a.high;
      } else {
        // Regular trump - doubles first, then by total
        const aIsDouble = a.high === a.low;
        const bIsDouble = b.high === b.low;
        if (aIsDouble && !bIsDouble) return -1;
        if (!aIsDouble && bIsDouble) return 1;
        return (b.high + b.low) - (a.high + a.low);
      }
    });
    
    // High trump is always a good lead
    if (trumpInHand.length > 0) {
      goodLeads.push(trumpInHand[0]!);
      
      // If we have 3+ trump including the high, lower trump are also good leads
      // because we can run trump and opponents will run out
      if (trumpInHand.length >= 3) {
        // Check if we have the highest trump
        let haveHighestTrump = false;
        
        if (trumpSuit === DOUBLES_AS_TRUMP) {
          // For doubles trump, check if we have 6-6
          haveHighestTrump = trumpInHand[0]!.high === 6 && trumpInHand[0]!.low === 6;
        } else {
          // For regular trump, check if we have the double of that suit
          haveHighestTrump = trumpInHand[0]!.high === trumpSuit && trumpInHand[0]!.low === trumpSuit;
        }
        
        if (haveHighestTrump) {
          // Add more trump as good leads (they'll win after we run trump)
          // Add up to 2 more (so we have 3 good trump leads total)
          for (let i = 1; i < Math.min(3, trumpInHand.length); i++) {
            goodLeads.push(trumpInHand[i]!);
          }
        }
      }
    }
    
    // 2. Check for non-trump good leads (only if all trump is accounted for)
    const totalTrump = 7; // There are always 7 trump dominoes
    const trumpPlayed = countPlayedTrump(state.tricks, state.trump);
    const ourTrump = trumpCount ?? trumpInHand.length;
    const trumpOutThere = totalTrump - trumpPlayed - ourTrump;
    
    if (trumpOutThere === 0) {
      // No trump remains - highest in each suit is a good lead
      for (const domino of hand) {
        // Skip if already marked as good lead (trump)
        if (goodLeads.some(gl => gl.id === domino.id)) continue;
        
        // Check each suit this domino belongs to
        const suits = domino.high === domino.low ? [domino.high] : [domino.high, domino.low];
        
        for (const suit of suits) {
          if (isHighestUnplayed(domino, suit, played)) {
            goodLeads.push(domino);
            break; // Only add once even if good in multiple suits
          }
        }
      }
    }
  }
  
  // 3. Rank all dominoes by strength (for general play decisions)
  // This includes both good leads and other dominoes
  const ranked = [...hand].sort((a, b) => {
    // Trump dominoes first
    const aIsTrump = trumpSuit !== TRUMP_NOT_SELECTED && (
      trumpSuit === DOUBLES_AS_TRUMP ? a.high === a.low : 
      (a.high === trumpSuit || a.low === trumpSuit)
    );
    const bIsTrump = trumpSuit !== TRUMP_NOT_SELECTED && (
      trumpSuit === DOUBLES_AS_TRUMP ? b.high === b.low : 
      (b.high === trumpSuit || b.low === trumpSuit)
    );
    
    if (aIsTrump && !bIsTrump) return -1;
    if (!aIsTrump && bIsTrump) return 1;
    
    // Within trump or non-trump, sort by value
    if (aIsTrump && bIsTrump) {
      if (trumpSuit === DOUBLES_AS_TRUMP) {
        // Doubles trump - by pip value
        return b.high - a.high;
      } else {
        // Regular trump - doubles first, then by total
        const aIsDouble = a.high === a.low;
        const bIsDouble = b.high === b.low;
        if (aIsDouble && !bIsDouble) return -1;
        if (!aIsDouble && bIsDouble) return 1;
        return (b.high + b.low) - (a.high + a.low);
      }
    }
    
    // Non-trump: doubles first, then by total
    const aIsDouble = a.high === a.low;
    const bIsDouble = b.high === b.low;
    if (aIsDouble && !bIsDouble) return -1;
    if (!aIsDouble && bIsDouble) return 1;
    
    const totalA = a.high + a.low;
    const totalB = b.high + b.low;
    if (totalA !== totalB) return totalB - totalA;
    return b.high - a.high;
  });
  
  return { goodLeads, ranked };
}

/**
 * Counts how many trump dominoes have been played
 */
function countPlayedTrump(tricks: Array<{ plays: Array<{ domino: Domino }> }>, trump: TrumpSelection): number {
  const trumpSuit = getTrumpSuit(trump);
  if (trumpSuit === TRUMP_NOT_SELECTED) return 0;
  
  let trumpPlayed = 0;
  
  for (const trick of tricks) {
    for (const play of trick.plays) {
      if (trumpSuit === DOUBLES_AS_TRUMP) {
        // Doubles trump
        if (play.domino.high === play.domino.low) {
          trumpPlayed++;
        }
      } else if (trumpSuit >= 0 && trumpSuit <= 6) {
        // Regular suit trump
        if (play.domino.high === trumpSuit || play.domino.low === trumpSuit) {
          trumpPlayed++;
        }
      }
    }
  }
  
  return trumpPlayed;
}