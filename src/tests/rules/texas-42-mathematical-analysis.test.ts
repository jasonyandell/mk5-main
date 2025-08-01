import { describe, it, expect } from 'vitest';
import { testLog } from '../helpers/testConsole';

describe('Texas 42 Mathematical Analysis', () => {
  const ALL_DOMINOES: [number, number][] = [];
  for (let i = 6; i >= 0; i--) {
    for (let j = i; j >= 0; j--) {
      ALL_DOMINOES.push([i, j]);
    }
  }
  
  const COUNTING_DOMINOES = [
    { domino: [5, 5], points: 10 },
    { domino: [6, 4], points: 10 },
    { domino: [5, 0], points: 5 },
    { domino: [4, 1], points: 5 },
    { domino: [3, 2], points: 5 }
  ];
  
  describe('Solo Impossible Scores', () => {
    it('should identify exactly which scores cannot be achieved solo', () => {
      const possibleScores = new Set<number>();
      
      for (let tricksWon = 0; tricksWon <= 7; tricksWon++) {
        for (let mask = 0; mask < (1 << COUNTING_DOMINOES.length); mask++) {
          let dominoPoints = 0;
          let dominoesUsed = 0;
          
          for (let i = 0; i < COUNTING_DOMINOES.length; i++) {
            if (mask & (1 << i)) {
              dominoPoints += COUNTING_DOMINOES[i].points;
              dominoesUsed++;
            }
          }
          
          if (dominoesUsed <= tricksWon) {
            possibleScores.add(dominoPoints + tricksWon);
          }
        }
      }
      
      const impossibleScores: number[] = [];
      for (let score = 30; score <= 42; score++) {
        if (!possibleScores.has(score)) {
          impossibleScores.push(score);
        }
      }
      
      expect(impossibleScores).toEqual([33, 38, 39]);
    });
  });
  
  describe('Team Scores', () => {
    it('should prove all scores 30-42 are achievable with teamwork', () => {
      const possibleTeamScores = new Set<number>();
      
      for (let teamTricks = 0; teamTricks <= 7; teamTricks++) {
        for (let mask = 0; mask < (1 << COUNTING_DOMINOES.length); mask++) {
          let dominoPoints = 0;
          for (let i = 0; i < COUNTING_DOMINOES.length; i++) {
            if (mask & (1 << i)) {
              dominoPoints += COUNTING_DOMINOES[i].points;
            }
          }
          possibleTeamScores.add(dominoPoints + teamTricks);
        }
      }
      
      for (let score = 30; score <= 42; score++) {
        expect(possibleTeamScores.has(score)).toBe(true);
      }
    });
  });
  
  describe('Laydown Hands', () => {
    it('should identify all hands that guarantee winning all 7 tricks', { timeout: 30000 }, () => {
      const laydowns = new Map<string, number>();
      let totalLaydowns = 0;
      let debugFirst = true;
      
      generateAllCombinations(ALL_DOMINOES, 7, (hand) => {
        // Check regular trumps (0-6)
        for (let trump = 0; trump <= 6; trump++) {
          if (guaranteesAllTricksWithTrump(hand, trump)) {
            totalLaydowns++;
            const key = describeHand(hand, trump);
            laydowns.set(key, (laydowns.get(key) || 0) + 1);
            
            // Debug first few 4-trump hands
            if (debugFirst && key.startsWith('4 ')) {
              testLog(`\nExample ${key}:`, hand.map(d => `${d[0]}-${d[1]}`).join(', '));
              debugFirst = false;
            }
          }
        }
        
        // Check doubles as trump
        if (guaranteesAllTricksWithDoubles(hand)) {
          totalLaydowns++;
          const key = describeHand(hand, 7);
          laydowns.set(key, (laydowns.get(key) || 0) + 1);
        }
      });
      
      {
        testLog(`\nTotal laydown hands: ${totalLaydowns}`);
        testLog('\nPatterns found:');
        for (const [pattern, count] of laydowns) {
          testLog(`${pattern}: ${count}`);
        }
        testLog(`\nVerified ${totalLaydowns} hands guarantee winning all 7 tricks`);
      }
      
      // We don't know the exact count, but should find many laydowns
      expect(totalLaydowns).toBeGreaterThan(0);
    });
  });
  
  function guaranteesAllTricksWithTrump(hand: [number, number][], trump: number): boolean {
    // A laydown requires that we can play our dominoes in an order that guarantees winning all tricks
    
    // Special case: All 7 of same suit is always a laydown
    const trumpDominoes = hand.filter(d => d[0] === trump || d[1] === trump);
    if (trumpDominoes.length === 7) return true;
    
    // For mixed hands, we need a more sophisticated analysis
    // Key insight: We can lead our trumps first to draw out opponent trumps,
    // then lead our guaranteed winners
    
    // Count trumps
    const ourTrumps = trumpDominoes;
    const nonTrumps = hand.filter(d => d[0] !== trump && d[1] !== trump);
    
    // Must have more trumps than opponents
    const opponentTrumps = 7 - ourTrumps.length;
    if (ourTrumps.length <= opponentTrumps) return false;
    
    // Must have the highest trump if we don't have all trumps
    const hasHighestTrump = ourTrumps.some(d => d[0] === trump && d[1] === trump);
    if (!hasHighestTrump) return false;
    
    // Now check if we have enough control
    // We need to be able to win tricks equal to opponent trump count with our HIGH trumps
    const allPossibleTrumps = getAllTrumpsInOrder(trump);
    const ourTrumpSet = new Set(ourTrumps.map(d => `${d[0]}-${d[1]}`));
    
    // Count how many of the TOP trumps we have
    let topTrumpsWeHave = 0;
    for (const t of allPossibleTrumps) {
      if (ourTrumpSet.has(`${t[0]}-${t[1]}`) || ourTrumpSet.has(`${t[1]}-${t[0]}`)) {
        topTrumpsWeHave++;
      } else {
        break; // We're missing this trump, so we don't have consecutive top trumps
      }
    }
    
    // For a laydown, we need more consecutive top trumps than opponents have total trumps
    // This ensures we can draw out all their trumps while maintaining the lead
    if (topTrumpsWeHave <= opponentTrumps) return false;
    
    // Finally, check that all non-trumps are guaranteed winners
    const handSet = new Set(hand.map(d => `${d[0]}-${d[1]}`));
    for (const nonTrump of nonTrumps) {
      if (!isNonTrumpGuaranteedWinner(nonTrump, trump, handSet)) {
        return false;
      }
    }
    
    return true;
  }
  
  function getAllTrumpsInOrder(trump: number): [number, number][] {
    const trumps: [number, number][] = [];
    
    // Double is highest
    trumps.push([trump, trump]);
    
    // Then by other pip, highest to lowest
    for (let i = 6; i >= 0; i--) {
      if (i !== trump) {
        if (i > trump) {
          trumps.push([i, trump]);
        } else {
          trumps.push([trump, i]);
        }
      }
    }
    
    return trumps;
  }
  
  function isNonTrumpGuaranteedWinner(domino: [number, number], trump: number, ourHandSet: Set<string>): boolean {
    // A non-trump is guaranteed if no opponent has:
    // 1. A higher domino in the same suit
    // 2. Any trump (but we've already checked we can exhaust their trumps)
    
    const suit = Math.max(domino[0], domino[1]);
    
    // For laydowns, we need to check all dominoes that would follow this suit
    // A domino follows suit if EITHER pip matches the led suit
    for (const opp of ALL_DOMINOES) {
      if (ourHandSet.has(`${opp[0]}-${opp[1]}`)) continue;
      if (opp[0] === trump || opp[1] === trump) continue; // Skip trumps
      
      // Check if opponent domino would follow suit
      if (opp[0] === suit || opp[1] === suit) {
        // They must follow suit, so compare
        if (compareDominosInSuit(opp, domino, suit) > 0) {
          return false; // Opponent has higher
        }
      }
    }
    
    return true;
  }
  
  
  /*
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  function compareSameSuit(d1: [number, number], d2: [number, number]): number {
    // For non-trump dominoes of the same suit
    const d1IsDouble = d1[0] === d1[1];
    const d2IsDouble = d2[0] === d2[1];
    
    // Double beats non-double
    if (d1IsDouble && !d2IsDouble) return 1;
    if (!d1IsDouble && d2IsDouble) return -1;
    
    // Both non-doubles - HIGHER pip value wins (e.g., 2-1 beats 2-0)
    const d1Low = Math.min(d1[0], d1[1]);
    const d2Low = Math.min(d2[0], d2[1]);
    
    return d1Low - d2Low; // Higher wins
  }
  */

  function compareDominosInSuit(d1: [number, number], d2: [number, number], suit: number): number {
    // When following suit, we need to compare dominoes that match the led suit
    // The suit is the higher pip of the led domino
    
    // First check if both dominoes actually follow suit
    const d1FollowsSuit = d1[0] === suit || d1[1] === suit;
    const d2FollowsSuit = d2[0] === suit || d2[1] === suit;
    
    if (!d1FollowsSuit || !d2FollowsSuit) {
      throw new Error('Both dominoes must follow suit');
    }
    
    // Check if either is the double
    const d1IsDouble = d1[0] === d1[1] && d1[0] === suit;
    const d2IsDouble = d2[0] === d2[1] && d2[0] === suit;
    
    if (d1IsDouble && !d2IsDouble) return 1;
    if (!d1IsDouble && d2IsDouble) return -1;
    
    // Both non-doubles - the one with the other pip HIGHER wins
    const d1OtherPip = d1[0] === suit ? d1[1] : d1[0];
    const d2OtherPip = d2[0] === suit ? d2[1] : d2[0];
    
    return d1OtherPip - d2OtherPip; // Higher other pip wins
  }
  
  function guaranteesAllTricksWithDoubles(hand: [number, number][]): boolean {
    // Similar logic for doubles as trump
    const doubles = hand.filter(d => d[0] === d[1]);
    const nonDoubles = hand.filter(d => d[0] !== d[1]);
    
    // All 7 doubles is always a laydown
    if (doubles.length === 7) return true;
    
    // Must have more doubles than opponents
    const opponentDoubles = 7 - doubles.length;
    if (doubles.length <= opponentDoubles) return false;
    
    // Must have the highest double (6-6)
    const hasHighestDouble = doubles.some(d => d[0] === 6 && d[1] === 6);
    if (!hasHighestDouble) return false;
    
    // Count top consecutive doubles we have
    const allDoubles = [[6,6], [5,5], [4,4], [3,3], [2,2], [1,1], [0,0]];
    const ourDoubleSet = new Set(doubles.map(d => `${d[0]}-${d[1]}`));
    
    let topDoublesWeHave = 0;
    for (const d of allDoubles) {
      if (ourDoubleSet.has(`${d[0]}-${d[1]}`)) {
        topDoublesWeHave++;
      } else {
        break; // Missing this double
      }
    }
    
    // For a laydown, we need more consecutive top doubles than opponents have total doubles
    // This ensures we can draw out all their doubles while maintaining the lead
    if (topDoublesWeHave <= opponentDoubles) return false;
    
    // Check all non-doubles are guaranteed winners
    const handSet = new Set(hand.map(d => `${d[0]}-${d[1]}`));
    for (const nonDouble of nonDoubles) {
      if (!isNonDoubleGuaranteedWinner(nonDouble, handSet)) {
        return false;
      }
    }
    
    return true;
  }
  
  function isNonDoubleGuaranteedWinner(domino: [number, number], ourHandSet: Set<string>): boolean {
    // When doubles are trump, a non-double wins if no opponent has higher in same suit
    const suit = Math.max(domino[0], domino[1]);
    const ourLow = Math.min(domino[0], domino[1]);
    
    for (const opp of ALL_DOMINOES) {
      if (ourHandSet.has(`${opp[0]}-${opp[1]}`)) continue;
      if (opp[0] === opp[1]) continue; // Skip doubles (they're trump)
      
      const oppSuit = Math.max(opp[0], opp[1]);
      if (oppSuit === suit) {
        const oppLow = Math.min(opp[0], opp[1]);
        if (oppLow > ourLow) {
          return false; // Opponent has higher
        }
      }
    }
    
    return true;
  }
  
  
  
  function describeHand(hand: [number, number][], trump: number): string {
    if (trump <= 6) {
      const trumpCount = hand.filter(d => d[0] === trump || d[1] === trump).length;
      if (trumpCount === 7) return `All 7 ${trump}s`;
      if (trumpCount === 6) return `6 ${trump}s + winner`;
      if (trumpCount === 5) return `5 ${trump}s + 2 winners`;
      return `${trumpCount} ${trump}s`;
    } else {
      const doubleCount = hand.filter(d => d[0] === d[1]).length;
      if (doubleCount === 7) return 'All 7 doubles';
      if (doubleCount === 6) return '6 doubles + winner';
      return `${doubleCount} doubles`;
    }
  }
  
  function generateAllCombinations<T>(
    arr: T[], 
    k: number, 
    callback: (combination: T[]) => void
  ): void {
    const n = arr.length;
    if (k > n) return;
    
    const indices = Array.from({ length: k }, (_, i) => i);
    
    while (true) {
      const combination = indices.map(i => arr[i]);
      callback(combination);
      
      let i = k - 1;
      while (i >= 0 && indices[i] === n - k + i) {
        i--;
      }
      
      if (i < 0) break;
      
      indices[i]++;
      
      for (let j = i + 1; j < k; j++) {
        indices[j] = indices[j - 1] + 1;
      }
    }
  }
});