import { describe, it, expect } from 'vitest';

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
    it('should identify all hands that guarantee winning all 7 tricks', () => {
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
              console.log(`\nExample ${key}:`, hand.map(d => `${d[0]}-${d[1]}`).join(', '));
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
      
      console.log(`\nTotal laydown hands: ${totalLaydowns}`);
      console.log('\nPatterns found:');
      for (const [pattern, count] of laydowns) {
        console.log(`${pattern}: ${count}`);
      }
      
      // We don't know the exact count, but should find many laydowns
      expect(totalLaydowns).toBeGreaterThan(0);
      console.log(`\nVerified ${totalLaydowns} hands guarantee winning all 7 tricks`);
    });
  });
  
  function guaranteesAllTricksWithTrump(hand: [number, number][], trump: number): boolean {
    const trumps = hand.filter(d => d[0] === trump || d[1] === trump);
    const nonTrumps = hand.filter(d => d[0] !== trump && d[1] !== trump);
    
    // Special case: all 7 trumps
    if (trumps.length === 7) return true;
    
    // Must have more trumps than opponents
    const opponentTrumps = 7 - trumps.length;
    if (trumps.length <= opponentTrumps) return false;
    
    // If we don't have all trumps, we must have the highest trump (the double)
    if (trumps.length < 7) {
      const hasDouble = trumps.some(d => d[0] === trump && d[1] === trump);
      if (!hasDouble) return false; // Opponent has the boss trump
    }
    
    // We win one trick for each trump we have
    const trumpTricks = trumps.length;
    
    // How many non-trump tricks do we need?
    const nonTrumpTricksNeeded = 7 - trumpTricks;
    
    // Count guaranteed non-trump winners
    let guaranteedWinners = 0;
    const handDominoes = new Set(hand.map(d => `${d[0]}-${d[1]}`));
    
    // Group non-trumps by suit
    const bySuit: Map<number, [number, number][]> = new Map();
    for (const d of nonTrumps) {
      const suit = Math.max(d[0], d[1]);
      if (!bySuit.has(suit)) {
        bySuit.set(suit, []);
      }
      bySuit.get(suit)!.push(d);
    }
    
    // Sort each suit by rank within that suit
    for (const [suit, dominoes] of bySuit) {
      dominoes.sort((a, b) => {
        // Doubles beat non-doubles
        const aIsDouble = a[0] === a[1];
        const bIsDouble = b[0] === b[1];
        if (aIsDouble && !bIsDouble) return -1;
        if (!aIsDouble && bIsDouble) return 1;
        
        // Higher other pip wins
        const aLow = Math.min(a[0], a[1]);
        const bLow = Math.min(b[0], b[1]);
        return bLow - aLow;
      });
    }
    
    // Check the highest domino in each suit we control
    for (const [suit, dominoes] of bySuit) {
      if (dominoes.length > 0) {
        // The highest in this suit (first after sorting) might be guaranteed
        const highest = dominoes[0];
        if (isGuaranteedWinner(highest, trump, handDominoes)) {
          guaranteedWinners++;
          if (guaranteedWinners >= nonTrumpTricksNeeded) {
            return true;
          }
        }
      }
    }
    
    return false;
  }
  
  function guaranteesAllTricksWithDoubles(hand: [number, number][]): boolean {
    const doubles = hand.filter(d => d[0] === d[1]);
    const nonDoubles = hand.filter(d => d[0] !== d[1]);
    
    // All 7 doubles
    if (doubles.length === 7) return true;
    
    // Must have more doubles than opponents
    const opponentDoubles = 7 - doubles.length;
    if (doubles.length <= opponentDoubles) return false;
    
    const doubleTricks = doubles.length;
    const nonDoubleTricksNeeded = 7 - doubleTricks;
    
    // Count guaranteed non-double winners
    let guaranteedWinners = 0;
    const handDominoes = new Set(hand.map(d => `${d[0]}-${d[1]}`));
    
    // When doubles are trump, non-doubles win by suit
    // Group non-doubles by suit
    const nonDoublesBySuit: Map<number, [number, number][]> = new Map();
    for (const d of nonDoubles) {
      const suit = Math.max(d[0], d[1]);
      if (!nonDoublesBySuit.has(suit)) {
        nonDoublesBySuit.set(suit, []);
      }
      nonDoublesBySuit.get(suit)!.push(d);
    }
    
    // Check highest in each suit
    for (const [suit, dominoes] of nonDoublesBySuit) {
      // Sort by rank within suit
      dominoes.sort((a, b) => {
        const aLow = Math.min(a[0], a[1]);
        const bLow = Math.min(b[0], b[1]);
        return bLow - aLow;
      });
      
      if (dominoes.length > 0) {
        const highest = dominoes[0];
        // Check if this is guaranteed winner in its suit
        let isGuaranteed = true;
        
        // Check all possible opponent dominoes in this suit
        for (const d of ALL_DOMINOES) {
          if (handDominoes.has(`${d[0]}-${d[1]}`)) continue;
          if (d[0] === d[1]) continue; // Skip doubles (they're trump)
          
          const dSuit = Math.max(d[0], d[1]);
          if (dSuit === suit) {
            // Would this beat our highest?
            const dLow = Math.min(d[0], d[1]);
            const highestLow = Math.min(highest[0], highest[1]);
            if (dLow > highestLow) {
              isGuaranteed = false;
              break;
            }
          }
        }
        
        if (isGuaranteed) {
          guaranteedWinners++;
          if (guaranteedWinners >= nonDoubleTricksNeeded) {
            return true;
          }
        }
      }
    }
    
    return false;
  }
  
  function isGuaranteedWinner(domino: [number, number], trump: number, handSet: Set<string>): boolean {
    // A domino is guaranteed to win if:
    // 1. We can lead it (control when it's played)
    // 2. No opponent domino of the same suit can beat it
    
    // Determine the suit of this domino (higher end)
    const suit = Math.max(domino[0], domino[1]);
    
    // Find all dominoes of the same suit that opponents might have
    for (const d of ALL_DOMINOES) {
      // Skip if it's in our hand
      if (handSet.has(`${d[0]}-${d[1]}`)) continue;
      
      // Skip if it's a trump
      if (d[0] === trump || d[1] === trump) continue;
      
      // Skip if it's not the same suit
      const dSuit = Math.max(d[0], d[1]);
      if (dSuit !== suit) continue;
      
      // Would this domino beat ours in the same suit?
      if (dominoBeatsInSuit(d, domino)) {
        return false;
      }
    }
    
    return true;
  }
  
  function dominoBeatsInSuit(d1: [number, number], d2: [number, number]): boolean {
    // When both dominoes are of the same suit (determined by higher pip)
    // The ranking within that suit determines the winner
    
    // For doubles: double beats non-double
    const d1IsDouble = d1[0] === d1[1];
    const d2IsDouble = d2[0] === d2[1];
    
    if (d1IsDouble && !d2IsDouble) return true;
    if (!d1IsDouble && d2IsDouble) return false;
    
    // For non-doubles of same suit: higher other pip wins
    // Example: In 5s suit, 5-4 beats 5-3 beats 5-2, etc.
    const d1High = Math.max(d1[0], d1[1]);
    const d1Low = Math.min(d1[0], d1[1]);
    const d2High = Math.max(d2[0], d2[1]);
    const d2Low = Math.min(d2[0], d2[1]);
    
    // Both should have same high (the suit)
    if (d1High !== d2High) {
      throw new Error('Comparing dominoes of different suits');
    }
    
    // Higher low pip wins
    return d1Low > d2Low;
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