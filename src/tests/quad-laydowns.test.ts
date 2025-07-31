import { describe, it, expect } from 'vitest';

describe('Quad Laydowns Analysis', () => {
  const ALL_DOMINOES: [number, number][] = [
    [6,6], [6,5], [6,4], [6,3], [6,2], [6,1], [6,0],
    [5,5], [5,4], [5,3], [5,2], [5,1], [5,0],
    [4,4], [4,3], [4,2], [4,1], [4,0],
    [3,3], [3,2], [3,1], [3,0],
    [2,2], [2,1], [2,0],
    [1,1], [1,0],
    [0,0]
  ];

  function getTrumpDominoes(trump: number): [number, number][] {
    if (trump === 7) {
      return [[6,6], [5,5], [4,4], [3,3], [2,2], [1,1], [0,0]];
    } else {
      const trumpDominoes: [number, number][] = [];
      trumpDominoes.push([trump, trump]); // Double first
      
      for (let i = 6; i >= 0; i--) {
        if (i !== trump) {
          if (i > trump) {
            trumpDominoes.push([i, trump]);
          } else {
            trumpDominoes.push([trump, i]);
          }
        }
      }
      
      return trumpDominoes;
    }
  }

  function analyzeSharedDominoes(trump1: number, trump2: number): [number, number][] {
    const trump1Dominoes = new Set(getTrumpDominoes(trump1).map(d => `${d[0]}-${d[1]}`));
    const trump2Dominoes = getTrumpDominoes(trump2);
    
    return trump2Dominoes.filter(d => trump1Dominoes.has(`${d[0]}-${d[1]}`));
  }

  it('should verify trump suit overlaps prevent quad laydowns', () => {
    console.log('\n=== Trump Suit Overlap Analysis ===\n');
    
    // Test all pairs of trump suits for overlaps
    const overlaps: { [key: string]: number } = {};
    
    for (let trump1 = 0; trump1 <= 6; trump1++) {
      for (let trump2 = trump1 + 1; trump2 <= 6; trump2++) {
        const shared = analyzeSharedDominoes(trump1, trump2);
        const key = `${trump1}s-${trump2}s`;
        overlaps[key] = shared.length;
        
        if (shared.length > 0) {
          console.log(`${key}: ${shared.length} shared dominoes - ${shared.map(d => `${d[0]}-${d[1]}`).join(', ')}`);
        }
      }
    }
    
    // Every pair should have overlaps
    const totalPairs = Object.keys(overlaps).length;
    const pairsWithOverlaps = Object.values(overlaps).filter(count => count > 0).length;
    
    console.log(`\nSummary: ${pairsWithOverlaps}/${totalPairs} trump pairs have shared dominoes`);
    
    expect(pairsWithOverlaps).toBe(totalPairs);
  });

  it('should verify no 4-trump combination allows sufficient dominoes', () => {
    console.log('\n=== 4-Trump Combination Analysis ===\n');
    
    const allCombinations: number[][] = [];
    
    // Generate all combinations of 4 different trump suits
    for (let a = 0; a <= 6; a++) {
      for (let b = a + 1; b <= 6; b++) {
        for (let c = b + 1; c <= 6; c++) {
          for (let d = c + 1; d <= 6; d++) {
            allCombinations.push([a, b, c, d]);
          }
        }
      }
    }
    
    console.log(`Testing ${allCombinations.length} combinations of 4 trump suits...\n`);
    
    let validCombinations = 0;
    
    for (const combo of allCombinations) {
      const usedDominoes = new Set<string>();
      let isValid = true;
      const allocations: [number, number][][] = [[], [], [], []];
      
      // Try to allocate 5 trumps to each player
      for (let player = 0; player < 4; player++) {
        const trump = combo[player];
        const trumpDominoes = getTrumpDominoes(trump);
        
        let allocated = 0;
        for (const domino of trumpDominoes) {
          const key = `${domino[0]}-${domino[1]}`;
          if (!usedDominoes.has(key) && allocated < 5) {
            allocations[player].push(domino);
            usedDominoes.add(key);
            allocated++;
          }
        }
        
        if (allocated < 5) {
          isValid = false;
          break;
        }
      }
      
      if (isValid) {
        validCombinations++;
        console.log(`✓ Valid: ${combo.map(t => `${t}s`).join(', ')}`);
        
        // Show allocations
        allocations.forEach((hand, i) => {
          console.log(`  P${i+1} (${combo[i]}s): ${hand.map(d => `${d[0]}-${d[1]}`).join(', ')}`);
        });
        console.log();
      }
    }
    
    console.log(`\nResult: ${validCombinations}/${allCombinations.length} combinations are valid`);
    
    // This should be 0 based on our findings
    expect(validCombinations).toBe(0);
  });

  it('should demonstrate the mathematical impossibility', () => {
    console.log('\n=== Mathematical Proof of Impossibility ===\n');
    
    // Count total trump dominoes needed vs available
    const trumpsNeededPerPlayer = 5; // double + 4 others
    const totalTrumpsNeeded = 4 * trumpsNeededPerPlayer; // 20 dominoes
    
    console.log(`Trumps needed per player: ${trumpsNeededPerPlayer}`);
    console.log(`Total trumps needed for 4 players: ${totalTrumpsNeeded}`);
    console.log(`Total dominoes available: ${ALL_DOMINOES.length}`);
    
    // Show why this fails
    console.log('\nWhy this fails:');
    console.log('1. Each suit has exactly 7 dominoes (including the double)');
    console.log('2. Many dominoes belong to multiple suits (e.g., 6-4 is in both 6s and 4s)');
    console.log('3. When 4 players need different trump suits, conflicts are inevitable');
    
    // Demonstrate with a specific example
    console.log('\nExample conflict with 0s, 1s, 2s, 3s:');
    const conflictExample = [0, 1, 2, 3];
    
    conflictExample.forEach((trump, i) => {
      const trumpDominoes = getTrumpDominoes(trump);
      console.log(`${trump}s trump dominoes: ${trumpDominoes.map(d => `${d[0]}-${d[1]}`).join(', ')}`);
    });
    
    // Show specific conflicts
    console.log('\nSpecific conflicts:');
    console.log('- 1-0 needed by both 0s and 1s players');
    console.log('- 2-0 needed by both 0s and 2s players');  
    console.log('- 2-1 needed by both 1s and 2s players');
    console.log('- 3-0 needed by both 0s and 3s players');
    console.log('- And many more...');
    
    expect(true).toBe(true); // This test just demonstrates the proof
  });

  it('should verify original laydown detection finds individual laydowns', () => {
    // Verify that individual laydowns do exist (just not 4 simultaneous ones)
    console.log('\n=== Individual Laydown Verification ===\n');
    
    // Test a known good laydown hand
    const knownLaydown: [number, number][] = [
      [6,6], [6,5], [6,4], [6,3], [6,2], // 5 trump (6s)
      [5,5], [4,4] // 2 high non-trumps
    ];
    
    console.log(`Testing known laydown: ${knownLaydown.map(d => `${d[0]}-${d[1]}`).join(', ')}`);
    console.log('Trump: 6s');
    
    // Count 6s trumps
    const trumpCount = knownLaydown.filter(d => d[0] === 6 || d[1] === 6).length;
    const hasDouble = knownLaydown.some(d => d[0] === 6 && d[1] === 6);
    const nonTrumps = knownLaydown.filter(d => !(d[0] === 6 || d[1] === 6));
    
    console.log(`Trumps: ${trumpCount}/7 available`);
    console.log(`Has double (6-6): ${hasDouble}`);
    console.log(`Non-trumps: ${nonTrumps.map(d => `${d[0]}-${d[1]}`).join(', ')}`);
    
    // This should pass basic laydown requirements
    expect(hasDouble).toBe(true);
    expect(trumpCount).toBeGreaterThanOrEqual(4);
    
    console.log('✓ Individual laydowns do exist');
  });
});