// Flexible search for quad laydowns - relax the strict "double + 4 highest" constraint
// Try different trump requirements and include doubles trump

const ALL_DOMINOES = [
  [6,6], [6,5], [6,4], [6,3], [6,2], [6,1], [6,0],
  [5,5], [5,4], [5,3], [5,2], [5,1], [5,0],
  [4,4], [4,3], [4,2], [4,1], [4,0],
  [3,3], [3,2], [3,1], [3,0],
  [2,2], [2,1], [2,0],
  [1,1], [1,0],
  [0,0]
];

function getTrumpDominoes(trump) {
  const trumpDominoes = [];
  
  for (const domino of ALL_DOMINOES) {
    if (trump === 7) {
      // Doubles trump
      if (domino[0] === domino[1]) {
        trumpDominoes.push(domino);
      }
    } else {
      // Regular suit trump
      if (domino[0] === trump || domino[1] === trump) {
        trumpDominoes.push(domino);
      }
    }
  }
  
  // Sort by trump strength (highest first)
  if (trump === 7) {
    trumpDominoes.sort((a, b) => b[0] - a[0]);
  } else {
    trumpDominoes.sort((a, b) => {
      const aIsDouble = a[0] === a[1] && a[0] === trump;
      const bIsDouble = b[0] === b[1] && b[0] === trump;
      
      if (aIsDouble && !bIsDouble) return -1;
      if (!aIsDouble && bIsDouble) return 1;
      
      const aOther = a[0] === trump ? a[1] : a[0];
      const bOther = b[0] === trump ? b[1] : b[0];
      
      return bOther - aOther;
    });
  }
  
  return trumpDominoes;
}

function tryFlexibleAllocation(trumpCombination, trumpRequirements) {
  // trumpRequirements: [numTrumps0, numTrumps1, numTrumps2, numTrumps3]
  const hands = [[], [], [], []];
  const used = new Set();
  
  // Get all trump dominoes for each player
  const playerTrumps = trumpCombination.map(trump => getTrumpDominoes(trump));
  
  // Try to allocate required trumps per player
  for (let player = 0; player < 4; player++) {
    const availableTrumps = playerTrumps[player];
    const needed = trumpRequirements[player];
    let allocated = 0;
    
    for (const domino of availableTrumps) {
      if (allocated >= needed) break;
      
      const key = `${domino[0]}-${domino[1]}`;
      if (!used.has(key)) {
        hands[player].push(domino);
        used.add(key);
        allocated++;
      }
    }
    
    if (allocated < needed) {
      return null; // Couldn't allocate enough trumps
    }
  }
  
  // Distribute remaining dominoes to fill hands to 7
  const remaining = ALL_DOMINOES.filter(d => !used.has(`${d[0]}-${d[1]}`));
  const totalTrumpsAllocated = trumpRequirements.reduce((sum, req) => sum + req, 0);
  const remainingNeeded = 28 - totalTrumpsAllocated;
  
  if (remaining.length !== remainingNeeded) {
    return null;
  }
  
  // Distribute remaining dominoes
  let remainingIndex = 0;
  for (let player = 0; player < 4; player++) {
    const stillNeed = 7 - hands[player].length;
    for (let i = 0; i < stillNeed; i++) {
      if (remainingIndex < remaining.length) {
        hands[player].push(remaining[remainingIndex++]);
      }
    }
  }
  
  // Verify all hands have 7 dominoes
  for (let player = 0; player < 4; player++) {
    if (hands[player].length !== 7) {
      return null;
    }
  }
  
  return hands;
}

function testLaydownRelaxed(hand, trump) {
  // More relaxed laydown test
  let trumpCount = 0;
  let hasHighestTrump = false;
  
  if (trump === 7) {
    // Doubles trump
    for (const domino of hand) {
      if (domino[0] === domino[1]) {
        trumpCount++;
        if (domino[0] === 6) hasHighestTrump = true;
      }
    }
  } else {
    // Regular suit trump
    for (const domino of hand) {
      if (domino[0] === trump || domino[1] === trump) {
        trumpCount++;
        if (domino[0] === trump && domino[1] === trump) hasHighestTrump = true;
      }
    }
  }
  
  // Relaxed requirements: has highest trump and at least 3 trumps
  return hasHighestTrump && trumpCount >= 3;
}

function flexibleSearch() {
  console.log("=".repeat(60));
  console.log("FLEXIBLE SEARCH FOR QUAD LAYDOWNS");
  console.log("Trying different trump requirements and including doubles");
  console.log("=".repeat(60));
  
  // Include doubles trump (7) in combinations
  const trumpValues = [0, 1, 2, 3, 4, 5, 6, 7];
  
  // Generate combinations including doubles
  const allCombinations = [];
  for (let a = 0; a < trumpValues.length; a++) {
    for (let b = a + 1; b < trumpValues.length; b++) {
      for (let c = b + 1; c < trumpValues.length; c++) {
        for (let d = c + 1; d < trumpValues.length; d++) {
          allCombinations.push([trumpValues[a], trumpValues[b], trumpValues[c], trumpValues[d]]);
        }
      }
    }
  }
  
  console.log(`Generated ${allCombinations.length} trump combinations (including doubles)\n`);
  
  // Try different trump requirements
  const trumpRequirementSets = [
    [5, 5, 5, 5], // Original: double + 4 others each
    [4, 4, 4, 4], // Relaxed: 4 trumps each
    [5, 4, 4, 3], // Mixed: descending requirements
    [4, 4, 3, 3], // Even more relaxed
    [5, 5, 4, 4], // Some high, some medium
    [3, 3, 3, 3], // Very relaxed: just 3 trumps each
  ];
  
  let totalTests = 0;
  let validAllocations = 0;
  let quadLaydowns = 0;
  
  for (const requirements of trumpRequirementSets) {
    console.log(`\nTesting with trump requirements: ${requirements.join(', ')}`);
    console.log("-".repeat(40));
    
    let testsThisSet = 0;
    let validThisSet = 0;
    
    for (const combo of allCombinations) {
      totalTests++;
      testsThisSet++;
      
      const hands = tryFlexibleAllocation(combo, requirements);
      
      if (!hands) continue;
      
      validAllocations++;
      validThisSet++;
      
      // Test laydowns
      const laydownResults = [];
      for (let player = 0; player < 4; player++) {
        const isLaydown = testLaydownRelaxed(hands[player], combo[player]);
        laydownResults.push(isLaydown);
      }
      
      const laydownCount = laydownResults.filter(Boolean).length;
      
      if (laydownCount === 4) {
        quadLaydowns++;
        console.log(`\n🎉 QUAD LAYDOWN FOUND! (#${quadLaydowns})`);
        console.log(`Trump combination: ${combo.map(t => t === 7 ? 'Doubles' : `${t}s`).join(', ')}`);
        console.log(`Requirements: ${requirements.join(', ')}`);
        console.log("=" .repeat(50));
        
        hands.forEach((hand, i) => {
          const trumpCount = combo[i] === 7 
            ? hand.filter(d => d[0] === d[1]).length
            : hand.filter(d => d[0] === combo[i] || d[1] === combo[i]).length;
          
          console.log(`Player ${i + 1} (${combo[i] === 7 ? 'Doubles' : combo[i]}s trump, ${trumpCount} trumps):`);
          console.log(`  ${hand.map(d => `${d[0]}-${d[1]}`).join(', ')}`);
          console.log(`  Laydown: ${laydownResults[i] ? '✅' : '❌'}`);
        });
        console.log("=" .repeat(50));
        
        // Don't break - continue searching for more
      }
      
      // Show progress occasionally
      if (testsThisSet % 20 === 0) {
        console.log(`  Tested ${testsThisSet} combinations, ${validThisSet} valid, ${quadLaydowns} quad laydowns found...`);
      }
    }
    
    console.log(`Requirements ${requirements.join(', ')}: ${validThisSet}/${testsThisSet} valid allocations`);
  }
  
  console.log(`\n${"=".repeat(60)}`);
  console.log("FLEXIBLE SEARCH RESULTS:");
  console.log(`Total tests: ${totalTests}`);
  console.log(`Valid allocations: ${validAllocations}`);
  console.log(`Quad laydowns found: ${quadLaydowns}`);
  
  if (quadLaydowns > 0) {
    console.log(`\n✅ SUCCESS: Found ${quadLaydowns} quad laydown scenario(s)!`);
    console.log("This proves that 4 simultaneous laydowns ARE possible in Texas 42.");
  } else {
    console.log(`\n❌ No quad laydowns found even with relaxed constraints.`);
    if (validAllocations > 0) {
      console.log("Valid allocations exist but none result in 4 laydowns.");
    } else {
      console.log("No valid allocations found at all.");
    }
  }
  console.log(`${"=".repeat(60)}`);
}

// Run the flexible search
flexibleSearch();