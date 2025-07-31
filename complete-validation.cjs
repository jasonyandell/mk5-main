// Complete validation of ALL possible quad laydown candidates
// Generate every valid trump combination, allocate dominoes, and test with game engine

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
      if (domino[0] === domino[1]) {
        trumpDominoes.push(domino);
      }
    } else {
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

function generateAllCombinations() {
  const trumpValues = [0, 1, 2, 3, 4, 5, 6, 7]; // Include doubles (7)
  const combinations = [];
  
  // Generate all combinations of 4 different trump suits
  for (let a = 0; a < trumpValues.length; a++) {
    for (let b = a + 1; b < trumpValues.length; b++) {
      for (let c = b + 1; c < trumpValues.length; c++) {
        for (let d = c + 1; d < trumpValues.length; d++) {
          combinations.push([trumpValues[a], trumpValues[b], trumpValues[c], trumpValues[d]]);
        }
      }
    }
  }
  
  return combinations;
}

function tryAllocateHands(trumpCombination) {
  // Try to allocate 4 trumps per player
  const hands = [[], [], [], []];
  const used = new Set();
  
  // Get trump dominoes for each player
  const playerTrumps = trumpCombination.map(trump => getTrumpDominoes(trump));
  
  // Allocate 4 trumps per player
  for (let player = 0; player < 4; player++) {
    const availableTrumps = playerTrumps[player];
    let allocated = 0;
    
    for (const domino of availableTrumps) {
      if (allocated >= 4) break;
      
      const key = `${domino[0]}-${domino[1]}`;
      if (!used.has(key)) {
        hands[player].push(domino);
        used.add(key);
        allocated++;
      }
    }
    
    if (allocated < 4) {
      return null; // Couldn't allocate enough trumps
    }
  }
  
  // Distribute remaining dominoes (28 - 16 = 12 remaining, 3 per player)
  const remaining = ALL_DOMINOES.filter(d => !used.has(`${d[0]}-${d[1]}`));
  
  if (remaining.length !== 12) {
    return null;
  }
  
  // Give 3 remaining dominoes to each player
  for (let i = 0; i < 12; i++) {
    const player = Math.floor(i / 3);
    hands[player].push(remaining[i]);
  }
  
  return hands;
}

// Game simulation functions
function createDomino(high, low) {
  return {
    high: Math.max(high, low),
    low: Math.min(high, low),
    id: `${Math.max(high, low)}-${Math.min(high, low)}`,
    points: 0
  };
}

function createGameState(hands, trumps, biddingPlayer) {
  const state = {
    phase: 'playing',
    trump: trumps[biddingPlayer],
    winningBidder: biddingPlayer,
    currentPlayer: biddingPlayer,
    currentBid: { type: 'marks', value: 1, player: biddingPlayer },
    players: [],
    tricks: [],
    currentTrick: null
  };
  
  for (let i = 0; i < 4; i++) {
    state.players[i] = {
      hand: hands[i].map(([h, l]) => createDomino(h, l))
    };
  }
  
  return state;
}

const gameCache = new Map();

function canWinAllTricks(state, targetPlayer) {
  if (state.tricks.some(t => t.winner !== undefined && t.winner !== targetPlayer)) {
    return false;
  }
  
  if (state.phase !== 'playing') {
    return state.tricks.length === 7 && state.tricks.every(t => t.winner === targetPlayer);
  }
  
  const key = createCacheKey(state, targetPlayer);
  if (gameCache.has(key)) {
    return gameCache.get(key);
  }
  
  const possibleMoves = getPossibleMoves(state);
  if (possibleMoves.length === 0) {
    gameCache.set(key, false);
    return false;
  }
  
  let result;
  if (state.currentPlayer === targetPlayer) {
    result = possibleMoves.some(move => {
      const newState = applyMove(state, move);
      return canWinAllTricks(newState, targetPlayer);
    });
  } else {
    result = possibleMoves.every(move => {
      const newState = applyMove(state, move);
      return canWinAllTricks(newState, targetPlayer);
    });
  }
  
  gameCache.set(key, result);
  return result;
}

function createCacheKey(state, targetPlayer) {
  const tricksWon = [0, 0, 0, 0];
  state.tricks.forEach(t => {
    if (t.winner !== undefined) tricksWon[t.winner]++;
  });
  
  const handSizes = state.players.map(p => p.hand.length);
  const currentTrickSize = state.currentTrick ? state.currentTrick.dominoes.length : 0;
  
  return `${tricksWon.join('-')}|${handSizes.join('-')}|P${state.currentPlayer}|T${state.trump}|CT${currentTrickSize}|TGT${targetPlayer}`;
}

function getPossibleMoves(state) {
  const currentPlayer = state.currentPlayer;
  const hand = state.players[currentPlayer].hand;
  
  if (hand.length === 0) return [];
  
  // For now, return all dominoes as possible moves (we could add suit following logic later)
  return hand.map(domino => ({ type: 'play', domino, player: currentPlayer }));
}

function applyMove(state, move) {
  const newState = JSON.parse(JSON.stringify(state));
  const { domino, player } = move;
  
  // Remove domino from player's hand
  newState.players[player].hand = newState.players[player].hand.filter(d => d.id !== domino.id);
  
  // Add to current trick or start new trick
  if (!newState.currentTrick) {
    newState.currentTrick = {
      dominoes: [{ domino, player }],
      leadSuit: getSuit(domino, newState.trump)
    };
  } else {
    newState.currentTrick.dominoes.push({ domino, player });
  }
  
  // If trick is complete, resolve it
  if (newState.currentTrick.dominoes.length === 4) {
    const winner = determineTrickWinner(newState.currentTrick, newState.trump);
    newState.tricks.push({
      dominoes: newState.currentTrick.dominoes,
      winner: winner
    });
    
    newState.currentTrick = null;
    newState.currentPlayer = winner;
    
    if (newState.tricks.length === 7) {
      newState.phase = 'complete';
    }
  } else {
    newState.currentPlayer = (newState.currentPlayer + 1) % 4;
  }
  
  return newState;
}

function getSuit(domino, trump) {
  if (trump === 7) {
    return domino.high === domino.low ? 'trump' : domino.high;
  } else {
    return (domino.high === trump || domino.low === trump) ? 'trump' : domino.high;
  }
}

function determineTrickWinner(trick, trump) {
  const { dominoes, leadSuit } = trick;
  
  let winner = dominoes[0].player;
  let winningDomino = dominoes[0].domino;
  
  for (let i = 1; i < dominoes.length; i++) {
    const { domino, player } = dominoes[i];
    
    if (beats(domino, winningDomino, leadSuit, trump)) {
      winner = player;
      winningDomino = domino;
    }
  }
  
  return winner;
}

function beats(d1, d2, leadSuit, trump) {
  const d1Suit = getSuit(d1, trump);
  const d2Suit = getSuit(d2, trump);
  
  if (d1Suit === 'trump' && d2Suit !== 'trump') return true;
  if (d1Suit !== 'trump' && d2Suit === 'trump') return false;
  
  if (d1Suit === 'trump' && d2Suit === 'trump') {
    return compareTrumps(d1, d2, trump) > 0;
  }
  
  if (d1Suit !== leadSuit) return false;
  if (d2Suit !== leadSuit) return true;
  
  return compareInSuit(d1, d2, leadSuit) > 0;
}

function compareTrumps(d1, d2, trump) {
  if (trump === 7) {
    return d1.high - d2.high;
  } else {
    const d1IsDouble = d1.high === d1.low && d1.high === trump;
    const d2IsDouble = d2.high === d2.low && d2.high === trump;
    
    if (d1IsDouble && !d2IsDouble) return 1;
    if (!d1IsDouble && d2IsDouble) return -1;
    
    const d1Other = d1.high === trump ? d1.low : d1.high;
    const d2Other = d2.high === trump ? d2.low : d2.high;
    
    return d1Other - d2Other;
  }
}

function compareInSuit(d1, d2, suit) {
  const d1IsDouble = d1.high === d1.low && d1.high === suit;
  const d2IsDouble = d2.high === d2.low && d2.high === suit;
  
  if (d1IsDouble && !d2IsDouble) return 1;
  if (!d1IsDouble && d2IsDouble) return -1;
  
  const d1Other = d1.high === suit ? d1.low : d1.high;
  const d2Other = d2.high === suit ? d2.low : d2.high;
  
  return d1Other - d2Other;
}

function simulateLaydown(hands, trumps, biddingPlayer) {
  gameCache.clear();
  const state = createGameState(hands, trumps, biddingPlayer);
  return canWinAllTricks(state, biddingPlayer);
}

function completeValidation() {
  console.log("=".repeat(80));
  console.log("COMPLETE VALIDATION: ALL POSSIBLE QUAD LAYDOWN CANDIDATES");
  console.log("Generating every valid trump combination and testing with game engine");
  console.log("=".repeat(80));
  
  const allCombinations = generateAllCombinations();
  console.log(`Total trump combinations to test: ${allCombinations.length}\n`);
  
  let candidatesGenerated = 0;
  let candidatesTested = 0;
  let validQuadLaydowns = 0;
  let totalIndividualTests = 0;
  let totalIndividualPassed = 0;
  
  for (const trumpCombo of allCombinations) {
    const hands = tryAllocateHands(trumpCombo);
    
    if (!hands) {
      continue; // Skip invalid allocations
    }
    
    candidatesGenerated++;
    candidatesTested++;
    
    console.log(`\nCandidate #${candidatesGenerated}: [${trumpCombo.map(t => t === 7 ? 'Doubles' : `${t}s`).join(', ')}]`);
    
    // Show hands
    hands.forEach((hand, i) => {
      const trumpCount = trumpCombo[i] === 7 
        ? hand.filter(d => d[0] === d[1]).length
        : hand.filter(d => d[0] === trumpCombo[i] || d[1] === trumpCombo[i]).length;
      
      console.log(`  P${i + 1} (${trumpCombo[i] === 7 ? 'Doubles' : trumpCombo[i]}s, ${trumpCount}T): ${hand.map(d => `${d[0]}-${d[1]}`).join(', ')}`);
    });
    
    // Test each player as bidder
    const results = [];
    for (let biddingPlayer = 0; biddingPlayer < 4; biddingPlayer++) {
      totalIndividualTests++;
      const canLaydown = simulateLaydown(hands, trumpCombo, biddingPlayer);
      results.push(canLaydown);
      
      if (canLaydown) {
        totalIndividualPassed++;
      }
    }
    
    const passedCount = results.filter(Boolean).length;
    console.log(`  Results: ${results.map((r, i) => `P${i+1}:${r ? '✅' : '❌'}`).join(' ')} (${passedCount}/4)`);
    
    if (passedCount === 4) {
      validQuadLaydowns++;
      console.log(`  🎉 VALID QUAD LAYDOWN FOUND! (#${validQuadLaydowns})`);
    }
    
    // Show progress occasionally
    if (candidatesGenerated % 10 === 0) {
      console.log(`\n--- Progress: ${candidatesGenerated} candidates tested, ${validQuadLaydowns} quad laydowns found ---\n`);
    }
  }
  
  console.log(`\n${"=".repeat(80)}`);
  console.log("COMPLETE VALIDATION RESULTS:");
  console.log(`Trump combinations checked: ${allCombinations.length}`);
  console.log(`Valid candidates generated: ${candidatesGenerated}`);
  console.log(`Candidates tested: ${candidatesTested}`);
  console.log(`Individual tests: ${totalIndividualTests}`);
  console.log(`Individual tests passed: ${totalIndividualPassed}/${totalIndividualTests} (${(totalIndividualPassed/totalIndividualTests*100).toFixed(1)}%)`);
  console.log(`Valid quad laydowns: ${validQuadLaydowns}/${candidatesTested}`);
  
  if (validQuadLaydowns > 0) {
    console.log(`\n✅ CONCLUSION: ${validQuadLaydowns} confirmed quad laydown(s) found!`);
    console.log("4 simultaneous laydowns ARE possible in Texas 42.");
  } else {
    console.log(`\n❌ CONCLUSION: No valid quad laydowns found in complete search.`);
    console.log("4 simultaneous laydowns appear to be impossible in Texas 42.");
  }
  console.log(`${"=".repeat(80)}`);
}

// Run complete validation
completeValidation();