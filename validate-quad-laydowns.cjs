// Validate quad laydown candidates through actual game simulation

// Extract the top 10 candidates from the flexible search results for validation
const quadLaydownCandidates = [
  {
    id: 1,
    trumps: [0, 1, 2, 3],
    hands: [
      [[0,0], [6,0], [5,0], [4,0], [6,6], [6,5], [6,4]], // P1 (0s trump)
      [[1,1], [6,1], [5,1], [4,1], [5,5], [5,4], [4,4]], // P2 (1s trump)
      [[2,2], [6,2], [5,2], [4,2], [3,2], [3,1], [3,0]], // P3 (2s trump)
      [[3,3], [6,3], [5,3], [4,3], [2,1], [2,0], [1,0]]  // P4 (3s trump)
    ]
  },
  {
    id: 2,
    trumps: [0, 1, 2, 4],
    hands: [
      [[0,0], [6,0], [5,0], [4,0], [6,6], [6,5], [6,3]], // P1 (0s trump)
      [[1,1], [6,1], [5,1], [4,1], [5,5], [5,3], [3,3]], // P2 (1s trump)
      [[2,2], [6,2], [5,2], [4,2], [3,2], [3,1], [3,0]], // P3 (2s trump)
      [[4,4], [6,4], [5,4], [4,3], [2,1], [2,0], [1,0]]  // P4 (4s trump)
    ]
  },
  {
    id: 3,
    trumps: [0, 1, 2, 5],
    hands: [
      [[0,0], [6,0], [5,0], [4,0], [6,6], [6,4], [6,3]], // P1 (0s trump)
      [[1,1], [6,1], [5,1], [4,1], [4,4], [4,3], [3,3]], // P2 (1s trump)
      [[2,2], [6,2], [5,2], [4,2], [3,2], [3,1], [3,0]], // P3 (2s trump)
      [[5,5], [6,5], [5,4], [5,3], [2,1], [2,0], [1,0]]  // P4 (5s trump)
    ]
  },
  {
    id: 4,
    trumps: [0, 1, 2, 6],
    hands: [
      [[0,0], [6,0], [5,0], [4,0], [5,5], [5,4], [5,3]], // P1 (0s trump)
      [[1,1], [6,1], [5,1], [4,1], [4,4], [4,3], [3,3]], // P2 (1s trump)
      [[2,2], [6,2], [5,2], [4,2], [3,2], [3,1], [3,0]], // P3 (2s trump)
      [[6,6], [6,5], [6,4], [6,3], [2,1], [2,0], [1,0]]  // P4 (6s trump)
    ]
  },
  {
    id: 5,
    trumps: [0, 1, 2, 7], // 7 = Doubles trump
    hands: [
      [[0,0], [6,0], [5,0], [4,0], [6,5], [6,4], [6,3]], // P1 (0s trump)
      [[1,1], [6,1], [5,1], [4,1], [5,4], [5,3], [4,3]], // P2 (1s trump)
      [[2,2], [6,2], [5,2], [4,2], [3,2], [3,1], [3,0]], // P3 (2s trump)
      [[6,6], [5,5], [4,4], [3,3], [2,1], [2,0], [1,0]]  // P4 (Doubles trump)
    ]
  },
  {
    id: 6,
    trumps: [0, 1, 3, 4],
    hands: [
      [[0,0], [6,0], [5,0], [4,0], [6,6], [6,5], [6,2]], // P1 (0s trump)
      [[1,1], [6,1], [5,1], [4,1], [5,5], [5,2], [3,2]], // P2 (1s trump)
      [[3,3], [6,3], [5,3], [4,3], [3,1], [3,0], [2,2]], // P3 (3s trump)
      [[4,4], [6,4], [5,4], [4,2], [2,1], [2,0], [1,0]]  // P4 (4s trump)
    ]
  },
  {
    id: 7,
    trumps: [1, 2, 3, 4],
    hands: [
      [[1,1], [6,1], [5,1], [4,1], [6,6], [6,5], [6,0]], // P1 (1s trump)
      [[2,2], [6,2], [5,2], [4,2], [5,5], [5,0], [3,2]], // P2 (2s trump)
      [[3,3], [6,3], [5,3], [4,3], [3,1], [3,0], [2,1]], // P3 (3s trump)
      [[4,4], [6,4], [5,4], [4,0], [2,0], [1,0], [0,0]]  // P4 (4s trump)
    ]
  },
  {
    id: 8,
    trumps: [2, 3, 4, 5],
    hands: [
      [[2,2], [6,2], [5,2], [4,2], [6,6], [6,1], [6,0]], // P1 (2s trump)
      [[3,3], [6,3], [5,3], [4,3], [4,0], [3,2], [3,1]], // P2 (3s trump)
      [[4,4], [6,4], [5,4], [4,1], [3,0], [2,1], [2,0]], // P3 (4s trump)
      [[5,5], [6,5], [5,1], [5,0], [1,1], [1,0], [0,0]]  // P4 (5s trump)
    ]
  },
  {
    id: 9,
    trumps: [3, 4, 5, 6],
    hands: [
      [[3,3], [6,3], [5,3], [4,3], [5,0], [4,1], [4,0]], // P1 (3s trump)
      [[4,4], [6,4], [5,4], [4,2], [3,2], [3,1], [3,0]], // P2 (4s trump)
      [[5,5], [6,5], [5,2], [5,1], [2,2], [2,1], [2,0]], // P3 (5s trump)
      [[6,6], [6,2], [6,1], [6,0], [1,1], [1,0], [0,0]]  // P4 (6s trump)
    ]
  },
  {
    id: 10,
    trumps: [0, 2, 4, 6],
    hands: [
      [[0,0], [6,0], [5,0], [4,0], [5,5], [5,3], [5,1]], // P1 (0s trump)
      [[2,2], [6,2], [5,2], [4,2], [4,1], [3,3], [3,2]], // P2 (2s trump)
      [[4,4], [6,4], [5,4], [4,3], [3,1], [3,0], [2,1]], // P3 (4s trump)
      [[6,6], [6,5], [6,3], [6,1], [2,0], [1,1], [1,0]]  // P4 (6s trump)
    ]
  }
];

// Import game simulation functions (inline to avoid module issues)
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
  
  // Set up hands
  for (let i = 0; i < 4; i++) {
    state.players[i] = {
      hand: hands[i].map(([h, l]) => createDomino(h, l))
    };
  }
  
  return state;
}

// Simple game simulation - check if bidding player can win all tricks
function simulateLaydown(hands, trumps, biddingPlayer) {
  const state = createGameState(hands, trumps, biddingPlayer);
  
  // Use exhaustive minimax to determine if bidding player can win all tricks
  return canWinAllTricks(state, biddingPlayer);
}

const gameCache = new Map();

function canWinAllTricks(state, targetPlayer) {
  // If any trick was already lost by target player, impossible
  if (state.tricks.some(t => t.winner !== undefined && t.winner !== targetPlayer)) {
    return false;
  }
  
  // If game is complete, check if target player won all tricks
  if (state.phase !== 'playing') {
    return state.tricks.length === 7 && state.tricks.every(t => t.winner === targetPlayer);
  }
  
  // Create cache key
  const key = createCacheKey(state, targetPlayer);
  if (gameCache.has(key)) {
    return gameCache.get(key);
  }
  
  // Get possible moves
  const possibleMoves = getPossibleMoves(state);
  if (possibleMoves.length === 0) {
    gameCache.set(key, false);
    return false;
  }
  
  let result;
  if (state.currentPlayer === targetPlayer) {
    // Target player's turn - needs to have at least one winning move
    result = possibleMoves.some(move => {
      const newState = applyMove(state, move);
      return canWinAllTricks(newState, targetPlayer);
    });
  } else {
    // Opponent's turn - target player needs to win regardless of opponent's move
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
  
  // For simplicity, return all dominoes in hand as possible moves
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
    
    // Check if game is over
    if (newState.tricks.length === 7) {
      newState.phase = 'complete';
    }
  } else {
    // Move to next player
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
  
  // Trump beats non-trump
  if (d1Suit === 'trump' && d2Suit !== 'trump') return true;
  if (d1Suit !== 'trump' && d2Suit === 'trump') return false;
  
  // Both trump
  if (d1Suit === 'trump' && d2Suit === 'trump') {
    return compareTrumps(d1, d2, trump) > 0;
  }
  
  // Neither trump - must follow suit to win
  if (d1Suit !== leadSuit) return false;
  if (d2Suit !== leadSuit) return true;
  
  // Both follow suit
  return compareInSuit(d1, d2, leadSuit) > 0;
}

function compareTrumps(d1, d2, trump) {
  if (trump === 7) {
    return d1.high - d2.high; // Higher double wins
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

function validateQuadLaydowns() {
  console.log("=".repeat(80));
  console.log("VALIDATING QUAD LAYDOWN CANDIDATES WITH GAME ENGINE");
  console.log("Testing each scenario with each player as bidder");
  console.log("=".repeat(80));
  
  let totalTests = 0;
  let totalPassed = 0;
  let validQuadLaydowns = 0;
  
  for (const candidate of quadLaydownCandidates) {
    console.log(`\n${"=".repeat(60)}`);
    console.log(`CANDIDATE #${candidate.id}: Trump combination [${candidate.trumps.map(t => t === 7 ? 'Doubles' : `${t}s`).join(', ')}]`);
    console.log(`${"=".repeat(60)}`);
    
    // Show the hands
    candidate.hands.forEach((hand, i) => {
      const trumpCount = candidate.trumps[i] === 7 
        ? hand.filter(d => d[0] === d[1]).length
        : hand.filter(d => d[0] === candidate.trumps[i] || d[1] === candidate.trumps[i]).length;
      
      console.log(`Player ${i + 1} (${candidate.trumps[i] === 7 ? 'Doubles' : candidate.trumps[i]}s trump, ${trumpCount} trumps):`);
      console.log(`  ${hand.map(d => `${d[0]}-${d[1]}`).join(', ')}`);
    });
    
    console.log(`\nTesting each player as bidder:`);
    
    let candidatePassed = true;
    const results = [];
    
    for (let biddingPlayer = 0; biddingPlayer < 4; biddingPlayer++) {
      totalTests++;
      gameCache.clear(); // Clear cache for each test
      
      const canLaydown = simulateLaydown(candidate.hands, candidate.trumps, biddingPlayer);
      results.push(canLaydown);
      
      if (canLaydown) {
        totalPassed++;
      } else {
        candidatePassed = false;
      }
      
      console.log(`  Player ${biddingPlayer + 1} as bidder: ${canLaydown ? '✅ LAYDOWN' : '❌ FAILED'}`);
    }
    
    console.log(`\nResult: ${results.filter(Boolean).length}/4 players can lay down`);
    
    if (candidatePassed) {
      validQuadLaydowns++;
      console.log(`🎉 VALID QUAD LAYDOWN - All 4 players can lay down!`);
    } else {
      console.log(`❌ NOT A QUAD LAYDOWN - ${4 - results.filter(Boolean).length} player(s) cannot lay down`);
    }
  }
  
  console.log(`\n${"=".repeat(80)}`);
  console.log("FINAL VALIDATION RESULTS:");
  console.log(`Candidates tested: ${quadLaydownCandidates.length}`);
  console.log(`Individual tests: ${totalTests}`);
  console.log(`Individual tests passed: ${totalPassed}/${totalTests} (${(totalPassed/totalTests*100).toFixed(1)}%)`);
  console.log(`Valid quad laydowns: ${validQuadLaydowns}/${quadLaydownCandidates.length}`);
  
  if (validQuadLaydowns > 0) {
    console.log(`\n✅ CONCLUSION: ${validQuadLaydowns} confirmed quad laydown(s) found!`);
    console.log("4 simultaneous laydowns ARE possible in Texas 42.");
  } else {
    console.log(`\n❌ CONCLUSION: No valid quad laydowns confirmed by game simulation.`);
    console.log("The flexible search had false positives - actual game play prevents these scenarios.");
  }
  console.log(`${"=".repeat(80)}`);
}

// Run validation
validateQuadLaydowns();