// Extracted game simulation utilities for laydown verification

function createDomino(high, low) {
  return {
    high: Math.max(high, low),
    low: Math.min(high, low),
    id: `${Math.max(high, low)}-${Math.min(high, low)}`,
    points: 0
  };
}

function createInitialState() {
  return {
    phase: 'playing',
    trump: 0,
    winningBidder: 0,
    currentPlayer: 0,
    currentBid: { type: 'marks', value: 1, player: 0 },
    players: [
      { hand: [] },
      { hand: [] },
      { hand: [] },
      { hand: [] }
    ],
    tricks: []
  };
}

function setupGameWithHands(hands, trumps) {
  const state = createInitialState();
  
  // Set up each player's hand
  for (let i = 0; i < 4; i++) {
    state.players[i].hand = hands[i].map(([h, l]) => createDomino(h, l));
  }
  
  // For now, just use player 0's trump (we'll test each player as bidder)
  state.trump = trumps[0];
  state.winningBidder = 0;
  state.currentPlayer = 0;
  
  return state;
}

// Cache for memoization
const simulationCache = new Map();

function getCacheKey(state) {
  const tricksByPlayer = [0, 0, 0, 0];
  state.tricks.forEach(t => {
    if (t.winner !== undefined) {
      tricksByPlayer[t.winner]++;
    }
  });
  const cardsLeft = state.players.map(p => p.hand.length).join('-');
  return `${tricksByPlayer.join('-')}|${cardsLeft}|P${state.currentPlayer}|T${state.trump}`;
}

// Simulate if a player can win all remaining tricks
function canPlayerWinAllTricks(state, targetPlayer) {
  // Check if target player has already lost any tricks
  if (state.tricks.some(t => t.winner !== undefined && t.winner !== targetPlayer)) {
    return false;
  }
  
  // If game is over, check if target player won all tricks
  if (state.phase !== 'playing') {
    return state.tricks.length === 7 && state.tricks.every(t => t.winner === targetPlayer);
  }
  
  const key = getCacheKey(state) + `|target${targetPlayer}`;
  if (simulationCache.has(key)) {
    return simulationCache.get(key);
  }
  
  // Get all possible next moves
  const transitions = getNextStates(state);
  if (transitions.length === 0) return false;
  
  let result;
  if (state.currentPlayer === targetPlayer) {
    // Target player's turn - they need at least one winning move
    result = transitions.some(t => canPlayerWinAllTricks(t.newState, targetPlayer));
  } else {
    // Opponent's turn - target player needs to win regardless of opponent's move
    result = transitions.every(t => canPlayerWinAllTricks(t.newState, targetPlayer));
  }
  
  simulationCache.set(key, result);
  return result;
}

// Simplified version of getNextStates - just the core logic needed
function getNextStates(state) {
  if (state.phase !== 'playing') return [];
  
  const currentPlayerHand = state.players[state.currentPlayer].hand;
  if (currentPlayerHand.length === 0) return [];
  
  const transitions = [];
  
  // For each domino in current player's hand
  for (const domino of currentPlayerHand) {
    // Create new state with this domino played
    const newState = JSON.parse(JSON.stringify(state));
    
    // Remove domino from hand
    const playerIndex = newState.currentPlayer;
    newState.players[playerIndex].hand = newState.players[playerIndex].hand.filter(
      d => d.id !== domino.id
    );
    
    // Add to current trick or start new trick
    if (!newState.currentTrick) {
      newState.currentTrick = {
        dominoes: [{ domino, player: playerIndex }],
        leadSuit: getSuit(domino, newState.trump)
      };
    } else {
      newState.currentTrick.dominoes.push({ domino, player: playerIndex });
    }
    
    // If trick is complete (4 dominoes), resolve it
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
    
    transitions.push({ action: `play-${domino.id}`, newState });
  }
  
  return transitions;
}

function getSuit(domino, trump) {
  if (trump === 7) {
    // Doubles trump
    if (domino.high === domino.low) return 'trump';
    return domino.high; // Suit is determined by higher pip
  } else {
    // Regular suit trump
    if (domino.high === trump || domino.low === trump) return 'trump';
    return domino.high; // Suit is determined by higher pip
  }
}

function determineTrickWinner(trick, trump) {
  const { dominoes, leadSuit } = trick;
  
  let winner = 0;
  let winningDomino = dominoes[0].domino;
  
  for (let i = 1; i < dominoes.length; i++) {
    const { domino } = dominoes[i];
    
    if (beats(domino, winningDomino, leadSuit, trump)) {
      winner = dominoes[i].player;
      winningDomino = domino;
    }
  }
  
  return winner;
}

function beats(domino1, domino2, leadSuit, trump) {
  const d1Suit = getSuit(domino1, trump);
  const d2Suit = getSuit(domino2, trump);
  
  // Trump beats non-trump
  if (d1Suit === 'trump' && d2Suit !== 'trump') return true;
  if (d1Suit !== 'trump' && d2Suit === 'trump') return false;
  
  // Both trump - compare trump values
  if (d1Suit === 'trump' && d2Suit === 'trump') {
    return compareTrumps(domino1, domino2, trump) > 0;
  }
  
  // Neither trump - only domino1 wins if it follows lead suit and is higher
  if (d1Suit !== leadSuit) return false;
  if (d2Suit !== leadSuit) return true;
  
  // Both follow suit - compare in suit
  return compareInSuit(domino1, domino2, leadSuit) > 0;
}

function compareTrumps(d1, d2, trump) {
  if (trump === 7) {
    // Doubles trump - higher double wins
    return d1.high - d2.high;
  } else {
    // Regular suit trump
    const d1IsDouble = d1.high === d1.low && d1.high === trump;
    const d2IsDouble = d2.high === d2.low && d2.high === trump;
    
    if (d1IsDouble && !d2IsDouble) return 1;
    if (!d1IsDouble && d2IsDouble) return -1;
    
    // Both doubles or both non-doubles
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

// Test if a specific player has a laydown
function testLaydown(hands, trumps, playerIndex) {
  simulationCache.clear(); // Clear cache for fresh test
  
  const state = setupGameWithHands(hands, trumps);
  state.trump = trumps[playerIndex];
  state.winningBidder = playerIndex;
  state.currentPlayer = playerIndex;
  
  return canPlayerWinAllTricks(state, playerIndex);
}

module.exports = {
  createDomino,
  createInitialState,
  setupGameWithHands,
  canPlayerWinAllTricks,
  testLaydown,
  simulationCache
};