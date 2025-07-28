import type { GameState, StateTransition, Bid } from '../types';
import { BID_TYPES, TRUMP_SUITS, GAME_CONSTANTS } from '../constants';
import { cloneGameState } from './state';
import { isValidBid, getValidPlays, getBidComparisonValue } from './rules';
import { dealDominoes } from './dominoes';
import { calculateTrickWinner, calculateTrickPoints, calculateRoundScore, isGameComplete } from './scoring';
import { getNextDealer, getPlayerLeftOfDealer, getNextPlayer } from './players';

/**
 * Core state machine function that returns all possible next states
 */
export function getNextStates(state: GameState): StateTransition[] {
  switch (state.phase) {
    case 'bidding':
      return getBiddingTransitions(state);
    case 'trump_selection':
      return getTrumpSelectionTransitions(state);
    case 'playing':
      return getPlayingTransitions(state);
    case 'scoring':
      return getScoringTransitions(state);
    default:
      return [];
  }
}

/**
 * Gets all valid bidding transitions
 */
function getBiddingTransitions(state: GameState): StateTransition[] {
  const transitions: StateTransition[] = [];
  
  // If bidding is complete (4 bids), determine next phase
  if (state.bids.length === 4) {
    const nonPassBids = state.bids.filter(b => b.type !== BID_TYPES.PASS);
    
    if (nonPassBids.length === 0) {
      // All passed - redeal
      const newState = cloneGameState(state);
      const hands = dealDominoes();
      newState.dealer = getNextDealer(state.dealer);
      newState.currentPlayer = getPlayerLeftOfDealer(newState.dealer);
      newState.bids = [];
      newState.currentBid = null;
      newState.players.forEach((player, i) => {
        player.hand = hands[i];
      });
      
      transitions.push({
        id: 'redeal',
        label: 'All passed - Redeal',
        newState
      });
    } else {
      // Find winning bidder and transition to trump selection
      const winningBid = nonPassBids.reduce((highest, current) => {
        const highestValue = getBidComparisonValue(highest);
        const currentValue = getBidComparisonValue(current);
        return currentValue > highestValue ? current : highest;
      });
      
      const newState = cloneGameState(state);
      newState.phase = 'trump_selection';
      newState.winningBidder = winningBid.player;
      newState.currentPlayer = winningBid.player;
      newState.currentBid = winningBid;
      
      transitions.push({
        id: 'select-trump',
        label: 'Select trump suit',
        newState
      });
    }
    
    return transitions;
  }
  
  // Check if current player has already bid
  const playerBids = state.bids.filter(b => b.player === state.currentPlayer);
  if (playerBids.length > 0) return [];
  
  const currentPlayerHand = state.players[state.currentPlayer].hand;
  
  // Generate pass bid
  const passBid: Bid = { type: BID_TYPES.PASS, player: state.currentPlayer };
  if (isValidBid(state, passBid)) {
    const newState = cloneGameState(state);
    newState.bids.push(passBid);
    newState.currentPlayer = getNextPlayer(state.currentPlayer);
    
    // If this completes bidding (4 bids), set up trump selection
    if (newState.bids.length === 4) {
      const nonPassBids = newState.bids.filter(b => b.type !== BID_TYPES.PASS);
      
      if (nonPassBids.length > 0) {
        // Find winning bidder and set trump selection state
        const winningBid = nonPassBids.reduce((highest, current) => {
          const highestValue = getBidComparisonValue(highest);
          const currentValue = getBidComparisonValue(current);
          return currentValue > highestValue ? current : highest;
        });
        
        newState.phase = 'trump_selection';
        newState.winningBidder = winningBid.player;
        newState.currentPlayer = winningBid.player;
        newState.currentBid = winningBid;
      }
      // All-pass case: keep bids as is for redeal transition
    }
    
    transitions.push({
      id: 'pass',
      label: 'Pass',
      newState
    });
  }
  
  // Generate point bids
  for (let points = GAME_CONSTANTS.MIN_BID; points <= GAME_CONSTANTS.MAX_BID; points++) {
    const bid: Bid = { type: BID_TYPES.POINTS, value: points, player: state.currentPlayer };
    if (isValidBid(state, bid, currentPlayerHand)) {
      const newState = cloneGameState(state);
      newState.bids.push(bid);
      newState.currentBid = bid;
      newState.currentPlayer = getNextPlayer(state.currentPlayer);
      
      transitions.push({
        id: `bid-${points}`,
        label: `Bid ${points} points`,
        newState
      });
    }
  }
  
  // Generate mark bids
  for (let marks = 1; marks <= 4; marks++) {
    const bid: Bid = { type: BID_TYPES.MARKS, value: marks, player: state.currentPlayer };
    if (isValidBid(state, bid, currentPlayerHand)) {
      const newState = cloneGameState(state);
      newState.bids.push(bid);
      newState.currentBid = bid;
      newState.currentPlayer = getNextPlayer(state.currentPlayer);
      
      transitions.push({
        id: `bid-${marks}-marks`,
        label: `Bid ${marks} mark${marks > 1 ? 's' : ''}`,
        newState
      });
    }
  }
  
  // Special contracts only in non-tournament mode
  if (!state.tournamentMode) {
    // Nello bids
    for (let marks = 1; marks <= 3; marks++) {
      const bid: Bid = { type: BID_TYPES.NELLO, value: marks, player: state.currentPlayer };
      if (isValidBid(state, bid, currentPlayerHand)) {
        const newState = cloneGameState(state);
        newState.bids.push(bid);
        newState.currentBid = bid;
        newState.currentPlayer = getNextPlayer(state.currentPlayer);
        
        transitions.push({
          id: `nello-${marks}`,
          label: `Nello ${marks}`,
          newState
        });
      }
    }
    
    // Splash bids
    for (let marks = 2; marks <= 3; marks++) {
      const bid: Bid = { type: BID_TYPES.SPLASH, value: marks, player: state.currentPlayer };
      if (isValidBid(state, bid, currentPlayerHand)) {
        const newState = cloneGameState(state);
        newState.bids.push(bid);
        newState.currentBid = bid;
        newState.currentPlayer = getNextPlayer(state.currentPlayer);
        
        transitions.push({
          id: `splash-${marks}`,
          label: `Splash ${marks}`,
          newState
        });
      }
    }
    
    // Plunge bids
    for (let marks = 4; marks <= 6; marks++) {
      const bid: Bid = { type: BID_TYPES.PLUNGE, value: marks, player: state.currentPlayer };
      if (isValidBid(state, bid, currentPlayerHand)) {
        const newState = cloneGameState(state);
        newState.bids.push(bid);
        newState.currentBid = bid;
        newState.currentPlayer = getNextPlayer(state.currentPlayer);
        
        transitions.push({
          id: `plunge-${marks}`,
          label: `Plunge ${marks}`,
          newState
        });
      }
    }
  }
  
  return transitions;
}

/**
 * Gets trump selection transitions
 */
function getTrumpSelectionTransitions(state: GameState): StateTransition[] {
  const transitions: StateTransition[] = [];
  
  if (!state.currentBid || state.winningBidder === null) {
    return transitions;
  }
  
  // Generate trump selection transitions
  Object.entries(TRUMP_SUITS).forEach(([name, trump]) => {
    const newState = cloneGameState(state);
    newState.phase = 'playing';
    newState.trump = trump;
    newState.currentPlayer = state.winningBidder!;
    
    transitions.push({
      id: `trump-${name.toLowerCase()}`,
      label: `Declare ${name} trump`,
      newState
    });
  });
  
  return transitions;
}

/**
 * Gets all valid playing transitions
 */
function getPlayingTransitions(state: GameState): StateTransition[] {
  const transitions: StateTransition[] = [];
  
  if (state.trump === null) return transitions;
  
  // If trick is complete, process it
  if (state.currentTrick.length === 4) {
    const winner = calculateTrickWinner(state.currentTrick, state.trump);
    const points = calculateTrickPoints(state.currentTrick);
    
    const newState = cloneGameState(state);
    newState.tricks.push({
      plays: [...state.currentTrick],
      winner,
      points
    });
    newState.currentTrick = [];
    newState.currentPlayer = winner;
    newState.teamScores[newState.players[winner].teamId] += points;
    
    if (newState.tricks.length === GAME_CONSTANTS.TRICKS_PER_HAND) {
      newState.phase = 'scoring';
    }
    
    transitions.push({
      id: 'complete-trick',
      label: `Player ${winner + 1} wins trick (${points} points)`,
      newState
    });
    
    return transitions;
  }
  
  // Generate play transitions for current player
  const player = state.players[state.currentPlayer];
  const validPlays = getValidPlays(player.hand, state.currentTrick, state.trump);
  
  validPlays.forEach(domino => {
    const newState = cloneGameState(state);
    const newPlayer = newState.players[state.currentPlayer];
    newPlayer.hand = newPlayer.hand.filter(d => d.id !== domino.id);
    newState.currentTrick.push({ player: state.currentPlayer, domino });
    newState.currentPlayer = getNextPlayer(state.currentPlayer);
    
    transitions.push({
      id: `play-${domino.id}`,
      label: `Play ${domino.high}-${domino.low}`,
      newState
    });
  });
  
  return transitions;
}

/**
 * Gets scoring phase transitions
 */
function getScoringTransitions(state: GameState): StateTransition[] {
  const transitions: StateTransition[] = [];
  
  const newMarks = calculateRoundScore(state);
  const newState = cloneGameState(state);
  newState.teamMarks = newMarks;
  
  if (isGameComplete(newMarks, state.gameTarget)) {
    newState.phase = 'game_end';
    
    transitions.push({
      id: 'score-hand',
      label: 'Score hand and end game',
      newState
    });
  } else {
    newState.phase = 'bidding';
    newState.dealer = getNextDealer(state.dealer);
    newState.currentPlayer = getPlayerLeftOfDealer(newState.dealer);
    
    // Deal new hands
    const hands = dealDominoes();
    newState.players.forEach((player, i) => {
      player.hand = hands[i];
    });
    
    // Reset round state
    newState.bids = [];
    newState.currentBid = null;
    newState.winningBidder = null;
    newState.trump = null;
    newState.tricks = [];
    newState.currentTrick = [];
    newState.teamScores = [0, 0];
    
    transitions.push({
      id: 'score-hand',
      label: 'Score hand and deal next',
      newState
    });
  }
  
  return transitions;
}