import type { GameState, GameAction, TrumpSelection, Bid, Play } from '../types';
import { EMPTY_BID } from '../types';
import { BID_TYPES, GAME_CONSTANTS } from '../constants';
import { isValidBid, getValidPlays, getBidComparisonValue, isValidTrump } from './rules';
import { dealDominoesWithSeed, getDominoSuit } from './dominoes';
import { calculateTrickWinner, calculateTrickPoints, calculateRoundScore, isGameComplete } from './scoring';
import { checkHandOutcome } from './handOutcome';
import { getNextDealer, getPlayerLeftOfDealer, getNextPlayer } from './players';
import { analyzeSuits } from './suit-analysis';

/**
 * Pure function that executes an action on a game state.
 * Never throws errors - invalid actions return unchanged state.
 * Always appends to actionHistory for complete audit trail.
 */
export function executeAction(state: GameState, action: GameAction): GameState {
  // Always record the action in history (even if invalid)
  const newState: GameState = {
    ...state,
    actionHistory: [...state.actionHistory, action]
  };

  // Process the action based on type
  switch (action.type) {
    case 'bid':
      return executeBid(newState, action.player, action.bid, action.value);
    
    case 'pass':
      return executePass(newState, action.player);
    
    case 'select-trump':
      return executeTrumpSelection(newState, action.player, action.trump);
    
    case 'play':
      return executePlay(newState, action.player, action.dominoId);
    
    case 'agree-complete-trick':
      return executeAgreement(newState, action.player, 'completeTrick');
    
    case 'agree-score-hand':
      return executeAgreement(newState, action.player, 'scoreHand');
    
    case 'complete-trick':
      return executeCompleteTrick(newState);
    
    case 'score-hand':
      return executeScoreHand(newState);
    
    case 'redeal':
      return executeRedeal(newState);
    
    default:
      // Unknown action type - return state with action recorded
      return newState;
  }
}

/**
 * Records a player's agreement for consensus actions
 */
function executeAgreement(state: GameState, player: number, type: 'completeTrick' | 'scoreHand'): GameState {
  // Validate player
  if (player < 0 || player >= 4) {
    return state; // Invalid player, no-op
  }

  // Check if already agreed
  if (state.consensus[type].has(player)) {
    return state; // Already agreed, no-op
  }

  // Record agreement
  return {
    ...state,
    consensus: {
      ...state.consensus,
      [type]: new Set([...state.consensus[type], player])
    }
  };
}

/**
 * Executes a bid action
 */
function executeBid(state: GameState, player: number, bidType: Bid['type'], value?: number): GameState {
  // Validate phase
  if (state.phase !== 'bidding') {
    return state; // Invalid phase, no-op
  }

  // Validate player
  const playerData = state.players[player];
  if (!playerData) {
    return state; // Invalid player, no-op
  }

  // Create bid object
  const bid: Bid = value !== undefined 
    ? { type: bidType, value, player }
    : { type: bidType, player };

  // Validate bid legality
  if (!isValidBid(state, bid, playerData.hand)) {
    return state; // Invalid bid, no-op
  }

  // Apply bid
  const newBids = [...state.bids, bid];
  let newPhase: GameState['phase'] = state.phase;
  let newWinningBidder = state.winningBidder;
  let newCurrentPlayer = getNextPlayer(player);
  let newCurrentBid = bid;

  // Check if bidding is complete
  if (newBids.length === 4) {
    const nonPassBids = newBids.filter(b => b.type !== BID_TYPES.PASS);
    
    if (nonPassBids.length > 0) {
      // Find winning bidder
      const winningBid = nonPassBids.reduce((highest, current) => {
        const highestValue = getBidComparisonValue(highest);
        const currentValue = getBidComparisonValue(current);
        return currentValue > highestValue ? current : highest;
      });
      
      newPhase = 'trump_selection';
      newWinningBidder = winningBid.player;
      newCurrentPlayer = winningBid.player;
      newCurrentBid = winningBid;
    }
  }

  return {
    ...state,
    bids: newBids,
    currentBid: newCurrentBid,
    phase: newPhase,
    winningBidder: newWinningBidder,
    currentPlayer: newCurrentPlayer
  };
}

/**
 * Executes a pass action
 */
function executePass(state: GameState, player: number): GameState {
  // Validate phase
  if (state.phase !== 'bidding') {
    return state; // Invalid phase, no-op
  }

  // Validate player
  const playerData = state.players[player];
  if (!playerData) {
    return state; // Invalid player, no-op
  }

  const passBid: Bid = { type: BID_TYPES.PASS, player };

  // Validate pass legality
  if (!isValidBid(state, passBid, playerData.hand)) {
    return state; // Invalid pass, no-op
  }

  // Apply pass
  const newBids = [...state.bids, passBid];
  let newPhase: GameState['phase'] = state.phase;
  let newWinningBidder = state.winningBidder;
  let newCurrentPlayer = getNextPlayer(player);
  let newCurrentBid = state.currentBid;

  // Check if bidding is complete
  if (newBids.length === 4) {
    const nonPassBids = newBids.filter(b => b.type !== BID_TYPES.PASS);
    
    if (nonPassBids.length > 0) {
      // Find winning bidder
      const winningBid = nonPassBids.reduce((highest, current) => {
        const highestValue = getBidComparisonValue(highest);
        const currentValue = getBidComparisonValue(current);
        return currentValue > highestValue ? current : highest;
      });
      
      newPhase = 'trump_selection';
      newWinningBidder = winningBid.player;
      newCurrentPlayer = winningBid.player;
      newCurrentBid = winningBid;
    }
    // All pass case handled by redeal action
  }

  return {
    ...state,
    bids: newBids,
    currentBid: newCurrentBid,
    phase: newPhase,
    winningBidder: newWinningBidder,
    currentPlayer: newCurrentPlayer
  };
}

/**
 * Executes trump selection
 */
function executeTrumpSelection(state: GameState, player: number, selection: TrumpSelection): GameState {
  // Validate phase
  if (state.phase !== 'trump_selection') {
    return state; // Invalid phase, no-op
  }

  // Validate player is winning bidder
  if (player !== state.winningBidder) {
    return state; // Not winning bidder, no-op
  }

  // Validate trump selection
  if (!isValidTrump(selection)) {
    return state; // Invalid trump, no-op
  }

  // Update suit analysis for all players
  const newPlayers = state.players.map(p => ({
    ...p,
    suitAnalysis: analyzeSuits(p.hand, selection)
  }));

  return {
    ...state,
    phase: 'playing',
    trump: selection,
    currentPlayer: player,
    players: newPlayers
  };
}

/**
 * Executes a domino play
 */
function executePlay(state: GameState, player: number, dominoId: string): GameState {
  // Validate phase
  if (state.phase !== 'playing') {
    return state; // Invalid phase, no-op
  }

  // Validate player
  const playerIndex = state.players.findIndex(p => p.id === player);
  if (playerIndex === -1) {
    return state; // Invalid player, no-op
  }

  const playerState = state.players[playerIndex];
  if (!playerState) {
    return state; // Invalid player state, no-op
  }
  const domino = playerState.hand.find(d => d.id === dominoId);

  if (!domino) {
    return state; // Player doesn't have domino, no-op
  }

  // Validate play legality
  const validPlays = getValidPlays(state, player);
  const isValid = validPlays.some(d => d.id === dominoId);

  if (!isValid) {
    return state; // Invalid play, no-op
  }

  // Create new player with domino removed
  const newPlayer: typeof playerState = {
    ...playerState,
    hand: playerState.hand.filter(d => d.id !== dominoId),
    suitAnalysis: analyzeSuits(
      playerState.hand.filter(d => d.id !== dominoId),
      state.trump
    )
  };

  // Update players array
  const newPlayers = [...state.players];
  newPlayers[playerIndex] = newPlayer;

  // Add to current trick
  const newCurrentTrick: Play[] = [...state.currentTrick, { player, domino }];

  // Set current suit if first play
  const newCurrentSuit = state.currentTrick.length === 0 
    ? getDominoSuit(domino, state.trump)
    : state.currentSuit;

  return {
    ...state,
    players: newPlayers,
    currentTrick: newCurrentTrick,
    currentSuit: newCurrentSuit,
    currentPlayer: getNextPlayer(player)
  };
}

/**
 * Executes trick completion
 */
function executeCompleteTrick(state: GameState): GameState {
  // Validate trick is complete
  if (state.currentTrick.length !== 4) {
    return state; // Trick not complete, no-op
  }

  // Check consensus (all 4 players must agree)
  if (state.consensus.completeTrick.size !== 4) {
    return state; // Not all agreed, no-op
  }

  // Calculate trick outcome
  const winner = calculateTrickWinner(state.currentTrick, state.trump, state.currentSuit);
  const points = calculateTrickPoints(state.currentTrick);

  // Get winner's team
  const winnerPlayer = state.players.find(p => p.id === winner);
  if (!winnerPlayer) {
    return state; // Invalid winner, no-op
  }

  // Update team scores
  const newTeamScores: [number, number] = [...state.teamScores];
  newTeamScores[winnerPlayer.teamId] += points + 1; // +1 for the trick

  // Add completed trick
  const newTricks = [...state.tricks, {
    plays: [...state.currentTrick],
    winner,
    points: points + 1,  // Include the 1 point for winning the trick
    ledSuit: state.currentSuit
  }];

  // Determine next phase
  let newPhase = state.phase;
  if (newTricks.length === GAME_CONSTANTS.TRICKS_PER_HAND) {
    newPhase = 'scoring';
  } else {
    // Check if hand outcome is determined
    const tempState = { ...state, tricks: newTricks, teamScores: newTeamScores };
    const outcome = checkHandOutcome(tempState);
    if (outcome.isDetermined) {
      newPhase = 'scoring';
    }
  }

  // Clear consensus for next trick
  return {
    ...state,
    tricks: newTricks,
    currentTrick: [],
    currentSuit: -1,
    teamScores: newTeamScores,
    currentPlayer: winner,
    phase: newPhase,
    consensus: {
      ...state.consensus,
      completeTrick: new Set()  // Clear consensus
    }
  };
}

/**
 * Executes hand scoring
 */
function executeScoreHand(state: GameState): GameState {
  // Validate phase
  if (state.phase !== 'scoring') {
    return state; // Invalid phase, no-op
  }

  // Check consensus (all 4 players must agree)
  if (state.consensus.scoreHand.size !== 4) {
    return state; // Not all agreed, no-op
  }

  // Calculate round score
  const newMarks = calculateRoundScore(state);

  // Check if game is complete
  if (isGameComplete(newMarks, state.gameTarget)) {
    // Game over
    const winner = newMarks[0] >= state.gameTarget ? 0 : 1;
    
    // Clear hands
    const newPlayers = state.players.map(p => ({
      ...p,
      hand: []
    }));

    return {
      ...state,
      phase: 'game_end',
      teamMarks: newMarks,
      isComplete: true,
      winner,
      players: newPlayers,
      consensus: {
        completeTrick: new Set(),
        scoreHand: new Set()  // Clear consensus
      }
    };
  }

  // Start new hand
  const newDealer = getNextDealer(state.dealer);
  const newCurrentPlayer = getPlayerLeftOfDealer(newDealer);
  const newShuffleSeed = state.shuffleSeed + 1000000;

  // Deal new hands
  const hands = dealDominoesWithSeed(newShuffleSeed);
  const newPlayers = state.players.map((player, i) => {
    const hand = hands[i];
    if (!hand) {
      // If deal fails, return unchanged state
      return player;
    }
    return {
      ...player,
      hand,
      suitAnalysis: analyzeSuits(hand) // No trump yet
    };
  });

  // Validate all hands were dealt
  if (newPlayers.some(p => p.hand.length === 0 && state.phase !== 'game_end')) {
    return state; // Deal failed, no-op
  }

  return {
    ...state,
    phase: 'bidding',
    dealer: newDealer,
    currentPlayer: newCurrentPlayer,
    shuffleSeed: newShuffleSeed,
    players: newPlayers,
    bids: [],
    currentBid: EMPTY_BID,
    winningBidder: -1,
    trump: { type: 'none' },
    tricks: [],
    currentTrick: [],
    currentSuit: -1,
    teamScores: [0, 0],
    teamMarks: newMarks,
    consensus: {
      completeTrick: new Set(),
      scoreHand: new Set()  // Clear consensus
    }
  };
}

/**
 * Executes redeal (all players passed)
 */
function executeRedeal(state: GameState): GameState {
  // Validate all passed
  const nonPassBids = state.bids.filter(b => b.type !== BID_TYPES.PASS);
  if (nonPassBids.length > 0) {
    return state; // Not all passed, no-op
  }

  // New dealer and shuffle
  const newDealer = getNextDealer(state.dealer);
  const newCurrentPlayer = getPlayerLeftOfDealer(newDealer);
  const newShuffleSeed = state.shuffleSeed + 1000000;

  // Deal new hands
  const hands = dealDominoesWithSeed(newShuffleSeed);
  const newPlayers = state.players.map((player, i) => {
    const hand = hands[i];
    if (!hand) {
      return player; // Deal failed, keep old hand
    }
    return {
      ...player,
      hand,
      suitAnalysis: analyzeSuits(hand) // No trump yet
    };
  });

  // Validate all hands were dealt
  if (newPlayers.some(p => p.hand.length === 0)) {
    return state; // Deal failed, no-op
  }

  return {
    ...state,
    dealer: newDealer,
    currentPlayer: newCurrentPlayer,
    shuffleSeed: newShuffleSeed,
    players: newPlayers,
    bids: [],
    currentBid: EMPTY_BID
  };
}