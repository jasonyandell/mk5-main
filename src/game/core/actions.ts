import type { GameState, GameAction, TrumpSelection, Bid, Play, Trick, LedSuit } from '../types';
import type { GameRules } from '../rulesets/types';
import { composeRules, baseRuleSet } from '../rulesets';
import { EMPTY_BID, NO_BIDDER, NO_LEAD_SUIT } from '../types';
import { BID_TYPES, GAME_CONSTANTS } from '../constants';
import { dealDominoesWithSeed } from './dominoes';
import { calculateTrickPoints, isGameComplete } from './scoring';
import { getNextDealer, getPlayerLeftOfDealer, getNextPlayer } from './players';
import { analyzeSuits } from './suit-analysis';

// Default rules (base rule set only, no special contracts)
const defaultRules = composeRules([baseRuleSet]);

/**
 * Pure function that executes an action on a game state.
 * Throws errors on invalid actions.
 * Always appends to actionHistory for complete audit trail.
 *
 * @param state - Current game state
 * @param action - Action to execute
 * @param rules - Game rules (defaults to base rule set if not provided)
 */
export function executeAction(state: GameState, action: GameAction, rules: GameRules = defaultRules): GameState {
  // Always record the action in history (even if invalid)
  const newState: GameState = {
    ...state,
    actionHistory: [...state.actionHistory, action]
  };

  // Process the action based on type
  switch (action.type) {
    case 'bid':
      return executeBid(newState, action.player, action.bid, action.value, rules);

    case 'pass':
      return executePass(newState, action.player, rules);

    case 'select-trump':
      return executeTrumpSelection(newState, action.player, action.trump, rules);

    case 'play':
      return executePlay(newState, action.player, action.dominoId, rules);

    case 'agree-complete-trick':
      return executeAgreement(newState, action.player, 'completeTrick');

    case 'agree-score-hand':
      return executeAgreement(newState, action.player, 'scoreHand');

    case 'complete-trick':
      return executeCompleteTrick(newState, rules);

    case 'score-hand':
      return executeScoreHand(newState, rules);

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
    throw new Error(`Invalid player ID: ${player}`);
    //return state; // Invalid player, no-op
  }

  // Check if already agreed
  if (state.consensus[type].has(player)) {
    throw new Error(`Player ${player} already agreed to ${type}`);
    //return state; // Already agreed, no-op
  }

  // Record agreement and advance to next player
  const nextPlayer = getNextPlayer(player);

  return {
    ...state,
    consensus: {
      ...state.consensus,
      [type]: new Set([...state.consensus[type], player])
    },
    // Advance to next player for their turn to agree
    currentPlayer: nextPlayer
  };
}

/**
 * Executes a bid action
 */
function executeBid(state: GameState, player: number, bidType: Bid['type'], value: number | undefined, rules: GameRules): GameState {
  // Validate phase
  if (state.phase !== 'bidding') {
    throw new Error('Invalid phase for bidding');
//    return state; // Invalid phase, no-op
  }

  // Validate player
  const playerData = state.players[player];
  if (!playerData) {
    throw new Error(`Invalid player ID: ${player}`);
//    return state; // Invalid player, no-op
  }

  // Create bid object
  const bid: Bid = value !== undefined
    ? { type: bidType, value, player }
    : { type: bidType, player };

  // Validate bid legality
  if (!rules.isValidBid(state, bid, playerData.hand)) {
    throw new Error('Invalid bid');
//    return state; // Invalid bid, no-op
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
        const highestValue = rules.getBidComparisonValue(highest);
        const currentValue = rules.getBidComparisonValue(current);
        return currentValue > highestValue ? current : highest;
      });

      newPhase = 'trump_selection';
      newWinningBidder = winningBid.player;
      // Use rules to determine who selects trump
      newCurrentPlayer = rules.getTrumpSelector(state, winningBid);
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
function executePass(state: GameState, player: number, rules: GameRules): GameState {
  // Validate phase
  if (state.phase !== 'bidding') {
    throw new Error('Invalid phase for passing');
//    return state; // Invalid phase, no-op
  }

  // Validate player
  const playerData = state.players[player];
  if (!playerData) {
    throw new Error(`Invalid player ID: ${player}`);
//    return state; // Invalid player, no-op
  }

  const passBid: Bid = { type: BID_TYPES.PASS, player };

  // Validate pass legality
  if (!rules.isValidBid(state, passBid, playerData.hand)) {
    throw new Error('Invalid pass');
//    return state; // Invalid pass, no-op
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
        const highestValue = rules.getBidComparisonValue(highest);
        const currentValue = rules.getBidComparisonValue(current);
        return currentValue > highestValue ? current : highest;
      });

      newPhase = 'trump_selection';
      newWinningBidder = winningBid.player;
      // Use rules to determine who selects trump
      newCurrentPlayer = rules.getTrumpSelector(state, winningBid);
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
function executeTrumpSelection(state: GameState, player: number, selection: TrumpSelection, rules: GameRules): GameState {
  // Validate phase
  if (state.phase !== 'trump_selection') {
    throw new Error('Invalid phase for trump selection');
//    return state; // Invalid phase, no-op
  }

  // Validate player is winning bidder (or trump selector in variants)
  if (player !== state.currentPlayer) {
    throw new Error('Only trump selector can select trump');
//    return state; // Not trump selector, no-op
  }

  // Validate trump selection
  if (!rules.isValidTrump(selection)) {
    throw new Error('Invalid trump selection');
//    return state; // Invalid trump, no-op
  }

  // Update suit analysis for all players
  const newPlayers = state.players.map(p => ({
    ...p,
    suitAnalysis: analyzeSuits(p.hand, selection)
  }));

  // Use rules to determine who leads first trick
  const firstLeader = rules.getFirstLeader(state, player, selection);

  return {
    ...state,
    phase: 'playing',
    trump: selection,
    currentPlayer: firstLeader,
    players: newPlayers
  };
}

/**
 * Executes a domino play
 */
function executePlay(state: GameState, player: number, dominoId: string, rules: GameRules): GameState {
  // Validate phase
  if (state.phase !== 'playing') {
    throw new Error('Invalid phase for domino play');
//    return state; // Invalid phase, no-op
  }

  // Validate player
  const playerIndex = state.players.findIndex(p => p.id === player);
  if (playerIndex === -1) {
    throw new Error(`Invalid player ID: ${player}`);
//    return state; // Invalid player, no-op
  }

  const playerState = state.players[playerIndex];
  if (!playerState) {
    throw new Error(`Invalid player state for player ID: ${player}`);
//    return state; // Invalid player state, no-op
  }
  const domino = playerState.hand.find(d => d.id === dominoId);

  if (!domino) {
    throw new Error(`Player ${player} does not have domino ${dominoId}`);
//    return state; // Player doesn't have domino, no-op
  }

  // Validate play legality
  const validPlays = rules.getValidPlays(state, player);
  const isValid = validPlays.some(d => d.id === dominoId);

  if (!isValid) {
    throw new Error(`Invalid play: ${dominoId} by player ${player}`);
//    return state; // Invalid play, no-op
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

  // Set current suit if first play (use rules to determine led suit)
  const newCurrentSuit = state.currentTrick.length === 0
    ? rules.getLedSuit(state, domino)
    : state.currentSuit;

  // Use rules to determine next player
  const nextPlayer = rules.getNextPlayer(state, player);

  return {
    ...state,
    players: newPlayers,
    currentTrick: newCurrentTrick,
    currentSuit: newCurrentSuit,
    currentPlayer: nextPlayer
  };
}

/**
 * Executes trick completion
 */
function executeCompleteTrick(state: GameState, rules: GameRules): GameState {
  // Validate trick is complete (use rules)
  if (!rules.isTrickComplete(state)) {
    // throw new Error('Trick not complete');
    //return state; // Trick not complete, no-op
  }

  // Check consensus (all 4 players must agree)
  // Even in nello where only 3 play, all 4 players must agree to advance
  if (state.consensus.completeTrick.size !== 4) {
    throw new Error('Not all players agreed to complete trick');
//    return state; // Not all agreed, no-op
  }

  // Calculate trick outcome (use rules)
  const winner = rules.calculateTrickWinner(state, state.currentTrick);
  const points = calculateTrickPoints(state.currentTrick);

  // Get winner's team
  const winnerPlayer = state.players.find(p => p.id === winner);
  if (!winnerPlayer) {

    throw new Error(`Invalid winner player index: ${winner}`);
//    return state; // Invalid winner, no-op
  }

  // Update team scores
  const newTeamScores: [number, number] = [...state.teamScores];
  newTeamScores[winnerPlayer.teamId] += points + 1; // +1 for the trick

  // Add completed trick
  const completedTrick: Trick = {
    plays: [...state.currentTrick],
    winner,
    points: points + 1,  // Include the 1 point for winning the trick
  };

  // Only add ledSuit if it's a valid suit (not -1)
  if (state.currentSuit >= 0 && state.currentSuit <= 7) {
    completedTrick.ledSuit = state.currentSuit as LedSuit;
  }

  const newTricks = [...state.tricks, completedTrick];

  // Create temporary state to check hand outcome
  const tempState: GameState = {
    ...state,
    tricks: newTricks,
    teamScores: newTeamScores
  };

  // Determine next phase
  let newPhase = state.phase;
  if (newTricks.length === GAME_CONSTANTS.TRICKS_PER_HAND) {
    newPhase = 'scoring';
  } else {
    // Check if hand outcome is determined (use rules)
    const outcome = rules.checkHandOutcome(tempState);
    if (outcome && outcome.isDetermined) {
      newPhase = 'scoring';
    }
  }

  // Clear consensus for next trick
  return {
    ...state,
    tricks: newTricks,
    currentTrick: [],
    currentSuit: NO_LEAD_SUIT,
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
function executeScoreHand(state: GameState, rules: GameRules = defaultRules): GameState {
  // Validate phase
  if (state.phase !== 'scoring') {
    throw new Error('Invalid phase for scoring');
    //return state; // Invalid phase, no-op
  }

  // Check consensus (all 4 players must agree)
  if (state.consensus.scoreHand.size !== 4) {
    throw new Error('Not all players agreed to score hand');
//    return state; // Not all agreed, no-op
  }

  // Calculate round score
  const newMarks = rules.calculateScore(state);

  // Check if game is complete
  if (isGameComplete(newMarks, state.gameTarget)) {
    // Game over - clear hands
    const newPlayers = state.players.map(p => ({
      ...p,
      hand: []
    }));

    return {
      ...state,
      phase: 'game_end',
      teamMarks: newMarks,
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
    throw new Error('Deal failed');
//    return state; // Deal failed, no-op
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
    winningBidder: NO_BIDDER,
    trump: { type: 'not-selected' },
    tricks: [],
    currentTrick: [],
    currentSuit: NO_LEAD_SUIT,
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
    throw new Error('Not all players passed');
//    return state; // Not all passed, no-op
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
    throw new Error('Deal failed');
//    return state; // Deal failed, no-op
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
