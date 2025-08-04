import type { GameState, GameAction, TrumpSelection, Bid, StateTransition } from '../types';
import { EMPTY_BID } from '../types';
import { cloneGameState } from './state';
import { BID_TYPES, TRUMP_SELECTIONS, GAME_CONSTANTS } from '../constants';
import { isValidBid, getValidPlays, getBidComparisonValue } from './rules';
import { dealDominoesWithSeed, getDominoSuit } from './dominoes';
import { calculateTrickWinner, calculateTrickPoints, calculateRoundScore, isGameComplete } from './scoring';
import { getNextDealer, getPlayerLeftOfDealer, getNextPlayer } from './players';
import { analyzeSuits } from './suit-analysis';

/**
 * Game Engine with action-based state management and history tracking
 */
export class GameEngine {
  private state: GameState;
  private history: GameAction[] = [];
  private stateSnapshots: GameState[] = [];

  constructor(initialState: GameState) {
    this.state = cloneGameState(initialState);
  }

  /**
   * Gets the current game state (read-only)
   */
  getState(): GameState {
    return this.state;
  }

  /**
   * Gets the action history
   */
  getHistory(): GameAction[] {
    return [...this.history];
  }

  /**
   * Executes an action and updates the state
   */
  executeAction(action: GameAction): void {
    // Save current state for undo
    this.stateSnapshots.push(cloneGameState(this.state));
    this.history.push(action);
    
    // Apply the action
    this.state = applyAction(this.state, action);
  }

  /**
   * Undoes the last action
   */
  undo(): boolean {
    if (this.stateSnapshots.length === 0) {
      return false;
    }
    
    this.history.pop();
    this.state = this.stateSnapshots.pop()!;
    return true;
  }

  /**
   * Gets all valid actions for the current state
   */
  getValidActions(): GameAction[] {
    return getValidActions(this.state);
  }

  /**
   * Resets the game engine to a new state
   */
  reset(newState: GameState): void {
    this.state = cloneGameState(newState);
    this.history = [];
    this.stateSnapshots = [];
  }
}

/**
 * Pure function that applies an action to a game state
 */
export function applyAction(state: GameState, action: GameAction): GameState {
  switch (action.type) {
    case 'bid':
      return applyBid(state, action.player, action.bidType, action.value);
    case 'pass':
      return applyPass(state, action.player);
    case 'select-trump':
      return applyTrumpSelection(state, action.player, action.selection);
    case 'play':
      return applyPlay(state, action.player, action.dominoId);
    case 'complete-trick':
      return applyCompleteTrick(state);
    case 'score-hand':
      return applyScoreHand(state);
    case 'redeal':
      return applyRedeal(state);
    default:
      return state;
  }
}

/**
 * Applies a bid action
 */
function applyBid(state: GameState, player: number, bidType: Bid['type'], value?: number): GameState {
  const newState = cloneGameState(state);
  const bid: Bid = { type: bidType, value, player };
  
  newState.bids.push(bid);
  newState.currentBid = bid;
  newState.currentPlayer = getNextPlayer(player);
  
  // Check if bidding is complete
  if (newState.bids.length === 4) {
    const nonPassBids = newState.bids.filter(b => b.type !== BID_TYPES.PASS);
    
    if (nonPassBids.length > 0) {
      // Find winning bidder
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
  }
  
  return newState;
}

/**
 * Applies a pass action
 */
function applyPass(state: GameState, player: number): GameState {
  const newState = cloneGameState(state);
  const passBid: Bid = { type: BID_TYPES.PASS, player };
  
  newState.bids.push(passBid);
  newState.currentPlayer = getNextPlayer(player);
  
  // Check if bidding is complete
  if (newState.bids.length === 4) {
    const nonPassBids = newState.bids.filter(b => b.type !== BID_TYPES.PASS);
    
    if (nonPassBids.length > 0) {
      // Find winning bidder
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
    // All pass case is handled in getValidActions
  }
  
  return newState;
}

/**
 * Applies trump selection
 */
function applyTrumpSelection(state: GameState, player: number, selection: TrumpSelection): GameState {
  const newState = cloneGameState(state);
  newState.phase = 'playing';
  newState.trump = selection;
  newState.currentPlayer = player;
  
  // Update suit analysis for all players after trump is declared
  newState.players.forEach(p => {
    p.suitAnalysis = analyzeSuits(p.hand, selection);
  });
  
  return newState;
}

/**
 * Applies a domino play
 */
function applyPlay(state: GameState, player: number, dominoId: string): GameState {
  const newState = cloneGameState(state);
  const playerState = newState.players[player];
  const domino = playerState.hand.find(d => d.id === dominoId);
  
  if (!domino) return state; // Invalid play
  
  // Remove domino from hand
  playerState.hand = playerState.hand.filter(d => d.id !== dominoId);
  
  // Add to current trick
  newState.currentTrick.push({ player, domino });
  
  // Set current suit if this is the first play of the trick
  if (state.currentTrick.length === 0) {
    newState.currentSuit = getDominoSuit(domino, state.trump);
  }
  
  // Update player's suit analysis
  playerState.suitAnalysis = analyzeSuits(playerState.hand, state.trump);
  
  // Move to next player
  newState.currentPlayer = getNextPlayer(player);
  
  return newState;
}

/**
 * Applies trick completion
 */
function applyCompleteTrick(state: GameState): GameState {
  if (state.currentTrick.length !== 4) return state; // Not ready to complete
  
  const newState = cloneGameState(state);
  const winner = calculateTrickWinner(state.currentTrick, state.trump, state.currentSuit);
  const points = calculateTrickPoints(state.currentTrick);
  
  // Add completed trick
  newState.tricks.push({
    plays: [...state.currentTrick],
    winner,
    points,
    ledSuit: state.currentSuit
  });
  
  // Clear current trick
  newState.currentTrick = [];
  newState.currentSuit = -1;
  
  // Award points to winning team
  newState.teamScores[newState.players[winner].teamId] += points + 1; // +1 for the trick
  
  // Set next player
  newState.currentPlayer = winner;
  
  // Check if hand is complete
  if (newState.tricks.length === GAME_CONSTANTS.TRICKS_PER_HAND) {
    newState.phase = 'scoring';
  }
  
  return newState;
}

/**
 * Applies hand scoring
 */
function applyScoreHand(state: GameState): GameState {
  const newState = cloneGameState(state);
  const newMarks = calculateRoundScore(state);
  newState.teamMarks = newMarks;
  
  if (isGameComplete(newMarks, state.gameTarget)) {
    newState.phase = 'game_end';
  } else {
    // Start new hand
    newState.phase = 'bidding';
    newState.dealer = getNextDealer(state.dealer);
    newState.currentPlayer = getPlayerLeftOfDealer(newState.dealer);
    
    // Deal new hands
    newState.shuffleSeed = state.shuffleSeed + 1000000;
    const hands = dealDominoesWithSeed(newState.shuffleSeed);
    newState.players.forEach((player, i) => {
      player.hand = hands[i];
      player.suitAnalysis = analyzeSuits(hands[i]); // No trump yet
    });
    
    // Reset round state
    newState.bids = [];
    newState.currentBid = EMPTY_BID;
    newState.winningBidder = -1;
    newState.trump = { type: 'none' };
    newState.tricks = [];
    newState.currentTrick = [];
    newState.currentSuit = -1;
    newState.teamScores = [0, 0];
  }
  
  return newState;
}

/**
 * Applies redeal (all players passed)
 */
function applyRedeal(state: GameState): GameState {
  const newState = cloneGameState(state);
  
  // New dealer and shuffle
  newState.dealer = getNextDealer(state.dealer);
  newState.currentPlayer = getPlayerLeftOfDealer(newState.dealer);
  newState.shuffleSeed = state.shuffleSeed + 1000000;
  
  // Deal new hands
  const hands = dealDominoesWithSeed(newState.shuffleSeed);
  newState.players.forEach((player, i) => {
    player.hand = hands[i];
    player.suitAnalysis = analyzeSuits(hands[i]); // No trump yet
  });
  
  // Reset bidding
  newState.bids = [];
  newState.currentBid = EMPTY_BID;
  
  return newState;
}

/**
 * Gets all valid actions for the current state
 */
export function getValidActions(state: GameState): GameAction[] {
  switch (state.phase) {
    case 'bidding':
      return getBiddingActions(state);
    case 'trump_selection':
      return getTrumpSelectionActions(state);
    case 'playing':
      return getPlayingActions(state);
    case 'scoring':
      return getScoringActions();
    default:
      return [];
  }
}

/**
 * Gets valid bidding actions
 */
function getBiddingActions(state: GameState): GameAction[] {
  const actions: GameAction[] = [];
  
  // Check if bidding is complete
  if (state.bids.length === 4) {
    const nonPassBids = state.bids.filter(b => b.type !== BID_TYPES.PASS);
    if (nonPassBids.length === 0) {
      // All passed - need redeal
      actions.push({ type: 'redeal' });
      return actions;
    }
    // Otherwise, bidding is complete - no more actions
    return actions;
  }
  
  // Check if current player has already bid
  if (state.bids.some(b => b.player === state.currentPlayer)) {
    return actions;
  }
  
  const currentPlayerHand = state.players[state.currentPlayer].hand;
  
  // Pass action
  actions.push({ type: 'pass', player: state.currentPlayer });
  
  // Point bids
  for (let points = GAME_CONSTANTS.MIN_BID; points <= GAME_CONSTANTS.MAX_BID; points++) {
    const bid: Bid = { type: BID_TYPES.POINTS, value: points, player: state.currentPlayer };
    if (isValidBid(state, bid, currentPlayerHand)) {
      actions.push({ type: 'bid', player: state.currentPlayer, bidType: BID_TYPES.POINTS, value: points });
    }
  }
  
  // Mark bids
  for (let marks = 1; marks <= 4; marks++) {
    const bid: Bid = { type: BID_TYPES.MARKS, value: marks, player: state.currentPlayer };
    if (isValidBid(state, bid, currentPlayerHand)) {
      actions.push({ type: 'bid', player: state.currentPlayer, bidType: BID_TYPES.MARKS, value: marks });
    }
  }
  
  // Special contracts in non-tournament mode
  if (!state.tournamentMode) {
    // Nello, Splash, Plunge - similar logic as above
    // Implementation omitted for brevity but would follow same pattern
  }
  
  return actions;
}

/**
 * Gets valid trump selection actions
 */
function getTrumpSelectionActions(state: GameState): GameAction[] {
  const actions: GameAction[] = [];
  
  if (state.winningBidder === -1) return actions;
  
  // Generate trump selection actions
  Object.values(TRUMP_SELECTIONS).forEach(trumpSelection => {
    actions.push({
      type: 'select-trump',
      player: state.winningBidder,
      selection: trumpSelection
    });
  });
  
  return actions;
}

/**
 * Gets valid playing actions
 */
function getPlayingActions(state: GameState): GameAction[] {
  const actions: GameAction[] = [];
  
  if (state.trump.type === 'none') return actions;
  
  // If trick is complete, complete it
  if (state.currentTrick.length === 4) {
    actions.push({ type: 'complete-trick' });
    return actions;
  }
  
  // Get valid plays for current player
  const validPlays = getValidPlays(state, state.currentPlayer);
  validPlays.forEach(domino => {
    actions.push({
      type: 'play',
      player: state.currentPlayer,
      dominoId: domino.id.toString()
    });
  });
  
  return actions;
}

/**
 * Gets valid scoring actions
 */
function getScoringActions(): GameAction[] {
  return [{ type: 'score-hand' }];
}

/**
 * Converts an action to a transition ID for compatibility
 */
export function actionToId(action: GameAction): string {
  switch (action.type) {
    case 'bid':
      if (action.bidType === BID_TYPES.POINTS) {
        return `bid-${action.value}`;
      } else if (action.bidType === BID_TYPES.MARKS) {
        return `bid-${action.value}-marks`;
      }
      return `${action.bidType}-${action.value}`;
    case 'pass':
      return 'pass';
    case 'select-trump':
      if (action.selection.type === 'suit') {
        const suitNames = ['blanks', 'ones', 'twos', 'threes', 'fours', 'fives', 'sixes'];
        return `trump-${suitNames[action.selection.suit!]}`;
      } else if (action.selection.type === 'doubles') {
        return 'trump-doubles';
      } else if (action.selection.type === 'no-trump') {
        return 'trump-no-trump';
      }
      return 'trump-none';
    case 'play':
      return `play-${action.dominoId}`;
    case 'complete-trick':
      return 'complete-trick';
    case 'score-hand':
      return 'score-hand';
    case 'redeal':
      return 'redeal';
    default:
      return 'unknown';
  }
}

/**
 * Converts an action to a human-readable label
 */
export function actionToLabel(action: GameAction): string {
  switch (action.type) {
    case 'bid':
      if (action.bidType === BID_TYPES.POINTS) {
        return `Bid ${action.value} points`;
      } else if (action.bidType === BID_TYPES.MARKS) {
        return `Bid ${action.value} mark${action.value! > 1 ? 's' : ''}`;
      }
      return `${action.bidType} ${action.value}`;
    case 'pass':
      return 'Pass';
    case 'select-trump':
      if (action.selection.type === 'suit') {
        const suitNames = ['Blanks', 'Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes'];
        return `Declare ${suitNames[action.selection.suit!]} trump`;
      } else if (action.selection.type === 'doubles') {
        return 'Declare Doubles trump';
      } else if (action.selection.type === 'no-trump') {
        return 'Declare No-trump';
      }
      return 'Select trump';
    case 'play':
      // Would need domino lookup for proper label
      return `Play domino ${action.dominoId}`;
    case 'complete-trick':
      return 'Complete trick';
    case 'score-hand':
      return 'Score hand';
    case 'redeal':
      return 'All passed - Redeal';
    default:
      return 'Unknown action';
  }
}

/**
 * Core state machine function that returns all possible next states
 * Now internally uses the action system for consistency
 */
export function getNextStates(state: GameState): StateTransition[] {
  const validActions = getValidActions(state);
  return validActions.map(action => ({
    id: actionToId(action),
    label: actionToLabel(action), 
    newState: applyAction(state, action)
  }));
}
