import type { GameState, GameAction, StateTransition, Bid, TrumpSelection, Domino } from '../types';
import { cloneGameState } from './state';
import { executeAction } from './actions';
import { BID_TYPES, TRUMP_SELECTIONS, GAME_CONSTANTS } from '../constants';
import { isValidBid, getValidPlays } from './rules';

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
    
    // Apply the action using pure executeAction
    this.state = executeAction(this.state, action);
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

// applyAction has been replaced by the pure executeAction from actions.ts
// The old apply functions have been removed - executeAction handles all actions purely
// Old apply functions removed - all action logic is now in the pure executeAction

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
      return getScoringActions(state);
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
  
  const currentPlayerData = state.players[state.currentPlayer];
  if (!currentPlayerData) {
    throw new Error(`Invalid current player index: ${state.currentPlayer}`);
  }
  const currentPlayerHand = currentPlayerData.hand;
  
  // Pass action
  actions.push({ type: 'pass', player: state.currentPlayer });
  
  // Point bids
  for (let points = GAME_CONSTANTS.MIN_BID; points <= GAME_CONSTANTS.MAX_BID; points++) {
    const bid: Bid = { type: BID_TYPES.POINTS, value: points, player: state.currentPlayer };
    if (isValidBid(state, bid, currentPlayerHand)) {
      actions.push({ type: 'bid', player: state.currentPlayer, bid: BID_TYPES.POINTS, value: points });
    }
  }
  
  // Mark bids
  for (let marks = 1; marks <= 4; marks++) {
    const bid: Bid = { type: BID_TYPES.MARKS, value: marks, player: state.currentPlayer };
    if (isValidBid(state, bid, currentPlayerHand)) {
      actions.push({ type: 'bid', player: state.currentPlayer, bid: BID_TYPES.MARKS, value: marks });
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
      trump: trumpSelection as TrumpSelection
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
  
  // If trick is complete, add consensus actions
  if (state.currentTrick.length === 4) {
    // Each player can agree to complete the trick
    for (let i = 0; i < 4; i++) {
      if (!state.consensus.completeTrick.has(i)) {
        actions.push({ type: 'agree-complete-trick', player: i });
      }
    }
    
    // If all have agreed, the trick can be completed
    if (state.consensus.completeTrick.size === 4) {
      actions.push({ type: 'complete-trick' });
    }
    return actions;
  }
  
  // Get valid plays for current player
  const validPlays = getValidPlays(state, state.currentPlayer);
  validPlays.forEach((domino: Domino) => {
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
function getScoringActions(state: GameState): GameAction[] {
  const actions: GameAction[] = [];
  
  // Each player can agree to score the hand
  for (let i = 0; i < 4; i++) {
    if (!state.consensus.scoreHand.has(i)) {
      actions.push({ type: 'agree-score-hand', player: i });
    }
  }
  
  // If all have agreed, the hand can be scored
  if (state.consensus.scoreHand.size === 4) {
    actions.push({ type: 'score-hand' });
  }
  
  return actions;
}

/**
 * Converts an action to a transition ID for compatibility
 */
export function actionToId(action: GameAction): string {
  switch (action.type) {
    case 'bid':
      if (action.bid === BID_TYPES.POINTS) {
        return `bid-${action.value}`;
      } else if (action.bid === BID_TYPES.MARKS) {
        return `bid-${action.value}-marks`;
      }
      return `${action.bid}-${action.value}`;
    case 'pass':
      return 'pass';
    case 'select-trump':
      if (action.trump.type === 'suit') {
        const suitNames = ['blanks', 'ones', 'twos', 'threes', 'fours', 'fives', 'sixes'];
        return `trump-${suitNames[action.trump.suit!]}`;
      } else if (action.trump.type === 'doubles') {
        return 'trump-doubles';
      } else if (action.trump.type === 'no-trump') {
        return 'trump-no-trump';
      }
      return 'trump-none';
    case 'play':
      return `play-${action.dominoId}`;
    case 'complete-trick':
      return 'complete-trick';
    case 'score-hand':
      return 'score-hand';
    case 'agree-complete-trick':
      return `agree-complete-trick-${action.player}`;
    case 'agree-score-hand':
      return `agree-score-hand-${action.player}`;
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
      if (action.bid === BID_TYPES.POINTS) {
        return `Bid ${action.value} points`;
      } else if (action.bid === BID_TYPES.MARKS) {
        return `Bid ${action.value} mark${action.value! > 1 ? 's' : ''}`;
      }
      return `${action.bid} ${action.value}`;
    case 'pass':
      return 'Pass';
    case 'select-trump':
      if (action.trump.type === 'suit') {
        const suitNames = ['Blanks', 'Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes'];
        return `Declare ${suitNames[action.trump.suit!]} trump`;
      } else if (action.trump.type === 'doubles') {
        return 'Declare Doubles trump';
      } else if (action.trump.type === 'no-trump') {
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
    case 'agree-complete-trick':
      return action.player === 0 ? 'Complete trick' : `Player ${action.player} agrees to complete trick`;
    case 'agree-score-hand':
      return action.player === 0 ? 'Score hand' : `Player ${action.player} agrees to score hand`;
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
    action: action,  // Include the action for audit trail
    newState: executeAction(state, action)  // Use pure executeAction
  }));
}
