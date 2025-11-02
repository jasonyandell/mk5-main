import type { GameState, GameAction, StateTransition, Bid, TrumpSelection, Domino } from '../types';
import type { GameLayer, GameRules } from '../layers/types';
import { cloneGameState } from './state';
import { executeAction } from './actions';
import { BID_TYPES, TRUMP_SELECTIONS, GAME_CONSTANTS } from '../constants';
import { composeActions, composeRules } from '../layers/compose';
import { baseLayer } from '../layers';

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
export function getValidActions(state: GameState, layers?: readonly GameLayer[], rules?: GameRules): GameAction[] {
  let baseActions: GameAction[];

  switch (state.phase) {
    case 'bidding':
      baseActions = getBiddingActions(state, rules);
      break;
    case 'trump_selection':
      baseActions = getTrumpSelectionActions(state);
      break;
    case 'playing':
      baseActions = getPlayingActions(state, rules);
      break;
    case 'scoring':
      baseActions = getScoringActions(state);
      break;
    default:
      baseActions = [];
  }

  // Apply layer transformations if provided
  if (layers && layers.length > 0) {
    return composeActions(layers, state, baseActions);
  }

  return baseActions;
}

/**
 * Gets valid bidding actions
 */
function getBiddingActions(state: GameState, rules?: GameRules): GameAction[] {
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

  // Helper to validate bids using composed rules or base layer as fallback
  const defaultRules = rules || composeRules([baseLayer]);
  const validateBid = (bid: Bid, hand?: Domino[]) =>
    defaultRules.isValidBid(state, bid, hand);

  // Pass action
  actions.push({ type: 'pass', player: state.currentPlayer });

  // Point bids
  for (let points = GAME_CONSTANTS.MIN_BID; points <= GAME_CONSTANTS.MAX_BID; points++) {
    const bid: Bid = { type: BID_TYPES.POINTS, value: points, player: state.currentPlayer };
    if (validateBid(bid, currentPlayerHand)) {
      actions.push({ type: 'bid', player: state.currentPlayer, bid: BID_TYPES.POINTS, value: points });
    }
  }

  // Mark bids
  for (let marks = 1; marks <= 4; marks++) {
    const bid: Bid = { type: BID_TYPES.MARKS, value: marks, player: state.currentPlayer };
    if (validateBid(bid, currentPlayerHand)) {
      actions.push({ type: 'bid', player: state.currentPlayer, bid: BID_TYPES.MARKS, value: marks });
    }
  }

  // Special contracts - always generate in base engine (variants will filter)
  // Nello bids
  for (let marks = 1; marks <= 4; marks++) {
    const bid: Bid = { type: BID_TYPES.NELLO, value: marks, player: state.currentPlayer };
    if (validateBid(bid, currentPlayerHand)) {
      actions.push({ type: 'bid', player: state.currentPlayer, bid: BID_TYPES.NELLO, value: marks });
    }
  }

  // Splash bids
  for (let marks = 2; marks <= 3; marks++) {
    const bid: Bid = { type: BID_TYPES.SPLASH, value: marks, player: state.currentPlayer };
    if (validateBid(bid, currentPlayerHand)) {
      actions.push({ type: 'bid', player: state.currentPlayer, bid: BID_TYPES.SPLASH, value: marks });
    }
  }

  // Plunge bids
  for (let marks = 4; marks <= 6; marks++) {
    const bid: Bid = { type: BID_TYPES.PLUNGE, value: marks, player: state.currentPlayer };
    if (validateBid(bid, currentPlayerHand)) {
      actions.push({ type: 'bid', player: state.currentPlayer, bid: BID_TYPES.PLUNGE, value: marks });
    }
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
  // Use currentPlayer (which may be partner for plunge/splash) not winningBidder
  Object.values(TRUMP_SELECTIONS).forEach(trumpSelection => {
    actions.push({
      type: 'select-trump',
      player: state.currentPlayer,
      trump: trumpSelection as TrumpSelection
    });
  });

  return actions;
}

/**
 * Gets valid playing actions
 */
function getPlayingActions(state: GameState, rules?: GameRules): GameAction[] {
  const actions: GameAction[] = [];

  if (state.trump.type === 'not-selected') return actions;

  // Check if trick is complete (use rules if provided, otherwise default to 4)
  const isTrickComplete = rules ? rules.isTrickComplete(state) : state.currentTrick.length === 4;

  // If trick is complete, add consensus actions
  if (isTrickComplete) {
    // All players who haven't agreed yet can agree (not just current player)
    // This is important for nello where the partner sits out but still needs to agree
    for (let playerId = 0; playerId < 4; playerId++) {
      if (!state.consensus.completeTrick.has(playerId)) {
        actions.push({ type: 'agree-complete-trick', player: playerId });
      }
    }

    // If all have agreed, the trick can be completed
    if (state.consensus.completeTrick.size === 4) {
      actions.push({ type: 'complete-trick' });
    }
    return actions;
  }
  
  // Get valid plays for current player using threaded rules
  const threadedRules = rules || composeRules([baseLayer]);
  const validPlays = threadedRules.getValidPlays(state, state.currentPlayer);
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
  
  // Only the current player can agree to score the hand
  if (!state.consensus.scoreHand.has(state.currentPlayer)) {
    actions.push({ type: 'agree-score-hand', player: state.currentPlayer });
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
        return `${action.value}`;
      } else if (action.bid === BID_TYPES.MARKS) {
        return `${action.value} mark${action.value! > 1 ? 's' : ''}`;
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
      return 'Next hand';
    case 'agree-complete-trick':
      return action.player === 0 ? 'Complete trick' : `Player ${action.player} agrees to complete trick`;
    case 'agree-score-hand':
      return action.player === 0 ? 'Next hand' : `Player ${action.player} agrees to next hand`;
    case 'redeal':
      return 'All passed - Redeal';
    default:
      return 'Unknown action';
  }
}

/**
 * Core state machine function that returns all possible next states
 * Now internally uses the action system for consistency
 *
 * @param state Current game state
 * @param layers Optional array of layers to compose for action generation
 * @param rules Optional composed rules for action execution
 */
export function getNextStates(
  state: GameState,
  layers?: readonly GameLayer[],
  rules?: GameRules
): StateTransition[] {
  const validActions = getValidActions(state, layers, rules);
  return validActions.map(action => ({
    id: actionToId(action),
    label: actionToLabel(action),
    action: action,  // Include the action for audit trail
    newState: executeAction(state, action, rules)  // Use pure executeAction with optional rules
  }));
}

