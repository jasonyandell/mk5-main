import type { GameState, StateTransition, GameAction, Domino } from '../types';
import type { ExecutionContext } from '../types/execution';
import { GAME_CONSTANTS } from '../constants';
import { EMPTY_BID, NO_LEAD_SUIT, NO_BIDDER } from '../types';
import { dealDominoesWithSeed } from './dominoes';
import { getPlayerLeftOfDealer } from './players';
import { executeAction } from './actions';
import { actionToId, actionToLabel } from './actions';
import { InvalidDealOverrideError } from '../types/config';

/**
 * Validates initialHands override with clear error messages.
 *
 * @throws {InvalidDealOverrideError} If hands invalid
 */
function validateInitialHands(hands: Domino[][]): void {
  // Check player count
  if (hands.length !== 4) {
    throw new InvalidDealOverrideError(
      `Must have exactly 4 players, got ${hands.length}`,
      { playerCount: hands.length }
    );
  }

  // Check each player has 7 dominoes
  hands.forEach((hand, i) => {
    if (hand.length !== 7) {
      throw new InvalidDealOverrideError(
        `Player ${i} must have 7 dominoes, got ${hand.length}`,
        { player: i, count: hand.length }
      );
    }
  });

  // Check for 28 unique dominoes
  const allDominoes = hands.flat();
  const ids = allDominoes.map(d => d.id);
  const uniqueIds = new Set(ids);

  if (uniqueIds.size !== 28) {
    const duplicates = ids.filter((id, i) => ids.indexOf(id) !== i);
    throw new InvalidDealOverrideError(
      `Must have 28 unique dominoes, got ${uniqueIds.size}`,
      { uniqueCount: uniqueIds.size, duplicates: Array.from(new Set(duplicates)) }
    );
  }

  // Verify all dominoes are valid (0-6 pips)
  allDominoes.forEach((d, i) => {
    if (d.high < 0 || d.high > 6 || d.low < 0 || d.low > 6) {
      throw new InvalidDealOverrideError(
        `Domino ${i} has invalid pips: [${d.high}, ${d.low}]`,
        { index: i, domino: d }
      );
    }
  });
}

/**
 * Creates the initial game state in setup phase
 */
export function createSetupState(options?: {
  shuffleSeed?: number,
  dealer?: number,
  playerTypes?: ('human' | 'ai')[],
  theme?: string,
  colorOverrides?: Record<string, string>
}): GameState {
  const dealer = options?.dealer ?? 3; // Start with dealer as player 3 for deterministic tests
  const currentPlayer = getPlayerLeftOfDealer(dealer); // Player to left of dealer bids first
  const shuffleSeed = options?.shuffleSeed ?? Date.now();
  const playerTypes = options?.playerTypes ?? ['human', 'ai', 'ai', 'ai'];

  const theme = options?.theme ?? 'business';
  const colorOverrides = options?.colorOverrides ?? {};

  return {
    // Event sourcing: initial configuration
    initialConfig: {
      playerTypes,
      shuffleSeed,
      theme,
      colorOverrides
    },

    phase: 'setup',
    players: [
      { id: 0, name: 'Player 1', hand: [], teamId: 0, marks: 0 },
      { id: 1, name: 'Player 2', hand: [], teamId: 1, marks: 0 },
      { id: 2, name: 'Player 3', hand: [], teamId: 0, marks: 0 },
      { id: 3, name: 'Player 4', hand: [], teamId: 1, marks: 0 },
    ],
    currentPlayer,
    dealer,
    bids: [],
    currentBid: EMPTY_BID,
    winningBidder: NO_BIDDER, // NO_BIDDER during bidding
    trump: { type: 'not-selected' as const }, // Never null, uses clear empty state
    tricks: [],
    currentTrick: [],
    currentSuit: NO_LEAD_SUIT, // NO_LEAD_SUIT when no trick in progress
    teamScores: [0, 0],
    teamMarks: [0, 0],
    gameTarget: GAME_CONSTANTS.DEFAULT_GAME_TARGET,
    shuffleSeed,
    // Player control types - default: player 0 human, rest AI
    playerTypes,
    // Action history for replay and debugging
    actionHistory: [],
    // Theme as first-class citizen
    theme,
    colorOverrides
  };
}

/**
 * Creates the initial game state with fresh hands dealt ready for bidding.
 *
 * Hand distribution priority:
 * 1. options.dealOverrides.initialHands (deterministic, exact hands)
 * 2. options.shuffleSeed (deterministic, shuffled)
 * 3. Date.now() seed (random, unrepeatable)
 *
 * Note: When both shuffleSeed and initialHands are specified:
 * - initialHands controls hand distribution
 * - shuffleSeed is still stored for other randomness (AI, tie-breaking, etc.)
 *
 * @throws {InvalidDealOverrideError} If dealOverrides.initialHands invalid
 */
export function createInitialState(options?: {
  shuffleSeed?: number,
  dealer?: number,
  playerTypes?: ('human' | 'ai')[],
  theme?: string,
  colorOverrides?: Record<string, string>,
  dealOverrides?: { initialHands?: Domino[][] },
  layers?: string[]
}): GameState {
  const dealer = options?.dealer ?? 3; // Start with dealer as player 3 for deterministic tests
  const currentPlayer = getPlayerLeftOfDealer(dealer); // Player to left of dealer bids first

  // Generate initial seed for deterministic shuffling
  const shuffleSeed = options?.shuffleSeed ?? Date.now();

  // Hand distribution: dealOverrides.initialHands > seed > random
  let hands: [Domino[], Domino[], Domino[], Domino[]];
  if (options?.dealOverrides?.initialHands) {
    validateInitialHands(options.dealOverrides.initialHands);
    hands = options.dealOverrides.initialHands as [Domino[], Domino[], Domino[], Domino[]];
  } else {
    hands = dealDominoesWithSeed(shuffleSeed);
  }

  const playerTypes = options?.playerTypes ?? ['human', 'ai', 'ai', 'ai'];
  const theme = options?.theme ?? 'business';
  const colorOverrides = options?.colorOverrides ?? {};

  const initialState = {
    // Event sourcing: initial configuration
    initialConfig: {
      playerTypes,
      shuffleSeed,
      theme,
      colorOverrides,
      ...(options?.dealOverrides && { dealOverrides: options.dealOverrides }),
      ...(options?.layers && options.layers.length > 0 && { layers: options.layers })
    },

    // Theme configuration (first-class citizen)
    theme,
    colorOverrides,

    // Game state
    phase: 'bidding' as const,
    players: [
      { id: 0, name: 'Player 1', hand: hands[0], teamId: 0 as const, marks: 0 },
      { id: 1, name: 'Player 2', hand: hands[1], teamId: 1 as const, marks: 0 },
      { id: 2, name: 'Player 3', hand: hands[2], teamId: 0 as const, marks: 0 },
      { id: 3, name: 'Player 4', hand: hands[3], teamId: 1 as const, marks: 0 },
    ],
    currentPlayer,
    dealer,
    bids: [],
    currentBid: EMPTY_BID,
    winningBidder: NO_BIDDER, // NO_BIDDER during bidding
    trump: { type: 'not-selected' as const }, // Never null, uses clear empty state
    tricks: [],
    currentTrick: [],
    currentSuit: NO_LEAD_SUIT, // NO_LEAD_SUIT when no trick in progress
    teamScores: [0, 0] as [number, number],
    teamMarks: [0, 0] as [number, number],
    gameTarget: GAME_CONSTANTS.DEFAULT_GAME_TARGET,
    shuffleSeed,
    // Player control types - default: player 0 human, rest AI
    playerTypes,
    // Action history for replay and debugging
    actionHistory: []
  };

  return initialState;
}

/**
 * Creates a deep copy of the game state for immutable operations
 */
export function cloneGameState(state: GameState): GameState {
  const clonedState: GameState = {
    ...state,
    // Clone initialConfig (spread all fields, then override arrays/objects that need cloning)
    initialConfig: {
      ...state.initialConfig,
      playerTypes: [...state.initialConfig.playerTypes],
      ...(state.initialConfig.colorOverrides ? { colorOverrides: { ...state.initialConfig.colorOverrides } } : {}),
      ...(state.initialConfig.layers ? { layers: [...state.initialConfig.layers] } : {})
    },
    players: state.players.map(player => ({
      ...player,
      hand: [...player.hand]
    })),
    bids: [...state.bids],
    tricks: state.tricks.map(trick => ({
      ...trick,
      plays: [...trick.plays]
    })),
    currentTrick: [...state.currentTrick],
    currentSuit: state.currentSuit,
    teamScores: [...state.teamScores] as [number, number],
    teamMarks: [...state.teamMarks] as [number, number],
    // Clone action history
    actionHistory: [...state.actionHistory]
  };

  return clonedState;
}

/**
 * Validates that a game state is well-formed
 */
export function validateGameState(state: GameState): string[] {
  const errors: string[] = [];

  // Check player count
  if (state.players.length !== GAME_CONSTANTS.PLAYERS) {
    errors.push(`Game must have exactly ${GAME_CONSTANTS.PLAYERS} players`);
  }

  // Check current player bounds
  if (state.currentPlayer < 0 || state.currentPlayer >= GAME_CONSTANTS.PLAYERS) {
    errors.push('Current player ID out of bounds');
  }

  // Check dealer bounds
  if (state.dealer < 0 || state.dealer >= GAME_CONSTANTS.PLAYERS) {
    errors.push('Dealer ID out of bounds');
  }

  // Check team scores and marks
  if (state.teamScores.length !== GAME_CONSTANTS.TEAMS) {
    errors.push(`Must have exactly ${GAME_CONSTANTS.TEAMS} team scores`);
  }
  if (state.teamMarks.length !== GAME_CONSTANTS.TEAMS) {
    errors.push(`Must have exactly ${GAME_CONSTANTS.TEAMS} team marks`);
  }

  // Check hand sizes (when not in setup)
  if (state.phase !== 'bidding' || state.bids.length === 0) {
    state.players.forEach((player, index) => {
      if (player.hand.length > GAME_CONSTANTS.HAND_SIZE) {
        errors.push(`Player ${index} has too many dominoes`);
      }
    });
  }

  return errors;
}

/**
 * Checks if the game is complete (any team reached target marks)
 */
export function isGameComplete(state: GameState): boolean {
  return state.teamMarks.some(marks => marks >= state.gameTarget);
}

/**
 * Gets the winning team (0 or 1), or null if game not complete
 */
export function getWinningTeam(state: GameState): number | null {
  if (!isGameComplete(state)) return null;
  
  return state.teamMarks[0] >= state.gameTarget ? 0 : 1;
}

/**
 * Advances game state to the next logical phase
 */
export function advanceToNextPhase(state: GameState): GameState {
  const newState = cloneGameState(state);

  switch (state.phase) {
    case 'setup':
      newState.phase = 'bidding';
      break;
    case 'bidding':
      newState.phase = 'trump_selection';
      break;
    case 'trump_selection':
      newState.phase = 'playing';
      break;
    case 'playing':
      newState.phase = 'scoring';
      break;
    case 'scoring':
      newState.phase = 'game_end';
      break;
    default:
      break;
  }

  return newState;
}

/**
 * Converts actions to state transitions (maps actions to ids/labels/newState).
 *
 * Uses the composed getValidActions from ExecutionContext - doesn't create its own composition.
 *
 * @param state Current game state
 * @param ctx Execution context with composed rules and getValidActions
 * @returns Array of state transitions with labels and new states
 */
export function getNextStates(
  state: GameState,
  ctx: ExecutionContext
): StateTransition[] {
  const validActions = ctx.getValidActions(state);

  return validActions.map((action: GameAction) => ({
    id: actionToId(action),
    label: actionToLabel(action),
    action: action,  // Include the action for audit trail
    newState: executeAction(state, action, ctx.rules)
  }));
}
