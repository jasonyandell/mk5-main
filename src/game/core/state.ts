import type { GameState, Player } from '../types';
import { GAME_CONSTANTS } from '../constants';
import { EMPTY_BID, NO_LEAD_SUIT, NO_BIDDER } from '../types';
import { dealDominoesWithSeed } from './dominoes';
import { getPlayerLeftOfDealer } from './players';
import { analyzeSuits } from './suit-analysis';

/**
 * Creates the initial game state in setup phase
 */
export function createSetupState(options?: { 
  shuffleSeed?: number, 
  dealer?: number, 
  tournamentMode?: boolean,
  playerTypes?: ('human' | 'ai')[]
}): GameState {
  const dealer = options?.dealer ?? 3; // Start with dealer as player 3 for deterministic tests
  const currentPlayer = getPlayerLeftOfDealer(dealer); // Player to left of dealer bids first
  
  return {
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
    tournamentMode: options?.tournamentMode ?? true,
    shuffleSeed: options?.shuffleSeed ?? Date.now(), // Initial seed for when dealing happens
    // Player control types - default: player 0 human, rest AI
    playerTypes: options?.playerTypes ?? ['human', 'ai', 'ai', 'ai'],
    // Consensus tracking for neutral actions
    consensus: {
      completeTrick: new Set<number>(),
      scoreHand: new Set<number>()
    },
    // Action history for replay and debugging
    actionHistory: [],
    // Test compatibility properties - empty hands in setup
    hands: {},
    bidWinner: -1, // -1 instead of null
    isComplete: false,
    winner: -1, // -1 instead of null
    // Theme as first-class citizen
    theme: 'business',
    colorOverrides: {}
  };
}

/**
 * Creates the initial game state with fresh hands dealt ready for bidding
 */
export function createInitialState(options?: { 
  shuffleSeed?: number, 
  dealer?: number, 
  tournamentMode?: boolean,
  playerTypes?: ('human' | 'ai')[],
  theme?: string,
  colorOverrides?: Record<string, string>
}): GameState {
  const dealer = options?.dealer ?? 3; // Start with dealer as player 3 for deterministic tests
  const currentPlayer = getPlayerLeftOfDealer(dealer); // Player to left of dealer bids first
  
  // Generate initial seed for deterministic shuffling
  const shuffleSeed = options?.shuffleSeed ?? Date.now();
  const hands = dealDominoesWithSeed(shuffleSeed);
  
  const initialState = {
    // Theme configuration (first-class citizen)
    theme: options?.theme ?? 'business',
    colorOverrides: options?.colorOverrides ?? {},
    
    // Game state
    phase: 'bidding' as const,
    players: [
      { id: 0, name: 'Player 1', hand: hands[0], teamId: 0 as const, marks: 0, suitAnalysis: analyzeSuits(hands[0]) },
      { id: 1, name: 'Player 2', hand: hands[1], teamId: 1 as const, marks: 0, suitAnalysis: analyzeSuits(hands[1]) },
      { id: 2, name: 'Player 3', hand: hands[2], teamId: 0 as const, marks: 0, suitAnalysis: analyzeSuits(hands[2]) },
      { id: 3, name: 'Player 4', hand: hands[3], teamId: 1 as const, marks: 0, suitAnalysis: analyzeSuits(hands[3]) },
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
    tournamentMode: options?.tournamentMode ?? true,
    shuffleSeed,
    // Player control types - default: player 0 human, rest AI
    playerTypes: options?.playerTypes ?? ['human', 'ai', 'ai', 'ai'],
    // Consensus tracking for neutral actions
    consensus: {
      completeTrick: new Set<number>(),
      scoreHand: new Set<number>()
    },
    // Action history for replay and debugging
    actionHistory: [],
    // Test compatibility properties
    hands: {
      0: hands[0],
      1: hands[1],
      2: hands[2],
      3: hands[3]
    },
    bidWinner: -1, // -1 instead of null
    isComplete: false,
    winner: -1, // -1 instead of null
  };
  
  return initialState;
}

/**
 * Creates a deep copy of the game state for immutable operations
 */
export function cloneGameState(state: GameState): GameState {
  const clonedState: GameState = {
    ...state,
    players: state.players.map(player => {
      const clonedPlayer: Player = {
        ...player,
        hand: [...player.hand]
      };
      
      if (player.suitAnalysis) {
        clonedPlayer.suitAnalysis = {
          count: { ...player.suitAnalysis.count },
          rank: {
            0: [...player.suitAnalysis.rank[0]],
            1: [...player.suitAnalysis.rank[1]],
            2: [...player.suitAnalysis.rank[2]],
            3: [...player.suitAnalysis.rank[3]],
            4: [...player.suitAnalysis.rank[4]],
            5: [...player.suitAnalysis.rank[5]],
            6: [...player.suitAnalysis.rank[6]],
            doubles: [...player.suitAnalysis.rank.doubles],
            trump: [...player.suitAnalysis.rank.trump]
          }
        };
      }
      
      return clonedPlayer;
    }),
    bids: [...state.bids],
    tricks: state.tricks.map(trick => ({
      ...trick,
      plays: [...trick.plays]
    })),
    currentTrick: [...state.currentTrick],
    currentSuit: state.currentSuit,
    teamScores: [...state.teamScores] as [number, number],
    teamMarks: [...state.teamMarks] as [number, number],
    // Clone consensus Sets
    consensus: {
      completeTrick: new Set(state.consensus.completeTrick),
      scoreHand: new Set(state.consensus.scoreHand)
    },
    // Clone action history
    actionHistory: [...state.actionHistory]
  };
  
  // Clone the hands object if it exists
  if (state.hands) {
    clonedState.hands = Object.fromEntries(
      Object.entries(state.hands).map(([playerId, hand]) => [
        playerId, 
        [...hand]
      ])
    );
  }
  
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