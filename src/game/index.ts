// Public API exports for the game engine
export type { 
  GameState, 
  Player, 
  Domino, 
  Bid, 
  Trick, 
  Play, 
  StateTransition,
  BidType,
  TrumpSelection,
  GameAction,
  GameHistory,
  GamePhase,
  GameConstants
} from './types';


export { 
  GAME_CONSTANTS, 
  BID_TYPES,
  GAME_PHASES, 
  TRUMP_SELECTIONS, 
  DOMINO_VALUES, 
  POINT_VALUES 
} from './constants';

// Core state management
export { 
  createInitialState, 
  createSetupState,
  cloneGameState, 
  validateGameState, 
  isGameComplete, 
  getWinningTeam,
  advanceToNextPhase
} from './core/state';

// Player utilities
export {
  getNextDealer,
  getPlayerLeftOfDealer
} from './core/players';

// Game actions and transitions
// Note: GameEngine class has been removed - use ExecutionContext pattern instead

// Pure action generation functions
export {
  generateStructuralActions  // Low-level: generates structural actions only (use ctx.getValidActions instead)
} from './layers/base';

// Action utilities
export {
  actionToId,
  actionToLabel
} from './core/actions';

// State transitions
export {
  getNextStates  // Requires ExecutionContext parameter
} from './core/state';

// Pure action execution
export { executeAction } from './core/actions';

// Event sourcing / replay
// Rule validation
// NOTE: For rule validation (isValidPlay, getValidPlays, isValidBid, etc.), use the threaded rules system:
// import { composeRules, baseLayer } from './layers';
// const rules = composeRules([baseLayer]);
// rules.isValidPlay(state, domino, playerId)
// rules.getValidPlays(state, playerId)
// rules.isValidBid(state, bid, playerHand)
// rules.getBidComparisonValue(bid)
// rules.isValidTrump(trump)
// rules.getLedSuit(state, domino)
// rules.isTrump(state, domino)
// rules.rankInTrick(state, led, domino)
// rules.calculateTrickWinner(state, trick)
export {
  getTrickWinner,
  getTrickPoints,
  determineTrickWinner,
  getTrumpValue
} from './core/rules';

// Domino utilities
// NOTE: For rule-aware functions (led suit, trump checks, rankings), use rules.* methods:
// rules.getLedSuit(state, domino), rules.isTrump(state, domino), rules.rankInTrick(state, led, domino)
export {
  createDominoes,
  shuffleDominoesWithSeed,
  dealDominoesWithSeed,
  getDominoPoints,
  isDouble,
  countDoubles
} from './core/dominoes';

// Scoring
// NOTE: For trick winner calculation, use rules.calculateTrickWinner(state, trick) instead
// NOTE: For game completion checks, use isGameComplete/getWinningTeam from state.ts (takes GameState)
//       The raw marks versions (isTargetReached/getWinnerFromMarks) are for internal scoring use.
export {
  calculateTrickPoints,
  calculateRoundScore,
  calculateGameSummary,
  calculateGameScore,
  isTargetReached,
  getWinnerFromMarks
} from './core/scoring';

// URL compression utilities (v2 only)
export {
  compressEvents,
  decompressEvents,
  encodeGameUrl,
  decodeGameUrl,
  stateToUrl,
  type URLData
} from './core/url-compression';
