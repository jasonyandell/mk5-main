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
// import { composeRules, baseRuleSet } from './layers';
// const rules = composeRules([baseRuleSet]);
// rules.isValidPlay(state, domino, playerId)
// rules.getValidPlays(state, playerId)
// rules.isValidBid(state, bid, playerHand)
// rules.getBidComparisonValue(bid)
// rules.isValidTrump(trump)
export {
  canFollowSuit,
  getTrickWinner,
  getTrickPoints,
  determineTrickWinner,
  getTrumpValue
} from './core/rules';

// Domino utilities
export { 
  createDominoes, 
  shuffleDominoesWithSeed,
  dealDominoesWithSeed,
  getDominoSuit, 
  getDominoValue, 
  getDominoPoints, 
  isDouble, 
  countDoubles 
} from './core/dominoes';

// Scoring
export { 
  calculateTrickWinner, 
  calculateTrickPoints, 
  calculateRoundScore, 
  calculateGameSummary,
  calculateGameScore,
  getWinningTeam as getWinningTeamFromMarks
} from './core/scoring';

// URL compression utilities (v2 only)
export {
  compressEvents,
  decompressEvents,
  encodeGameUrl,
  decodeGameUrl,
  type URLData
} from './core/url-compression';
