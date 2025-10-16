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
  GameConstants,
  PlayerView,
  PublicPlayer
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

// Game actions and transitions are now exported from gameEngine

// Action-based game engine
export { 
  GameEngine, 
  getValidActions, 
  actionToId, 
  actionToLabel,
  getNextStates 
} from './core/gameEngine';

// Pure action execution
export { executeAction } from './core/actions';

// Event sourcing / replay
export {
  replayActions,
  createInitialStateWithVariants,
  applyActionWithHistory
} from './core/replay';

// Player view system
export { getPlayerView } from './core/playerView';

// Rule validation
export { 
  isValidBid, 
  isValidOpeningBid,
  isValidPlay, 
  getValidPlays, 
  canFollowSuit, 
  getBidComparisonValue,
  getTrickWinner,
  getTrickPoints,
  determineTrickWinner,
  isValidTrump,
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