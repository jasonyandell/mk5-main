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
  Trump,
  GamePhase,
  GameConstants
} from './types';

export { 
  GAME_CONSTANTS, 
  BID_TYPES, 
  TRUMP_SUITS, 
  DOMINO_VALUES, 
  POINT_VALUES 
} from './constants';

// Core state management
export { 
  createInitialState, 
  cloneGameState, 
  validateGameState, 
  isGameComplete, 
  getWinningTeam,
  advanceToNextPhase
} from './core/state';

// Game actions and transitions
export { getNextStates } from './core/actions';

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
  dealDominoes, 
  shuffleDominoes,
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