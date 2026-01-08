export interface Domino {
  high: number;
  low: number;
  id: string | number;
  points?: number;
}

export type ConsensusAction = 'complete-trick' | 'score-hand';
export type PlayerType = 'human' | 'ai';

// ============= SUIT CONSTANTS =============
// Define all constants first so they can be used in type definitions
export const BLANKS = 0 as const;
export const ACES = 1 as const;
export const DEUCES = 2 as const;
export const TRES = 3 as const;
export const FOURS = 4 as const;
export const FIVES = 5 as const;
export const SIXES = 6 as const;
export const CALLED = 7 as const;
export const NO_TRUMP = 8 as const;

// Semantic constants for different "-1" contexts
export const NO_LEAD_SUIT = -1 as const;       // No domino has been led in current trick
export const TRUMP_NOT_SELECTED = -1 as const; // Trump hasn't been selected yet
export const PLAYED_AS_TRUMP = -1 as const;    // Domino is being played/analyzed as trump
export const NO_BIDDER = -1 as const;          // No player has won the bid yet

// ============= SUIT TYPES =============
// The 7 natural suits (pip values) - self-documenting with constant names
export type RegularSuit = 
  | typeof BLANKS   // 0
  | typeof ACES     // 1
  | typeof DEUCES   // 2
  | typeof TRES     // 3
  | typeof FOURS    // 4
  | typeof FIVES    // 5
  | typeof SIXES;   // 6

// What suit can be led (includes called suit 7 when dominoes are absorbed)
export type LedSuit = RegularSuit | typeof CALLED;
export type LedSuitOrNone = LedSuit | typeof NO_LEAD_SUIT;

// What can be selected as trump
export type TrumpSuit = RegularSuit | typeof CALLED | typeof NO_TRUMP;
export type TrumpSuitOrNone = TrumpSuit | typeof TRUMP_NOT_SELECTED;

// Enum-like object for convenient access
export const SUIT = {
  BLANKS: 0,
  ACES: 1,
  DEUCES: 2,
  TRES: 3,
  FOURS: 4,
  FIVES: 5,
  SIXES: 6,
  CALLED: 7,  // The 8th suit - where absorbed dominoes go
  NO_TRUMP: 8,
  NONE: -1
} as const;

// Type derived from the SUIT object
export type SuitValue = typeof SUIT[keyof typeof SUIT];

export interface SuitCount {
  0: number; // blanks
  1: number; // ones  
  2: number; // twos
  3: number; // threes
  4: number; // fours
  5: number; // fives
  6: number; // sixes
  doubles: number; // count of doubles
  trump: number; // count of trump dominoes (identical to trump suit when trump is declared)
}

export interface SuitRanking {
  0: Domino[]; // blanks
  1: Domino[]; // ones
  2: Domino[]; // twos
  3: Domino[]; // threes
  4: Domino[]; // fours
  5: Domino[]; // fives
  6: Domino[]; // sixes
  doubles: Domino[]; // all doubles
  trump: Domino[]; // all trump dominoes (identical to trump suit when trump is declared)
}

export interface SuitAnalysis {
  count: SuitCount;
  rank: SuitRanking;
}

export interface Player {
  id: number;
  name: string;
  hand: Domino[];
  teamId: 0 | 1;
  marks: number;
}

/**
 * Base bid types - always available (invariants)
 */
export type BaseBidType = 'pass' | 'points' | 'marks';

/**
 * Special bid types - compositional, enabled by layers
 * - splash: Requires splashLayer
 * - plunge: Requires plungeLayer
 */
export type SpecialBidType = 'splash' | 'plunge';

/**
 * All possible bid types in Texas 42.
 *
 * Base types (constants - always available):
 * - pass: Decline to bid
 * - points: Bid number of points (30-42)
 * - marks: Bid number of marks (1-7)
 *
 * Special types (compositional - enabled by layers):
 * - splash: Auto-bid requiring 3+ doubles (2-3 marks)
 * - plunge: Auto-bid requiring 4+ doubles (4+ marks)
 */
export type BidType = BaseBidType | SpecialBidType;

// Clean Trump type - no legacy support
export interface TrumpSelection {
  type: 'not-selected' | 'suit' | 'doubles' | 'no-trump' | 'nello' | 'sevens';
  suit?: RegularSuit;  // Only when type === 'suit'
}

export interface Bid {
  type: BidType;
  value?: number;
  player: number;
}

// Empty state for current bid instead of null
export const EMPTY_BID: Bid = {
  type: 'pass',
  player: -1
};

// Helper function to check if a bid is empty
export function isEmptyBid(bid: Bid): boolean {
  return bid.player === -1 && bid.type === 'pass';
}

export interface Play {
  player: number;
  domino: Domino;
}

export interface PlayedDomino {
  player: number;
  domino: Domino;
}

export interface Trick {
  plays: Play[];
  winner?: number;
  points: number;
  ledSuit?: LedSuit;
}

export type GamePhase = 'setup' | 'bidding' | 'trump_selection' | 'playing' | 'scoring' | 'game_end' | 'one-hand-complete';

export interface GameState {
  // Event sourcing: source of truth (config + actions = state)
  initialConfig: import('./types/config').GameConfig;

  // Theme configuration (first-class citizen)
  theme: string; // DaisyUI theme name (default: 'business')
  colorOverrides: Record<string, string>; // CSS variable overrides (e.g., '--p': '71.9967% 0.123825 62.756393')

  // Game state (all derived from initialConfig + actionHistory)
  phase: GamePhase;
  players: Player[];
  currentPlayer: number;
  dealer: number;
  bids: Bid[];
  currentBid: Bid; // Never null, uses EMPTY_BID
  winningBidder: number; // -1 during bidding instead of null
  trump: TrumpSelection; // Never null, uses { type: 'not-selected' }
  tricks: Trick[];
  currentTrick: Play[];
  currentSuit: LedSuitOrNone; // -1 when no trick in progress
  teamScores: [number, number];
  teamMarks: [number, number];
  gameTarget: number;
  shuffleSeed: number; // Seed for deterministic shuffling
  // Player control types - who is human vs AI (supports drop-in/drop-out)
  playerTypes: ('human' | 'ai')[];
  // Action history for replay and debugging
  actionHistory: GameAction[];
}

export type FilteredGameState = Omit<GameState, 'players'> & {
  players: Array<{
    id: number;
    name: string;
    teamId: 0 | 1;
    marks: number;
    hand: Domino[];  // Empty array if observer can't see this hand
    handCount: number;
  }>;
};

export interface StateTransition {
  id: string;
  label: string;
  action: GameAction;  // The action that creates this transition
  newState: GameState;
}

// Simplified Game Action types - pure data, no nesting
// Note: autoExecute and meta are optional action transformer extensions (not core game logic)
export type GameAction =
  | { type: 'bid'; player: number; bid: BidType; value?: number; autoExecute?: boolean; meta?: Record<string, unknown> }
  | { type: 'pass'; player: number; autoExecute?: boolean; meta?: Record<string, unknown> }
  | { type: 'select-trump'; player: number; trump: TrumpSelection; autoExecute?: boolean; meta?: Record<string, unknown> }
  | { type: 'play'; player: number; dominoId: string; autoExecute?: boolean; meta?: Record<string, unknown> }
  | { type: 'complete-trick'; autoExecute?: boolean; meta?: Record<string, unknown> }
  | { type: 'score-hand'; autoExecute?: boolean; meta?: Record<string, unknown> }
  | { type: 'agree-trick'; player: number; autoExecute?: boolean; meta?: Record<string, unknown> }
  | { type: 'agree-score'; player: number; autoExecute?: boolean; meta?: Record<string, unknown> }
  | { type: 'redeal'; autoExecute?: boolean; meta?: Record<string, unknown> }
  | { type: 'retry-one-hand'; autoExecute?: boolean; meta?: Record<string, unknown> }  // Retry one-hand mode with same seed
  | { type: 'new-one-hand'; autoExecute?: boolean; meta?: Record<string, unknown> }    // Start new one-hand mode with new seed

// History tracking for undo/redo
export interface GameHistory {
  actions: GameAction[];
  stateSnapshots: GameState[];
}

export interface GameConstants {
  TOTAL_DOMINOES: 28;
  HAND_SIZE: 7;
  TOTAL_POINTS: 42;
  TRICKS_PER_HAND: 7;
  PLAYERS: 4;
  TEAMS: 2;
  MIN_BID: 30;
  MAX_BID: 41;
  DEFAULT_GAME_TARGET: 7;
}
