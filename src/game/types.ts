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
export const DOUBLES_AS_TRUMP = 7 as const;
export const NO_TRUMP = 8 as const;

// Semantic constants for different "-1" contexts
export const NO_LEAD_SUIT = -1 as const;       // No domino has been led in current trick
export const TRUMP_NOT_SELECTED = -1 as const; // Trump hasn't been selected yet
export const PLAYED_AS_TRUMP = -1 as const;    // Domino is being played/analyzed as trump
export const NO_BIDDER = -1 as const;          // No player has won the bid yet

// @deprecated Use specific semantic constants above
export const NO_SUIT = NO_LEAD_SUIT;

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

// What suit can be led (includes doubles as suit 7 when doubles are trump)
export type LedSuit = RegularSuit | typeof DOUBLES_AS_TRUMP;
export type LedSuitOrNone = LedSuit | typeof NO_LEAD_SUIT;

// What can be selected as trump
export type TrumpSuit = RegularSuit | typeof DOUBLES_AS_TRUMP | typeof NO_TRUMP;
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
  DOUBLES: 7,  // Only valid when doubles are trump
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
  suitAnalysis?: SuitAnalysis;
}

export type BidType = 'pass' | 'points' | 'marks' | 'nello' | 'splash' | 'plunge';

// Clean Trump type - no legacy support
export interface TrumpSelection {
  type: 'not-selected' | 'suit' | 'doubles' | 'no-trump';
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

export type GamePhase = 'setup' | 'bidding' | 'trump_selection' | 'playing' | 'scoring' | 'game_end';

export interface GameState {
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
  tournamentMode: boolean;
  shuffleSeed: number; // Seed for deterministic shuffling
  // Player control types - who is human vs AI (supports drop-in/drop-out)
  playerTypes: ('human' | 'ai')[];
  // Consensus tracking for neutral actions
  consensus: {
    completeTrick: Set<number>;  // Players who agreed to complete trick
    scoreHand: Set<number>;       // Players who agreed to score hand
  };
  // Action history for replay and debugging
  actionHistory: GameAction[];
  // Additional properties for test compatibility
  hands?: { [playerId: number]: Domino[] };
  bidWinner?: number; // -1 instead of null
  isComplete?: boolean;
  winner?: number; // -1 instead of null
  // Pure AI scheduling - part of game state for determinism
  aiSchedule: {
    [playerId: number]: {
      transition: StateTransition;
      executeAtTick: number;  // Game tick when this should execute
    }
  };
  // Game's internal clock (not wall time) for pure deterministic timing
  currentTick: number;
}

export interface StateTransition {
  id: string;
  label: string;
  action: GameAction;  // The action that creates this transition
  newState: GameState;
}

// Simplified Game Action types - pure data, no nesting
export type GameAction = 
  | { type: 'bid'; player: number; bid: BidType; value?: number }
  | { type: 'pass'; player: number }
  | { type: 'select-trump'; player: number; trump: TrumpSelection }
  | { type: 'play'; player: number; dominoId: string }
  | { type: 'agree-complete-trick'; player: number }  // Consensus action
  | { type: 'agree-score-hand'; player: number }     // Consensus action
  | { type: 'complete-trick' }  // Executed when all agree
  | { type: 'score-hand' }      // Executed when all agree
  | { type: 'redeal' }

// History tracking for undo/redo
export interface GameHistory {
  actions: GameAction[];
  stateSnapshots: GameState[];
}


// Type-safe public player without hand visibility
export interface PublicPlayer {
  id: number;
  name: string;
  teamId: 0 | 1;
  marks: number;
  handCount: number;  // No hand field exists in type!
}

// Player-specific view with privacy
export interface PlayerView {
  playerId: number;
  phase: GamePhase;
  self: { id: number; hand: Domino[] };  // Only self has hands
  players: PublicPlayer[];  // Others have no hand field
  validTransitions: StateTransition[];  // Only transitions this player can take
  consensus: {
    completeTrick: Set<number>;
    scoreHand: Set<number>;
  };
  currentTrick: Play[];
  tricks: Trick[];
  teamScores: [number, number];
  teamMarks: [number, number];
  trump: TrumpSelection;
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