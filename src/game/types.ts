export interface Domino {
  high: number;
  low: number;
  id: string | number;
  points?: number;
}

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
export type Trump = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | { suit: string | number; followsSuit: boolean }; // 0-6: suits, 7: doubles, 8: no-trump

export interface Bid {
  type: BidType;
  value?: number;
  player: number;
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
}

export type GamePhase = 'setup' | 'bidding' | 'trump_selection' | 'playing' | 'scoring' | 'game_end';

export interface GameState {
  phase: GamePhase;
  players: Player[];
  currentPlayer: number;
  dealer: number;
  bids: Bid[];
  currentBid: Bid | null;
  winningBidder: number | null;
  trump: Trump | null;
  tricks: Trick[];
  currentTrick: Play[];
  teamScores: [number, number];
  teamMarks: [number, number];
  gameTarget: number;
  tournamentMode: boolean;
  shuffleSeed: number; // Seed for deterministic shuffling
  // Additional properties for test compatibility
  hands?: { [playerId: number]: Domino[] };
  bidWinner?: number | null;
  isComplete?: boolean;
  winner?: number | null;
}

export interface StateTransition {
  id: string;
  label: string;
  newState: GameState;
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