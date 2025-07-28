import type { Trump, BidType, GameConstants } from './types';

export const GAME_CONSTANTS: GameConstants = {
  TOTAL_DOMINOES: 28,
  HAND_SIZE: 7,
  TOTAL_POINTS: 42,
  TRICKS_PER_HAND: 7,
  PLAYERS: 4,
  TEAMS: 2,
  MIN_BID: 30,
  MAX_BID: 41,
  DEFAULT_GAME_TARGET: 7,
};

export const BID_TYPES: Record<string, BidType> = {
  PASS: 'pass',
  POINTS: 'points',
  MARKS: 'marks',
  NELLO: 'nello',
  SPLASH: 'splash',
  PLUNGE: 'plunge',
} as const;

export const TRUMP_SUITS: Record<string, Trump> = {
  BLANKS: 0,
  ONES: 1,
  TWOS: 2,
  THREES: 3,
  FOURS: 4,
  FIVES: 5,
  SIXES: 6,
} as const;

export const SUIT_VALUES = TRUMP_SUITS;

export const DOMINO_VALUES = [
  [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6],
  [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6],
  [2, 2], [2, 3], [2, 4], [2, 5], [2, 6],
  [3, 3], [3, 4], [3, 5], [3, 6],
  [4, 4], [4, 5], [4, 6],
  [5, 5], [5, 6],
  [6, 6]
];

export const POINT_VALUES = new Map([
  [10, 10], // 5-5 = 10 points
  [20, 10], // 6-4 = 10 points  
  [5, 5],   // 5-0 = 5 points
  [5, 5],   // 4-1 = 5 points
  [5, 5],   // 3-2 = 5 points
  [11, 5],  // 6-5 = 5 points
  [12, 2],  // 6-6 = 2 points
]);

export const DOUBLES = [
  [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]
];