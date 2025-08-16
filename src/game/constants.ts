import type { GameConstants, TrumpSelection } from './types';

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

export const BID_TYPES = {
  PASS: 'pass' as const,
  POINTS: 'points' as const,
  MARKS: 'marks' as const,
  NELLO: 'nello' as const,
  SPLASH: 'splash' as const,
  PLUNGE: 'plunge' as const,
};


// New TrumpSelection constants
export const TRUMP_SELECTIONS: Record<string, TrumpSelection> = {
  BLANKS: { type: 'suit', suit: 0 },
  ONES: { type: 'suit', suit: 1 },
  TWOS: { type: 'suit', suit: 2 },
  THREES: { type: 'suit', suit: 3 },
  FOURS: { type: 'suit', suit: 4 },
  FIVES: { type: 'suit', suit: 5 },
  SIXES: { type: 'suit', suit: 6 },
  DOUBLES: { type: 'doubles' },
  NO_TRUMP: { type: 'no-trump' },
} as const;

export const SUIT_VALUES = {
  BLANKS: 0,
  ONES: 1,
  TWOS: 2,
  THREES: 3,
  FOURS: 4,
  FIVES: 5,
  SIXES: 6,
} as const;


export const DOMINO_VALUES: readonly [number, number][] = [
  [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6],
  [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6],
  [2, 2], [2, 3], [2, 4], [2, 5], [2, 6],
  [3, 3], [3, 4], [3, 5], [3, 6],
  [4, 4], [4, 5], [4, 6],
  [5, 5], [5, 6],
  [6, 6]
] as const;

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