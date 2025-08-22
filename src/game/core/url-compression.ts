import type { GameState } from '../types';
import { createInitialState } from './state';
import { dealDominoesWithSeed } from './dominoes';
import { getPlayerLeftOfDealer } from './players';
import { analyzeSuits } from './suit-analysis';

/**
 * Minimal state representation for URL storage
 * Only includes data that cannot be derived
 */
export interface MinimalGameState {
  s: number;  // shuffleSeed - determines hand dealing
  d?: number; // dealer (only if not default 3)
  t?: number; // gameTarget (only if not default 7)
  m?: boolean; // tournamentMode (only if false)
  p?: ('h' | 'a')[]; // player types: 'h'uman or 'a'i (only if not default [h,a,a,a])
}

/**
 * Compressed action representation
 */
export interface CompressedAction {
  i: string; // id (can be further shortened)
}

/**
 * URL data structure
 */
export interface URLData {
  v: 1; // version for future compatibility
  s: MinimalGameState;
  a: CompressedAction[];
}

/**
 * Compress a full GameState to minimal representation
 */
export function compressGameState(state: GameState): MinimalGameState {
  const minimal: MinimalGameState = {
    s: state.shuffleSeed
  };
  
  // Only include non-default values
  if (state.dealer !== 3) minimal.d = state.dealer;
  if (state.gameTarget !== 7) minimal.t = state.gameTarget;
  if (!state.tournamentMode) minimal.m = false;
  
  // Only include player types if not default [human, ai, ai, ai]
  const defaultTypes = ['human', 'ai', 'ai', 'ai'];
  if (state.playerTypes.some((t, i) => t !== defaultTypes[i])) {
    minimal.p = state.playerTypes.map(t => t === 'human' ? 'h' : 'a');
  }
  
  return minimal;
}

/**
 * Expand minimal state to full GameState
 */
export function expandMinimalState(minimal: MinimalGameState): GameState {
  // Create a state with the specific seed and player types
  const playerTypes = minimal.p ? 
    minimal.p.map(t => t === 'h' ? 'human' : 'ai') as ('human' | 'ai')[] :
    undefined;
  
  const state = createInitialState(playerTypes ? { playerTypes } : undefined);
  
  // Override with minimal state values
  state.shuffleSeed = minimal.s;
  if (minimal.d !== undefined) {
    state.dealer = minimal.d;
    // Update current player based on dealer
    state.currentPlayer = getPlayerLeftOfDealer(state.dealer);
  }
  if (minimal.t !== undefined) state.gameTarget = minimal.t;
  if (minimal.m !== undefined) state.tournamentMode = minimal.m;
  
  // Recreate hands with the seed
  const hands = dealDominoesWithSeed(state.shuffleSeed);
  
  // Update player hands and recalculate suit analysis
  state.players.forEach((player, i) => {
    const hand = hands[i];
    if (!hand) {
      throw new Error(`No hand dealt for player ${i}`);
    }
    player.hand = hand;
    player.suitAnalysis = analyzeSuits(hand);
  });
  
  // Update deprecated hands property
  state.hands = {
    0: hands[0]!,
    1: hands[1]!,
    2: hands[2]!,
    3: hands[3]!
  };
  
  return state;
}

/**
 * Compress action IDs to save space
 * Maps common actions to single characters
 */
const ACTION_COMPRESSION: Record<string, string> = {
  'pass': 'p',
  'bid-30': '30',
  'bid-31': '31',
  'bid-32': '32',
  'bid-33': '33',
  'bid-34': '34',
  'bid-35': '35',
  'bid-36': '36',
  'bid-37': '37',
  'bid-38': '38',
  'bid-39': '39',
  'bid-40': '40',
  'bid-41': '41',
  'bid-1-marks': 'm1',
  'bid-2-marks': 'm2',
  'bid-3-marks': 'm3',
  'redeal': 'r',
  'score-hand': 'sh',
  'complete-trick': 'ct',
  // Trump selection
  'trump-blanks': 't0',
  'trump-ones': 't1',
  'trump-twos': 't2',
  'trump-threes': 't3',
  'trump-fours': 't4',
  'trump-fives': 't5',
  'trump-sixes': 't6',
  'trump-doubles': 't7',
  'trump-no-trump': 't8',
  // Common dominoes (we'll add more as needed)
  'play-0-0': '00',
  'play-1-0': '10',
  'play-1-1': '11',
  'play-2-0': '20',
  'play-2-1': '21',
  'play-2-2': '22',
  'play-3-0': '30d', // 'd' suffix to distinguish from bid-30
  'play-3-1': '31d',
  'play-3-2': '32d',
  'play-3-3': '33d',
  'play-4-0': '40d',
  'play-4-1': '41d',
  'play-4-2': '42',
  'play-4-3': '43',
  'play-4-4': '44',
  'play-5-0': '50',
  'play-5-1': '51',
  'play-5-2': '52',
  'play-5-3': '53',
  'play-5-4': '54',
  'play-5-5': '55',
  'play-6-0': '60',
  'play-6-1': '61',
  'play-6-2': '62',
  'play-6-3': '63',
  'play-6-4': '64',
  'play-6-5': '65',
  'play-6-6': '66',
};

// Reverse mapping for decompression
const ACTION_DECOMPRESSION: Record<string, string> = Object.entries(ACTION_COMPRESSION)
  .reduce((acc, [k, v]) => ({ ...acc, [v]: k }), {});

/**
 * Compress an action ID
 */
export function compressActionId(id: string): string {
  return ACTION_COMPRESSION[id] || id;
}

/**
 * Decompress an action ID
 */
export function decompressActionId(compressed: string): string {
  return ACTION_DECOMPRESSION[compressed] || compressed;
}

/**
 * Encode URL data to base64
 */
export function encodeURLData(data: URLData): string {
  const json = JSON.stringify(data);
  // Use base64url encoding (URL-safe)
  return btoa(json)
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=/g, '');
}

/**
 * Decode base64 URL data
 */
export function decodeURLData(encoded: string): URLData {
  // Restore base64 padding
  const base64 = encoded
    .replace(/-/g, '+')
    .replace(/_/g, '/')
    + '=='.slice(0, (4 - encoded.length % 4) % 4);
  
  const json = atob(base64);
  return JSON.parse(json);
}