/**
 * URL Compression System
 *
 * Pure isomorphism: URL ↔ Game State
 * - stateToUrl(state) generates shareable URL from any game state
 * - decodeGameUrl(url) + replay reconstructs identical state
 *
 * Encoding:
 * - Actions: base66 (1 char = 1 event)
 * - Seeds: base36
 * - Initial hands: bit-packed base64url (28 dominoes × 5 bits = 24 chars)
 *
 * URL params: s (seed) XOR i (initialHands), p, d, l, t, v, a
 */

import type { Domino, FilteredGameState, GameState } from '../types';
import { actionToId } from './actions';

// Primary tier: 65 most common events (1 char each)
const EVENT_TO_CHAR: Record<string, string> = {
  // Bidding (A-R, 18 events)
  'pass': 'A',
  'redeal': 'B',
  'bid-30': 'C',
  'bid-31': 'D',
  'bid-32': 'E',
  'bid-33': 'F',
  'bid-34': 'G',
  'bid-35': 'H',
  'bid-36': 'I',
  'bid-37': 'J',
  'bid-38': 'K',
  'bid-39': 'L',
  'bid-40': 'M',
  'bid-41': 'N',
  'bid-42': 'O',
  'bid-1-marks': 'P',
  'bid-2-marks': 'Q',
  'bid-3-marks': 'R',
  
  // Trump (S-a, 9 events)
  // NOTE: Changed from 'ones/twos/threes' to 'aces/deuces/tres' for consistency
  // with game-terms.ts naming convention (2025-01-20).
  'trump-blanks': 'S',
  'trump-aces': 'T',
  'trump-deuces': 'U',
  'trump-tres': 'V',
  'trump-fours': 'W',
  'trump-fives': 'X',
  'trump-sixes': 'Y',
  'trump-doubles': 'Z',
  'trump-no-trump': 'a',
  
  // Dominoes (b-2, 28 events)
  'play-0-0': 'b',
  'play-1-0': 'c',
  'play-1-1': 'd',
  'play-2-0': 'e',
  'play-2-1': 'f',
  'play-2-2': 'g',
  'play-3-0': 'h',
  'play-3-1': 'i',
  'play-3-2': 'j',
  'play-3-3': 'k',
  'play-4-0': 'l',
  'play-4-1': 'm',
  'play-4-2': 'n',
  'play-4-3': 'o',
  'play-4-4': 'p',
  'play-5-0': 'q',
  'play-5-1': 'r',
  'play-5-2': 's',
  'play-5-3': 't',
  'play-5-4': 'u',
  'play-5-5': 'v',
  'play-6-0': 'w',
  'play-6-1': 'x',
  'play-6-2': 'y',
  'play-6-3': 'z',
  'play-6-4': '0',
  'play-6-5': '1',
  'play-6-6': '2',
  
  // Consensus (3-4, 2 events)
  'complete-trick': '3',
  'score-hand': '4',
  // Codes 5-9, -, _, . now available for future use (reclaimed from removed agree actions)
};

// Reverse mapping for decompression
const CHAR_TO_EVENT: Record<string, string> = Object.fromEntries(
  Object.entries(EVENT_TO_CHAR).map(([k, v]) => [v, k])
);

// Secondary tier: Rare events (2 chars: ~X)
const ESCAPED_EVENT_TO_CHAR: Record<string, string> = {
  'bid-plunge': 'A',
  'bid-splash': 'B',
  // 'bid-nello': 'C', // BROKEN: Nello is not a bid type, it's a trump selection (marks bid + nello trump). Fix planned for URL system refactor.
  'bid-84-honors': 'D',
  'bid-sevens': 'E', // BROKEN: Sevens is not a bid type, it's a trump selection (marks bid + sevens trump). Fix planned for URL system refactor.
  'timeout-p0': 'F',
  'timeout-p1': 'G',
  'timeout-p2': 'H',
  'timeout-p3': 'I',
  // Room for 56 more future events
};

// Reverse mapping for escaped events
const ESCAPED_CHAR_TO_EVENT: Record<string, string> = Object.fromEntries(
  Object.entries(ESCAPED_EVENT_TO_CHAR).map(([k, v]) => [v, k])
);

// ============================================================================
// Initial Hands Encoding (bit-packed base64url)
// ============================================================================

/**
 * Convert domino to canonical index (0-27) using triangular numbers.
 * Matches createDominoes() order: 0-0, 1-0, 1-1, 2-0, 2-1, 2-2, ...
 */
function dominoToIndex(d: Domino): number {
  return d.high * (d.high + 1) / 2 + d.low;
}

/**
 * Convert canonical index (0-27) back to Domino.
 * Inverse of dominoToIndex using triangular number formula.
 */
function indexToDomino(idx: number): Domino {
  // high = floor((sqrt(8*idx + 1) - 1) / 2)
  const high = Math.floor((Math.sqrt(8 * idx + 1) - 1) / 2);
  const low = idx - high * (high + 1) / 2;
  return { high, low, id: `${high}-${low}` };
}

/**
 * Encode 4 hands (28 dominoes) to compact base64url string (~24 chars).
 * Each domino index (0-27) uses 5 bits. 28 × 5 = 140 bits → 18 bytes.
 */
function encodeInitialHands(hands: Domino[][]): string {
  const indices = hands.flat().map(dominoToIndex);
  if (indices.length !== 28) {
    throw new Error(`Expected 28 dominoes, got ${indices.length}`);
  }

  // Pack 28 5-bit values into 18 bytes
  const bytes = new Uint8Array(18);
  let bitPos = 0;
  for (const idx of indices) {
    const byteIdx = Math.floor(bitPos / 8);
    const bitOffset = bitPos % 8;
    bytes[byteIdx]! |= (idx << bitOffset) & 0xFF;
    if (bitOffset > 3 && byteIdx + 1 < bytes.length) {
      bytes[byteIdx + 1]! |= idx >> (8 - bitOffset);
    }
    bitPos += 5;
  }

  // Base64url encode (no padding)
  const binary = String.fromCharCode(...bytes);
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=/g, '');
}

/**
 * Decode base64url string back to 4 hands (28 dominoes).
 * Inverse of encodeInitialHands.
 */
function decodeInitialHands(encoded: string): Domino[][] {
  // Base64url decode
  const b64 = encoded.replace(/-/g, '+').replace(/_/g, '/');
  const binary = atob(b64);
  const bytes = Uint8Array.from(binary, c => c.charCodeAt(0));

  if (bytes.length !== 18) {
    throw new Error(`Invalid initialHands encoding: expected 18 bytes, got ${bytes.length}`);
  }

  // Unpack 28 5-bit values
  const indices: number[] = [];
  let bitPos = 0;
  for (let i = 0; i < 28; i++) {
    const byteIdx = Math.floor(bitPos / 8);
    const bitOffset = bitPos % 8;
    let value = (bytes[byteIdx]! >> bitOffset) & 0x1F;
    if (bitOffset > 3 && byteIdx + 1 < bytes.length) {
      value |= ((bytes[byteIdx + 1]! << (8 - bitOffset)) & 0x1F);
    }
    if (value < 0 || value > 27) {
      throw new Error(`Invalid domino index ${value} at position ${i}`);
    }
    indices.push(value);
    bitPos += 5;
  }

  // Validate uniqueness
  const uniqueIndices = new Set(indices);
  if (uniqueIndices.size !== 28) {
    throw new Error(`Duplicate dominoes in URL: expected 28 unique, got ${uniqueIndices.size}`);
  }

  // Convert to Domino[][] (4 players × 7 dominoes)
  const dominoes = indices.map(indexToDomino);
  return [
    dominoes.slice(0, 7),
    dominoes.slice(7, 14),
    dominoes.slice(14, 21),
    dominoes.slice(21, 28)
  ];
}

/**
 * Compress an array of event IDs to a compact string
 */
export function compressEvents(events: string[]): string {
  return events.map(event => {
    if (EVENT_TO_CHAR[event]) {
      return EVENT_TO_CHAR[event];
    }
    if (ESCAPED_EVENT_TO_CHAR[event]) {
      return '~' + ESCAPED_EVENT_TO_CHAR[event];
    }
    // Unknown event - shouldn't happen in production
    console.warn(`Unknown event: ${event}`);
    return '~?';
  }).join('');
}

/**
 * Decompress a compact string back to event IDs
 */
export function decompressEvents(compressed: string): string[] {
  const events: string[] = [];
  
  for (let i = 0; i < compressed.length; i++) {
    if (compressed[i] === '~' && i + 1 < compressed.length) {
      // Escaped event - read next char
      i++;
      const nextChar = compressed[i];
      if (nextChar) {
        const escaped = ESCAPED_CHAR_TO_EVENT[nextChar];
        events.push(escaped || 'unknown');
      } else {
        events.push('unknown');
      }
    } else {
      // Regular event
      const currentChar = compressed[i];
      if (currentChar) {
        const event = CHAR_TO_EVENT[currentChar];
        events.push(event || 'unknown');
      } else {
        events.push('unknown');
      }
    }
  }
  
  return events;
}

/**
 * Encode a complete game state to a URL.
 *
 * seed and initialHands are mutually exclusive:
 * - If initialHands provided, uses `i=` param (omits `s=`)
 * - Otherwise uses `s=` param for seed-based games
 */
export function encodeGameUrl(
  seed: number | undefined,
  actions: string[],
  playerTypes?: ('human' | 'ai')[],
  dealer?: number,
  theme?: string,
  colorOverrides?: Record<string, string>,
  sectionName?: string,
  layers?: string[],
  initialHands?: Domino[][]
): string {
  const params = new URLSearchParams();

  // Seed XOR initialHands (mutually exclusive)
  if (initialHands) {
    params.set('i', encodeInitialHands(initialHands));
  } else if (seed !== undefined) {
    params.set('s', seed.toString(36));
  }

  // Only include player types if not default
  if (playerTypes) {
    const playerStr = playerTypes.map(t => t[0]).join('');
    if (playerStr !== 'haaa') {
      params.set('p', playerStr);
    }
  }

  // Only include dealer if not default (3)
  if (dealer !== undefined && dealer !== 3) {
    params.set('d', dealer.toString());
  }

  // Theme parameters
  // Only include theme if not default ('business')
  if (theme && theme !== 'business') {
    params.set('t', theme);
  }

  // Only include color overrides if present
  if (colorOverrides && Object.keys(colorOverrides).length > 0) {
    // Convert color overrides to compact format
    const colorPairs: string[] = [];
    Object.entries(colorOverrides).forEach(([varName, colorValue]) => {
      // Convert "--p" to "p", "--pc" to "pc", etc.
      const key = varName.replace('--', '');
      const v = colorValue.trim();

      // Detect format using robust regex
      const isOKLCH = /^(\d+(?:\.\d+)?)%\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)$/.test(v);
      const isHSL = /^(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)%\s+(\d+(?:\.\d+)?)%$/.test(v);

      let encoded: string;
      if (isOKLCH) {
        // Parse OKLCH values
        const match = v.match(/^(\d+(?:\.\d+)?)%\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)$/);
        if (match && match[1] && match[2] && match[3]) {
          const l = parseFloat(match[1]).toFixed(2); // L with 2 decimals
          const c = Math.round(parseFloat(match[2]) * 100); // C × 100 as int
          const h = Math.round(parseFloat(match[3])); // H as int
          encoded = `o${key}${l},${c},${h}`;
        } else {
          encoded = `o${key}${v.replace(/\s+/g, ',').replace(/%/g, '')}`;
        }
      } else if (isHSL) {
        // Parse HSL values
        const match = v.match(/^(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)%\s+(\d+(?:\.\d+)?)%$/);
        if (match && match[1] && match[2] && match[3]) {
          const h = Math.round(parseFloat(match[1])); // H as int
          const s = Math.round(parseFloat(match[2])); // S as int
          const l = Math.round(parseFloat(match[3])); // L as int
          encoded = `h${key}${h},${s},${l}`;
        } else {
          encoded = `h${key}${v.replace(/\s+/g, ',').replace(/%/g, '')}`;
        }
      } else {
        // Fallback for unexpected formats
        encoded = `${key}${v.replace(/\s+/g, ',').replace(/%/g, '')}`;
      }
      colorPairs.push(encoded);
    });
    params.set('v', colorPairs.join(';'));
  }

  // Section identifier (human-readable) e.g., one_hand, one_trick
  if (sectionName && sectionName.trim()) {
    params.set('h', sectionName);
  }

  // Enabled layers (short codes, comma-separated)
  if (layers && layers.length > 0) {
    const codes = layers
      .map(rs => LAYER_CODES[rs])
      .filter(code => code !== undefined)
      .join(',');
    if (codes) {
      params.set('l', codes);
    }
  }

  // Compressed actions - ALWAYS LAST since it changes most frequently
  params.set('a', compressEvents(actions));

  return '?' + params.toString();
}

/**
 * Decode a URL to extract game state.
 *
 * seed and initialHands are mutually exclusive:
 * - URLs with `i=` have initialHands, seed will be 0
 * - URLs with `s=` have seed, initialHands will be undefined
 */
export function decodeGameUrl(urlString: string): URLData {
  // Handle both full URLs and just query strings
  const queryString = urlString.includes('?')
    ? urlString.split('?')[1]
    : urlString;

  const params = new URLSearchParams(queryString);

  // Empty URL is ok - no game to load
  if (!params.toString()) {
    return {
      seed: 0,
      actions: [],
      playerTypes: ['human', 'ai', 'ai', 'ai'],
      dealer: 3,
      theme: 'business',
      colorOverrides: {},
      scenario: ''
    };
  }

  // Parse theme (default 'business')
  const theme = params.get('t') || 'business';

  // Parse section/scenario identifier (optional)
  const scenario = params.get('h') || '';

  // Parse color overrides (compact format)
  const colorOverrides: Record<string, string> = {};
  const colorStr = params.get('v');
  if (colorStr) {
    const entries = colorStr.split(';');
    entries.forEach(entry => {
      // Compact format: o{var}{L},{C},{H} or h{var}{H},{S},{L}
      // Examples: "op73.54,21,181" or "hpc240,60,50"

      // Check if it starts with 'o' (OKLCH) or 'h' (HSL)
      if (entry.startsWith('o')) {
        // OKLCH format
        const content = entry.substring(1); // Remove 'o' prefix
        // Find where the variable name ends and values begin
        const firstNumberIndex = content.search(/\d/);
        if (firstNumberIndex === -1) return;

        const varName = '--' + content.substring(0, firstNumberIndex);
        const values = content.substring(firstNumberIndex).split(',');

        if (values.length === 3 && values[0] && values[1] && values[2]) {
          const l = parseFloat(values[0]); // L is already with decimals
          const c = parseFloat(values[1]) / 100; // C was multiplied by 100
          const h = parseFloat(values[2]); // H is integer
          colorOverrides[varName] = `${l}% ${c} ${h}`;
        }
      } else if (entry.startsWith('h')) {
        // HSL format
        const content = entry.substring(1); // Remove 'h' prefix
        // Find where the variable name ends and values begin
        const firstNumberIndex = content.search(/\d/);
        if (firstNumberIndex === -1) return;

        const varName = '--' + content.substring(0, firstNumberIndex);
        const values = content.substring(firstNumberIndex).split(',');

        if (values.length === 3 && values[0] && values[1] && values[2]) {
          const h = parseFloat(values[0]); // H is integer
          const s = parseFloat(values[1]); // S is integer percentage
          const l = parseFloat(values[2]); // L is integer percentage
          colorOverrides[varName] = `${h} ${s}% ${l}%`;
        }
      } else {
        // Fallback for any unexpected format
        console.warn('Unknown color format in URL:', entry);
      }
    });
  }

  // Parse initialHands XOR seed (mutually exclusive)
  const initialHandsStr = params.get('i');
  const seedStr = params.get('s');

  let seed = 0;
  let initialHands: Domino[][] | undefined;

  if (initialHandsStr) {
    // initialHands takes priority
    initialHands = decodeInitialHands(initialHandsStr);
  } else if (seedStr) {
    seed = parseInt(seedStr, 36);
    if (isNaN(seed)) {
      throw new Error('Invalid URL: seed must be a valid number');
    }
  }

  // Parse actions
  const actionsStr = params.get('a') || '';
  const actions = actionsStr ? decompressEvents(actionsStr) : [];

  // Parse player types
  const playerStr = params.get('p') || 'haaa';
  const playerTypes = playerStr.split('').map(c =>
    c === 'h' ? 'human' : 'ai'
  ) as ('human' | 'ai')[];

  // Validate player count
  if (playerTypes.length !== 4) {
    throw new Error('Invalid URL: must have exactly 4 players');
  }

  // Parse dealer (default 3)
  const dealerStr = params.get('d');
  const dealer = dealerStr ? parseInt(dealerStr, 10) : 3;
  if (isNaN(dealer) || dealer < 0 || dealer > 3) {
    throw new Error('Invalid URL: dealer must be 0-3');
  }

  // Parse enabled layers (short codes)
  const layersStr = params.get('l');
  const layers = layersStr
    ? layersStr.split(',')
        .map(code => LAYER_CODE_TO_NAME[code])
        .filter((name): name is string => name !== undefined)
    : undefined;

  const result: URLData = {
    seed,
    actions,
    playerTypes,
    dealer,
    theme,
    colorOverrides,
    scenario
  };

  if (layers && layers.length > 0) {
    result.layers = layers;
  }

  if (initialHands) {
    result.initialHands = initialHands;
  }

  return result;
}

// Layer short codes (for URL encoding)
// All layers use the same unified encoding via the `l=` URL parameter
const LAYER_CODES: Record<string, string> = {
  // Game variant layers
  'nello': 'n',
  'plunge': 'p',
  'splash': 'S',  // Changed from 's' to avoid collision with speed
  'sevens': 'v',
  'tournament': 't',
  // Behavior layers
  'oneHand': 'o',
  'speed': 's',
  'hints': 'h'
};

// Reverse mapping for decoding
const LAYER_CODE_TO_NAME: Record<string, string> = Object.fromEntries(
  Object.entries(LAYER_CODES).map(([k, v]) => [v, k])
);

// Export types for use in other modules
export interface URLData {
  seed: number;
  actions: string[];
  playerTypes: ('human' | 'ai')[];
  dealer: number;
  theme: string;
  colorOverrides: Record<string, string>;
  scenario: string;
  layers?: string[];
  initialHands?: Domino[][];  // Mutually exclusive with seed
}

// ============================================================================
// State → URL (the missing piece!)
// ============================================================================

/**
 * Generate a shareable URL from game state.
 *
 * This is the key function for the URL ↔ State isomorphism.
 * Any game state (GameState or FilteredGameState) can be serialized to a URL
 * that, when replayed, produces identical state.
 */
export function stateToUrl(state: GameState | FilteredGameState): string {
  const config = state.initialConfig;
  const actionIds = state.actionHistory.map(actionToId);

  // Mutual exclusion: initialHands OR seed
  if (config.dealOverrides?.initialHands) {
    return encodeGameUrl(
      undefined,  // no seed when using initialHands
      actionIds,
      config.playerTypes,
      state.dealer,
      config.theme,
      config.colorOverrides,
      undefined,  // sectionName
      config.layers,
      config.dealOverrides.initialHands
    );
  }

  return encodeGameUrl(
    state.shuffleSeed,
    actionIds,
    config.playerTypes,
    state.dealer,
    config.theme,
    config.colorOverrides,
    undefined,  // sectionName
    config.layers
  );
}
