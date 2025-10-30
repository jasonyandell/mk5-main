/**
 * URL Compression System
 * Uses base66 encoding: 1 character = 1 event
 * Reduces URLs by ~96% (8KB -> 350 chars)
 * Seeds use base36 encoding for additional compression
 *
 * IMPORTANT: Completeness comes first - the URL must capture the complete game state
 * to enable proper replay and sharing. After ensuring completeness, we optimize for
 * minimal URL length.
 */

/**
 * FUTURE: replay-url capability
 * This module's encodeGameUrl() and decodeGameUrl() will be integrated
 * into the protocol layer to enable URL-based game replay:
 * - GameKernel.getView() generates GameView for clients with 'replay-url' capability
 * - Clients can use replayUrl to replay full game without storing full state
 * - See src/kernel/GameKernel.ts and src/shared/multiplayer/protocol.ts
 */

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
  'trump-blanks': 'S',
  'trump-ones': 'T',
  'trump-twos': 'U',
  'trump-threes': 'V',
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
  
  // Consensus (3-., 10 events)
  'complete-trick': '3',
  'score-hand': '4',
  'agree-complete-trick-0': '5',
  'agree-complete-trick-1': '6',
  'agree-complete-trick-2': '7',
  'agree-complete-trick-3': '8',
  'agree-score-hand-0': '9',
  'agree-score-hand-1': '-',
  'agree-score-hand-2': '_',
  'agree-score-hand-3': '.',
};

// Reverse mapping for decompression
const CHAR_TO_EVENT: Record<string, string> = Object.fromEntries(
  Object.entries(EVENT_TO_CHAR).map(([k, v]) => [v, k])
);

// Secondary tier: Rare events (2 chars: ~X)
const ESCAPED_EVENT_TO_CHAR: Record<string, string> = {
  'bid-plunge': 'A',
  'bid-splash': 'B',
  'bid-nello': 'C',
  'bid-84-honors': 'D',
  'bid-sevens': 'E',
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
 * Encode a complete game state to a URL
 */
export function encodeGameUrl(
  seed: number,
  actions: string[],
  playerTypes?: ('human' | 'ai')[],
  dealer?: number,
  theme?: string,
  colorOverrides?: Record<string, string>,
  sectionName?: string
): string {
  const params = new URLSearchParams();
  
  // Theme parameters FIRST (so they survive truncation)
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
  
  // Seed as base36 for compression
  params.set('s', seed.toString(36));
  
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

  // Compressed actions - always last since it's the longest
  params.set('a', compressEvents(actions));
  
  // Section identifier (human-readable) e.g., one_hand, one_trick
  if (sectionName && sectionName.trim()) {
    params.set('h', sectionName);
  }
  
  return '?' + params.toString();
}

/**
 * Decode a URL to extract game state
 */
export function decodeGameUrl(urlString: string): {
  seed: number;
  actions: string[];
  playerTypes: ('human' | 'ai')[];
  dealer: number;
  theme: string;
  colorOverrides: Record<string, string>;
  scenario: string;
} {
  // Handle both full URLs and just query strings
  const queryString = urlString.includes('?') 
    ? urlString.split('?')[1] 
    : urlString;
    
  const params = new URLSearchParams(queryString);
  
  // Check version for backward compatibility
  const version = params.get('v');
  
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
  
  // Parse color overrides (new compact format)
  const colorOverrides: Record<string, string> = {};
  const colorStr = params.get('v');
  if (colorStr) {
    const entries = colorStr.split(';');
    entries.forEach(entry => {
      // New compact format: o{var}{L},{C},{H} or h{var}{H},{S},{L}
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
  
  // Parse seed
  const seedStr = params.get('s');
  if (!seedStr) {
    // Defer decision to caller: use 0 when seed is not present
    // Callers can detect and supply a seed and update URL.
    return {
      seed: 0,
      actions: [],
      playerTypes: ['human', 'ai', 'ai', 'ai'],
      dealer: 3,
      theme,
      colorOverrides,
      scenario
    };
  }
  
  // Handle both old decimal format (v2) and new base36 format
  let seed: number;
  if (version === '2') {
    // Old format: decimal
    seed = parseInt(seedStr, 10);
  } else {
    // New format: base36
    seed = parseInt(seedStr, 36);
  }
  
  if (isNaN(seed)) {
    throw new Error('Invalid URL: seed must be a valid number');
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

  return { seed, actions, playerTypes, dealer, theme, colorOverrides, scenario };
}

// Export types for use in other modules
export interface URLData {
  seed: number;
  actions: string[];
  playerTypes: ('human' | 'ai')[];
  dealer: number;
  theme: string;
  colorOverrides: Record<string, string>;
  scenario: string;
}
