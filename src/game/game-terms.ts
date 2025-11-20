/**
 * Centralized game terminology and display names
 *
 * This module provides type-safe utilities for converting game data structures
 * to human-readable strings. All display name logic should live here.
 *
 * Philosophy:
 * - Single source of truth for all game terminology
 * - Type-safe: uses discriminated unions and type predicates
 * - Pure functions: no side effects
 * - Export both capitalized (for display) and lowercase (for identifiers)
 */

import type { TrumpSelection, LedSuit, RegularSuit, LedSuitOrNone } from './types';
import { BLANKS, ACES, DEUCES, TRES, FOURS, FIVES, SIXES, DOUBLES_AS_TRUMP, PLAYED_AS_TRUMP } from './types';

// ============================================================================
// Suit Display Names
// ============================================================================

/**
 * Capitalized suit names for UI display
 * Maps LedSuit (0-7) to display string
 */
export const SUIT_NAMES: Record<LedSuit, string> = {
  [BLANKS]: 'Blanks',
  [ACES]: 'Aces',
  [DEUCES]: 'Deuces',
  [TRES]: 'Tres',
  [FOURS]: 'Fours',
  [FIVES]: 'Fives',
  [SIXES]: 'Sixes',
  [DOUBLES_AS_TRUMP]: 'Doubles'
} as const;

/**
 * Alternative naming convention (Ones instead of Aces)
 */
export const SUIT_NAMES_NUMERIC: Record<LedSuit, string> = {
  [BLANKS]: 'Blanks',
  [ACES]: 'Ones',
  [DEUCES]: 'Twos',
  [TRES]: 'Threes',
  [FOURS]: 'Fours',
  [FIVES]: 'Fives',
  [SIXES]: 'Sixes',
  [DOUBLES_AS_TRUMP]: 'Doubles'
} as const;

/**
 * Lowercase suit identifiers for URLs and file names
 */
export const SUIT_IDENTIFIERS: Record<RegularSuit, string> = {
  [BLANKS]: 'blanks',
  [ACES]: 'aces',
  [DEUCES]: 'deuces',
  [TRES]: 'tres',
  [FOURS]: 'fours',
  [FIVES]: 'fives',
  [SIXES]: 'sixes'
} as const;

/**
 * Get display name for a suit (0-7)
 *
 * @param suit - The suit number (0-7, where 7 is doubles)
 * @param options.lowercase - Return lowercase version
 * @param options.numeric - Use numeric names (Ones instead of Aces)
 * @returns Display string like "Blanks" or "blanks"
 */
export function getSuitName(
  suit: LedSuit,
  options: { lowercase?: boolean; numeric?: boolean } = {}
): string {
  const { lowercase = false, numeric = false } = options;
  const names = numeric ? SUIT_NAMES_NUMERIC : SUIT_NAMES;
  const name = names[suit];
  return lowercase ? name.toLowerCase() : name;
}

/**
 * Get identifier for a regular suit (0-6) suitable for URLs/files
 *
 * @param suit - The suit number (0-6)
 * @returns Identifier like "blanks", "aces"
 */
export function getSuitIdentifier(suit: RegularSuit): string {
  return SUIT_IDENTIFIERS[suit];
}

// ============================================================================
// Trump Display Names
// ============================================================================

/**
 * Get human-readable trump selection display
 *
 * @param trump - The trump selection
 * @param options.lowercase - Return lowercase version
 * @param options.numeric - Use numeric names (Ones instead of Aces)
 * @param options.includeArticle - Include "the" for suit trumps
 * @returns Display string like "Blanks", "Doubles", "No Trump", "Not Selected"
 *
 * @example
 * getTrumpDisplay({ type: 'suit', suit: 0 }) // "Blanks"
 * getTrumpDisplay({ type: 'suit', suit: 0 }, { lowercase: true }) // "blanks"
 * getTrumpDisplay({ type: 'doubles' }) // "Doubles"
 * getTrumpDisplay({ type: 'no-trump' }) // "No Trump"
 * getTrumpDisplay({ type: 'not-selected' }) // "Not Selected"
 */
export function getTrumpDisplay(
  trump: TrumpSelection,
  options: { lowercase?: boolean; numeric?: boolean; includeArticle?: boolean } = {}
): string {
  const { lowercase = false, numeric = false, includeArticle = false } = options;

  switch (trump.type) {
    case 'not-selected':
      return lowercase ? 'not selected' : 'Not Selected';

    case 'no-trump':
      return lowercase ? 'no trump' : 'No Trump';

    case 'doubles':
      return lowercase ? 'doubles' : 'Doubles';

    case 'suit': {
      if (trump.suit === undefined) {
        return lowercase ? 'unknown' : 'Unknown';
      }
      const suitName = getSuitName(trump.suit as LedSuit, { lowercase, numeric });
      if (includeArticle && !lowercase) {
        return `the ${suitName}`;
      }
      return suitName;
    }

    case 'nello':
      return lowercase ? 'nello' : 'Nello';

    case 'sevens':
      return lowercase ? 'sevens' : 'Sevens';

    default:
      return lowercase ? 'unknown' : 'Unknown';
  }
}

/**
 * Get trump identifier suitable for URLs and file names
 *
 * @param trump - The trump selection
 * @returns Identifier like "blanks", "doubles", "no-trump", "nello"
 *
 * @example
 * getTrumpIdentifier({ type: 'suit', suit: 0 }) // "blanks"
 * getTrumpIdentifier({ type: 'doubles' }) // "doubles"
 * getTrumpIdentifier({ type: 'no-trump' }) // "no-trump"
 */
export function getTrumpIdentifier(trump: TrumpSelection): string {
  switch (trump.type) {
    case 'suit':
      return trump.suit !== undefined ? getSuitIdentifier(trump.suit as RegularSuit) : 'unknown';
    case 'doubles':
      return 'doubles';
    case 'no-trump':
      return 'no-trump';
    case 'nello':
      return 'nello';
    case 'sevens':
      return 'sevens';
    case 'not-selected':
      return 'not-selected';
    default:
      return 'unknown';
  }
}

// ============================================================================
// Bid Type Display Names
// ============================================================================

/**
 * Get display label for a bid action
 *
 * @param bid - Bid object with type and optional value
 * @returns Display string like "30", "2 marks", "nello", "Pass"
 *
 * @example
 * getBidLabel({ type: 'points', value: 30 }) // "30"
 * getBidLabel({ type: 'marks', value: 2 }) // "2 marks"
 * getBidLabel({ type: 'marks', value: 1 }) // "1 mark"
 * getBidLabel({ type: 'nello', value: 2 }) // "nello"
 * getBidLabel({ type: 'pass' }) // "Pass"
 */
export function getBidLabel(bid: { type: string; value?: number | undefined }): string {
  switch (bid.type) {
    case 'points':
      return `${bid.value}`;

    case 'marks':
      return bid.value === 1 ? '1 mark' : `${bid.value} marks`;

    case 'pass':
      return 'Pass';

    // Special contracts
    case 'nello':
      return 'nello';

    case 'splash':
      return 'splash';

    case 'plunge':
      return 'plunge';

    case 'sevens':
      return 'sevens';

    default:
      // Generic fallback for unknown bid types
      return bid.value !== undefined ? `${bid.type} ${bid.value}` : bid.type;
  }
}

/**
 * Get verbose bid description including value
 *
 * @param bid - Bid object with type and optional value
 * @returns Description like "Bid 30", "Bid 2 marks", "Bid nello", "Pass"
 *
 * @example
 * getBidDescription({ type: 'points', value: 30 }) // "Bid 30"
 * getBidDescription({ type: 'marks', value: 2 }) // "Bid 2 marks"
 * getBidDescription({ type: 'nello', value: 2 }) // "Bid nello"
 */
export function getBidDescription(bid: { type: string; value?: number | undefined }): string {
  if (bid.type === 'pass') {
    return 'Pass';
  }
  return `Bid ${getBidLabel(bid)}`;
}

// ============================================================================
// Action Display Names
// ============================================================================

/**
 * Get display label for a trump selection action
 *
 * @param trump - Trump selection
 * @param options.includeVerb - Include "Declare" prefix
 * @param options.numeric - Use numeric names
 * @returns Label like "Blanks trump" or "Declare Blanks trump"
 *
 * @example
 * getTrumpActionLabel({ type: 'suit', suit: 0 }) // "Blanks trump"
 * getTrumpActionLabel({ type: 'suit', suit: 0 }, { includeVerb: true }) // "Declare Blanks trump"
 * getTrumpActionLabel({ type: 'doubles' }) // "Doubles trump"
 * getTrumpActionLabel({ type: 'no-trump' }) // "No-trump"
 */
export function getTrumpActionLabel(
  trump: TrumpSelection,
  options: { includeVerb?: boolean; numeric?: boolean } = {}
): string {
  const { includeVerb = false, numeric = false } = options;
  const prefix = includeVerb ? 'Declare ' : '';

  switch (trump.type) {
    case 'suit': {
      if (trump.suit === undefined) return `${prefix}Unknown trump`;
      const suitName = getSuitName(trump.suit as LedSuit, { numeric });
      return `${prefix}${suitName} trump`;
    }

    case 'doubles':
      return `${prefix}Doubles trump`;

    case 'no-trump':
      return includeVerb ? `${prefix}No-trump` : 'No-trump';

    case 'nello':
      return includeVerb ? `${prefix}Nello` : 'Nello';

    case 'sevens':
      return includeVerb ? `${prefix}Sevens` : 'Sevens';

    default:
      return `${prefix}Select trump`;
  }
}

// ============================================================================
// Suit Context Display (for tooltips and hints)
// ============================================================================

/**
 * Get suit name with trump annotation for display
 *
 * @param suit - The suit being displayed
 * @param trump - Current trump selection
 * @param options.lowercase - Return lowercase version
 * @returns Display string like "Blanks (Trump)" or just "Blanks"
 *
 * @example
 * getSuitWithTrumpContext(0, { type: 'suit', suit: 0 }) // "Blanks (Trump)"
 * getSuitWithTrumpContext(0, { type: 'doubles' }) // "Blanks"
 */
export function getSuitWithTrumpContext(
  suit: LedSuit,
  trump: TrumpSelection,
  options: { lowercase?: boolean } = {}
): string {
  const { lowercase = false } = options;
  const suitName = getSuitName(suit, { lowercase });

  // Check if this suit is trump
  const isTrump = trump.type === 'suit' && trump.suit === suit;

  if (isTrump) {
    return lowercase ? `${suitName} (trump)` : `${suitName} (Trump)`;
  }

  return suitName;
}

// ============================================================================
// Parsing Utilities (String → Number)
// ============================================================================

/**
 * Led suit name mappings (for strength table, hints, etc.)
 * Maps LedSuitOrNone (-1 to 7) to identifier strings
 */
export const LED_SUIT_NAMES: Record<LedSuitOrNone, string> = {
  [PLAYED_AS_TRUMP]: 'played-as-trump',
  [BLANKS]: 'led-blanks',
  [ACES]: 'led-aces',
  [DEUCES]: 'led-deuces',
  [TRES]: 'led-tres',
  [FOURS]: 'led-fours',
  [FIVES]: 'led-fives',
  [SIXES]: 'led-sixes',
  [DOUBLES_AS_TRUMP]: 'led-doubles'
} as const;

/**
 * Suit name to number mapping for parsing
 * Accepts lowercase suit names, returns RegularSuit number
 */
export const SUIT_NAME_TO_NUMBER: Record<string, RegularSuit> = {
  'blanks': BLANKS,
  'aces': ACES,
  'deuces': DEUCES,
  'tres': TRES,
  'fours': FOURS,
  'fives': FIVES,
  'sixes': SIXES
} as const;

/**
 * Trump name to number mapping for parsing
 * Accepts lowercase trump names, returns number (0-8)
 * Note: 0-6 are suits, 7 is doubles, 8 is no-trump
 */
export const TRUMP_NAME_TO_NUMBER: Record<string, number> = {
  'blanks': 0,
  'aces': 1,
  'deuces': 2,
  'tres': 3,
  'fours': 4,
  'fives': 5,
  'sixes': 6,
  'doubles': 7,
  'no-trump': 8
} as const;

/**
 * Parse a suit name string to RegularSuit number
 *
 * @param name - Suit name (case-insensitive)
 * @returns RegularSuit number (0-6) or undefined if invalid
 *
 * @example
 * parseSuitName('blanks') // 0
 * parseSuitName('ACES') // 1
 * parseSuitName('invalid') // undefined
 */
export function parseSuitName(name: string): RegularSuit | undefined {
  return SUIT_NAME_TO_NUMBER[name.toLowerCase()];
}

/**
 * Parse a trump string to TrumpSelection object
 *
 * @param trumpStr - Trump identifier string
 * @returns TrumpSelection object
 *
 * @example
 * parseTrumpString('blanks') // { type: 'suit', suit: 0 }
 * parseTrumpString('doubles') // { type: 'doubles' }
 * parseTrumpString('no-trump') // { type: 'no-trump' }
 * parseTrumpString('nello') // { type: 'nello' }
 * parseTrumpString('invalid') // { type: 'no-trump' } (fallback)
 */
export function parseTrumpString(trumpStr: string): TrumpSelection {
  const normalized = trumpStr.toLowerCase();

  if (normalized === 'no-trump') {
    return { type: 'no-trump' };
  } else if (normalized === 'doubles') {
    return { type: 'doubles' };
  } else if (normalized === 'nello') {
    return { type: 'nello' };
  } else if (normalized === 'sevens') {
    return { type: 'sevens' };
  } else {
    const suit = parseSuitName(normalized);
    if (suit !== undefined) {
      return { type: 'suit', suit };
    }
  }

  // Fallback for invalid input
  return { type: 'no-trump' };
}

/**
 * Get led suit name for display in hints, strength table, etc.
 *
 * @param suit - Led suit or none (-1 to 7)
 * @returns Identifier like "led-blanks", "played-as-trump"
 *
 * @example
 * getLedSuitName(0) // "led-blanks"
 * getLedSuitName(-1) // "played-as-trump"
 * getLedSuitName(7) // "led-doubles"
 */
export function getLedSuitName(suit: LedSuitOrNone): string {
  return LED_SUIT_NAMES[suit] ?? `led-suit-${suit}`;
}
