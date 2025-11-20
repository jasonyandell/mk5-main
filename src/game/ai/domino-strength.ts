/**
 * Simplified Domino Strength Analysis
 * 
 * Clean, single-responsibility functions for analyzing domino strength
 * in different play contexts.
 */

import type { Domino, TrumpSelection, GameState, Play, LedSuit, LedSuitOrNone, RegularSuit } from '../types';
import { PLAYED_AS_TRUMP } from '../types';
import { getTrickWinner } from '../core/rules';
import { getTrumpSuit, isTrump, trumpToNumber, isRegularSuitTrump, isDoublesTrump, dominoHasSuit, getNonSuitPip } from '../core/dominoes';
import { getDominoStrength } from './strength-table.generated';
import { getPlayedDominoesFromTricks } from '../core/domino-tracking';

/**
 * Analysis of a domino's strength when played as a specific suit
 */
export interface DominoStrength {
  domino: Domino;
  playedAsSuit: LedSuitOrNone;  // Which suit we're analyzing it as (-1 for trump context)
  isTrump: boolean;      // Is this domino trump?
  beatenBy: Domino[];    // Dominoes that can follow this suit AND beat us
  beats: Domino[];       // Dominoes that can follow this suit AND lose to us
  cannotFollow: Domino[];  // Dominoes that cannot follow this suit (win by default)
}

/**
 * Get the suits a domino can be played as (for AI analysis)
 */
export function getPlayableSuits(domino: Domino, trump: TrumpSelection): LedSuit[] {
  const trumpSuit = getTrumpSuit(trump);

  // When doubles are trump
  if (isDoublesTrump(trumpSuit)) {
    if (domino.high === domino.low) {
      return [7];  // Doubles can only be played as doubles suit
    }
    // Non-doubles can be played as either pip
    const suits = new Set<RegularSuit>();
    suits.add(domino.high as RegularSuit);
    suits.add(domino.low as RegularSuit);
    return Array.from(suits);
  }

  // When regular suit is trump
  if (isRegularSuitTrump(trumpSuit)) {
    // Trump dominoes can only be played as trump
    if (dominoHasSuit(domino, trumpSuit)) {
      return [trumpSuit as RegularSuit];
    }
    // Non-trump doubles can only be played as their pip
    if (domino.high === domino.low) {
      return [domino.high as RegularSuit];
    }
    // Non-trump non-doubles can be played as either pip
    const suits = new Set<RegularSuit>();
    suits.add(domino.high as RegularSuit);
    suits.add(domino.low as RegularSuit);
    return Array.from(suits);
  }
  
  // No-trump: dominoes can be played as their natural suits
  if (domino.high === domino.low) {
    return [domino.high as RegularSuit];
  }
  const suits = new Set<RegularSuit>();
  suits.add(domino.high as RegularSuit);
  suits.add(domino.low as RegularSuit);
  return Array.from(suits);
}

/**
 * Get all unplayed dominoes (excluding those in our hand and already played)
 */
function getUnplayedDominoes(state: GameState, playerId: number): Domino[] {
  // Get dominoes in our hand
  const ourHand = new Set(state.players[playerId]?.hand.map(d => d.id) ?? []);

  // Get played dominoes from completed tricks
  const played = getPlayedDominoesFromTricks(state.tricks);
  
  // Generate all possible dominoes and filter
  const unplayed: Domino[] = [];
  for (let high = 6; high >= 0; high--) {
    for (let low = high; low >= 0; low--) {
      const id = `${high}-${low}`;
      if (!ourHand.has(id) && !played.has(id)) {
        unplayed.push({ high, low, id });
      }
    }
  }
  
  return unplayed;
}

/**
 * Get dominoes excluded from play (in our hand or already played)
 */
function getExcludedDominoes(state: GameState, playerId: number): Set<string> {
  const excluded = new Set<string>();

  // Add dominoes in our hand
  for (const domino of state.players[playerId]?.hand ?? []) {
    excluded.add(domino.id.toString());
  }

  // Add played dominoes from completed tricks
  const played = getPlayedDominoesFromTricks(state.tricks);
  for (const dominoId of played) {
    excluded.add(dominoId);
  }

  return excluded;
}

/**
 * Fast lookup using precomputed table
 */
export function analyzeDominoAsSuitFast(
  domino: Domino,
  playedAsSuit: LedSuitOrNone,
  trump: TrumpSelection,
  state: GameState,
  playerId: number
): DominoStrength {
  // Get precomputed entry
  const entry = getDominoStrength(domino, trump, playedAsSuit);
  if (!entry) {
    // Fallback to runtime computation if key not found
    console.warn(`No precomputed entry for domino ${domino.id}, falling back to runtime`);
    return analyzeDominoAsSuit(domino, playedAsSuit, trump, state, playerId);
  }
  
  // Filter out excluded dominoes
  const excluded = getExcludedDominoes(state, playerId);
  
  // Convert IDs back to Domino objects, filtering out excluded ones
  const idToDomino = (id: string): Domino => {
    const parts = id.split('-').map(Number);
    return { high: parts[0]!, low: parts[1]!, id };
  };
  
  return {
    domino,
    playedAsSuit: playedAsSuit,
    isTrump: isTrump(domino, trump),
    beatenBy: entry.beatenBy.filter(id => !excluded.has(id)).map(idToDomino),
    beats: entry.beats.filter(id => !excluded.has(id)).map(idToDomino),
    cannotFollow: entry.cannotFollow.filter(id => !excluded.has(id)).map(idToDomino)
  };
}

/**
 * Check if a domino can follow a specific suit
 */
function canFollowSuit(domino: Domino, suit: LedSuit, trump: TrumpSelection): boolean {
  // Trump dominoes can always "follow" (they trump in)
  if (isTrump(domino, trump)) {
    return true;
  }

  // Non-trump dominoes can follow if they contain the suit
  return dominoHasSuit(domino, suit);
}

/**
 * Analyze a domino when played as a specific suit (or as trump)
 */
export function analyzeDominoAsSuit(
  domino: Domino,
  playedAsSuit: LedSuitOrNone,  // PLAYED_AS_TRUMP means played as trump
  trump: TrumpSelection,
  state: GameState,
  playerId: number
): DominoStrength {
  const unplayed = [domino, ...getUnplayedDominoes(state, playerId)]; // Unplayed dominoes + this domino
  const beatenBy: Domino[] = [];
  const beats: Domino[] = [];
  const cannotFollow: Domino[] = [];
  
  // Determine the effective suit for this analysis
  // If playedAsSuit is PLAYED_AS_TRUMP, the domino is trump and leads naturally
  const effectiveSuit = playedAsSuit;
  
  // Check each unplayed domino
  for (const opponent of unplayed) {
    // First check if opponent can follow the suit
    const canFollow = playedAsSuit === PLAYED_AS_TRUMP 
      ? true  // If we're playing as trump, consider all responses
      : canFollowSuit(opponent, playedAsSuit as LedSuit, trump);
    
    if (!canFollow) {
      // Opponent cannot follow suit - we win by default
      cannotFollow.push(opponent);
      continue;
    }
    
    // Opponent can follow - determine who wins in fair competition
    const trick: Play[] = [
      { player: 0, domino },
      { player: 1, domino: opponent }
    ];
    
    const winner = getTrickWinner(trick, trump, effectiveSuit);
    
    if (winner === 1) {
      // Opponent can follow AND beats us
      beatenBy.push(opponent);
    } else {
      // Opponent can follow but loses to us
      beats.push(opponent);
    }
  }
  
  // Sort beats and beatenBy to show trumps first
  const sortWithTrumpsFirst = (a: Domino, b: Domino) => {
    const aIsTrump = isTrump(a, trump);
    const bIsTrump = isTrump(b, trump);
    if (aIsTrump && !bIsTrump) return -1;
    if (!aIsTrump && bIsTrump) return 1;
    // Within trumps or non-trumps, sort by value (higher first)
    if (a.high !== b.high) return b.high - a.high;
    return b.low - a.low;
  };
  
  beats.sort(sortWithTrumpsFirst);
  beatenBy.sort(sortWithTrumpsFirst);
  cannotFollow.sort(sortWithTrumpsFirst);
  
  return {
    domino,
    playedAsSuit: playedAsSuit,  // PLAYED_AS_TRUMP indicates trump context
    isTrump: isTrump(domino, trump),
    beatenBy,
    beats,
    cannotFollow
  };
}

/**
 * Get all strength analyses for a domino (one per playable suit)
 */
export function analyzeDomino(
  domino: Domino,
  trump: TrumpSelection,
  state: GameState,
  playerId: number
): DominoStrength[] {
  const results: DominoStrength[] = [];
  
  if (isTrump(domino, trump)) {
    // Trump domino - analyze as trump only
    results.push(analyzeDominoAsSuit(domino, PLAYED_AS_TRUMP, trump, state, playerId));
  } else {
    // Non-trump domino - analyze for each suit
    const suits = getPlayableSuits(domino, trump);
    for (const suit of suits) {
      results.push(analyzeDominoAsSuit(domino, suit, trump, state, playerId));
    }
  }
  
  return results;
}

/**
 * Orient a domino for display based on context
 * 
 * Rules:
 * 1. Trump dominoes always show trump side first
 * 2. Non-trump dominoes show context suit first
 * 3. Doubles and same-suit default to high-low
 */
export function orientDomino(
  domino: Domino,
  context: { suit: number | null, trump: TrumpSelection }
): string {
  // Rule 1: Trump dominoes always show trump side first
  if (isTrump(domino, context.trump)) {
    const trumpValue = trumpToNumber(context.trump);
    if (trumpValue !== null && trumpValue !== 7) {
      // Suit trump - show trump suit first
      if (domino.high === trumpValue) {
        return `${domino.high}-${domino.low}`;
      } else if (domino.low === trumpValue) {
        return `${domino.low}-${domino.high}`;
      }
    }
    // Doubles trump or default
    return `${domino.high}-${domino.low}`;
  }
  
  // Rule 2: Non-trump dominoes show context suit first (if specified)
  if (context.suit !== null && context.suit >= 0) {
    const nonSuitPip = getNonSuitPip(domino, context.suit);
    if (nonSuitPip !== null) {
      // Domino has context suit on one pip - show suit first
      return `${context.suit}-${nonSuitPip}`;
    }
  }
  
  // Rule 3: Default to high-low
  return `${domino.high}-${domino.low}`;
}

/**
 * Format strength analysis for display
 */
export function formatStrengthAnalysis(
  strength: DominoStrength,
  trump: TrumpSelection,
  indent: string = '  '
): string {
  const context = { suit: strength.playedAsSuit, trump };
  
  // Format label
  const label = strength.playedAsSuit === -1 
    ? (trump.type === 'doubles' ? 'doubles' : `${trumpToNumber(trump)}s`)
    : `${strength.playedAsSuit}s`;
  
  // Format beatenBy list with proper orientation
  let beatenByStr: string;
  if (strength.beatenBy.length === 0) {
    beatenByStr = 'None';
  } else if (strength.beatenBy.length > 8) {
    const formatted = strength.beatenBy.slice(0, 8)
      .map(d => `[${orientDomino(d, context)}]`)
      .join(' ');
    beatenByStr = `${formatted} ... (${strength.beatenBy.length} total)`;
  } else {
    beatenByStr = strength.beatenBy
      .map(d => `[${orientDomino(d, context)}]`)
      .join(' ');
  }
  
  return `${indent}${label}: ${beatenByStr}`;
}