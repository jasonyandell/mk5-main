/**
 * Simplified Domino Strength Analysis
 * 
 * Clean, single-responsibility functions for analyzing domino strength
 * in different play contexts.
 */

import type { Domino, TrumpSelection, GameState, Play } from '../types';
import { getTrickWinner } from '../core/rules';
import { trumpToNumber } from '../core/dominoes';

/**
 * Analysis of a domino's strength when played as a specific suit
 */
export interface DominoStrength {
  domino: Domino;
  playedAsSuit: number;  // Which suit we're analyzing it as
  isTrump: boolean;      // Is this domino trump?
  beatenBy: Domino[];    // Dominoes that can follow this suit AND beat us
  beats: Domino[];       // Dominoes that can follow this suit AND lose to us
  cannotFollow: Domino[];  // Dominoes that cannot follow this suit (win by default)
}

/**
 * Check if a domino is trump
 */
function isTrump(domino: Domino, trump: TrumpSelection): boolean {
  const trumpValue = trumpToNumber(trump);
  if (trumpValue === null) return false;
  
  if (trumpValue === 7) {
    // Doubles trump
    return domino.high === domino.low;
  } else {
    // Suit trump
    return domino.high === trumpValue || domino.low === trumpValue;
  }
}

/**
 * Get the suits a domino can be played as
 * Trump dominoes can only be played as trump
 * Non-trump dominoes can be played as either of their suits
 */
export function getPlayableSuits(domino: Domino, trump: TrumpSelection): number[] {
  if (isTrump(domino, trump)) {
    // Trump dominoes can only be played as trump
    // Return empty array to indicate "trump only" context
    return [];
  }
  
  // Non-trump dominoes can be played as either suit
  const suits = new Set([domino.high, domino.low]);
  return Array.from(suits);
}

/**
 * Get all unplayed dominoes (excluding those in our hand and already played)
 */
function getUnplayedDominoes(state: GameState, playerId: number): Domino[] {
  // Get dominoes in our hand
  const ourHand = new Set(state.players[playerId].hand.map(d => d.id));
  
  // Get played dominoes
  const played = new Set<string>();
  for (const trick of state.tricks) {
    for (const play of trick.plays) {
      played.add(play.domino.id.toString());
    }
  }
  
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
 * Check if a domino can follow a specific suit
 */
function canFollowSuit(domino: Domino, suit: number, trump: TrumpSelection): boolean {
  // Trump dominoes can always "follow" (they trump in)
  if (isTrump(domino, trump)) {
    return true;
  }
  
  // Non-trump dominoes can follow if they contain the suit
  return domino.high === suit || domino.low === suit;
}

/**
 * Analyze a domino when played as a specific suit (or as trump)
 */
export function analyzeDominoAsSuit(
  domino: Domino,
  playedAsSuit: number | null,  // null means played as trump
  trump: TrumpSelection,
  state: GameState,
  playerId: number
): DominoStrength {
  const unplayed = getUnplayedDominoes(state, playerId);
  const beatenBy: Domino[] = [];
  const beats: Domino[] = [];
  const cannotFollow: Domino[] = [];
  
  // Determine the effective suit for this analysis
  // If playedAsSuit is null, the domino is trump and leads naturally
  const effectiveSuit = playedAsSuit ?? -1;
  
  // Check each unplayed domino
  for (const opponent of unplayed) {
    // First check if opponent can follow the suit
    const canFollow = playedAsSuit === null 
      ? true  // If we're playing as trump, consider all responses
      : canFollowSuit(opponent, playedAsSuit, trump);
    
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
    playedAsSuit: playedAsSuit ?? -1,  // -1 indicates trump context
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
    results.push(analyzeDominoAsSuit(domino, null, trump, state, playerId));
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
    if (domino.high === context.suit && domino.low !== context.suit) {
      return `${domino.high}-${domino.low}`;
    } else if (domino.low === context.suit && domino.high !== context.suit) {
      return `${domino.low}-${domino.high}`;
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