/**
 * Simplified Domino Strength Analysis
 * 
 * Clean, single-responsibility functions for analyzing domino strength
 * in different play contexts.
 */

import type { Domino, TrumpSelection, GameState, Play, LedSuit, LedSuitOrNone } from '../types';
import { PLAYED_AS_TRUMP } from '../types';
import { getTrickWinner } from '../core/rules';
import { getTrumpSuit, isTrump, isRegularSuitTrump, isDoublesTrump, dominoHasSuit, getNonSuitPip } from '../core/dominoes';
import { getPlayedDominoesFromTricks } from '../core/domino-tracking';
import { suitsWithTrumpBase } from '../layers/compose';

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
 * Check if a domino can be played in response to a led suit.
 *
 * NOTE: This has DIFFERENT semantics from dominoBelongsToSuit in dominoes.ts!
 *
 * - dominoBelongsToSuit: Can this domino satisfy the "must follow suit" rule?
 *   → Trump dominoes do NOT belong to non-trump suits (can't use them to follow)
 *
 * - canPlayIntoSuit: Can this domino be played in response to this suit?
 *   → Trump dominoes CAN respond by trumping in
 *
 * This function is used for AI strength analysis - modeling what opponents
 * MIGHT play in response, not what they're REQUIRED to play.
 *
 * Example with 4s trump, 0s led:
 * - dominoBelongsToSuit(4-0, 0, trump) = FALSE (it's trump, can't follow 0s)
 * - canPlayIntoSuit(4-0, 0, trump) = TRUE (can trump in on 0s)
 */
function canPlayIntoSuit(domino: Domino, suit: LedSuit, trump: TrumpSelection): boolean {
  // Trump dominoes can always respond (they trump in)
  if (isTrump(domino, trump)) {
    return true;
  }

  // Non-trump dominoes can respond if they contain the suit
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
    // First check if opponent can play into the suit (includes trumping in)
    const canFollow = playedAsSuit === PLAYED_AS_TRUMP
      ? true  // If we're playing as trump, consider all responses
      : canPlayIntoSuit(opponent, playedAsSuit as LedSuit, trump);
    
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
    const suits = suitsWithTrumpBase(state, domino);
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
    const trumpSuit = getTrumpSuit(context.trump);
    if (isRegularSuitTrump(trumpSuit)) {
      // Suit trump - show trump suit first
      if (domino.high === trumpSuit) {
        return `${domino.high}-${domino.low}`;
      } else if (domino.low === trumpSuit) {
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
  const trumpSuit = getTrumpSuit(trump);
  const label = strength.playedAsSuit === -1
    ? (isDoublesTrump(trumpSuit) ? 'doubles' : `${trumpSuit}s`)
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