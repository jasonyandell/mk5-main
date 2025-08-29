/**
 * AI Utilities Module
 * 
 * Provides straightforward analysis of game state for AI decision making.
 * Returns what the AI has and what it means in context.
 */

import type { Domino, TrumpSelection, Trick, Play, GameState } from '../types';
import { getDominoValue, trumpToNumber, getDominoPoints, getDominoSuit } from '../core/dominoes';
import { getValidPlays, getTrickWinner } from '../core/rules';

/**
 * Context for AI analysis - just the essentials
 */
export interface AIContext {
  state: GameState;
  playerId: number;
}

/**
 * Analysis of a single domino in context
 */
export interface DominoAnalysis {
  domino: Domino;
  value: number;             // Numeric value for trick-taking
  points: number;            // Counting points (0, 5, or 10)
  isTrump: boolean;          // Is this trump?
  wouldBeatTrick: boolean;   // Would beat current trick if played
  beatenBy: Domino[] | undefined;       // Unplayed dominoes that can beat this (undefined if not playable)
  beats: Domino[] | undefined;           // Unplayed dominoes this can beat (undefined if not playable)
}

/**
 * Complete analysis for AI decision making
 */
export interface AIAnalysis {
  dominoes: DominoAnalysis[];     // Analysis of each domino in hand, sorted by effective position
  trumpRemaining: number;          // Trump not in our hand or played
}

/**
 * Main analysis function - provides complete context analysis for AI
 * 
 * @param state Complete game state
 * @param playerId Player to analyze
 * @param forceAnalysis If true, analyze all dominoes even if not currently playable (useful during bidding)
 * @returns Analysis of hand in context
 */
export function analyzeHand(state: GameState, playerId: number, forceAnalysis: boolean = false): AIAnalysis {
  const player = state.players[playerId];
  if (!player) throw new Error(`Player ${playerId} not found`);
  
  const hand = player.hand;
  const { trump, tricks, currentTrick, currentSuit } = state;
  
  // Get played dominoes
  const played = getPlayedDominoes(tricks);
  
  // Count trump
  const trumpPlayed = countPlayedTrump(tricks, trump);
  const trumpInHand = hand.filter(d => isTrump(d, trump)).length;
  const trumpRemaining = Math.max(0, 7 - trumpPlayed - trumpInHand);
  
  // Determine which dominoes are playable using the battle-tested method
  const playableDominoes = getValidPlays(state, playerId);
  const playableIds = new Set(playableDominoes.map(d => d.id.toString()));
  
  // Create set of dominoes in our hand (these can't beat each other in actual play)
  const handDominoIds = new Set(hand.map(d => d.id.toString()));
  
  // Analyze each domino
  const dominoes = hand.map(domino => {
    // If forceAnalysis is true, analyze even if not currently playable
    const shouldAnalyze = forceAnalysis || playableIds.has(domino.id.toString());
    const beatenBy = shouldAnalyze 
      ? getDominoesCanBeat(domino, trump, played, currentSuit, currentTrick, handDominoIds)
      : undefined;
    const beats = shouldAnalyze
      ? getDominoesBeaten(domino, trump, played, currentSuit, currentTrick, handDominoIds)
      : undefined;
    
    return {
      domino,
      value: getDominoValue(domino, trump),
      points: getDominoPoints(domino),
      isTrump: isTrump(domino, trump),
      wouldBeatTrick: wouldBeatCurrentTrick(domino, currentTrick, trump, currentSuit),
      beatenBy,
      beats
    };
  });
  
  // Sort by effective position (fewer beats = stronger)
  dominoes.sort((a, b) => {
    // Unplayable at the end
    if (a.beatenBy === undefined) return 1;
    if (b.beatenBy === undefined) return -1;
    
    // Fewer beats = stronger position (goes first)
    const countDiff = a.beatenBy.length - b.beatenBy.length;
    if (countDiff !== 0) return countDiff;
    
    // When tied on defensive strength, prefer offensive strength
    // More dominoes we can beat = stronger
    if (a.beats !== undefined && b.beats !== undefined) {
      const beatsDiff = b.beats.length - a.beats.length;
      if (beatsDiff !== 0) return beatsDiff;
    }
    
    // Final tiebreak by value
    return b.value - a.value;
  });
  
  return {
    dominoes,
    trumpRemaining
  };
}

// Helper functions (simplified and focused)

function getPlayedDominoes(tricks: Trick[]): Set<string> {
  const played = new Set<string>();
  tricks.forEach(trick => {
    trick.plays.forEach(play => {
      played.add(play.domino.id.toString());
    });
  });
  return played;
}

function countPlayedTrump(tricks: Trick[], trump: TrumpSelection): number {
  const trumpValue = trumpToNumber(trump);
  if (trumpValue === null) return 0;
  
  let count = 0;
  for (const trick of tricks) {
    for (const play of trick.plays) {
      if (isTrump(play.domino, trump)) {
        count++;
      }
    }
  }
  return count;
}

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

// Note: findUnbeatableLeads functionality is now integrated into beatenBy
// A domino with beatenBy.length === 0 is unbeatable

function wouldBeatCurrentTrick(
  domino: Domino,
  currentTrick: Play[],
  trump: TrumpSelection,
  currentSuit?: number
): boolean {
  // If leading a trick, any domino we play is provisionally "winning" until others respond
  if (currentTrick.length === 0) return true;

  // Use the led suit from state (passed in) when following
  const leadSuit = currentSuit ?? -1;

  // Simulate adding our play as a synthetic player and ask the rules engine
  const simulated = [...currentTrick, { player: 999, domino }];
  const winner = getTrickWinner(simulated, trump, leadSuit);
  return winner === 999;
}

/**
 * Gets all unplayed dominoes that can beat this domino in the current context
 * 
 * @param domino The domino to check
 * @param trump Trump selection
 * @param played Set of played domino IDs
 * @param currentSuit The suit being followed (if any)
 * @param currentTrick Current trick plays
 * @param handDominoIds Set of domino IDs in our hand (these can't beat each other)
 * @returns Array of unplayed dominoes that could beat this
 */
function getDominoesCanBeat(
  domino: Domino,
  trump: TrumpSelection,
  played: Set<string>,
  currentSuit: number | undefined,
  currentTrick: Play[],
  handDominoIds: Set<string> = new Set()
): Domino[] {
  // Simulate adding our domino to the current trick
  const simulatedTrick = [...currentTrick, { player: 0, domino }];
  
  // Determine the effective suit for this play
  let effectiveSuit: number;
  if (currentTrick.length === 0) {
    // We're leading - the suit is determined by our domino
    effectiveSuit = getDominoSuit(domino, trump);
  } else {
    // We're following - use the led suit
    effectiveSuit = currentSuit ?? -1;
  }
  
  const result: Domino[] = [];
  
  // Check all possible dominoes
  for (let high = 6; high >= 0; high--) {
    for (let low = high; low >= 0; low--) {
      const testId = `${high}-${low}`;
      
      // Skip if already played
      if (played.has(testId)) continue;
      
      // Skip our own domino
      if (testId === domino.id.toString()) continue;
      
      // Skip dominoes in our hand (they can't beat each other in actual play)
      if (handDominoIds.has(testId)) continue;
      
      // Create test domino and simulate it being played
      const testDomino: Domino = { high, low, id: testId };
      const testTrick = [...simulatedTrick, { player: 1, domino: testDomino }];
      
      // Use getTrickWinner to determine if the test domino would win
      const winner = getTrickWinner(testTrick, trump, effectiveSuit);
      
      // If the test domino (player 1) wins over our domino (player 0), add it to results
      if (winner === 1) {
        result.push(testDomino);
      }
    }
  }
  
  return result;
}

/**
 * Gets all unplayed dominoes that this domino can beat in the current context
 * 
 * @param domino The domino to check
 * @param trump Trump selection
 * @param played Set of played domino IDs
 * @param currentSuit The suit being followed (if any)
 * @param currentTrick Current trick plays
 * @param handDominoIds Set of domino IDs in our hand (these can't beat each other)
 * @returns Array of unplayed dominoes that this could beat
 */
function getDominoesBeaten(
  domino: Domino,
  trump: TrumpSelection,
  played: Set<string>,
  currentSuit: number | undefined,
  currentTrick: Play[],
  handDominoIds: Set<string> = new Set()
): Domino[] {
  // Determine the effective suit for this play
  const effectiveSuit = currentTrick.length === 0
    ? getDominoSuit(domino, trump)
    : (currentSuit ?? -1);

  const result: Domino[] = [];

  // Check all possible dominoes
  for (let high = 6; high >= 0; high--) {
    for (let low = high; low >= 0; low--) {
      const testId = `${high}-${low}`;

      // Skip if already played
      if (played.has(testId)) continue;

      // Skip our own domino
      if (testId === domino.id.toString()) continue;

      // Skip dominoes in our hand (they can't beat each other in actual play)
      if (handDominoIds.has(testId)) continue;

      // Create test domino and simulate a two-play trick (us then them)
      const testDomino: Domino = { high, low, id: testId };
      const testTrick: Play[] = [
        { player: 0, domino },
        { player: 1, domino: testDomino }
      ];

      // Use rules engine to decide if we beat them
      const winner = getTrickWinner(testTrick, trump, effectiveSuit);
      if (winner === 0) {
        result.push(testDomino);
      }
    }
  }

  return result;
}

// Note: orderByEffectivePosition is no longer needed
// The default sort in analyzeHand now sorts by effective position