/**
 * Lexicographic Hand Strength Evaluation
 * 
 * Evaluates hand strength by ranking dominoes by how many can beat them,
 * then comparing hands lexicographically (like comparing words alphabetically).
 * 
 * A hand with ranks [0,0,1,2,3,4,5] is stronger than [0,1,1,2,3,4,5]
 * because it has two unbeatables vs one.
 */

import type { Domino, TrumpSelection, GameState } from '../types';
import { analyzeHand } from './utilities';

/**
 * Domino with its beaten-by rank
 */
interface RankedDomino {
  domino: Domino;
  rank: number; // Number of dominoes that can beat this one
}

/**
 * Result of lexicographic evaluation for a specific trump
 */
interface LexicographicEvaluation {
  trump: TrumpSelection;
  rankedHand: RankedDomino[];
  score: number[]; // The lexicographic array [0,0,1,2,3,4,5] for comparison
}

/**
 * Calculate lexicographic hand strength for a specific trump selection
 */
function evaluateHandWithTrump(
  hand: Domino[],
  trump: TrumpSelection,
  state: GameState
): LexicographicEvaluation {
  // Create a minimal state for analysis with this trump
  const minimalState: GameState = {
    ...state,
    phase: 'playing',
    players: [
      { ...state.players[0], hand },
      { ...state.players[1], hand: [] },
      { ...state.players[2], hand: [] },
      { ...state.players[3], hand: [] }
    ],
    currentPlayer: 0,
    trump,
    currentTrick: [],
    currentSuit: -1,
  };

  // Analyze the hand with this trump (force analysis even if not in playing phase)
  const analysis = analyzeHand(minimalState, 0, true);
  
  // Create ranked dominoes array
  const rankedHand: RankedDomino[] = hand.map(domino => {
    const dominoAnalysis = analysis.dominoes.find(da => 
      da.domino.high === domino.high && da.domino.low === domino.low
    );
    
    // Get the beaten-by count (rank)
    const rank = dominoAnalysis?.beatenBy?.length ?? 28; // If not found, worst case
    
    return {
      domino,
      rank
    };
  });

  // Sort by rank (ascending - best dominoes first)
  rankedHand.sort((a, b) => a.rank - b.rank);
  
  // Extract just the ranks for lexicographic comparison
  const score = rankedHand.map(rd => rd.rank);
  
  return {
    trump,
    rankedHand,
    score
  };
}

/**
 * Compare two lexicographic scores
 * Returns negative if a < b, positive if a > b, 0 if equal
 * LOWER is better (fewer dominoes can beat)
 */
function compareLexicographic(a: number[], b: number[]): number {
  const minLength = Math.min(a.length, b.length);
  
  for (let i = 0; i < minLength; i++) {
    if (a[i] < b[i]) return -1; // a is better
    if (a[i] > b[i]) return 1;  // b is better
  }
  
  // If all compared elements are equal, shorter array is better
  // (shouldn't happen with dominoes, but just in case)
  return a.length - b.length;
}

/**
 * Calculate the best possible lexicographic hand strength
 * 
 * Same signature as calculateHandStrengthWithTrump but uses
 * lexicographic evaluation instead of point-based scoring.
 * 
 * Returns a score where LOWER is better (to match comparison logic)
 * The score is constructed from the lexicographic array.
 */
export function calculateLexicographicStrength(
  hand: Domino[],
  trump: TrumpSelection | undefined,
  state: GameState,
  analyzingPlayerId: number = 0
): number {
  // If trump is already determined, evaluate with that trump
  if (trump) {
    const evaluation = evaluateHandWithTrump(hand, trump, state);
    // Convert lexicographic array to a single score
    // Weight earlier positions more heavily (they matter more)
    let score = 0;
    for (let i = 0; i < evaluation.score.length; i++) {
      // Each position is worth 100^(6-i) to maintain lexicographic ordering
      score += evaluation.score[i] * Math.pow(100, 6 - i);
    }
    return score;
  }

  // No trump specified - evaluate all possible trumps
  const possibleTrumps: TrumpSelection[] = [
    { type: 'doubles' },
    { type: 'suit', suit: 0 },
    { type: 'suit', suit: 1 },
    { type: 'suit', suit: 2 },
    { type: 'suit', suit: 3 },
    { type: 'suit', suit: 4 },
    { type: 'suit', suit: 5 },
    { type: 'suit', suit: 6 },
  ];

  let bestEvaluation: LexicographicEvaluation | null = null;
  
  for (const possibleTrump of possibleTrumps) {
    const evaluation = evaluateHandWithTrump(hand, possibleTrump, state);
    
    if (!bestEvaluation || compareLexicographic(evaluation.score, bestEvaluation.score) < 0) {
      bestEvaluation = evaluation;
    }
  }

  if (!bestEvaluation) {
    return 999999; // Worst possible score
  }

  // Convert best lexicographic array to a single score
  let score = 0;
  for (let i = 0; i < bestEvaluation.score.length; i++) {
    score += bestEvaluation.score[i] * Math.pow(100, 6 - i);
  }
  
  return score;
}

/**
 * Get detailed lexicographic analysis for debugging
 */
export function getLexicographicAnalysis(
  hand: Domino[],
  state: GameState
): { trump: TrumpSelection; rankedHand: RankedDomino[]; score: number[] }[] {
  const possibleTrumps: TrumpSelection[] = [
    { type: 'doubles' },
    { type: 'suit', suit: 0 },
    { type: 'suit', suit: 1 },
    { type: 'suit', suit: 2 },
    { type: 'suit', suit: 3 },
    { type: 'suit', suit: 4 },
    { type: 'suit', suit: 5 },
    { type: 'suit', suit: 6 },
  ];

  const evaluations = possibleTrumps.map(trump => 
    evaluateHandWithTrump(hand, trump, state)
  );

  // Sort by lexicographic score (best first)
  evaluations.sort((a, b) => compareLexicographic(a.score, b.score));
  
  return evaluations;
}