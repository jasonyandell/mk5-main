/**
 * Multi-Trump Analysis
 * 
 * Analyzes dominoes under all possible trump contexts to show
 * how they perform as different suits.
 * 
 * Updated to use the simplified domino-strength module.
 */

import type { Domino, TrumpSelection, GameState } from '../types';
import { analyzeDomino, orientDomino } from './domino-strength';

/**
 * Analysis of a domino under different trump contexts
 */
export interface MultiTrumpAnalysis {
  domino: Domino;
  contexts: TrumpContextAnalysis[];
}

/**
 * Analysis for a specific trump context
 */
export interface TrumpContextAnalysis {
  trump: TrumpSelection;
  label: string; // e.g., "2s", "doubles", "no-trump"
  suit: number | undefined; // The suit being analyzed (for non-trump contexts)
  isTrump: boolean;
  beatenBy: Domino[];
  beats: Domino[];
}


/**
 * Analyze a domino under all relevant trump contexts
 * 
 * Simplified version using the new domino-strength module
 */
export function analyzeMultiTrump(
  domino: Domino,
  state: GameState,
  playerId: number = 0,
  currentTrump?: TrumpSelection
): MultiTrumpAnalysis {
  const trump = currentTrump || { type: 'suit', suit: 0 };
  
  // Get strength analyses for all playable suits
  const strengthAnalyses = analyzeDomino(domino, trump, state, playerId);
  
  // Convert to TrumpContextAnalysis format for backward compatibility
  const contexts: TrumpContextAnalysis[] = strengthAnalyses.map(strength => {
    // Determine label
    let label: string;
    if (strength.playedAsSuit === -1) {
      // Trump context
      if (trump.type === 'doubles') {
        label = 'doubles';
      } else if (trump.type === 'suit') {
        label = `${trump.suit}s`;
      } else {
        label = 'unknown';
      }
    } else {
      // Specific suit context
      label = `${strength.playedAsSuit}s`;
    }
    
    return {
      trump,
      label,
      suit: strength.playedAsSuit >= 0 ? strength.playedAsSuit : undefined,
      isTrump: strength.isTrump,
      beatenBy: strength.beatenBy,
      beats: strength.beats
    };
  });
  
  return {
    domino,
    contexts
  };
}

/**
 * Format a domino oriented by the context (suit being played or trump)
 * Delegates to the simplified orientDomino function
 */
function formatDominoByContext(d: Domino, context: TrumpContextAnalysis): string {
  return orientDomino(d, { 
    suit: context.suit !== undefined ? context.suit : null, 
    trump: context.trump 
  });
}

/**
 * Format multi-trump analysis for display
 */
export function formatMultiTrumpAnalysis(
  analysis: MultiTrumpAnalysis,
  indent: string = '  '
): string[] {
  const lines: string[] = [];
  
  // Process each context separately to maintain trump-specific formatting
  for (const context of analysis.contexts) {
    const beatenByStr = context.beatenBy.length === 0 
      ? 'None' 
      : context.beatenBy.length > 8
        ? `[${context.beatenBy.slice(0, 8).map(d => formatDominoByContext(d, context)).join('] [')}] ... (${context.beatenBy.length} total)`
        : `[${context.beatenBy.map(d => formatDominoByContext(d, context)).join('] [')}]`;
    
    lines.push(`${indent}${context.label}: ${beatenByStr}`);
  }
  
  return lines;
}