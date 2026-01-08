import type { GameState, Domino } from '../types';

/**
 * Centralized utilities for tracking played dominoes across the game.
 * Extracted from duplicated code in handOutcome.ts, suit-analysis.ts, and domino-strength.ts
 */

/**
 * Gets all dominoes that have been played in completed tricks.
 * Does NOT include dominoes from the current in-progress trick.
 *
 * @param tricks - Array of completed tricks
 * @returns Set of domino IDs (e.g., "6-5", "3-3") that have been played
 */
export function getPlayedDominoesFromTricks(
  tricks: Array<{ plays: Array<{ domino: Domino }> }>
): Set<string> {
  const played = new Set<string>();

  for (const trick of tricks) {
    for (const play of trick.plays) {
      played.add(play.domino.id.toString());
    }
  }

  return played;
}

/**
 * Gets ALL played dominoes including both completed tricks and current in-progress trick.
 *
 * @param state - Current game state
 * @returns Set of domino IDs (e.g., "6-5", "3-3") that have been played
 */
export function getAllPlayedDominoes(state: GameState): Set<string> {
  const played = new Set<string>();

  // Add dominoes from completed tricks
  for (const trick of state.tricks) {
    for (const play of trick.plays) {
      played.add(play.domino.id.toString());
    }
  }

  // Add dominoes from current in-progress trick
  for (const play of state.currentTrick) {
    played.add(play.domino.id.toString());
  }

  return played;
}
