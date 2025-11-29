/**
 * Constraint Tracker for Monte Carlo AI
 *
 * Tracks what we know about opponent hands from observed play.
 * Uses existing game rules to determine suit membership - NEVER re-derives.
 *
 * Key insight: When a player doesn't follow suit, they reveal they have
 * NO dominoes of that suit. Combined with tracking played dominoes,
 * we can narrow down what each player might hold.
 */

import type { GameState, Domino, Play, LedSuit } from '../types';
import type { GameRules } from '../layers/types';
import { createDominoes, dominoBelongsToSuit } from '../core/dominoes';

/**
 * Constraints on what dominoes each player can hold.
 *
 * We track:
 * - Which dominoes have been played (no one has them)
 * - Which suits each player is void in (they showed they couldn't follow)
 * - Which dominoes we (the AI) hold (known exactly)
 */
export interface HandConstraints {
  /** Domino IDs that have been played and are no longer in anyone's hand */
  played: Set<string>;

  /** For each player (0-3), the suits they are known to be void in */
  voidInSuit: Map<number, Set<LedSuit>>;

  /** The AI player's own hand (known exactly) */
  myHand: Set<string>;

  /** The AI player's index */
  myPlayerIndex: number;
}

/**
 * Build constraints from the current game state.
 *
 * Walks through completed tricks to infer:
 * 1. What's been played (simple tracking)
 * 2. Who is void in which suits (inference from not following)
 *
 * @param state Current game state
 * @param myPlayerIndex The AI player's index (0-3)
 * @param rules Composed game rules (for getLedSuit)
 * @returns Constraints that must be respected when sampling opponent hands
 */
export function buildConstraints(
  state: GameState,
  myPlayerIndex: number,
  rules: GameRules
): HandConstraints {
  const played = new Set<string>();
  const voidInSuit = new Map<number, Set<LedSuit>>();

  // Initialize void sets for all players
  for (let i = 0; i < 4; i++) {
    voidInSuit.set(i, new Set<LedSuit>());
  }

  // Process completed tricks to build constraints
  for (const trick of state.tricks) {
    processTrickForConstraints(trick.plays, state, rules, played, voidInSuit);
  }

  // Process current trick (may be incomplete)
  if (state.currentTrick.length > 0) {
    processTrickForConstraints(state.currentTrick, state, rules, played, voidInSuit);
  }

  // Build my hand set
  const myPlayer = state.players[myPlayerIndex];
  const myHand = new Set<string>();
  if (myPlayer) {
    for (const domino of myPlayer.hand) {
      myHand.add(String(domino.id));
    }
  }

  return {
    played,
    voidInSuit,
    myHand,
    myPlayerIndex
  };
}

/**
 * Process a single trick to update constraints.
 *
 * For each play:
 * 1. Mark the domino as played
 * 2. If not the lead and didn't follow suit, player is void in led suit
 */
function processTrickForConstraints(
  plays: Play[],
  state: GameState,
  rules: GameRules,
  played: Set<string>,
  voidInSuit: Map<number, Set<LedSuit>>
): void {
  if (plays.length === 0) return;

  // Get the lead play and its suit
  const leadPlay = plays[0];
  if (!leadPlay) return;

  const ledSuit = rules.getLedSuit(state, leadPlay.domino);

  // Process each play in the trick
  for (let i = 0; i < plays.length; i++) {
    const play = plays[i];
    if (!play) continue;

    // Mark domino as played
    played.add(String(play.domino.id));

    // For non-lead plays, check if they followed suit
    if (i > 0) {
      // Use dominoBelongsToSuit - the unified function that correctly handles trump
      // A trump domino (like 4-0 when 4s are trump) does NOT belong to non-trump suits
      const followedSuit = dominoBelongsToSuit(play.domino, ledSuit, state.trump);

      if (!followedSuit) {
        // Player didn't follow suit - they're void in the led suit
        // They played a domino that doesn't belong to the led suit, so they must
        // not have any dominoes that belong to that suit.
        const playerVoids = voidInSuit.get(play.player);
        if (playerVoids) {
          playerVoids.add(ledSuit);
        }
      }
    }
  }
}

/**
 * Get all dominoes that COULD be in a specific player's hand.
 *
 * Filters out:
 * - Dominoes that have been played
 * - Dominoes in the AI's hand
 * - Dominoes that belong to suits the player is void in
 *
 * @param constraints The built constraints
 * @param playerIndex Which player to get candidates for
 * @param trump Current trump selection (needed for suit membership)
 * @returns Array of dominoes that could be in this player's hand
 */
export function getCandidateDominoes(
  constraints: HandConstraints,
  playerIndex: number,
  trump: import('../types').TrumpSelection
): Domino[] {
  // Start with all 28 dominoes
  const allDominoes = createDominoes();

  // Get this player's void suits
  const voidSuits = constraints.voidInSuit.get(playerIndex) || new Set<LedSuit>();

  // Filter to candidates
  return allDominoes.filter(domino => {
    const id = String(domino.id);

    // Exclude played dominoes
    if (constraints.played.has(id)) return false;

    // Exclude AI's own hand
    if (constraints.myHand.has(id)) return false;

    // Exclude dominoes in suits the player is void in
    // A domino is excluded if it belongs to any void suit
    // dominoBelongsToSuit handles trump correctly (4-0 doesn't belong to 0s if 4s trump)
    for (const voidSuit of voidSuits) {
      if (dominoBelongsToSuit(domino, voidSuit, trump)) {
        return false;
      }
    }

    return true;
  });
}

/**
 * Get the pool of dominoes that need to be distributed to opponents.
 *
 * This is all dominoes except:
 * - Played dominoes
 * - AI's own hand
 *
 * @param constraints The built constraints
 * @returns Array of dominoes in the distribution pool
 */
export function getDistributionPool(constraints: HandConstraints): Domino[] {
  const allDominoes = createDominoes();

  return allDominoes.filter(domino => {
    const id = String(domino.id);
    return !constraints.played.has(id) && !constraints.myHand.has(id);
  });
}

/**
 * Calculate how many dominoes each player should have.
 *
 * At any point in the game:
 * - Each player starts with 7
 * - Players lose 1 domino per completed trick
 * - Current trick plays count as "in flight" (domino left hand but trick not complete)
 *
 * @param state Current game state
 * @returns Array of [p0count, p1count, p2count, p3count]
 */
export function getExpectedHandSizes(state: GameState): [number, number, number, number] {
  const completedTricks = state.tricks.length;
  const baseSize = 7 - completedTricks;

  // Track who has played in current trick
  const playedInCurrentTrick = new Set<number>();
  for (const play of state.currentTrick) {
    playedInCurrentTrick.add(play.player);
  }

  // Players who played in current trick have one fewer domino
  return [
    baseSize - (playedInCurrentTrick.has(0) ? 1 : 0),
    baseSize - (playedInCurrentTrick.has(1) ? 1 : 0),
    baseSize - (playedInCurrentTrick.has(2) ? 1 : 0),
    baseSize - (playedInCurrentTrick.has(3) ? 1 : 0)
  ];
}
