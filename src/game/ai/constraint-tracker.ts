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
import { createDominoes } from '../core/dominoes';

/** Cached set of all 28 dominoes (avoid repeated allocation) */
const ALL_DOMINOES = createDominoes();
Object.freeze(ALL_DOMINOES);

/**
 * Pre-computed canFollow lookup table.
 * Maps dominoId -> Set of suits the domino can follow.
 * Built once per PIMC evaluation (trump is fixed).
 */
export type CanFollowCache = Map<string, Set<LedSuit>>;

/**
 * Build a canFollow lookup table for all dominoes and suits.
 * Call once per PIMC evaluation and pass to getCandidateDominoes.
 *
 * @param state Game state (only trump matters)
 * @param rules Composed game rules
 * @returns Cache mapping dominoId -> set of suits it can follow
 */
export function buildCanFollowCache(state: GameState, rules: GameRules): CanFollowCache {
  const cache: CanFollowCache = new Map();

  for (const domino of ALL_DOMINOES) {
    const suits = new Set<LedSuit>();
    // Check all possible led suits (0-7)
    for (let suit = 0; suit <= 7; suit++) {
      if (rules.canFollow(state, suit as LedSuit, domino)) {
        suits.add(suit as LedSuit);
      }
    }
    cache.set(String(domino.id), suits);
  }

  return cache;
}

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
      // Use rules.canFollow - the composed rule that handles all trump variations
      const followedSuit = rules.canFollow(state, ledSuit, play.domino);

      if (!followedSuit) {
        // Player didn't follow suit - they're void in the led suit
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
 * @param state Current game state (for canFollow rule)
 * @param rules Composed game rules
 * @param canFollowCache Optional pre-computed cache (for performance)
 * @returns Array of dominoes that could be in this player's hand
 */
export function getCandidateDominoes(
  constraints: HandConstraints,
  playerIndex: number,
  state: GameState,
  rules: GameRules,
  canFollowCache?: CanFollowCache
): Domino[] {
  // Get this player's void suits
  const voidSuits = constraints.voidInSuit.get(playerIndex) || new Set<LedSuit>();

  // Filter to candidates (using cached ALL_DOMINOES)
  return ALL_DOMINOES.filter(domino => {
    const id = String(domino.id);

    // Exclude played dominoes
    if (constraints.played.has(id)) return false;

    // Exclude AI's own hand
    if (constraints.myHand.has(id)) return false;

    // Exclude dominoes in suits the player is void in
    // A domino is excluded if it can follow any void suit
    if (canFollowCache) {
      // Use pre-computed cache (fast path)
      const dominoSuits = canFollowCache.get(id);
      if (dominoSuits) {
        for (const voidSuit of voidSuits) {
          if (dominoSuits.has(voidSuit)) {
            return false;
          }
        }
      }
    } else {
      // Fallback to direct rule calls (slow path)
      for (const voidSuit of voidSuits) {
        if (rules.canFollow(state, voidSuit, domino)) {
          return false;
        }
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
  return ALL_DOMINOES.filter(domino => {
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
