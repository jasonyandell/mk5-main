/**
 * Hand Sampler for Monte Carlo AI
 *
 * Generates random but constraint-respecting opponent hand distributions.
 * Uses backtracking search with MRV heuristic to guarantee finding a solution
 * if one exists.
 *
 * Invariant: A valid distribution MUST always exist (the real game state is one).
 * If no solution is found, that indicates a bug in constraint tracking.
 */

import type { Domino, GameState } from '../types';
import type { GameRules } from '../layers/types';
import type { HandConstraints, CanFollowCache } from './constraint-tracker';
import { getCandidateDominoes, getDistributionPool } from './constraint-tracker';

/** Random number generator interface (for testability) */
export interface RandomGenerator {
  /** Returns a random float in [0, 1) */
  random(): number;
}

/** Default RNG using Math.random */
export const defaultRng: RandomGenerator = {
  random: () => Math.random()
};

/**
 * Result of sampling opponent hands.
 * Maps player index to their sampled hand.
 */
export type SampledHands = Map<number, Domino[]>;

/**
 * Shuffle array in place using Fisher-Yates algorithm.
 */
function shuffle<T>(array: T[], rng: RandomGenerator): void {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(rng.random() * (i + 1));
    [array[i], array[j]] = [array[j]!, array[i]!];
  }
}

/**
 * Backtracking search to find a valid assignment of dominoes to opponents.
 *
 * Uses MRV (Minimum Remaining Values) heuristic: always assigns to the
 * player with minimum slack first. This aggressive pruning makes the
 * algorithm efficient in practice.
 *
 * @returns true if a valid assignment was found, false otherwise
 */
function sampleWithBacktracking(
  opponents: number[],
  remaining: Map<number, number>,
  available: Set<string>,
  candidatesPerOpponent: Map<number, Set<string>>,
  hands: Map<number, string[]>,
  rng: RandomGenerator
): boolean {
  // Find player with minimum slack (MRV heuristic)
  let minSlack = Infinity;
  let mostConstrained: number | null = null;

  for (const opp of opponents) {
    const need = remaining.get(opp) ?? 0;
    if (need === 0) continue;

    const candidates = candidatesPerOpponent.get(opp)!;
    let availableCount = 0;
    for (const id of candidates) {
      if (available.has(id)) availableCount++;
    }
    const slack = availableCount - need;

    // Early pruning: if any player can't be satisfied, backtrack immediately
    if (slack < 0) return false;

    if (slack < minSlack) {
      minSlack = slack;
      mostConstrained = opp;
    }
  }

  // All players satisfied
  if (mostConstrained === null) return true;

  // Get shuffled candidates for randomness in the solution
  const candidates = candidatesPerOpponent.get(mostConstrained)!;
  const availableCandidates = [...candidates].filter(id => available.has(id));
  shuffle(availableCandidates, rng);

  // Try each candidate with backtracking
  for (const candidateId of availableCandidates) {
    // Choose
    available.delete(candidateId);
    hands.get(mostConstrained)!.push(candidateId);
    remaining.set(mostConstrained, (remaining.get(mostConstrained) ?? 1) - 1);

    // Recurse
    if (sampleWithBacktracking(opponents, remaining, available, candidatesPerOpponent, hands, rng)) {
      return true;
    }

    // Backtrack
    available.add(candidateId);
    hands.get(mostConstrained)!.pop();
    remaining.set(mostConstrained, (remaining.get(mostConstrained) ?? 0) + 1);
  }

  return false; // No valid assignment found in this branch
}

/**
 * Sample opponent hands that respect all known constraints.
 *
 * Algorithm (backtracking with MRV heuristic):
 * 1. Get the pool of dominoes to distribute (excludes played + AI's hand)
 * 2. For each opponent, build candidate set (respects void constraints)
 * 3. Use backtracking search to find a valid assignment
 * 4. Guaranteed to find a solution if one exists
 *
 * @param constraints Built constraints from game history
 * @param expectedSizes How many dominoes each player should have [p0, p1, p2, p3]
 * @param state Current game state
 * @param rules Composed game rules
 * @param rng Random number generator
 * @param canFollowCache Optional pre-computed canFollow cache (for performance)
 * @returns Map of player index -> sampled hand
 * @throws Error if no valid assignment exists (indicates bug in constraint tracking)
 */
export function sampleOpponentHands(
  constraints: HandConstraints,
  expectedSizes: [number, number, number, number],
  state: GameState,
  rules: GameRules,
  rng: RandomGenerator = defaultRng,
  canFollowCache?: CanFollowCache
): SampledHands {
  const myIndex = constraints.myPlayerIndex;
  const opponents = [0, 1, 2, 3].filter(i => i !== myIndex);

  const pool = getDistributionPool(constraints);
  const available = new Set(pool.map(d => String(d.id)));

  // Build candidate sets (using cache if provided)
  const candidatesPerOpponent = new Map<number, Set<string>>();
  for (const opp of opponents) {
    const candidates = getCandidateDominoes(constraints, opp, state, rules, canFollowCache);
    candidatesPerOpponent.set(opp, new Set(candidates.map(d => String(d.id))));
  }

  // Track remaining need per player
  const remaining = new Map<number, number>();
  for (const opp of opponents) {
    remaining.set(opp, expectedSizes[opp] ?? 0);
  }

  // Result hands (as string IDs for backtracking)
  const handIds = new Map<number, string[]>();
  for (const opp of opponents) {
    handIds.set(opp, []);
  }

  // Run backtracking search
  const success = sampleWithBacktracking(
    opponents,
    remaining,
    available,
    candidatesPerOpponent,
    handIds,
    rng
  );

  if (!success) {
    throw new Error(
      `No valid hand distribution exists. ` +
      `This indicates a bug in constraint tracking.\n` +
      `Pool size: ${pool.length}`
    );
  }

  // Convert string IDs back to Domino objects
  const hands = new Map<number, Domino[]>();
  for (const opp of opponents) {
    const ids = handIds.get(opp)!;
    const dominoes = ids.map(id => pool.find(d => String(d.id) === id)!);
    hands.set(opp, dominoes);
  }

  return hands;
}

/**
 * Sample hands for bidding phase (no constraints).
 *
 * During bidding, we know only our own hand. The other 21 dominoes
 * are randomly distributed among the 3 other players (7 each).
 *
 * @param allDominoes Complete set of 28 dominoes
 * @param myHand The bidder's 7 dominoes
 * @param myPlayerIndex The bidder's player index (0-3)
 * @param rng Random number generator
 * @returns Map of player index -> sampled hand (for all 3 opponents)
 */
export function sampleBiddingHands(
  allDominoes: Domino[],
  myHand: Domino[],
  myPlayerIndex: number,
  rng: RandomGenerator = defaultRng
): SampledHands {
  // Get dominoes not in my hand
  const myHandIds = new Set(myHand.map(d => d.id));
  const pool = allDominoes.filter(d => !myHandIds.has(d.id));

  // Shuffle the pool
  const shuffled = [...pool];
  shuffle(shuffled, rng);

  // Deal 7 to each opponent
  const hands = new Map<number, Domino[]>();
  const opponents = [0, 1, 2, 3].filter(i => i !== myPlayerIndex);

  for (let i = 0; i < 3; i++) {
    const oppIndex = opponents[i]!;
    hands.set(oppIndex, shuffled.slice(i * 7, (i + 1) * 7));
  }

  return hands;
}

/**
 * Create a seeded random number generator.
 * Uses a simple LCG for reproducibility in tests.
 */
export function createSeededRng(seed: number): RandomGenerator {
  let state = seed;

  return {
    random(): number {
      // LCG parameters (same as glibc)
      state = (state * 1103515245 + 12345) & 0x7fffffff;
      return state / 0x7fffffff;
    }
  };
}

// ============================================================================
// Endgame Enumeration
// ============================================================================

/**
 * A single valid assignment of dominoes to opponents.
 * Maps player index -> array of domino IDs.
 */
export type HandAssignment = Map<number, string[]>;

/**
 * Count all valid opponent hand configurations without materializing them.
 *
 * Uses the same backtracking algorithm as sampleWithBacktracking but only counts.
 * This is used to decide whether enumeration is cheaper than sampling.
 *
 * @param opponents Array of opponent player indices
 * @param expectedSizes How many dominoes each opponent should have
 * @param candidatesPerOpponent Pre-built candidate sets per opponent
 * @param available Set of available domino IDs
 * @returns Number of valid configurations, capped at limit+1 to avoid over-counting
 */
export function countValidConfigurations(
  constraints: HandConstraints,
  expectedSizes: [number, number, number, number],
  state: GameState,
  rules: GameRules,
  limit: number,
  canFollowCache?: CanFollowCache
): number {
  const myIndex = constraints.myPlayerIndex;
  const opponents = [0, 1, 2, 3].filter(i => i !== myIndex);

  const pool = getDistributionPool(constraints);
  const available = new Set(pool.map(d => String(d.id)));

  // Build candidate sets
  const candidatesPerOpponent = new Map<number, Set<string>>();
  for (const opp of opponents) {
    const candidates = getCandidateDominoes(constraints, opp, state, rules, canFollowCache);
    candidatesPerOpponent.set(opp, new Set(candidates.map(d => String(d.id))));
  }

  // Track remaining need per player
  const remaining = new Map<number, number>();
  for (const opp of opponents) {
    remaining.set(opp, expectedSizes[opp] ?? 0);
  }

  // Counter for valid configurations
  let count = 0;

  function countRecursive(availableSet: Set<string>): void {
    // Early termination if we've exceeded the limit
    if (count > limit) return;

    // Find player with minimum slack (MRV heuristic)
    let minSlack = Infinity;
    let mostConstrained: number | null = null;

    for (const opp of opponents) {
      const need = remaining.get(opp) ?? 0;
      if (need === 0) continue;

      const candidates = candidatesPerOpponent.get(opp)!;
      let availableCount = 0;
      for (const id of candidates) {
        if (availableSet.has(id)) availableCount++;
      }
      const slack = availableCount - need;

      // Early pruning: if any player can't be satisfied, backtrack immediately
      if (slack < 0) return;

      if (slack < minSlack) {
        minSlack = slack;
        mostConstrained = opp;
      }
    }

    // All players satisfied - found a valid configuration!
    if (mostConstrained === null) {
      count++;
      return;
    }

    // Get candidates for most constrained player
    const candidates = candidatesPerOpponent.get(mostConstrained)!;
    const availableCandidates = [...candidates].filter(id => availableSet.has(id));

    // Try each candidate with backtracking
    for (const candidateId of availableCandidates) {
      if (count > limit) return;

      // Choose
      availableSet.delete(candidateId);
      remaining.set(mostConstrained, (remaining.get(mostConstrained) ?? 1) - 1);

      // Recurse
      countRecursive(availableSet);

      // Backtrack
      availableSet.add(candidateId);
      remaining.set(mostConstrained, (remaining.get(mostConstrained) ?? 0) + 1);
    }
  }

  countRecursive(available);

  return count;
}

/**
 * Enumerate all valid opponent hand configurations.
 *
 * Instead of sampling randomly, this generates all possible valid assignments.
 * Use when the number of configurations is small enough (< threshold).
 *
 * @param constraints Built constraints from game history
 * @param expectedSizes How many dominoes each player should have
 * @param state Current game state
 * @param rules Composed game rules
 * @param maxConfigurations Maximum number to enumerate (safety limit)
 * @param canFollowCache Optional pre-computed canFollow cache
 * @returns Array of all valid hand assignments, each as a SampledHands map
 */
export function enumerateAllConfigurations(
  constraints: HandConstraints,
  expectedSizes: [number, number, number, number],
  state: GameState,
  rules: GameRules,
  maxConfigurations: number = 1000,
  canFollowCache?: CanFollowCache
): SampledHands[] {
  const myIndex = constraints.myPlayerIndex;
  const opponents = [0, 1, 2, 3].filter(i => i !== myIndex);

  const pool = getDistributionPool(constraints);
  const available = new Set(pool.map(d => String(d.id)));

  // Build candidate sets
  const candidatesPerOpponent = new Map<number, Set<string>>();
  for (const opp of opponents) {
    const candidates = getCandidateDominoes(constraints, opp, state, rules, canFollowCache);
    candidatesPerOpponent.set(opp, new Set(candidates.map(d => String(d.id))));
  }

  // Track remaining need per player
  const remaining = new Map<number, number>();
  for (const opp of opponents) {
    remaining.set(opp, expectedSizes[opp] ?? 0);
  }

  // Current assignment being built (domino IDs per player)
  const currentAssignment = new Map<number, string[]>();
  for (const opp of opponents) {
    currentAssignment.set(opp, []);
  }

  // Collect all valid configurations
  const configurations: SampledHands[] = [];

  function enumerateRecursive(availableSet: Set<string>): void {
    // Safety limit
    if (configurations.length >= maxConfigurations) return;

    // Find player with minimum slack (MRV heuristic for efficient enumeration)
    let minSlack = Infinity;
    let mostConstrained: number | null = null;

    for (const opp of opponents) {
      const need = remaining.get(opp) ?? 0;
      if (need === 0) continue;

      const candidates = candidatesPerOpponent.get(opp)!;
      let availableCount = 0;
      for (const id of candidates) {
        if (availableSet.has(id)) availableCount++;
      }
      const slack = availableCount - need;

      // Early pruning
      if (slack < 0) return;

      if (slack < minSlack) {
        minSlack = slack;
        mostConstrained = opp;
      }
    }

    // All players satisfied - save this configuration
    if (mostConstrained === null) {
      // Convert current assignment to SampledHands (Domino objects)
      const sampledHands: SampledHands = new Map();
      for (const opp of opponents) {
        const ids = currentAssignment.get(opp)!;
        const dominoes = ids.map(id => pool.find(d => String(d.id) === id)!);
        sampledHands.set(opp, dominoes);
      }
      configurations.push(sampledHands);
      return;
    }

    // Get sorted candidates for deterministic enumeration
    const candidates = candidatesPerOpponent.get(mostConstrained)!;
    const availableCandidates = [...candidates].filter(id => availableSet.has(id)).sort();

    // Try each candidate
    for (const candidateId of availableCandidates) {
      if (configurations.length >= maxConfigurations) return;

      // Choose
      availableSet.delete(candidateId);
      currentAssignment.get(mostConstrained)!.push(candidateId);
      remaining.set(mostConstrained, (remaining.get(mostConstrained) ?? 1) - 1);

      // Recurse
      enumerateRecursive(availableSet);

      // Backtrack
      availableSet.add(candidateId);
      currentAssignment.get(mostConstrained)!.pop();
      remaining.set(mostConstrained, (remaining.get(mostConstrained) ?? 0) + 1);
    }
  }

  enumerateRecursive(available);

  return configurations;
}
