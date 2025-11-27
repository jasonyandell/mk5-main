/**
 * Hand Sampler for Monte Carlo AI
 *
 * Generates random but constraint-respecting opponent hand distributions.
 * Uses dynamic greedy algorithm (min-slack first) for deterministic O(n) sampling.
 *
 * Invariant: A valid distribution MUST always exist (the real game state is one).
 * If Hall's condition is violated, that's a bug in constraint tracking.
 */

import type { Domino, TrumpSelection } from '../types';
import type { HandConstraints } from './constraint-tracker';
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
 * Sample opponent hands that respect all known constraints.
 *
 * Algorithm (dynamic greedy, min-slack first):
 * 1. Get the pool of dominoes to distribute (excludes played + AI's hand)
 * 2. For each opponent, build candidate set (respects void constraints)
 * 3. Repeatedly:
 *    - Find player with minimum slack (available candidates - remaining need)
 *    - Randomly assign one domino from their available candidates
 * 4. Deterministic guarantee: always succeeds if constraints are satisfiable
 *
 * @param constraints Built constraints from game history
 * @param expectedSizes How many dominoes each player should have [p0, p1, p2, p3]
 * @param trump Current trump selection (needed for suit membership)
 * @param rng Random number generator
 * @returns Map of player index -> sampled hand
 * @throws Error if Hall's condition violated (indicates bug in constraint tracking)
 */
export function sampleOpponentHands(
  constraints: HandConstraints,
  expectedSizes: [number, number, number, number],
  trump: TrumpSelection,
  rng: RandomGenerator = defaultRng
): SampledHands {
  const myIndex = constraints.myPlayerIndex;
  const opponents = [0, 1, 2, 3].filter(i => i !== myIndex);

  const pool = getDistributionPool(constraints);
  const available = new Set(pool.map(d => String(d.id)));

  // Build candidate sets
  const candidatesPerOpponent = new Map<number, Set<string>>();
  for (const opp of opponents) {
    const candidates = getCandidateDominoes(constraints, opp, trump);
    candidatesPerOpponent.set(opp, new Set(candidates.map(d => String(d.id))));
  }

  // Track remaining need per player
  const remaining = new Map<number, number>();
  for (const opp of opponents) {
    remaining.set(opp, expectedSizes[opp] ?? 0);
  }

  // Result hands
  const hands = new Map<number, Domino[]>();
  for (const opp of opponents) {
    hands.set(opp, []);
  }

  // Assign one domino at a time, always picking most constrained player
  while (true) {
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

      if (slack < minSlack) {
        minSlack = slack;
        mostConstrained = opp;
      }
    }

    if (mostConstrained === null) break;

    if (minSlack < 0) {
      // This should never happen if constraints are correct
      throw new Error(
        `Hall's condition violated for player ${mostConstrained}. ` +
        `This indicates a bug in constraint tracking.\n` +
        `Pool size: ${pool.length}`
      );
    }

    // Random selection from available candidates
    const candidates = candidatesPerOpponent.get(mostConstrained)!;
    const availableCandidates = [...candidates].filter(id => available.has(id));
    const idx = Math.floor(rng.random() * availableCandidates.length);
    const selectedId = availableCandidates[idx]!;

    const domino = pool.find(d => String(d.id) === selectedId)!;
    hands.get(mostConstrained)!.push(domino);
    available.delete(selectedId);
    remaining.set(mostConstrained, (remaining.get(mostConstrained) ?? 1) - 1);
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
