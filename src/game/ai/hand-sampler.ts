/**
 * Hand Sampler for Monte Carlo AI
 *
 * Generates random but constraint-respecting opponent hand distributions.
 * Uses rejection sampling when constraints are tight.
 *
 * Invariant: A valid distribution MUST always exist (the real game state is one).
 * If sampling repeatedly fails, that's a bug in constraint tracking.
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
 * Algorithm:
 * 1. Get the pool of dominoes to distribute (excludes played + AI's hand)
 * 2. For each opponent, filter pool to candidates (respects void constraints)
 * 3. Use constraint propagation + random assignment
 * 4. If invalid, retry (rejection sampling)
 *
 * @param constraints Built constraints from game history
 * @param expectedSizes How many dominoes each player should have [p0, p1, p2, p3]
 * @param trump Current trump selection (needed for suit membership)
 * @param rng Random number generator
 * @param maxAttempts Maximum sampling attempts before giving up
 * @returns Map of player index -> sampled hand
 * @throws Error if no valid distribution found after maxAttempts (indicates bug)
 */
export function sampleOpponentHands(
  constraints: HandConstraints,
  expectedSizes: [number, number, number, number],
  trump: TrumpSelection,
  rng: RandomGenerator = defaultRng,
  maxAttempts: number = 1000
): SampledHands {
  const myIndex = constraints.myPlayerIndex;
  const opponents = [0, 1, 2, 3].filter(i => i !== myIndex);

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const result = trySampleOnce(constraints, expectedSizes, trump, opponents, rng);
    if (result !== null) {
      return result;
    }
  }

  // This should never happen if constraints are correct
  // Add debug info about why sampling failed
  const pool = getDistributionPool(constraints);
  const candCounts: Record<number, number> = {};
  for (const opp of opponents) {
    candCounts[opp] = getCandidateDominoes(constraints, opp, trump).length;
  }

  throw new Error(
    `Failed to sample valid hand distribution after ${maxAttempts} attempts. ` +
    `This indicates a bug in constraint tracking.\n` +
    `Pool size: ${pool.length}, Opponent candidates: ${JSON.stringify(candCounts)}, ` +
    `Expected: ${JSON.stringify(expectedSizes)}`
  );
}

/**
 * Attempt to sample a valid distribution once.
 * Returns null if this attempt failed (constraints couldn't be satisfied).
 */
function trySampleOnce(
  constraints: HandConstraints,
  expectedSizes: [number, number, number, number],
  trump: TrumpSelection,
  opponents: number[],
  rng: RandomGenerator
): SampledHands | null {
  // Get the pool of dominoes to distribute
  const pool = getDistributionPool(constraints);
  const poolSet = new Set(pool.map(d => String(d.id)));

  // Get candidate dominoes for each opponent
  const candidatesPerOpponent = new Map<number, Set<string>>();
  for (const opp of opponents) {
    const candidates = getCandidateDominoes(constraints, opp, trump);
    candidatesPerOpponent.set(opp, new Set(candidates.map(d => String(d.id))));
  }

  // Result hands
  const hands = new Map<number, Domino[]>();
  for (const opp of opponents) {
    hands.set(opp, []);
  }

  // Track which dominoes are still available
  const available = new Set(poolSet);

  // Sort opponents by constraint tightness (most constrained first)
  // This improves success rate of rejection sampling
  const sortedOpponents = [...opponents].sort((a, b) => {
    const candA = candidatesPerOpponent.get(a)?.size ?? 0;
    const candB = candidatesPerOpponent.get(b)?.size ?? 0;
    const needA = expectedSizes[a] ?? 0;
    const needB = expectedSizes[b] ?? 0;
    // Ratio of candidates to need - lower is more constrained
    const ratioA = needA > 0 ? candA / needA : Infinity;
    const ratioB = needB > 0 ? candB / needB : Infinity;
    return ratioA - ratioB;
  });

  // Assign dominoes to each opponent
  for (const opp of sortedOpponents) {
    const need = expectedSizes[opp] ?? 0;
    const candidates = candidatesPerOpponent.get(opp);
    if (!candidates) continue;

    // Get available candidates for this opponent
    const availableCandidates: string[] = [];
    for (const id of candidates) {
      if (available.has(id)) {
        availableCandidates.push(id);
      }
    }

    // Check if we have enough candidates
    if (availableCandidates.length < need) {
      return null; // Failed - not enough candidates
    }

    // Randomly select 'need' dominoes from available candidates
    const selected = randomSelect(availableCandidates, need, rng);

    // Convert IDs back to dominoes and add to hand
    const oppHand = hands.get(opp)!;
    for (const id of selected) {
      const domino = pool.find(d => String(d.id) === id);
      if (domino) {
        oppHand.push(domino);
        available.delete(id);
      }
    }
  }

  // Verify all dominoes were distributed
  if (available.size !== 0) {
    return null; // Failed - leftover dominoes
  }

  // Verify each opponent got the right count
  for (const opp of opponents) {
    const hand = hands.get(opp);
    const expected = expectedSizes[opp] ?? 0;
    if (!hand || hand.length !== expected) {
      return null; // Failed - wrong count
    }
  }

  return hands;
}

/**
 * Randomly select k items from an array without replacement.
 * Uses Fisher-Yates partial shuffle for efficiency.
 */
function randomSelect<T>(items: T[], k: number, rng: RandomGenerator): T[] {
  if (k >= items.length) {
    return [...items];
  }

  // Copy array for shuffling
  const arr = [...items];
  const result: T[] = [];

  for (let i = 0; i < k; i++) {
    const remaining = arr.length - i;
    const idx = i + Math.floor(rng.random() * remaining);

    // Swap
    const temp = arr[i]!;
    arr[i] = arr[idx]!;
    arr[idx] = temp;

    result.push(arr[i]!);
  }

  return result;
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
