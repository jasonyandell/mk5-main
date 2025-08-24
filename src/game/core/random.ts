/**
 * Seeded random number generator using Linear Congruential Generator (LCG)
 * This ensures deterministic randomness for game replay
 */
export class SeededRandom {
  private seed: number;

  constructor(seed: number) {
    this.seed = seed;
  }

  /**
   * Generate next random number between 0 and 1
   * Uses Park and Miller's "minimal standard" LCG constants
   */
  next(): number {
    // LCG formula: (a * seed) % m
    // Using Park & Miller constants: a = 16807, m = 2^31 - 1
    const a = 16807;
    const m = 2147483647; // 2^31 - 1
    
    this.seed = (a * this.seed) % m;
    return this.seed / m;
  }

  /**
   * Generate random integer between min (inclusive) and max (exclusive)
   */
  nextInt(min: number, max: number): number {
    return Math.floor(this.next() * (max - min)) + min;
  }
}

/**
 * Create a seeded RNG from a seed value
 * Ensures seed is a positive integer
 */
export function createSeededRandom(seed: number): SeededRandom {
  // Ensure seed is a positive integer
  const safeSeed = Math.abs(Math.floor(seed)) || 1;
  return new SeededRandom(safeSeed);
}

/**
 * Simple hash function for generating deterministic values
 */
export function createHash(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return Math.abs(hash);
}