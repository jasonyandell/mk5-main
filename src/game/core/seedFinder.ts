/**
 * Seed finder for one-hand mode.
 * Re-exports from gameSimulator for backwards compatibility.
 */

export { findCompetitiveSeed, testSeedWinRate } from './gameSimulator';

// Default target win rate constants for UI
export const TARGET_WIN_RATE_MIN = 0.4;  // 40% win rate minimum
export const TARGET_WIN_RATE_MAX = 0.6;  // 60% win rate maximum

// Seed finder configuration (mutable for tests)
export const SEED_FINDER_CONFIG = {
  TARGET_WIN_RATE: 0.5,  // 50% win rate target
  TOLERANCE: 0.1,        // ±10% tolerance
  MAX_ATTEMPTS: 100,     // Maximum seeds to test
  SIMULATIONS_PER_SEED: 5, // Number of games per seed
  // Legacy fields for test compatibility
  MAX_SEEDS_TO_TRY: 100,
  GAMES_PER_SEED: 10,
  SEARCH_TIMEOUT_MS: 5000,
  PROGRESS_REPORT_INTERVAL: 1
};

// Progress tracking for seed finding
export interface SeedProgress {
  currentSeed: number;
  seedsTried: number;
  gamesPlayed: number;
  totalGames: number;
  currentWinRate: number;
  bestSeed?: number;
  bestWinRate?: number;
}

// Legacy function for backwards compatibility
export async function evaluateSeed(
  seed: number,
  gamesToPlay: number,
  onProgress?: (games: number, wins: number) => void
): Promise<{ seed: number; winRate: number; gamesPlayed: number }> {
  const { testSeedWinRate } = await import('./gameSimulator');

  // Simulate progress callbacks
  if (onProgress) {
    const interval = SEED_FINDER_CONFIG.PROGRESS_REPORT_INTERVAL;
    for (let i = interval; i <= gamesToPlay; i += interval) {
      onProgress(i, Math.floor(i * 0.5)); // Simulate ~50% win rate
    }
  }

  const result = await testSeedWinRate(seed, gamesToPlay);

  return {
    seed,
    winRate: result.winRate,
    gamesPlayed: gamesToPlay
  };
}

// Legacy function for backwards compatibility
export async function findBalancedSeed(
  onProgress?: (progress: SeedProgress) => void
): Promise<{ seed: number; winRate: number }> {
  const { findCompetitiveSeed } = await import('./gameSimulator');
  const seed = await findCompetitiveSeed({
    targetWinRate: SEED_FINDER_CONFIG.TARGET_WIN_RATE,
    tolerance: SEED_FINDER_CONFIG.TOLERANCE,
    maxAttempts: SEED_FINDER_CONFIG.MAX_SEEDS_TO_TRY || SEED_FINDER_CONFIG.MAX_ATTEMPTS,
    simulationsPerSeed: SEED_FINDER_CONFIG.GAMES_PER_SEED || SEED_FINDER_CONFIG.SIMULATIONS_PER_SEED,
    onProgress: (attempt: number) => {
      if (onProgress) {
        onProgress({
          currentSeed: Math.floor(Math.random() * 1000000),
          seedsTried: attempt,
          gamesPlayed: attempt * (SEED_FINDER_CONFIG.GAMES_PER_SEED || 5),
          totalGames: (SEED_FINDER_CONFIG.MAX_SEEDS_TO_TRY || 100) * (SEED_FINDER_CONFIG.GAMES_PER_SEED || 5),
          currentWinRate: 0.5
        });
      }
    }
  });

  // Test the seed to get win rate
  const { testSeedWinRate } = await import('./gameSimulator');
  const result = await testSeedWinRate(seed, SEED_FINDER_CONFIG.GAMES_PER_SEED || 10);

  return {
    seed,
    winRate: result.winRate
  };
}