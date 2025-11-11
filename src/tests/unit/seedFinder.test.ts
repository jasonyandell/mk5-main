import { describe, it, expect } from 'vitest';
import { testSeedWinRate, findCompetitiveSeed, SEED_FINDER_CONFIG } from '../../game/ai/gameSimulator';

// Override for fast tests
SEED_FINDER_CONFIG.MAX_SEEDS_TO_TRY = 100;
SEED_FINDER_CONFIG.GAMES_PER_SEED = 10;
SEED_FINDER_CONFIG.SEARCH_TIMEOUT_MS = 4000;

describe('Game Simulator - Seed Testing', () => {
  describe('testSeedWinRate', () => {
    it('should test a seed and return win rate', async () => {
      const result = await testSeedWinRate(12345, SEED_FINDER_CONFIG.GAMES_PER_SEED);

      expect(result).toBeDefined();
      expect(result.winRate).toBeGreaterThanOrEqual(0);
      expect(result.winRate).toBeLessThanOrEqual(1);
      expect(result.avgScore).toBeGreaterThanOrEqual(0);
    });

    it('should return consistent results for the same seed', async () => {
      const seed = 54321;
      const simulations = 5;

      const result1 = await testSeedWinRate(seed, simulations);
      const result2 = await testSeedWinRate(seed, simulations);

      // Results should be identical for same seed
      expect(result1.winRate).toBe(result2.winRate);
      expect(result1.avgScore).toBe(result2.avgScore);
    });

    it('should respect simulations count', async () => {
      const seed = 11111;
      const simulations = 3;

      const result = await testSeedWinRate(seed, simulations);

      expect(result).toBeDefined();
      expect(result.winRate).toBeGreaterThanOrEqual(0);
      expect(result.winRate).toBeLessThanOrEqual(1);
    });
  });

  describe('findCompetitiveSeed', () => {
    it('should find a seed within acceptable win rate range', async () => {
      let progressCalls = 0;

      const seed = await findCompetitiveSeed({
        targetWinRate: SEED_FINDER_CONFIG.TARGET_WIN_RATE,
        tolerance: SEED_FINDER_CONFIG.TOLERANCE,
        maxAttempts: SEED_FINDER_CONFIG.MAX_SEEDS_TO_TRY,
        simulationsPerSeed: SEED_FINDER_CONFIG.SIMULATIONS_PER_SEED,
        onProgress: (attempt) => {
          progressCalls++;
          expect(attempt).toBeGreaterThan(0);
          expect(attempt).toBeLessThanOrEqual(SEED_FINDER_CONFIG.MAX_SEEDS_TO_TRY);
        }
      });

      expect(seed).toBeGreaterThan(0);
      expect(seed).toBeLessThan(1000000);
      expect(progressCalls).toBeGreaterThan(0);
    });

    it('should call progress callback during search', async () => {
      let lastAttempt = 0;
      const progressAttempts: number[] = [];

      await findCompetitiveSeed({
        targetWinRate: 0.5,
        tolerance: 0.1,
        maxAttempts: 10,
        simulationsPerSeed: 3,
        onProgress: (attempt) => {
          progressAttempts.push(attempt);
          expect(attempt).toBeGreaterThan(lastAttempt);
          lastAttempt = attempt;
        }
      });

      expect(progressAttempts.length).toBeGreaterThan(0);
      expect(progressAttempts.length).toBeLessThanOrEqual(10);
    });

    it('should return a valid seed even if no perfect match found', async () => {
      // Use very tight constraints to force fallback
      const seed = await findCompetitiveSeed({
        targetWinRate: 0.5,
        tolerance: 0.01,  // Very tight tolerance
        maxAttempts: 3,    // Very few attempts
        simulationsPerSeed: 2
      });

      expect(seed).toBeGreaterThan(0);
      expect(seed).toBeLessThan(1000000);
    });

    it('should respect maxAttempts limit', async () => {
      let attempts = 0;
      const maxAttempts = 5;

      await findCompetitiveSeed({
        targetWinRate: 0.5,
        tolerance: 0.1,
        maxAttempts,
        simulationsPerSeed: 2,
        onProgress: () => {
          attempts++;
        }
      });

      expect(attempts).toBeLessThanOrEqual(maxAttempts);
    });
  });

  describe('SEED_FINDER_CONFIG', () => {
    it('should have valid configuration values', () => {
      expect(SEED_FINDER_CONFIG.TARGET_WIN_RATE).toBeGreaterThan(0);
      expect(SEED_FINDER_CONFIG.TARGET_WIN_RATE).toBeLessThan(1);
      expect(SEED_FINDER_CONFIG.TOLERANCE).toBeGreaterThan(0);
      expect(SEED_FINDER_CONFIG.TOLERANCE).toBeLessThan(1);
      expect(SEED_FINDER_CONFIG.MAX_ATTEMPTS).toBeGreaterThan(0);
      expect(SEED_FINDER_CONFIG.SIMULATIONS_PER_SEED).toBeGreaterThan(0);
    });

    it('should have legacy fields for backward compatibility', () => {
      expect(SEED_FINDER_CONFIG.MAX_SEEDS_TO_TRY).toBeDefined();
      expect(SEED_FINDER_CONFIG.GAMES_PER_SEED).toBeDefined();
      expect(SEED_FINDER_CONFIG.SEARCH_TIMEOUT_MS).toBeDefined();
      expect(SEED_FINDER_CONFIG.PROGRESS_REPORT_INTERVAL).toBeDefined();
    });
  });
});
