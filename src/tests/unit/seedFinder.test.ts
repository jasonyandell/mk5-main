import { describe, it, expect } from 'vitest';
import { testSeedWinRate, findCompetitiveSeed, SEED_FINDER_CONFIG } from '../../game/ai/gameSimulator';
import { createSeededRng } from '../../game/ai/hand-sampler';

// Override for fast tests
SEED_FINDER_CONFIG.MAX_ATTEMPTS = 100;
SEED_FINDER_CONFIG.SIMULATIONS_PER_SEED = 10;

describe('Game Simulator - Seed Testing', () => {
  // Use random strategy for fast simulation tests
  const fastConfig = { aiStrategyConfig: { type: 'random' as const } };

  describe('testSeedWinRate', () => {
    it('should test a seed and return win rate', async () => {
      const result = await testSeedWinRate(12345, SEED_FINDER_CONFIG.SIMULATIONS_PER_SEED, fastConfig);

      expect(result).toBeDefined();
      expect(result.winRate).toBeGreaterThanOrEqual(0);
      expect(result.winRate).toBeLessThanOrEqual(1);
      expect(result.avgScore).toBeGreaterThanOrEqual(0);
    });

    it('should return consistent results for the same seed with injected RNG', async () => {
      const seed = 54321;
      const simulations = 5;

      // Create configs with same seeded RNG
      const config1 = { aiStrategyConfig: { type: 'random' as const, rng: createSeededRng(42) } };
      const config2 = { aiStrategyConfig: { type: 'random' as const, rng: createSeededRng(42) } };

      const result1 = await testSeedWinRate(seed, simulations, config1);
      const result2 = await testSeedWinRate(seed, simulations, config2);

      // Results should be identical for same seed
      expect(result1.winRate).toBe(result2.winRate);
      expect(result1.avgScore).toBe(result2.avgScore);
    });

    it('should respect simulations count', async () => {
      const seed = 11111;
      const simulations = 3;

      const result = await testSeedWinRate(seed, simulations, fastConfig);

      expect(result).toBeDefined();
      expect(result.winRate).toBeGreaterThanOrEqual(0);
      expect(result.winRate).toBeLessThanOrEqual(1);
    });
  });

  describe('findCompetitiveSeed', () => {
    it('should call progress callback during search', async () => {
      let lastAttempt = 0;
      const progressAttempts: number[] = [];

      await findCompetitiveSeed({
        targetWinRate: 0.5,
        tolerance: 0.1,
        maxAttempts: 10,
        simulationsPerSeed: 3,
        aiStrategyConfig: { type: 'random' },
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
        simulationsPerSeed: 2,
        aiStrategyConfig: { type: 'random' }
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
        aiStrategyConfig: { type: 'random' },
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

  });
});
