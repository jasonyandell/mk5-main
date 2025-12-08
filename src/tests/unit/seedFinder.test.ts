import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { testSeedWinRate, findCompetitiveSeed, SEED_FINDER_CONFIG } from '../../game/ai/gameSimulator';
import { setDefaultAIStrategy, getDefaultAIStrategy, resetRandomStrategy, setRandomStrategySeed, type AIStrategyType } from '../../game/ai/actionSelector';

// Override for fast tests
SEED_FINDER_CONFIG.MAX_ATTEMPTS = 100;
SEED_FINDER_CONFIG.SIMULATIONS_PER_SEED = 10;

// Use random strategy for fast simulation tests
let originalStrategy: AIStrategyType;
beforeAll(() => {
  originalStrategy = getDefaultAIStrategy();
  setDefaultAIStrategy('random');
});
afterAll(() => {
  setDefaultAIStrategy(originalStrategy);
  resetRandomStrategy();
});

describe('Game Simulator - Seed Testing', () => {
  describe('testSeedWinRate', () => {
    it('should test a seed and return win rate', async () => {
      const result = await testSeedWinRate(12345, SEED_FINDER_CONFIG.SIMULATIONS_PER_SEED);

      expect(result).toBeDefined();
      expect(result.winRate).toBeGreaterThanOrEqual(0);
      expect(result.winRate).toBeLessThanOrEqual(1);
      expect(result.avgScore).toBeGreaterThanOrEqual(0);
    });

    it('should return consistent results for the same seed', async () => {
      const seed = 54321;
      const simulations = 5;

      // Use seeded RNG for deterministic behavior
      setRandomStrategySeed(42);
      const result1 = await testSeedWinRate(seed, simulations);

      // Reset to same seed for second run
      setRandomStrategySeed(42);
      const result2 = await testSeedWinRate(seed, simulations);

      // Results should be identical for same seed
      expect(result1.winRate).toBe(result2.winRate);
      expect(result1.avgScore).toBe(result2.avgScore);

      // Reset to non-deterministic for other tests
      resetRandomStrategy();
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

  });
});
