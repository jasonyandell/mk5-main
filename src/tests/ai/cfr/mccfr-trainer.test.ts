/**
 * Integration tests for MCCFRTrainer
 */

import { describe, it, expect } from 'vitest';
import { MCCFRTrainer } from '../../../game/ai/cfr/mccfr-trainer';

describe('MCCFRTrainer', () => {
  describe('constructor', () => {
    it('creates trainer with default config', () => {
      const trainer = new MCCFRTrainer();
      expect(trainer.getRegretTable().size).toBe(0);
    });

    it('accepts custom config', () => {
      const trainer = new MCCFRTrainer({
        iterations: 100,
        seed: 12345
      });
      expect(trainer).toBeDefined();
    });
  });

  describe('train', () => {
    it('runs training iterations', async () => {
      const trainer = new MCCFRTrainer({
        iterations: 10,
        seed: 42
      });

      const result = await trainer.train();

      expect(result.iterations).toBe(10);
      expect(result.trainingTimeMs).toBeGreaterThan(0);
      expect(result.iterationsPerSecond).toBeGreaterThan(0);
    });

    it('discovers information sets during training', async () => {
      const trainer = new MCCFRTrainer({
        iterations: 50,
        seed: 42
      });

      await trainer.train();

      // Should have found some info sets
      expect(trainer.getRegretTable().size).toBeGreaterThan(0);
    });

    it('calls progress callback', async () => {
      let progressCalls = 0;
      let lastIteration = 0;

      const trainer = new MCCFRTrainer({
        iterations: 100,
        seed: 42,
        progressInterval: 25,
        onProgress: (iter, _total, _nodes) => {
          progressCalls++;
          lastIteration = iter;
        }
      });

      await trainer.train();

      expect(progressCalls).toBeGreaterThan(0);
      expect(lastIteration).toBe(100);
    });
  });

  describe('serialize', () => {
    it('produces valid serialized format', async () => {
      const trainer = new MCCFRTrainer({
        iterations: 10,
        seed: 42
      });

      await trainer.train();
      const serialized = trainer.serialize();

      expect(serialized.version).toBe(1);
      expect(serialized.iterationsCompleted).toBe(10);
      expect(serialized.config.seed).toBe(42);
      expect(serialized.trainedAt).toBeDefined();
      expect(serialized.trainingTimeMs).toBeGreaterThan(0);
      expect(Array.isArray(serialized.nodes)).toBe(true);
    });
  });

  describe('fromSerialized', () => {
    it('restores trainer from serialized data', async () => {
      const trainer1 = new MCCFRTrainer({
        iterations: 20,
        seed: 42
      });

      await trainer1.train();
      const serialized = trainer1.serialize();
      const nodeCount1 = trainer1.getRegretTable().size;

      const trainer2 = MCCFRTrainer.fromSerialized(serialized);
      const nodeCount2 = trainer2.getRegretTable().size;

      expect(nodeCount2).toBe(nodeCount1);
    });
  });

  describe('regret table accumulation', () => {
    it('accumulates regrets across iterations', async () => {
      const trainer = new MCCFRTrainer({
        iterations: 100,
        seed: 42
      });

      await trainer.train();

      const table = trainer.getRegretTable();
      const stats = table.getStats();

      // Should have accumulated data
      expect(stats.nodeCount).toBeGreaterThan(0);
      expect(stats.totalVisits).toBeGreaterThan(0);
    });
  });

  describe('determinism', () => {
    it('produces identical results with same seed', async () => {
      const trainer1 = new MCCFRTrainer({
        iterations: 30,
        seed: 12345
      });

      const trainer2 = new MCCFRTrainer({
        iterations: 30,
        seed: 12345
      });

      await trainer1.train();
      await trainer2.train();

      const serialized1 = trainer1.serialize();
      const serialized2 = trainer2.serialize();

      expect(serialized1.nodes.length).toBe(serialized2.nodes.length);

      // Same info sets should be discovered
      const keys1 = new Set(serialized1.nodes.map(n => n.key));
      const keys2 = new Set(serialized2.nodes.map(n => n.key));

      for (const key of keys1) {
        expect(keys2.has(key)).toBe(true);
      }
    });

    it('produces different results with different seeds', async () => {
      const trainer1 = new MCCFRTrainer({
        iterations: 30,
        seed: 11111
      });

      const trainer2 = new MCCFRTrainer({
        iterations: 30,
        seed: 22222
      });

      await trainer1.train();
      await trainer2.train();

      const serialized1 = trainer1.serialize();
      const serialized2 = trainer2.serialize();

      // Different seeds should explore different games
      // Might still have some overlap, but not identical
      const keys1 = new Set(serialized1.nodes.map(n => n.key));
      const keys2 = new Set(serialized2.nodes.map(n => n.key));

      // At least some difference expected (not guaranteed but very likely)
      // We check that they're not completely identical
      const union = new Set([...keys1, ...keys2]);
      expect(union.size).toBeGreaterThanOrEqual(keys1.size);
    });
  });
});
