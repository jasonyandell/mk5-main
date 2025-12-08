/**
 * Unit tests for RegretTable
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { RegretTable } from '../../../game/ai/cfr/regret-table';
import type { MCCFRConfig } from '../../../game/ai/cfr/types';

describe('RegretTable', () => {
  let table: RegretTable;

  beforeEach(() => {
    table = new RegretTable();
  });

  describe('getNode', () => {
    it('creates a new node for unknown info set', () => {
      const node = table.getNode('test-info-set');
      expect(node).toBeDefined();
      expect(node.regrets.size).toBe(0);
      expect(node.strategySum.size).toBe(0);
      expect(node.visitCount).toBe(0);
    });

    it('returns same node for same info set', () => {
      const node1 = table.getNode('test-info-set');
      node1.visitCount = 5;
      const node2 = table.getNode('test-info-set');
      expect(node2.visitCount).toBe(5);
    });
  });

  describe('getStrategy', () => {
    it('returns uniform distribution for new info set', () => {
      const actions = ['6-4', '5-5', '3-2'];
      const strategy = table.getStrategy('new-info-set', actions);

      expect(strategy.size).toBe(3);
      for (const action of actions) {
        expect(strategy.get(action)).toBeCloseTo(1 / 3, 5);
      }
    });

    it('uses regret matching for positive regrets', () => {
      const node = table.getNode('test-info-set');
      node.regrets.set('6-4', 10);
      node.regrets.set('5-5', 20);
      node.regrets.set('3-2', 0);

      const strategy = table.getStrategy('test-info-set', ['6-4', '5-5', '3-2']);

      // Total positive regret = 30
      expect(strategy.get('6-4')).toBeCloseTo(10 / 30, 5);
      expect(strategy.get('5-5')).toBeCloseTo(20 / 30, 5);
      expect(strategy.get('3-2')).toBe(0);
    });

    it('returns uniform for all non-positive regrets', () => {
      const node = table.getNode('test-info-set');
      node.regrets.set('6-4', -5);
      node.regrets.set('5-5', 0);
      node.regrets.set('3-2', -10);

      const strategy = table.getStrategy('test-info-set', ['6-4', '5-5', '3-2']);

      for (const action of ['6-4', '5-5', '3-2']) {
        expect(strategy.get(action)).toBeCloseTo(1 / 3, 5);
      }
    });
  });

  describe('getAverageStrategy', () => {
    it('returns uniform for unknown info set', () => {
      const strategy = table.getAverageStrategy('unknown', ['6-4', '5-5']);
      expect(strategy.get('6-4')).toBeCloseTo(0.5, 5);
      expect(strategy.get('5-5')).toBeCloseTo(0.5, 5);
    });

    it('normalizes strategy sums to probabilities', () => {
      const node = table.getNode('test-info-set');
      node.strategySum.set('6-4', 100);
      node.strategySum.set('5-5', 300);
      node.strategySum.set('3-2', 100);

      const strategy = table.getAverageStrategy('test-info-set', ['6-4', '5-5', '3-2']);

      expect(strategy.get('6-4')).toBeCloseTo(0.2, 5);
      expect(strategy.get('5-5')).toBeCloseTo(0.6, 5);
      expect(strategy.get('3-2')).toBeCloseTo(0.2, 5);
    });
  });

  describe('updateRegrets', () => {
    it('updates regrets based on counterfactual values', () => {
      const actionValues = new Map<string, number>([
        ['6-4', 10],
        ['5-5', 20],
        ['3-2', 5]
      ]);
      const expectedValue = 12; // Weighted average
      const opponentReachProb = 0.5;

      table.updateRegrets('test-info-set', actionValues, expectedValue, opponentReachProb);

      const node = table.getNode('test-info-set');
      // Regret = (value - expectedValue) * opponentReachProb
      expect(node.regrets.get('6-4')).toBeCloseTo((10 - 12) * 0.5, 5);
      expect(node.regrets.get('5-5')).toBeCloseTo((20 - 12) * 0.5, 5);
      expect(node.regrets.get('3-2')).toBeCloseTo((5 - 12) * 0.5, 5);
    });

    it('accumulates regrets over multiple updates', () => {
      const actionValues = new Map<string, number>([['6-4', 10]]);

      table.updateRegrets('test-info-set', actionValues, 5, 1.0);
      table.updateRegrets('test-info-set', actionValues, 5, 1.0);

      const node = table.getNode('test-info-set');
      expect(node.regrets.get('6-4')).toBe(10); // (10-5)*1 + (10-5)*1
    });
  });

  describe('updateStrategySum', () => {
    it('updates strategy sum and visit count', () => {
      const strategy = new Map<string, number>([
        ['6-4', 0.5],
        ['5-5', 0.5]
      ]);

      table.updateStrategySum('test-info-set', strategy, 1.0);

      const node = table.getNode('test-info-set');
      expect(node.visitCount).toBe(1);
      expect(node.strategySum.get('6-4')).toBe(0.5);
      expect(node.strategySum.get('5-5')).toBe(0.5);
    });

    it('weights by reach probability', () => {
      const strategy = new Map<string, number>([
        ['6-4', 0.5],
        ['5-5', 0.5]
      ]);

      table.updateStrategySum('test-info-set', strategy, 0.5);

      const node = table.getNode('test-info-set');
      expect(node.strategySum.get('6-4')).toBe(0.25); // 0.5 * 0.5
      expect(node.strategySum.get('5-5')).toBe(0.25);
    });
  });

  describe('serialize/deserialize', () => {
    it('round-trips correctly', () => {
      // Set up some data
      const node = table.getNode('test-info-set');
      node.regrets.set('6-4', 10);
      node.regrets.set('5-5', -5);
      node.strategySum.set('6-4', 100);
      node.strategySum.set('5-5', 200);
      node.visitCount = 50;

      const config: MCCFRConfig = {
        iterations: 1000,
        seed: 42
      };

      // Serialize
      const serialized = table.serialize(config, 1000, 5000);
      expect(serialized.version).toBe(1);
      expect(serialized.iterationsCompleted).toBe(1000);

      // Deserialize
      const restored = RegretTable.deserialize(serialized);
      const restoredNode = restored.getNode('test-info-set');

      expect(restoredNode.regrets.get('6-4')).toBe(10);
      expect(restoredNode.regrets.get('5-5')).toBe(-5);
      expect(restoredNode.strategySum.get('6-4')).toBe(100);
      expect(restoredNode.strategySum.get('5-5')).toBe(200);
      expect(restoredNode.visitCount).toBe(50);
    });
  });

  describe('size', () => {
    it('returns number of unique info sets', () => {
      expect(table.size).toBe(0);

      table.getNode('info-set-1');
      expect(table.size).toBe(1);

      table.getNode('info-set-2');
      expect(table.size).toBe(2);

      table.getNode('info-set-1'); // Same as before
      expect(table.size).toBe(2);
    });
  });

  describe('applyDiscount', () => {
    it('discounts regrets and strategy sums', () => {
      const node = table.getNode('test-info-set');
      node.regrets.set('6-4', 100);
      node.strategySum.set('6-4', 200);

      table.applyDiscount(0.5);

      expect(node.regrets.get('6-4')).toBe(50);
      expect(node.strategySum.get('6-4')).toBe(100);
    });
  });
});
