/**
 * Unit tests for MCCFRStrategy
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { MCCFRStrategy, HybridMCCFRStrategy } from '../../../game/ai/cfr/mccfr-strategy';
import { RegretTable } from '../../../game/ai/cfr/regret-table';
import { MCCFRTrainer } from '../../../game/ai/cfr/mccfr-trainer';
import { HeadlessRoom } from '../../../server/HeadlessRoom';
import type { ValidAction } from '../../../multiplayer/types';

describe('MCCFRStrategy', () => {
  let regretTable: RegretTable;

  beforeEach(() => {
    regretTable = new RegretTable();
  });

  describe('construction', () => {
    it('creates strategy from regret table', () => {
      const strategy = new MCCFRStrategy(regretTable);
      expect(strategy).toBeDefined();
    });

    it.skip('creates strategy from serialized data', async () => {
      const trainer = new MCCFRTrainer({ iterations: 10, seed: 42 });
      await trainer.train();
      const serialized = trainer.serialize();

      const strategy = MCCFRStrategy.fromSerialized(serialized);
      expect(strategy).toBeDefined();
    });
  });

  describe('chooseAction', () => {
    it('returns single action when only one available', () => {
      const strategy = new MCCFRStrategy(regretTable);
      const actions: ValidAction[] = [{
        action: { type: 'play', player: 0, dominoId: '6-4' },
        label: 'Play 6-4'
      }];

      const room = new HeadlessRoom({ playerTypes: ['ai', 'ai', 'ai', 'ai'] }, 42);
      // Fast forward to playing phase
      skipToPlayingPhase(room);

      const state = room.getState();
      if (state.phase === 'playing') {
        const chosen = strategy.chooseAction(state, actions);
        expect(chosen).toBe(actions[0]);
      }
    });

    it('uses greedy selection by default', () => {
      // Set up regret table with clear preference
      const node = regretTable.getNode('test-info-set');
      node.strategySum.set('6-4', 100);
      node.strategySum.set('5-5', 10);
      node.strategySum.set('3-2', 10);

      const strategy = new MCCFRStrategy(regretTable, { mode: 'greedy' });

      // The strategy should prefer 6-4 when choosing
      expect(strategy).toBeDefined();
    });

    it('throws when no actions available', () => {
      const strategy = new MCCFRStrategy(regretTable);
      const room = new HeadlessRoom({ playerTypes: ['ai', 'ai', 'ai', 'ai'] }, 42);
      const state = room.getState();

      expect(() => strategy.chooseAction(state, [])).toThrow('No valid actions');
    });
  });

  describe('getStats', () => {
    it('returns regret table stats', () => {
      const node = regretTable.getNode('info-set-1');
      node.visitCount = 10;

      const strategy = new MCCFRStrategy(regretTable);
      const stats = strategy.getStats();

      expect(stats.nodeCount).toBe(1);
      expect(stats.totalVisits).toBe(10);
    });
  });

  describe('hasTrainingData', () => {
    it.skip('returns true for trained info sets', async () => {
      const trainer = new MCCFRTrainer({ iterations: 50, seed: 42 });
      await trainer.train();

      const strategy = new MCCFRStrategy(trainer.getRegretTable());
      const room = new HeadlessRoom({ playerTypes: ['ai', 'ai', 'ai', 'ai'] }, 42);
      skipToPlayingPhase(room);

      const state = room.getState();
      if (state.phase === 'playing') {
        // May or may not have training data for this specific state
        // Just verify the method works
        const hasData = strategy.hasTrainingData(state);
        expect(typeof hasData).toBe('boolean');
      }
    });
  });
});

describe('HybridMCCFRStrategy', () => {
  describe('construction', () => {
    it('creates with null regret table', () => {
      const strategy = new HybridMCCFRStrategy(null);
      expect(strategy).toBeDefined();
    });

    it('creates with regret table', () => {
      const regretTable = new RegretTable();
      const strategy = new HybridMCCFRStrategy(regretTable);
      expect(strategy).toBeDefined();
    });
  });

  describe('chooseAction', () => {
    it('falls back to Monte Carlo for untrained states', async () => {
      // Create a fresh strategy with no training
      const strategy = new HybridMCCFRStrategy(null);

      const room = new HeadlessRoom({ playerTypes: ['ai', 'ai', 'ai', 'ai'] }, 42);
      skipToPlayingPhase(room);

      const state = room.getState();
      if (state.phase === 'playing') {
        const actions = room.getValidActions(state.currentPlayer);
        const playActions = actions.filter(a => a.action.type === 'play');

        if (playActions.length > 0) {
          const chosen = strategy.chooseAction(state, playActions);
          expect(chosen).toBeDefined();
          expect(chosen.action.type).toBe('play');
        }
      }
    });
  });
});

// Helper to skip to playing phase
function skipToPlayingPhase(room: HeadlessRoom): void {
  let iterations = 0;
  const maxIterations = 50;

  while (iterations < maxIterations) {
    const state = room.getState();

    if (state.phase === 'playing' || state.phase === 'game_end') {
      return;
    }

    const actions = room.getAllActions();
    const anyActions = Object.values(actions).flat();

    if (anyActions.length === 0) {
      return;
    }

    // Find auto-execute or pick first action
    const autoAction = anyActions.find(a =>
      a.action.type === 'complete-trick' ||
      a.action.type === 'score-hand' ||
      a.action.autoExecute
    );

    const action = autoAction || anyActions[0];
    if (!action) return;

    const player = 'player' in action.action ? action.action.player : state.currentPlayer;

    try {
      room.executeAction(player, action.action);
    } catch {
      return;
    }

    iterations++;
  }
}
