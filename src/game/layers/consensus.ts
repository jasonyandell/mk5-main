/**
 * Consensus layer - gates progress actions with player acknowledgment.
 *
 * Derives acknowledgment state from actionHistory (pure function).
 * Without this layer, complete-trick/score-hand execute immediately.
 * With this layer, all 4 players must agree first.
 */

import type { Layer } from './types';
import type { GameState, GameAction } from '../types';

export const consensusLayer: Layer = {
  name: 'consensus',

  getValidActions: (state: GameState, prev: GameAction[]): GameAction[] => {
    // Gate complete-trick
    const hasCompleteTrick = prev.some(a => a.type === 'complete-trick');
    if (hasCompleteTrick) {
      const acks = countAcksSinceLastProgress(state.actionHistory, 'agree-trick', 'complete-trick');
      if (acks.size < 4) {
        const filtered = prev.filter(a => a.type !== 'complete-trick');
        for (let p = 0; p < 4; p++) {
          if (!acks.has(p)) {
            filtered.push({ type: 'agree-trick', player: p });
          }
        }
        return filtered;
      }
    }

    // Gate score-hand
    const hasScoreHand = prev.some(a => a.type === 'score-hand');
    if (hasScoreHand) {
      const acks = countAcksSinceLastProgress(state.actionHistory, 'agree-score', 'score-hand');
      if (acks.size < 4) {
        const filtered = prev.filter(a => a.type !== 'score-hand');
        for (let p = 0; p < 4; p++) {
          if (!acks.has(p)) {
            filtered.push({ type: 'agree-score', player: p });
          }
        }
        return filtered;
      }
    }

    return prev;
  }
};

/**
 * Count player acknowledgments since the last progress action.
 * Pure function - scans history backwards.
 */
function countAcksSinceLastProgress(
  history: GameAction[],
  ackType: 'agree-trick' | 'agree-score',
  progressType: 'complete-trick' | 'score-hand'
): Set<number> {
  const acks = new Set<number>();
  for (let i = history.length - 1; i >= 0; i--) {
    const action = history[i];
    if (!action) continue;
    if (action.type === progressType) break;
    if (action.type === ackType && 'player' in action) {
      acks.add(action.player);
    }
  }
  return acks;
}
