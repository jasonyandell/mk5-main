/**
 * Consensus layer - gates progress actions with human player acknowledgment.
 *
 * Derives acknowledgment state from actionHistory (pure function).
 * Without this layer, complete-trick/score-hand execute immediately.
 * With this layer, all HUMAN players must agree first (AI doesn't vote).
 *
 * Reads state.playerTypes to determine which players are human.
 */

import type { Layer } from './types';
import type { GameState, GameAction } from '../types';

export const consensusLayer: Layer = {
  name: 'consensus',

  getValidActions: (state: GameState, prev: GameAction[]): GameAction[] => {
    // Determine which players are human
    const humanPlayers = new Set(
      state.playerTypes
        .map((type, i) => type === 'human' ? i : -1)
        .filter(i => i >= 0)
    );

    // If no humans, no consensus needed - pass through
    if (humanPlayers.size === 0) return prev;

    // Gate complete-trick
    const hasCompleteTrick = prev.some(a => a.type === 'complete-trick');
    if (hasCompleteTrick) {
      const acks = countAcksSinceLastProgress(state.actionHistory, 'agree-trick', 'complete-trick');
      const humanAcks = new Set([...acks].filter(p => humanPlayers.has(p)));

      if (humanAcks.size < humanPlayers.size) {
        const filtered = prev.filter(a => a.type !== 'complete-trick');
        for (const p of humanPlayers) {
          if (!humanAcks.has(p)) {
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
      const humanAcks = new Set([...acks].filter(p => humanPlayers.has(p)));

      if (humanAcks.size < humanPlayers.size) {
        const filtered = prev.filter(a => a.type !== 'score-hand');
        for (const p of humanPlayers) {
          if (!humanAcks.has(p)) {
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
