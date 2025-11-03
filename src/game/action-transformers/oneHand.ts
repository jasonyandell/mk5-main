import type { ActionTransformerFactory } from './types';

/**
 * One-hand mode: Skip bidding with scripted actions, end after one hand.
 *
 * Mechanics:
 * - Inject scripted bidding/trump actions at start of bidding (with autoExecute: true)
 * - GameHost will auto-execute these immediately
 * - After scoring, end game instead of dealing new hand
 *
 * Implementation:
 * - Emits scripted sequence when bids array is empty
 * - Replaces score-hand action with auto-executed version that ends game
 * - All changes via action list transformation - no state mutation
 */
export const oneHandActionTransformer: ActionTransformerFactory = () => (base) => (state) => {
  const baseActions = base(state);

  // At start of bidding (no bids yet), inject scripted sequence
  if (state.phase === 'bidding') {
    switch (state.bids.length) {
      case 0:
        return [{
          type: 'pass' as const,
          player: 0,
          autoExecute: true,
          meta: { scriptId: 'one-hand-setup', step: 1 }
        }];
      case 1:
        return [{
          type: 'pass' as const,
          player: 1,
          autoExecute: true,
          meta: { scriptId: 'one-hand-setup', step: 2 }
        }];
      case 2:
        return [{
          type: 'pass' as const,
          player: 2,
          autoExecute: true,
          meta: { scriptId: 'one-hand-setup', step: 3 }
        }];
      case 3:
        return [{
          type: 'bid' as const,
          player: 3,
          bid: 'points' as const,
          value: 30,
          autoExecute: true,
          meta: { scriptId: 'one-hand-setup', step: 4 }
        }];
      default:
        break;
    }
  }

  // During trump selection, inject scripted trump choice
  if (state.phase === 'trump_selection') {
    return [
      {
        type: 'select-trump' as const,
        player: 3,
        trump: { type: 'suit' as const, suit: 4 }, // Fours as trump
        autoExecute: true,
        meta: { scriptId: 'one-hand-setup', step: 5 }
      }
    ];
  }

  // After scoring phase, prevent new hand - end game instead
  if (state.phase === 'scoring' && state.consensus.scoreHand.size === 4) {
    // Filter out the regular score-hand action and replace with auto-executed version
    return [
      ...baseActions.filter(a => a.type !== 'score-hand'),
      {
        type: 'score-hand' as const,
        autoExecute: true,
        meta: { scriptId: 'one-hand-end' }
      }
    ];
  }

  return baseActions;
};
