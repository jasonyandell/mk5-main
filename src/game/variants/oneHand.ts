import type { VariantFactory } from './types';

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
export const oneHandVariant: VariantFactory = () => (base) => (state) => {
  const baseActions = base(state);

  // At start of bidding (no bids yet), inject scripted sequence
  if (state.phase === 'bidding' && state.bids.length === 0) {
    return [
      // Player 3 bids 30 points
      {
        type: 'bid' as const,
        player: 3,
        bid: 'points' as const,
        value: 30,
        autoExecute: true,
        meta: { scriptId: 'one-hand-setup', step: 1 }
      },
      // Other players pass
      {
        type: 'pass' as const,
        player: 0,
        autoExecute: true,
        meta: { scriptId: 'one-hand-setup', step: 2 }
      },
      {
        type: 'pass' as const,
        player: 1,
        autoExecute: true,
        meta: { scriptId: 'one-hand-setup', step: 3 }
      },
      {
        type: 'pass' as const,
        player: 2,
        autoExecute: true,
        meta: { scriptId: 'one-hand-setup', step: 4 }
      }
    ];
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
