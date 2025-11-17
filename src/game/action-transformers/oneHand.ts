import type { ActionTransformerFactory } from './types';

/**
 * One-hand mode ActionTransformer: Automate bidding/trump and score-hand consensus.
 *
 * Responsibilities (ActionTransformer layer):
 * - Inject scripted bidding/trump actions at start (with autoExecute: true)
 * - Auto-execute score-hand action to skip manual consensus
 * - Script actions execute with system authority, bypassing session checks
 *
 * Phase transition is handled by oneHandRuleSet (RuleSet layer):
 * - See src/game/rulesets/oneHand.ts for terminal phase logic
 * - RuleSet returns 'one-hand-complete' phase instead of 'bidding'
 *
 * Separation of concerns:
 * - ActionTransformer = automation (what actions to inject/modify)
 * - RuleSet = execution logic (how game transitions between phases)
 *
 * Implementation:
 * - Emits scripted sequence when bids array is empty
 * - Replaces score-hand action with auto-executed version for smooth UX
 * - All changes via action list transformation - no state mutation
 */
export const oneHandActionTransformer: ActionTransformerFactory = () => (base) => (state) => {
  const baseActions = base(state);

  // At start of bidding (no bids yet), inject scripted sequence
  if (state.phase === 'bidding') {
    // Use currentPlayer to ensure we inject actions in correct turn order
    const currentPlayer = state.currentPlayer;

    switch (state.bids.length) {
      case 0:
        return [{
          type: 'pass' as const,
          player: currentPlayer,
          autoExecute: true,
          meta: { scriptId: 'one-hand-setup', step: 1, authority: 'system' as const }
        }];
      case 1:
        return [{
          type: 'pass' as const,
          player: currentPlayer,
          autoExecute: true,
          meta: { scriptId: 'one-hand-setup', step: 2, authority: 'system' as const }
        }];
      case 2:
        return [{
          type: 'pass' as const,
          player: currentPlayer,
          autoExecute: true,
          meta: { scriptId: 'one-hand-setup', step: 3, authority: 'system' as const }
        }];
      case 3:
        return [{
          type: 'bid' as const,
          player: currentPlayer,
          bid: 'points' as const,
          value: 30,
          autoExecute: true,
          meta: { scriptId: 'one-hand-setup', step: 4, authority: 'system' as const }
        }];
      default:
        break;
    }
  }

  // During trump selection, inject scripted trump choice
  if (state.phase === 'trump_selection') {
    // Winner of bid (stored in winningBidder) selects trump
    return [
      {
        type: 'select-trump' as const,
        player: state.winningBidder,
        trump: { type: 'suit' as const, suit: 4 }, // Fours as trump
        autoExecute: true,
        meta: { scriptId: 'one-hand-setup', step: 5, authority: 'system' as const }
      }
    ];
  }

  // After scoring phase, auto-execute score-hand for smooth UX
  // Phase transition to 'one-hand-complete' is handled by oneHandRuleSet
  if (state.phase === 'scoring' && state.consensus.scoreHand.size === 4) {
    // Replace score-hand with auto-executed version to skip manual button click
    return [
      ...baseActions.filter(a => a.type !== 'score-hand'),
      {
        type: 'score-hand' as const,
        autoExecute: true,
        meta: { scriptId: 'one-hand-scoring', authority: 'system' as const }
      }
    ];
  }

  return baseActions;
};
