import type { ActionTransformerFactory } from './types';

/**
 * Tournament mode: Disable special contracts (nello, splash, plunge).
 * Only standard point and mark bids allowed.
 */
export const tournamentVariant: ActionTransformerFactory = () => (base) => (state) => {
  const actions = base(state);

  // Filter out special bid types during bidding phase
  if (state.phase !== 'bidding') {
    return actions;
  }

  return actions.filter(action =>
    action.type !== 'bid' ||
    !['nello', 'splash', 'plunge'].includes(action.bid)
  );
};
