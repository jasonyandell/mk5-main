/**
 * Tournament Layer - Disables special contracts.
 *
 * Tournament mode restricts play to standard bids only:
 * - Standard point bids (30-42)
 * - Standard mark bids (1-4 marks)
 * - No special contracts: nello, splash, plunge, sevens
 *
 * This is a filtering Layer that:
 * - Removes special bid actions during bidding phase (splash, plunge)
 * - Removes special trump selections during trump_selection phase (nello, sevens)
 */

import type { Layer } from './types';

export const tournamentLayer: Layer = {
  name: 'tournament',

  getValidActions: (state, prev) => {
    // Filter during bidding phase: remove special bid types (splash, plunge)
    if (state.phase === 'bidding') {
      return prev.filter(action =>
        action.type !== 'bid' ||
        !['splash', 'plunge'].includes(action.bid)
      );
    }

    // Filter during trump selection: remove special trump types (nello, sevens)
    if (state.phase === 'trump_selection') {
      return prev.filter(action =>
        action.type !== 'select-trump' ||
        !['nello', 'sevens'].includes(action.trump?.type || '')
      );
    }

    return prev;
  }
};
