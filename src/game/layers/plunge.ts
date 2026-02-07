/**
 * Plunge layer - Special bid requiring 4+ doubles.
 *
 * From docs/rules.md §8.A:
 * - Requires 4+ doubles in hand
 * - Bid value: Automatic based on current high bid (4+ marks, jumps over existing bids)
 * - Partner declares trump and leads
 * - Must win all 7 tricks (early termination if opponents win any trick)
 */

import { createDoublesBidLayer } from './doubles-bid-factory';

export const plungeLayer = createDoublesBidLayer({
  name: 'plunge',
  minDoubles: 4,
  minValue: 4
  // maxValue: undefined (no upper limit)
});
