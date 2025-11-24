/**
 * Splash Layer - Special 2-3 mark bid requiring 3+ doubles.
 *
 * From docs/rules.md §8.A:
 * - Requires 3+ doubles in hand
 * - Bid value: Automatic based on current high bid (2-3 marks, jumps over existing bids)
 * - Partner declares trump and leads
 * - Must win all 7 tricks (early termination if opponents win any trick)
 */

import { createDoublesBidLayer } from './doubles-bid-factory';

export const splashRuleSet = createDoublesBidLayer({
  name: 'splash',
  minDoubles: 3,
  minValue: 2,
  maxValue: 3
});
