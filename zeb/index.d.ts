export * from './pkg/zeb.js';
export { default } from './pkg/zeb.js';

import type { Declaration, Forecast } from './pkg/zeb.js';

/** What `chooseBid` needs besides the forecast. */
export interface BidOptions {
  /** The lowest legal bid, 30..42 (42 is a mark bid). */
  need: number;
  /** The make probability a bid must reach; default 0.8. */
  cutoff?: number;
  /** The declarations the bidder may lean on; default every one forecast. */
  legalDeclarations?: readonly Declaration[];
}

/** A bid and the declaration it leans on. */
export interface BidChoice {
  bid: number;
  decl: Declaration;
  /** The forecast probability of making `bid` under `decl`. */
  pmake: number;
}

/**
 * The bid a forecast supports: the highest target, at least `need`, that
 * some allowed declaration makes with probability at least `cutoff`, and
 * the declaration most likely to make it (the first listed, on a tie).
 * Null means pass. A plain function of the forecast; the auction's own
 * rules (a partner's pass, a forced 30) are the caller's.
 */
export function chooseBid(forecast: readonly Forecast[], options: BidOptions): BidChoice | null;
