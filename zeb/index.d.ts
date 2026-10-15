// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: MIT

export * from './pkg/zeb.js';
export { default } from './pkg/zeb.js';

import type { Declaration, Forecast } from './pkg/zeb.js';

/** What `chooseBid` needs besides the forecast. */
export interface BidOptions {
  /** The lowest legal bid, 30..42 (42 is a mark bid); with `match`, any legal bid (a seat facing an 84 needs more). */
  need: number;
  /** The make probability a bid must reach; default 0.8. */
  cutoff?: number;
  /** The declarations the bidder may lean on; default every one forecast. */
  legalDeclarations?: readonly Declaration[];
  /**
   * Opt in to match-aware bidding: the hand-only call plus one more option, a
   * mark bid, taken when its match equity beats the hand-only call's. Absent,
   * `chooseBid` is the hand-only rule, as in 0.3.0.
   */
  match?: MatchState;
  /**
   * Opt in to the overcall, tuned for opponents who contest openings (needs
   * `match`, whose `holder` and `passesAfter` it reads; the score is read only
   * by the mark bid). When an opponent holds the high bid: the forecast is
   * calibrated (logit p~ = -0.0162 + 0.9237 logit p), and a hand-only pass
   * becomes the one-mark call, `need`..42, with the largest chance of our
   * side taking the hand's mark, if that beats passing (the lowest bid on a
   * tie; its `pmake` is the calibrated one). Prices use the contested field's
   * constants (refit on a table that overcalls back, #170). Against a table
   * that does not contest, the refit gives back about 3 points of game share
   * relative to 0.4.0's constants. Default false: 0.4.0's behavior exactly.
   */
  overcall?: boolean;
}

/** The game and the auction at the bidder's turn, for `BidOptions.match`. */
export interface MatchState {
  /** The bidder's side's marks. */
  us: number;
  /** The other side's marks. */
  them: number;
  /** Marks that win the game; default 7. */
  marksToWin?: number;
  /** True when the bidder's side shakes first on the next hand (the dealer's side does not). */
  weShakeNext: boolean;
  /** Who holds the high bid now: nobody, the bidder's partner, or an opponent. */
  holder: 'none' | 'partner' | 'opp';
  /** The seats still to speak after the bidder, 0..3. */
  passesAfter: number;
  /** The highest legal bid at this turn; a mark bid is available only at 84 or more. */
  maxBid: number;
  /** True in a points-to-win game: the match equity is the marks game's, so there is no mark option. */
  pointsToWin?: boolean;
}

/** `matchPrice`'s inputs: the match state, and the two calls being priced. */
export interface MatchPriceInput {
  us: number;
  them: number;
  marksToWin?: number;
  weShakeNext: boolean;
  holder: 'none' | 'partner' | 'opp';
  /** The seats still to speak after the bidder, 0..3 (`MatchState.passesAfter`). */
  after: number;
  /** The hand-only call: a bid 30..42, or 0 to pass. */
  c0: number;
  /** The make probability of `c0`. */
  pC0: number;
  /** The mark bid: 84, or one mark over an 84. */
  markCall: number;
  /** The forecast's probability of all 42 points (the best declaration's). */
  p42: number;
  /** Price with the contested field's constants (`BidOptions.overcall`'s); default false, 0.4.0's. */
  contested?: boolean;
}

/** The chance our side wins the game after each call, and the call with the larger one. */
export interface MatchPrice {
  eC0: number;
  eMark: number;
  /** `markCall` if `eMark > eC0`, else `c0`: a tie keeps the hand-only call. */
  call: number;
}

/** Price the hand-only call against the mark bid: the numbers behind `chooseBid` with `match`. */
export function matchPrice(input: MatchPriceInput): MatchPrice;

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
 *
 * With `options.match` it also weighs a mark bid (`match.maxBid`: 84, or one
 * mark over an 84) at the declaration most likely to take all 42 points, and
 * returns it as `{ bid: maxBid, decl, pmake: p42 }` when it beats the
 * hand-only call in match equity. Behind a partner's bid (`holder: 'partner'`)
 * the hand-only call is a pass.
 *
 * With `options.overcall` (and `match`), an opponent's bid also calibrates the
 * forecast, and a hand-only pass over it becomes the one-mark overcall that
 * beats passing, when there is one.
 */
export function chooseBid(forecast: readonly Forecast[], options: BidOptions): BidChoice | null;
