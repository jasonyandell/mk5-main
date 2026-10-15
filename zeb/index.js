// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: MIT

/**
 * @jasonyandell/zeb — Zeb, the Texas 42 scenario player, in WebAssembly.
 *
 * Load the module once — `await init()` in a browser or Web Worker (it
 * fetches zeb_bg.wasm beside the glue), `initSync({ module: bytes })` in
 * Node — then `forecast`, `play` and `analyze` answer synchronously. A call
 * takes tens of milliseconds to seconds, so run them off the UI thread.
 */
import { calibrate, checkMatch, chooseWithMatch, overcallBid } from './match.js';

export { default, initSync, forecast, play, analyze, version, netId } from './pkg/zeb.js';
export { matchPrice } from './match.js';

/** The bid a forecast supports, or null to pass: the contract is in index.d.ts. */
export function chooseBid(forecast, { need, cutoff = 0.8, legalDeclarations, match, overcall = false } = {}) {
  if (!Array.isArray(forecast)) throw new TypeError('forecast must be the array forecast() returned');
  // With `match`, `need` may exceed 42: a seat facing an 84 has no point bid, only the mark option.
  if (!Number.isInteger(need) || need < 30 || (match === undefined && need > 42)) {
    throw new RangeError(`need must be an integer bid in 30..42, not ${need}`);
  }
  if (typeof cutoff !== 'number' || !(cutoff > 0 && cutoff <= 1)) {
    throw new RangeError(`cutoff must be a probability in (0, 1], not ${cutoff}`);
  }
  if (legalDeclarations !== undefined && !Array.isArray(legalDeclarations)) {
    throw new TypeError('legalDeclarations must be an array of declarations');
  }
  if (typeof overcall !== 'boolean') throw new TypeError('overcall must be a boolean');
  if (overcall && match === undefined) throw new TypeError('overcall needs match (it reads match.holder and match.passesAfter)');
  if (match !== undefined) checkMatch(match);
  let allowed = forecast.filter((f) => legalDeclarations === undefined || legalDeclarations.includes(f.decl));
  if (match === undefined) return handOnly(allowed, need, cutoff);
  // `overcall`: over an opponent's bid the forecast is calibrated, and a pass becomes the overcall that beats it.
  const contest = overcall && match.holder === 'opp';
  if (contest) allowed = calibrate(allowed);
  // Behind the partner's bid the hand-only call is a pass (Plunge's rule), and above 42 there is no point bid.
  let hand = match.holder === 'partner' || need > 42 ? null : handOnly(allowed, need, cutoff);
  if (hand === null && contest && need <= 42) hand = overcallBid(allowed, need, match.passesAfter);
  return chooseWithMatch(allowed, hand, match, overcall);
}

function handOnly(allowed, need, cutoff) {
  for (let bid = 42; bid >= need; bid--) {
    let best = null;
    for (const f of allowed) {
      const pmake = f.pmake[bid - 30];
      if (pmake >= cutoff && (best === null || pmake > best.pmake)) best = { bid, decl: f.decl, pmake };
    }
    if (best !== null) return best;
  }
  return null;
}
