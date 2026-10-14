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
export { default, initSync, forecast, play, analyze, version, netId } from './pkg/zeb.js';

/** The bid a forecast supports, or null to pass: the contract is in index.d.ts. */
export function chooseBid(forecast, { need, cutoff = 0.8, legalDeclarations } = {}) {
  if (!Array.isArray(forecast)) throw new TypeError('forecast must be the array forecast() returned');
  if (!Number.isInteger(need) || need < 30 || need > 42) {
    throw new RangeError(`need must be an integer bid in 30..42, not ${need}`);
  }
  if (typeof cutoff !== 'number' || !(cutoff > 0 && cutoff <= 1)) {
    throw new RangeError(`cutoff must be a probability in (0, 1], not ${cutoff}`);
  }
  if (legalDeclarations !== undefined && !Array.isArray(legalDeclarations)) {
    throw new TypeError('legalDeclarations must be an array of declarations');
  }
  const allowed = forecast.filter((f) => legalDeclarations === undefined || legalDeclarations.includes(f.decl));
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
