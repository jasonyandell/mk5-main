// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

/**
 * @jasonyandell/walt — Walt, the level-1 scenario player for Texas 42, in WebAssembly.
 *
 * Load the module once — `await init()` in a browser or Web Worker (it fetches
 * walt_bg.wasm beside the glue), `initSync({ module: bytes })` in Node — then
 * `play` answers synchronously. A call takes milliseconds to a few hundred
 * milliseconds, so run it off the UI thread.
 */
import { handle } from './pkg/walt.js';

export { default, initSync, version } from './pkg/walt.js';

const U64 = 1n << 64n;

function integer(name, value, lo, hi) {
  if (!Number.isInteger(value) || value < lo || value > hi) {
    throw new RangeError(`${name} must be an integer in ${lo}..${hi}, not ${value}`);
  }
  return value;
}

function tiles(name, value) {
  if (!Array.isArray(value)) throw new TypeError(`${name} must be an array of tile ids`);
  return value.map((t) => integer(`${name} tile`, t, 0, 27));
}

/** The request as `walt1 play` text (WALT.md §6): what `ask` takes. */
export function requestText({ decl, bidder, bid, seat, hand, plays = [], worlds, minds, seed } = {}) {
  const lines = [
    'walt1 play',
    `decl ${integer('decl', decl, 0, 9)}`,
    `bid ${integer('bid', bid, 30, 42)}`,
    `seat ${integer('seat', seat, 0, 3)}`,
    `bidder ${integer('bidder', bidder, 0, 3)}`,
    `hand ${tiles('hand', hand).join(' ')}`,
  ];
  if (!Array.isArray(plays)) throw new TypeError('plays must be an array of [seat, tile]');
  const flat = plays.map((p) => {
    if (!Array.isArray(p) || p.length !== 2) throw new TypeError('each play must be [seat, tile]');
    return `${integer('play seat', p[0], 0, 3)} ${integer('play tile', p[1], 0, 27)}`;
  });
  lines.push(`plays ${flat.join(' ')}`);
  if (worlds !== undefined) lines.push(`n ${integer('worlds', worlds, 1, 4096)}`);
  if (minds !== undefined) lines.push(`n0 ${integer('minds', minds, 1, 4096)}`);
  if (seed !== undefined) {
    const ok = typeof seed === 'bigint' || Number.isSafeInteger(seed);
    if (!ok || BigInt(seed) < 0n || BigInt(seed) >= U64) {
      throw new RangeError('seed must be a non-negative integer below 2^64 (a bigint past 2^53)');
    }
    lines.push(`seed ${BigInt(seed)}`);
  }
  return lines.join('\n') + '\n';
}

/** One `walt1 play` request as text in, the response line parsed; an error line throws. */
export function ask(text) {
  const out = JSON.parse(handle(text));
  if (out.error !== undefined) throw new Error(`walt: ${out.error}`);
  return out;
}

/** The seat's move: the contract is in index.d.ts. */
export function play(request) {
  const out = ask(requestText(request));
  return {
    tile: out.choice,
    forced: out.forced,
    q: out.opts.map(([tile, bp]) => [tile, bp / 10000]),
    leader: out.leader,
    points: out.points,
  };
}
