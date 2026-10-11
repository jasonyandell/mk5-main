// The package through its WebAssembly: bit-identical to the Python referee
// on the fixtures, deterministic, and refusing bad requests in plain words.
// Run with `npm test` (which builds pkg/ first).
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { analyze, forecast, initSync, netId, play, version } from '../index.js';

const read = (path) => readFileSync(new URL(path, import.meta.url));
initSync({ module: read('../pkg/zeb_bg.wasm') });

const fixtures = JSON.parse(read('./fixtures/parity.json'));
const HAND = [27, 26, 25, 20, 19, 14, 9]; // 6-6 6-5 6-4 5-5 5-4 4-4 3-3

/** The fixture's request as a caller passes it: seeds past 2^53 as bigints. */
function request({ seed, ...rest }) {
  const big = BigInt(seed);
  return { ...rest, seed: big <= Number.MAX_SAFE_INTEGER ? Number(big) : big };
}

test('every position answers as the Python referee does, to the bit', () => {
  assert.equal(fixtures.net, netId(), 'the fixtures were made with another net');
  for (const [i, { request: r, referee }] of fixtures.positions.entries()) {
    const { tile, forced } = referee;
    const q = referee.q ?? [];
    assert.deepEqual(play(request(r)), { tile, forced, q }, `position ${i}: play`);
    assert.deepEqual(analyze(request(r)), { tile, forced, q, pmake: referee.pmake ?? [] }, `position ${i}: analyze`);
    if (referee.forecast) {
      const { hand, decl, worlds, seed } = r;
      const got = forecast(request({ hand, declarations: [decl], worlds, seed }));
      assert.deepEqual(got, [referee.forecast], `position ${i}: forecast`);
    }
  }
});

test('the fixtures cover the ground', () => {
  const cases = fixtures.positions;
  const decls = new Set(cases.map((c) => c.request.decl));
  for (const d of [0, 1, 2, 3, 4, 5, 6, 7, 9]) assert.ok(decls.has(d), `declaration ${d}`);
  assert.ok(cases.filter((c) => c.referee.forecast).length >= 9, 'an opening per declaration');
  assert.ok(cases.some((c) => c.referee.forced), 'a forced move');
  assert.ok(cases.some((c) => c.request.seat % 2 !== c.request.bidder % 2), 'a defender');
  assert.ok(cases.some((c) => BigInt(c.request.seed) > BigInt(Number.MAX_SAFE_INTEGER)), 'a bigint seed');
});

test('the same request gives the same answer; another seed, another sample', () => {
  const ask = { hand: HAND, declarations: [6, 9], worlds: 48, seed: 5 };
  assert.deepEqual(forecast(ask), forecast(ask));
  assert.notDeepEqual(forecast(ask), forecast({ ...ask, seed: 6 }));
  assert.deepEqual(forecast({ ...ask, seed: 5n }), forecast(ask), 'a number and a bigint seed agree');
  const pos = { decl: 6, bidder: 1, bid: 34, seat: 0, hand: HAND, plays: [[1, 0], [2, 1], [3, 2]] };
  assert.deepEqual(play(pos), play({ ...pos }));
  assert.deepEqual(analyze(pos), analyze({ ...pos, worlds: 160, seed: 0 }), 'the defaults are 160 worlds, seed 0');
});

test('a forecast defaults to the nine straight declarations', () => {
  const all = forecast({ hand: HAND, worlds: 8 });
  assert.deepEqual(all.map((f) => f.decl), [0, 1, 2, 3, 4, 5, 6, 7, 9]);
  for (const f of all) {
    assert.equal(f.pmake.length, 13);
    assert.ok(f.pmake.every((p, i) => p >= 0 && p <= 1 && (i === 0 || p <= f.pmake[i - 1])), 'a tail');
  }
  assert.deepEqual(forecast({ hand: HAND, declarations: [] }), []);
});

test('version and netId are the receipts', () => {
  assert.equal(version(), JSON.parse(read('../package.json')).version);
  const digest = createHash('sha256').update(read('../net/margin_d250k_h128x64.fnet')).digest('hex');
  assert.equal(netId(), digest);
});

test('bad requests are refused in plain words', () => {
  const pos = { decl: 6, bidder: 0, bid: 30, seat: 0, hand: HAND, plays: [] };
  const refused = [
    [() => forecast(null), /a request is an object/],
    [() => forecast([HAND]), /a request is an object/],
    [() => forecast({ hand: HAND, world: 8 }), /unknown field "world"/],
    [() => forecast({ get hand() { throw new Error('no'); } }), /reading hand threw/],
    [() => forecast({}), /hand is required/],
    [() => forecast({ hand: HAND.slice(1) }), /a hand is 7 tiles, not 6/],
    [() => forecast({ hand: [27, 27, 25, 20, 19, 14, 9] }), /tile 27 appears twice/],
    [() => forecast({ hand: [28, 26, 25, 20, 19, 14, 9] }), /hand\[0\] must be an integer from 0 to 27/],
    [() => forecast({ hand: [-1, 26, 25, 20, 19, 14, 9] }), /hand\[0\] must be an integer from 0 to 27/],
    [() => forecast({ hand: [1.5, 26, 25, 20, 19, 14, 9] }), /hand\[0\] must be an integer/],
    [() => forecast({ hand: 'sixes' }), /hand must be an array/],
    [() => forecast({ hand: HAND, declarations: [8] }), /declaration 8 .* nello/],
    [() => forecast({ hand: HAND, declarations: [10] }), /declarations\[0\] must be an integer from 0 to 9/],
    [() => forecast({ hand: HAND, declarations: [6, 6] }), /declaration 6 is listed twice/],
    [() => forecast({ hand: HAND, worlds: 0 }), /worlds must be an integer from 1 to 1024/],
    [() => forecast({ hand: HAND, worlds: 1025 }), /worlds must be an integer from 1 to 1024/],
    [() => forecast({ hand: HAND, worlds: 2.5 }), /worlds must be an integer/],
    [() => forecast({ hand: HAND, seed: -1 }), /seed must be a non-negative integer/],
    [() => forecast({ hand: HAND, seed: -1n }), /seed must be a non-negative integer/],
    [() => forecast({ hand: HAND, seed: 2n ** 64n }), /seed must be a non-negative integer/],
    [() => forecast({ hand: HAND, seed: 2 ** 60 }), /seed must be .* a number up to 2\^53/],
    [() => play({ ...pos, decl: undefined }), /decl is required/],
    [() => play({ ...pos, bid: 29 }), /bid must be an integer from 30 to 42/],
    [() => play({ ...pos, bidder: 4 }), /bidder must be an integer from 0 to 3/],
    [() => play({ ...pos, seat: 1 }), /it is seat 0's turn, not seat 1's/],
    [() => play({ ...pos, plays: [[0]] }), /plays\[0\] must be a \[seat, tile\] pair/],
    [() => play({ ...pos, plays: [[0, 0]] }), /play 0: seat 0 does not hold tile 0/],
    [() => play({ ...pos, plays: [[1, 0]] }), /play 0: it is seat 0's turn, not seat 1's/],
    [() => play({ ...pos, plays: [[0, 27], [1, 26]] }), /play 1: tile 26 was dealt to seat 0, not seat 1/],
    [() => play({ ...pos, plays: [[0, 27], [1, 27]] }), /play 1: tile 27 was already played/],
    [() => play({ ...pos, bidder: 3, plays: [[3, 21], [0, 9]] }), /play 1: seat 0 must follow suit/],
    [() => play({ ...pos, plays: [[0, 27], [1, 0], [2, 1], [3, 2]] }), /no deal of the unseen tiles fits/],
    [
      () => play({ ...pos, plays: [[0, 27], [1, 0], [2, 21], [3, 22], [0, 26], [1, 23]] }),
      /play 5: seat 1 already failed to follow this suit/,
    ],
  ];
  for (const [call, message] of refused) assert.throws(call, message);
  // A refusal leaves the module working.
  assert.equal(play(pos).forced, false);
});
