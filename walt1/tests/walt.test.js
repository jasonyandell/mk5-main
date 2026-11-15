// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { initSync, play, ask, requestText, version } from '../index.js';

const here = (p) => new URL(p, import.meta.url);
initSync({ module: readFileSync(here('../pkg/walt_bg.wasm')) });

const fixtures = readFileSync(here('./fixtures/hand0.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
const opening = { decl: 6, bidder: 0, bid: 35, seat: 0, hand: [1, 6, 7, 8, 18, 19, 26], seed: 1000 };

test('the version is the package version', () => {
  assert.equal(version(), JSON.parse(readFileSync(here('../package.json'), 'utf8')).version);
});

test('the wasm answers every fixture request exactly as the native binary did', () => {
  assert.equal(fixtures.length, 28);
  for (const { req, res } of fixtures) assert.deepEqual(ask(req), JSON.parse(res));
});

test('play: an opening lead', () => {
  const move = play(opening);
  assert.equal(move.forced, false);
  assert.deepEqual(move.q.map(([t]) => t), [1, 6, 7, 8, 18, 19, 26]);
  assert.ok(move.q.every(([, p]) => p >= 0 && p <= 1));
  // The bidding side plays a tile with the largest estimate.
  assert.equal(Math.max(...move.q.map(([, p]) => p)), move.q.find(([t]) => t === move.tile)[1]);
  assert.equal(move.leader, 0);
  assert.deepEqual(move.points, [0, 0]);
  assert.deepEqual(play(opening), move, 'the same request, the same move');
});

test('play: the object and the text are one request', () => {
  assert.equal(requestText({ ...opening, plays: [], worlds: 40, minds: 8 }), fixtures[0].req);
  assert.equal(play({ ...opening, worlds: 40, minds: 8 }).tile, JSON.parse(fixtures[0].res).choice);
});

test('play: a forced tile is not planned', () => {
  const move = play({ decl: 6, bidder: 0, bid: 35, seat: 1, hand: [3, 4, 12, 16, 22, 23, 24], plays: [[0, 19]] });
  assert.deepEqual(move, { tile: 16, forced: true, q: [], leader: 0, points: [0, 0] });
});

test('play: every declaration, with defaults', () => {
  for (const decl of [0, 1, 2, 3, 4, 5, 6, 7, 9]) {
    const move = play({ decl, bidder: 2, bid: 30, seat: 2, hand: [0, 5, 9, 14, 20, 21, 27] });
    assert.equal(move.q.length, 7);
    assert.ok([0, 5, 9, 14, 20, 21, 27].includes(move.tile));
  }
});

test('play: a seed past 2^53 is a bigint', () => {
  assert.match(requestText({ ...opening, seed: 0xB7E151628AED2A6Bn }), /seed 13249961062380153451\n$/);
  assert.throws(() => play({ ...opening, seed: 2 ** 60 }), RangeError);
  assert.throws(() => play({ ...opening, seed: -1 }), RangeError);
});

test('play: malformed requests throw before the planner runs', () => {
  assert.throws(() => play({ ...opening, decl: 10 }), RangeError);
  assert.throws(() => play({ ...opening, bid: 29 }), RangeError);
  assert.throws(() => play({ ...opening, hand: [1, 6, 7, 8, 18, 19, 28] }), RangeError);
  assert.throws(() => play({ ...opening, plays: [[0]] }), TypeError);
  assert.throws(() => play({ ...opening, hand: '1 6 7' }), TypeError);
  assert.throws(() => play({ ...opening, worlds: 0 }), RangeError);
});

test('play: a record that breaks the rules is an Error naming the play', () => {
  assert.throws(() => play({ ...opening, decl: 8 }), /walt: decl must be/);
  assert.throws(() => play({ ...opening, plays: [[1, 3]] }), /not that seat's turn/);
  assert.throws(() => play({ ...opening, seat: 1 }), /seat 0 is to act/);
  assert.throws(() => play({ ...opening, hand: [1, 6, 7, 8, 18, 19, 19] }), /seven different tiles/);
  assert.throws(
    () => play({ decl: 6, bidder: 0, bid: 35, seat: 1, hand: [3, 4, 12, 16, 22, 23, 24], plays: [[0, 19], [1, 3]] }),
    /could have followed suit/,
  );
});
