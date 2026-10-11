// chooseBid: a pure function of the forecast, no WebAssembly.
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { chooseBid } from '../index.js';

/** A forecast row whose tail is `p(target)`. */
const row = (decl, p) => ({ decl, mean: 0, pmake: Array.from({ length: 13 }, (_, i) => p(30 + i)) });

const sixes = row(6, (b) => (b <= 34 ? 0.9 : b <= 36 ? 0.82 : 0.4));
const fives = row(5, (b) => (b <= 36 ? 0.85 : 0.1));
const weak = row(0, () => 0.3);

test('the highest target some declaration clears, and the likeliest declaration there', () => {
  assert.deepEqual(chooseBid([sixes, fives, weak], { need: 30 }), { bid: 36, decl: 5, pmake: 0.85 });
  assert.deepEqual(chooseBid([sixes, weak], { need: 30 }), { bid: 36, decl: 6, pmake: 0.82 });
});

test('never below need; null to pass', () => {
  assert.deepEqual(chooseBid([sixes], { need: 36 }), { bid: 36, decl: 6, pmake: 0.82 });
  assert.equal(chooseBid([sixes], { need: 37 }), null);
  assert.equal(chooseBid([weak], { need: 30 }), null);
  assert.equal(chooseBid([], { need: 30 }), null);
});

test('the cutoff sets the temperament', () => {
  assert.deepEqual(chooseBid([sixes], { need: 30, cutoff: 0.85 }), { bid: 34, decl: 6, pmake: 0.9 });
  assert.deepEqual(chooseBid([weak], { need: 30, cutoff: 0.3 }), { bid: 42, decl: 0, pmake: 0.3 });
});

test('legal declarations narrow the choice', () => {
  assert.deepEqual(chooseBid([sixes, fives], { need: 30, legalDeclarations: [6] }), { bid: 36, decl: 6, pmake: 0.82 });
  assert.equal(chooseBid([sixes, fives], { need: 30, legalDeclarations: [] }), null);
});

test('a tie goes to the declaration listed first', () => {
  const twin = { ...fives, decl: 4 };
  assert.equal(chooseBid([fives, twin], { need: 30 }).decl, 5);
  assert.equal(chooseBid([twin, fives], { need: 30 }).decl, 4);
});

test('bad options are refused', () => {
  assert.throws(() => chooseBid([sixes], { need: 29 }), RangeError);
  assert.throws(() => chooseBid([sixes], { need: 43 }), RangeError);
  assert.throws(() => chooseBid([sixes], { need: 30.5 }), RangeError);
  assert.throws(() => chooseBid([sixes], {}), RangeError);
  assert.throws(() => chooseBid([sixes], { need: 30, cutoff: 0 }), RangeError);
  assert.throws(() => chooseBid([sixes], { need: 30, cutoff: 1.5 }), RangeError);
  assert.throws(() => chooseBid(sixes, { need: 30 }), TypeError);
  assert.throws(() => chooseBid([sixes], { need: 30, legalDeclarations: 6 }), TypeError);
});
