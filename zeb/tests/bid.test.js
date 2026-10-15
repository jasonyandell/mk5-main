// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: MIT

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

test('match: mark bid, ties, and the options that remove it', () => {
  const sure = row(6, () => 0.95);
  const match = { us: 0, them: 6, weShakeNext: false, holder: 'none', passesAfter: 0, maxBid: 84 };
  // Down 6-0 a set loses the game at 42 or 84 alike, and a make at 84 is two marks: all upside.
  assert.deepEqual(chooseBid([sure], { need: 30, match }), { bid: 84, decl: 6, pmake: 0.95 });
  assert.deepEqual(chooseBid([sure], { need: 30, match: { ...match, maxBid: 42 } }), chooseBid([sure], { need: 30 }));
  assert.deepEqual(chooseBid([sure], { need: 30, match: { ...match, pointsToWin: true } }), chooseBid([sure], { need: 30 }));
  // Up 6-0 the 42 dominates (the book's rule): the hand-only call stands.
  assert.equal(chooseBid([sure], { need: 30, match: { ...match, us: 6, them: 0 } }).bid, 42);
  // Behind the partner the hand-only call is a pass; facing an 84 there is no point bid.
  assert.equal(chooseBid([weak], { need: 30, match: { ...match, holder: 'partner' } }), null);
  assert.equal(chooseBid([weak], { need: 126, match: { ...match, us: 6, them: 0, holder: 'opp', maxBid: 126 } }), null);
  assert.throws(() => chooseBid([sure], { need: 30, match: { ...match, holder: 'me' } }), RangeError);
  assert.throws(() => chooseBid([sure], { need: 30, match: { ...match, weShakeNext: 0 } }), TypeError);
  assert.throws(() => chooseBid([sure], { need: 43 }), RangeError);
});

test('overcall: opt-in, needs match, acts only over an opponent', () => {
  const mid = row(6, (b) => (b <= 33 ? 0.7 : 0.2)); // under the 0.8 rule at every bid
  const match = { us: 3, them: 3, weShakeNext: true, holder: 'opp', passesAfter: 1, maxBid: 31 };
  assert.equal(chooseBid([mid], { need: 31, match }), null);
  const over = chooseBid([mid], { need: 31, match, overcall: true });
  assert.equal(over.bid, 31);
  assert.equal(over.decl, 6);
  assert.ok(over.pmake > 0.6 && over.pmake < 0.7, 'the forecast is calibrated, a little down');
  // Not behind a partner, not with nobody holding, not without the option.
  for (const holder of ['partner', 'none']) {
    assert.equal(chooseBid([mid], { need: 31, match: { ...match, holder }, overcall: true }), null);
  }
  assert.throws(() => chooseBid([mid], { need: 31, overcall: true }), TypeError);
  assert.throws(() => chooseBid([mid], { need: 31, match, overcall: 1 }), TypeError);
  // A sure hand still bids 42 over an opponent, with the calibrated pmake.
  const sure = row(6, () => 0.95);
  const bid = chooseBid([sure], { need: 31, match, overcall: true });
  assert.equal(bid.bid, 42);
  assert.ok(bid.pmake < 0.95 && bid.pmake > 0.9);
});
