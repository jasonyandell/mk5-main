// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

import { readFileSync } from 'node:fs';
import { describe, test, expect } from 'vitest';
import type { GameState } from '../../../game/types';
import { createInitialState, getNextStates, getPlayerLeftOfDealer } from '../../../game';
import { createTestContext } from '../../helpers/executionContext';

/**
 * The engine's bids against the arena's and Silas's (silas/tests/data/auction-legal.txt,
 * written from arena/auction.py by arena/test_silas_fixtures.py; silas/tests/auction.rs
 * reads the same file). After every legal prefix of an auction, the engine offers
 * exactly the calls the fixture lists, five marks included: the rules set no
 * ceiling on mark bids beyond the one-mark steps, and three raises from 84 reach five.
 */

const FIXTURE = new URL('../../../../silas/tests/data/auction-legal.txt', import.meta.url);

/** Each line: the prefix of calls and the calls legal after it, as the table says them. */
function fixture(): Array<{ prefix: string[]; legal: string[] }> {
  return readFileSync(FIXTURE, 'utf8')
    .split('\n')
    .filter((line) => line.length > 0 && !line.startsWith('#'))
    .map((line) => {
      const [prefix = '', legal = ''] = line.split(':');
      const words = (s: string) => s.trim().split(/\s+/).filter((w) => w !== '-' && w !== '');
      return { prefix: words(prefix), legal: words(legal) };
    });
}

/** A call as the table says it, from an engine transition id; null for any other action. */
function called(id: string): string | null {
  if (id === 'pass') return 'pass';
  const points = /^bid-(\d+)$/.exec(id);
  if (points) return points[1] ?? null;
  const marks = /^bid-(\d+)-marks$/.exec(id);
  if (marks) return String(42 * Number(marks[1]));
  return null;
}

describe('the engine bids as the arena and Silas do', () => {
  const ctx = createTestContext();

  function bidding(dealer: number): GameState {
    const state = createInitialState({ shuffleSeed: 12345 });
    state.phase = 'bidding';
    state.dealer = dealer;
    state.currentPlayer = getPlayerLeftOfDealer(dealer);
    state.bids = [];
    return state;
  }

  function after(state: GameState, call: string): void {
    const transition = getNextStates(state, ctx).find((t) => called(t.id) === call);
    expect(transition, `the engine offers ${call}`).toBeDefined();
    if (transition) Object.assign(state, transition.newState);
  }

  test('after every legal prefix, the legal calls are the fixture\'s', () => {
    const lines = fixture();
    expect(lines.length).toBe(834);
    for (const dealer of [3, 0]) {
      for (const { prefix, legal } of lines) {
        const state = bidding(dealer);
        prefix.forEach((call) => after(state, call));
        const offered = getNextStates(state, ctx)
          .map((t) => called(t.id))
          .filter((c): c is string => c !== null);
        expect([...offered].sort(), `after ${prefix.join(' ') || 'no calls'}`).toEqual([...legal].sort());
      }
    }
  });

  test('four passes leave no bid, only the redeal', () => {
    const state = bidding(3);
    ['pass', 'pass', 'pass', 'pass'].forEach((call) => after(state, call));
    const ids = getNextStates(state, ctx).map((t) => t.id);
    expect(ids).toEqual(['redeal']);
  });
});
