import { describe, it, expect } from 'vitest';
import type { Bid, GameState } from '../../../game/types';
import { createInitialState, getNextStates, getPlayerLeftOfDealer } from '../../../game';
import { highestNamableMarks } from '../../../game/core/rules';
import { createTestContext } from '../../helpers/executionContext';

/**
 * wiki/topics/rules-of-42.md § Bidding: two marks is the most an opening can name, and
 * every bid after it adds one mark. The rules set no ceiling of their own, so the engine
 * offers five marks after four (issue #142; the arena and Silas offer it too).
 */
describe('Feature: Mark bids have no ceiling beyond the one-mark steps', () => {
  const ctx = createTestContext();

  const marks = (value: number, player: number): Bid => ({ type: 'marks', value, player });

  function bidding(): GameState {
    const state = createInitialState({ shuffleSeed: 12345 });
    state.phase = 'bidding';
    state.dealer = 3;
    state.currentPlayer = getPlayerLeftOfDealer(3);
    state.bids = [];
    return state;
  }

  function offered(state: GameState): string[] {
    return getNextStates(state, ctx).map((t) => t.id).filter((id) => /^bid-\d+-marks$/.test(id));
  }

  function play(state: GameState, id: string): void {
    const transition = getNextStates(state, ctx).find((t) => t.id === id);
    expect(transition, `the engine offers ${id}`).toBeDefined();
    if (transition) Object.assign(state, transition.newState);
  }

  describe('highestNamableMarks', () => {
    it('is two as an opening, and after point bids', () => {
      expect(highestNamableMarks([])).toBe(2);
      expect(highestNamableMarks([{ type: 'points', value: 41, player: 0 }])).toBe(2);
    });

    it('is one more than the highest mark bid, from one mark up', () => {
      expect(highestNamableMarks([marks(1, 0)])).toBe(2);
      expect(highestNamableMarks([marks(2, 0)])).toBe(3);
      expect(highestNamableMarks([marks(2, 0), marks(3, 1), marks(4, 2)])).toBe(5);
      expect(highestNamableMarks([marks(6, 0)])).toBe(7);
    });
  });

  it('an opening names one or two marks', () => {
    expect(offered(bidding())).toEqual(['bid-1-marks', 'bid-2-marks']);
  });

  it('three raises from two marks reach five, the fourth bidder\'s call', () => {
    const state = bidding();
    play(state, 'bid-2-marks');
    expect(offered(state)).toEqual(['bid-3-marks']);
    play(state, 'bid-3-marks');
    expect(offered(state)).toEqual(['bid-4-marks']);
    play(state, 'bid-4-marks');
    expect(offered(state)).toEqual(['bid-5-marks']);
    play(state, 'bid-5-marks');
    expect(state.currentBid).toMatchObject({ type: 'marks', value: 5 });
  });

  it('offers only the next mark above any mark bid, however high', () => {
    for (const high of [5, 6, 9]) {
      const state = bidding();
      state.bids = [marks(high, 3)];
      state.currentBid = marks(high, 3);
      expect(offered(state)).toEqual([`bid-${high + 1}-marks`]);
    }
  });
});
