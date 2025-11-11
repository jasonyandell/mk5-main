import { describe, it, expect } from 'vitest';
import { createInitialState } from '../../game/core/state';
import { getNextStates } from '../../game/core/state';
import { createTestContext } from '../helpers/executionContext';
import { NO_LEAD_SUIT, NO_BIDDER } from '../../game/types';
import type { StateTransition } from '../../game/types';

describe('Refactor Validation', () => {
  it('should have no nullable fields in initial state', () => {
    const state = createInitialState({ shuffleSeed: 12345 });

    // Core assertions - these should never be null after refactor
    expect(state.winningBidder).toBe(NO_BIDDER);
    expect(state.trump).toEqual({ type: 'not-selected' });
    expect(state.currentSuit).toBe(NO_LEAD_SUIT);
    // New assertion for currentBid - should be EMPTY_BID instead of null
    expect(state.currentBid).toEqual({ type: 'pass', player: NO_BIDDER });
  });

  it('should work with getNextStates using ExecutionContext pattern', () => {

    const ctx = createTestContext();
    const state = createInitialState({ shuffleSeed: 12345 });

    // Use ExecutionContext for getting transitions
    const transitions = getNextStates(state, ctx);

    expect(transitions.length).toBeGreaterThan(0);

    // Each transition should have the expected properties
    transitions.forEach((transition: StateTransition) => {
      expect(transition).toHaveProperty('id');
      expect(transition).toHaveProperty('label');
      expect(transition).toHaveProperty('newState');

      // New states should also follow the no-null rule
      expect(transition.newState.winningBidder).toBeGreaterThanOrEqual(NO_BIDDER);
      expect(transition.newState.trump).toBeDefined();
      expect(transition.newState.trump.type).toBeDefined();
      expect(transition.newState.currentSuit).toBeGreaterThanOrEqual(NO_LEAD_SUIT);
    });
  });
});
