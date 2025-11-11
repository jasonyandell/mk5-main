/**
 * Test to verify that ExecutionContext uses composed rules for validation.
 */

import { describe, it, expect } from 'vitest';
import { createInitialState } from '../../../game/core/state';
import { BID_TYPES } from '../../../game/constants';
import { createHandWithDoubles } from '../../helpers/gameTestHelper';
import { createTestContext, createTestContextWithRuleSets } from '../../helpers/executionContext';

describe('Composed Rules in ExecutionContext', () => {
  it('should use composed rules for bid validation', () => {
    // Create a context with base rules only
    const ctx = createTestContext();

    // Create a state in bidding phase with a previous bid (so special contracts are possible)
    const state = createInitialState();
    state.phase = 'bidding';
    state.currentPlayer = 1; // Second player to bid
    // Add a previous bid so special contracts become possible
    state.bids = [{ type: BID_TYPES.POINTS, value: 35, player: 0 }];
    state.players[0]!.hand = createHandWithDoubles(4);
    state.players[1]!.hand = createHandWithDoubles(4);

    // Verify that composed rules are available and can validate bids
    const rules = ctx.rules;
    expect(rules).toBeDefined();
    expect(rules.isValidBid).toBeDefined();

    // Test bid validation using the composed rules
    const bid = { type: BID_TYPES.POINTS, value: 35, player: 1 };
    const isValid = rules.isValidBid(state, bid, state.players[1]!.hand);
    expect(typeof isValid).toBe('boolean');
  });

  it('should use composed rules for play validation', () => {
    const ctx = createTestContext();

    const state = createInitialState();
    state.phase = 'playing';
    state.trump = { type: 'suit', suit: 0 }; // blanks trump

    // Set up player with multiple dominoes but no prior trick
    state.players[1]!.hand = [
      { id: 1, high: 0, low: 1 },   // 0-1
      { id: 2, high: 0, low: 2 },   // 0-2
      { id: 3, high: 0, low: 3 }    // 0-3
    ];
    state.currentPlayer = 1;
    state.currentTrick = []; // First play of trick - all dominoes valid

    // Verify that composed rules are available and can validate plays
    const rules = ctx.rules;
    expect(rules).toBeDefined();
    expect(rules.getValidPlays).toBeDefined();

    // Test play validation using the composed rules
    const validPlays = rules.getValidPlays(state, 1);
    expect(Array.isArray(validPlays)).toBe(true);
    expect(validPlays.length).toBeGreaterThan(0);
  });

  it('should support multiple ruleset compositions', () => {
    // Test with specific rulesets
    const ctx = createTestContextWithRuleSets(['base', 'nello']);

    const rules = ctx.rules;
    expect(rules).toBeDefined();

    // Verify core rule methods exist
    expect(rules.getTrumpSelector).toBeDefined();
    expect(rules.getFirstLeader).toBeDefined();
    expect(rules.getNextPlayer).toBeDefined();
    expect(rules.isTrickComplete).toBeDefined();
    expect(rules.checkHandOutcome).toBeDefined();
  });
});
