/**
 * Test to verify that gameEngine uses threaded rules for validation.
 */

import { describe, it, expect } from 'vitest';
import { getValidActions } from '../../../game/core/gameEngine';
import { composeRules } from '../../../game/rulesets/compose';
import type { GameState, Bid, Domino } from '../../../game/types';
import type { GameRuleSet } from '../../../game/rulesets/types';
import { createInitialState } from '../../../game/core/state';
import { BID_TYPES } from '../../../game/constants';
import { baseRuleSet } from '../../../game/rulesets/base';
import { createHandWithDoubles } from '../../helpers/gameTestHelper';

describe('Threaded Rules in GameEngine', () => {
  it('should use composed rules for bid validation in getBiddingActions', () => {
    // Create a restrictive ruleset that blocks all NELLO bids through rules
    const restrictiveLayer: GameRuleSet = {
      name: 'restrictive',
      rules: {
        isValidBid: (_state: GameState, bid: Bid, _playerHand: unknown, prev: boolean) => {
          // Block all NELLO bids
          if (bid.type === BID_TYPES.NELLO) {
            return false;
          }
          return prev;
        }
      }
    };

    // Create a state in bidding phase with opening bid position
    const state = createInitialState();
    state.phase = 'bidding';
    state.currentPlayer = 0;
    state.bids = [];
    state.players[0]!.hand = createHandWithDoubles(4);

    // With base rules only: NELLO bids should not be generated
    // (base ruleSet's isValidBid returns false for NELLO)
    const baseOnlyRules = composeRules([baseRuleSet]);
    const actionsBaseOnly = getValidActions(state, undefined, baseOnlyRules);
    const nelloBidsBaseOnly = actionsBaseOnly.filter(
      a => a.type === 'bid' && a.bid === BID_TYPES.NELLO
    );
    expect(nelloBidsBaseOnly.length).toBe(0); // Base ruleset doesn't validate NELLO

    // With permissive rules (no validation): NELLO bids should be generated
    // (permissive ruleset allows all bids that base ruleset allows, plus NELLO)
    const permissiveLayer: GameRuleSet = {
      name: 'permissive',
      rules: {
        isValidBid: (_state: GameState, bid: Bid, _playerHand: unknown, prev: boolean) => {
          // Allow NELLO bids
          if (bid.type === BID_TYPES.NELLO && bid.value !== undefined && bid.value >= 1) {
            return true;
          }
          return prev;
        }
      }
    };
    const permissiveRules = composeRules([baseRuleSet, permissiveLayer]);
    const actionsPermissive = getValidActions(state, undefined, permissiveRules);
    const nelloBidsPermissive = actionsPermissive.filter(
      a => a.type === 'bid' && a.bid === BID_TYPES.NELLO
    );
    expect(nelloBidsPermissive.length).toBeGreaterThan(0); // Permissive ruleset allows NELLO

    // With restrictive ruleSet: rules should filter out NELLO bids during validation
    const restrictiveRules = composeRules([baseRuleSet, permissiveLayer, restrictiveLayer]);
    const actionsWithRestriction = getValidActions(state, undefined, restrictiveRules);
    const nelloBidsWithRestriction = actionsWithRestriction.filter(
      a => a.type === 'bid' && a.bid === BID_TYPES.NELLO
    );
    expect(nelloBidsWithRestriction.length).toBe(0); // Restrictive ruleset blocks NELLO through rules
  });

  it('should use composed rules for play validation in getPlayingActions', () => {
    // Create a test ruleset that restricts plays to only the first domino
    const restrictiveLayer: GameRuleSet = {
      name: 'restrictive-plays',
      rules: {
        getValidPlays: (_state: GameState, _playerId: number, prev: Domino[]): Domino[] => {
          // Return only the first domino from prev result
          if (prev.length > 0) {
            const first = prev[0];
            return first ? [first] : [];
          }
          return [];
        }
      }
    };

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

    // Without ruleSet, all dominoes should be playable (first play of trick)
    const actionsWithoutLayer = getValidActions(state);
    expect(actionsWithoutLayer.length).toBe(3);

    // With restrictive ruleSet, should only return first domino
    const rules = composeRules([restrictiveLayer]);
    const actionsWithLayer = getValidActions(state, [restrictiveLayer], rules);

    expect(actionsWithLayer.length).toBe(1);
    expect(actionsWithLayer[0]!.type).toBe('play');
    if (actionsWithLayer[0]!.type === 'play') {
      expect(actionsWithLayer[0]!.dominoId).toBe('1');
    }
  });
});
