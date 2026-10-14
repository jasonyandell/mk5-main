// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

/**
 * Every trump option names the trump selector (#97).
 *
 * The trump selector is `state.currentPlayer` once bidding completes:
 * analyzeBiddingCompletion sets it from rules.getTrumpSelector, which plunge and
 * splash turn to the bidder's partner. Every layer that offers a trump - base,
 * nello, sevens - must name that player, or the selector cannot pick the option
 * and the bidder sees one that executeTrumpSelection refuses.
 */

import { describe, it, expect } from 'vitest';
import { HeadlessRoom } from '../../../server/HeadlessRoom';
import { HandBuilder } from '../../helpers';
import { createInitialState, executeAction } from '../../../game';
import type { GameAction, GameState } from '../../../game/types';
import type { ExecutionContext } from '../../../game/types/execution';
import {
  baseLayer,
  nelloLayer,
  sevensLayer,
  consensusLayer,
  composeRules,
  composeGetValidActions,
  getPartner,
  type Layer
} from '../../../game/layers';
import { generateStructuralActions } from '../../../game/layers/base';

// P0 holds four doubles (plunge- and splash-eligible)
const hands = [
  HandBuilder.fromStrings(['6-6', '5-5', '4-4', '3-3', '6-5', '5-4', '4-3']),
  HandBuilder.fromStrings(['0-1', '0-2', '0-3', '1-2', '1-3', '2-3', '2-4']),
  HandBuilder.fromStrings(['6-4', '6-3', '6-2', '6-1', '5-3', '5-2', '5-1']),
  HandBuilder.fromStrings(['0-0', '1-1', '2-2', '0-4', '0-5', '0-6', '1-4'])
];

function trumpOptions(actions: GameAction[]): Extract<GameAction, { type: 'select-trump' }>[] {
  return actions.filter((a): a is Extract<GameAction, { type: 'select-trump' }> => a.type === 'select-trump');
}

describe.each([
  { contract: 'plunge', layers: ['plunge', 'nello', 'sevens'] },
  { contract: 'splash', layers: ['splash', 'nello', 'sevens'] }
])('$contract composed with nello and sevens', ({ contract, layers }) => {
  function roomAfterContractBid(): HeadlessRoom {
    const room = new HeadlessRoom({
      playerTypes: ['human', 'human', 'human', 'human'],
      layers,
      shuffleSeed: 4242,
      dealOverrides: { initialHands: hands }
    }, 4242);

    // Bidding opens left of the dealer; seats pass until P0, who bids the contract
    while (room.getState().currentPlayer !== 0) {
      room.executeAction(room.getState().currentPlayer, { type: 'pass', player: room.getState().currentPlayer });
    }
    const bid = room.getValidActions(0).find(a => a.action.type === 'bid' && a.action.bid === contract);
    expect(bid, `${contract} bid offered to P0`).toBeDefined();
    room.executeAction(0, bid!.action);
    while (room.getState().phase === 'bidding') {
      room.executeAction(room.getState().currentPlayer, { type: 'pass', player: room.getState().currentPlayer });
    }
    return room;
  }

  it('the partner selects, and every trump option names the partner', () => {
    const room = roomAfterContractBid();
    const state = room.getState();
    expect(state.phase).toBe('trump_selection');
    expect(state.winningBidder).toBe(0);
    expect(state.currentPlayer).toBe(getPartner(0));

    const unfiltered = room.getAllActions()['__unfiltered__']!.map(v => v.action);
    const options = trumpOptions(unfiltered);
    expect(options.length).toBeGreaterThan(0);
    expect(options.every(a => a.player === state.currentPlayer)).toBe(true);
    expect(trumpOptions(room.getValidActions(0).map(v => v.action))).toHaveLength(0);

    const pick = trumpOptions(room.getValidActions(2).map(v => v.action))[0]!;
    room.executeAction(2, pick);
    expect(room.getState().phase).toBe('playing');
  });

  it('offers no nello or sevens on a must-win-all contract (they ride a marks bid)', () => {
    const room = roomAfterContractBid();
    const types = trumpOptions(room.getValidActions(2).map(v => v.action)).map(a => a.trump.type);
    expect(types).not.toContain('nello');
    expect(types).not.toContain('sevens');
  });
});

/**
 * The real layers give a partner the trump only on plunge/splash bids, where
 * nello and sevens are not offered. This layer hands the partner the trump on a
 * marks bid, so nello and sevens must follow the selector, not the bidder.
 */
const partnerNamesTrumpOnMarks: Layer = {
  name: 'partner-names-trump-on-marks',
  rules: {
    getTrumpSelector: (_state, bid, prev) => bid.type === 'marks' ? getPartner(bid.player) : prev
  }
};

function contextWith(layers: Layer[]): ExecutionContext {
  const rules = composeRules(layers);
  return {
    layers,
    rules,
    getValidActions: composeGetValidActions(layers, (state: GameState) => generateStructuralActions(state, rules))
  };
}

describe.each(['nello', 'sevens'] as const)('%s when the partner names trump on a marks bid', (special) => {
  it('is offered to the partner, who can select it', () => {
    const ctx = contextWith([baseLayer, partnerNamesTrumpOnMarks, nelloLayer, sevensLayer, consensusLayer]);

    let state = createInitialState({ shuffleSeed: 7 });
    const bidder = state.currentPlayer;
    state = executeAction(state, { type: 'bid', player: bidder, bid: 'marks', value: 1 }, ctx.rules);
    while (state.phase === 'bidding') {
      state = executeAction(state, { type: 'pass', player: state.currentPlayer }, ctx.rules);
    }

    const partner = getPartner(bidder);
    expect(state.winningBidder).toBe(bidder);
    expect(state.currentPlayer).toBe(partner);

    const option = trumpOptions(ctx.getValidActions(state)).find(a => a.trump.type === special);
    expect(option).toEqual({ type: 'select-trump', player: partner, trump: { type: special } });

    const after = executeAction(state, option!, ctx.rules);
    expect(after.trump.type).toBe(special);
    expect(after.phase).toBe('playing');
  });
});
