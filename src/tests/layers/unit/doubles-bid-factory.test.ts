/**
 * Unit tests for doubles-bid-factory layers (plunge and splash).
 *
 * These layers share identical logic via createDoublesBidLayer factory:
 * - Plunge: 4+ doubles, 4+ marks, no cap
 * - Splash: 3+ doubles, 2-3 marks, capped at 3
 *
 * This file tests the ACTUAL overrides, not passthrough behavior.
 * Passthrough behavior is verified in compose-rules.test.ts.
 */

import { describe, it, expect } from 'vitest';
import { plungeLayer } from '../../../game/layers/plunge';
import { splashLayer } from '../../../game/layers/splash';
import { composeRules } from '../../../game/layers/compose';
import { StateBuilder, HandBuilder, createTestLayer, PLAYER_SENTINEL } from '../../helpers';
import { BID_TYPES } from '../../../game/constants';
import type { Bid, BidType } from '../../../game/types';

/**
 * Parameterized test suite for plunge and splash.
 * Tests only the behaviors that differ or are unique to these layers.
 */
describe.each([
  {
    name: 'plunge',
    layer: plungeLayer,
    minDoubles: 4,
    minValue: 4,
    maxValue: undefined
  },
  {
    name: 'splash',
    layer: splashLayer,
    minDoubles: 3,
    minValue: 2,
    maxValue: 3
  }
])('$name layer', ({ name, layer, minDoubles, minValue, maxValue }) => {

  describe('getValidActions (bid generation)', () => {
    it(`adds ${name} bid when player has ${minDoubles}+ doubles`, () => {
      const hand = HandBuilder.withDoubles(minDoubles);
      const state = StateBuilder.inBiddingPhase()
        .withCurrentPlayer(1)
        .withBids([])
        .withPlayerHand(1, hand)
        .build();

      const baseActions = [{ type: 'pass' as const, player: 1 }];
      const actions = layer.getValidActions?.(state, baseActions) ?? [];

      expect(actions).toHaveLength(2);
      const bidAction = actions.find(a => a.type === 'bid');
      expect(bidAction).toBeDefined();
      expect(bidAction?.type === 'bid' && bidAction.bid).toBe(name);
      expect(bidAction?.type === 'bid' && bidAction.value).toBe(minValue);
    });

    it(`does NOT add ${name} bid when player has ${minDoubles - 1} doubles`, () => {
      const hand = HandBuilder.withDoubles(minDoubles - 1);
      const state = StateBuilder.inBiddingPhase()
        .withCurrentPlayer(0)
        .withPlayerHand(0, hand)
        .build();

      const actions = layer.getValidActions?.(state, []) ?? [];
      expect(actions.filter(a => a.type === 'bid')).toHaveLength(0);
    });

    it('calculates bid value as max(minValue, highestMarksBid + 1)', () => {
      const hand = HandBuilder.withDoubles(minDoubles + 1);
      const state = StateBuilder.inBiddingPhase()
        .withCurrentPlayer(2)
        .withBids([
          { type: BID_TYPES.MARKS, value: minValue, player: 0 }
        ])
        .withPlayerHand(2, hand)
        .build();

      const actions = layer.getValidActions?.(state, []) ?? [];
      const bidAction = actions.find(a => a.type === 'bid');

      const expectedValue = maxValue
        ? Math.min(maxValue, minValue + 1)
        : minValue + 1;
      expect(bidAction?.type === 'bid' && bidAction.value).toBe(expectedValue);
    });

    if (maxValue !== undefined) {
      it(`caps bid value at ${maxValue}`, () => {
        const hand = HandBuilder.withDoubles(6);
        const state = StateBuilder.inBiddingPhase()
          .withCurrentPlayer(0)
          .withBids([{ type: BID_TYPES.MARKS, value: maxValue, player: 1 }])
          .withPlayerHand(0, hand)
          .build();

        const actions = layer.getValidActions?.(state, []) ?? [];
        const bidAction = actions.find(a => a.type === 'bid');

        expect(bidAction?.type === 'bid' && bidAction.value).toBe(maxValue);
      });
    }

    it('does NOT add bid when not in bidding phase', () => {
      const hand = HandBuilder.withDoubles(7);
      const state = StateBuilder.inBiddingPhase()
        .with({ phase: 'trump_selection' })
        .withPlayerHand(0, hand)
        .build();

      const actions = layer.getValidActions?.(state, []) ?? [];
      expect(actions.filter(a => a.type === 'bid')).toHaveLength(0);
    });
  });

  describe('getTrumpSelector', () => {
    it(`returns partner for ${name} bid`, () => {
      // Use TestLayer to verify the layer returns partner
      const testLayer = createTestLayer({
        getTrumpSelector: () => PLAYER_SENTINEL
      });
      const rules = composeRules([testLayer, layer]);

      const state = StateBuilder.inBiddingPhase().build();
      const bid: Bid = { type: name as BidType, value: minValue, player: 0 };

      expect(rules.getTrumpSelector(state, bid)).toBe(2); // Partner of 0
    });

    it.each([0, 1, 2, 3])('returns correct partner for bidder %i', (bidder) => {
      const testLayer = createTestLayer();
      const rules = composeRules([testLayer, layer]);

      const state = StateBuilder.inBiddingPhase().build();
      const bid: Bid = { type: name as BidType, value: minValue, player: bidder };
      const expectedPartner = (bidder + 2) % 4;

      expect(rules.getTrumpSelector(state, bid)).toBe(expectedPartner);
    });

    it(`passes through for non-${name} bids`, () => {
      const testLayer = createTestLayer({
        getTrumpSelector: () => PLAYER_SENTINEL
      });
      const rules = composeRules([testLayer, layer]);

      const state = StateBuilder.inBiddingPhase().build();
      const bid: Bid = { type: BID_TYPES.MARKS, value: 2, player: 1 };

      expect(rules.getTrumpSelector(state, bid)).toBe(PLAYER_SENTINEL);
    });
  });

  describe('checkHandOutcome (early termination)', () => {
    it('returns not determined when bidding team winning all tricks', () => {
      const state = StateBuilder.inBiddingPhase()
        .withWinningBid(0, { type: name as BidType, value: minValue, player: 0 })
        .with({
          players: [
            { id: 0, name: 'P0', teamId: 0, marks: 0, hand: [] },
            { id: 1, name: 'P1', teamId: 1, marks: 0, hand: [] },
            { id: 2, name: 'P2', teamId: 0, marks: 0, hand: [] },
            { id: 3, name: 'P3', teamId: 1, marks: 0, hand: [] }
          ],
          tricks: [
            {
              plays: [
                { player: 0, domino: { id: '1-0', high: 1, low: 0 } },
                { player: 1, domino: { id: '2-0', high: 2, low: 0 } },
                { player: 2, domino: { id: '3-0', high: 3, low: 0 } },
                { player: 3, domino: { id: '4-0', high: 4, low: 0 } }
              ],
              winner: 0, // Team 0 wins
              points: 10
            }
          ]
        })
        .build();

      const testLayer = createTestLayer({
        checkHandOutcome: () => ({ isDetermined: false })
      });
      const rules = composeRules([testLayer, layer]);
      const outcome = rules.checkHandOutcome(state);

      expect(outcome.isDetermined).toBe(false);
    });

    it('terminates early when opponents win any trick', () => {
      const state = StateBuilder.inBiddingPhase()
        .withWinningBid(0, { type: name as BidType, value: minValue, player: 0 })
        .with({
          players: [
            { id: 0, name: 'P0', teamId: 0, marks: 0, hand: [] },
            { id: 1, name: 'P1', teamId: 1, marks: 0, hand: [] },
            { id: 2, name: 'P2', teamId: 0, marks: 0, hand: [] },
            { id: 3, name: 'P3', teamId: 1, marks: 0, hand: [] }
          ],
          tricks: [
            {
              plays: [
                { player: 0, domino: { id: '1-0', high: 1, low: 0 } },
                { player: 1, domino: { id: '6-0', high: 6, low: 0 } },
                { player: 2, domino: { id: '2-0', high: 2, low: 0 } },
                { player: 3, domino: { id: '3-0', high: 3, low: 0 } }
              ],
              winner: 1, // Team 1 wins - contract fails!
              points: 0
            }
          ]
        })
        .build();

      const testLayer = createTestLayer({
        checkHandOutcome: () => ({ isDetermined: false })
      });
      const rules = composeRules([testLayer, layer]);
      const outcome = rules.checkHandOutcome(state);

      expect(outcome.isDetermined).toBe(true);
      expect(outcome.isDetermined && outcome.reason).toContain('Defending team won trick');
      expect(outcome.isDetermined && outcome.decidedAtTrick).toBe(1);
    });

    it(`passes through for non-${name} bids`, () => {
      const state = StateBuilder.inBiddingPhase()
        .withWinningBid(0, { type: BID_TYPES.MARKS, value: 2, player: 0 })
        .with({
          players: [
            { id: 0, name: 'P0', teamId: 0, marks: 0, hand: [] },
            { id: 1, name: 'P1', teamId: 1, marks: 0, hand: [] },
            { id: 2, name: 'P2', teamId: 0, marks: 0, hand: [] },
            { id: 3, name: 'P3', teamId: 1, marks: 0, hand: [] }
          ],
          tricks: [
            {
              plays: [
                { player: 0, domino: { id: '1-0', high: 1, low: 0 } },
                { player: 1, domino: { id: '6-0', high: 6, low: 0 } },
                { player: 2, domino: { id: '2-0', high: 2, low: 0 } },
                { player: 3, domino: { id: '3-0', high: 3, low: 0 } }
              ],
              winner: 1, // Opponents win - but it's not plunge/splash
              points: 0
            }
          ]
        })
        .build();

      const testLayer = createTestLayer({
        checkHandOutcome: () => ({ isDetermined: false })
      });
      const rules = composeRules([testLayer, layer]);
      const outcome = rules.checkHandOutcome(state);

      expect(outcome.isDetermined).toBe(false);
    });
  });

  describe('getBidComparisonValue', () => {
    it(`treats ${name} bid as marks (value * 42)`, () => {
      const testLayer = createTestLayer({
        getBidComparisonValue: () => 0
      });
      const rules = composeRules([testLayer, layer]);

      const bid: Bid = { type: name as BidType, value: minValue, player: 0 };
      expect(rules.getBidComparisonValue(bid)).toBe(minValue * 42);
    });

    it('passes through for other bid types', () => {
      const testLayer = createTestLayer({
        getBidComparisonValue: () => 999
      });
      const rules = composeRules([testLayer, layer]);

      const bid: Bid = { type: BID_TYPES.POINTS, value: 30, player: 0 };
      expect(rules.getBidComparisonValue(bid)).toBe(999);
    });
  });

  describe('isValidBid', () => {
    it(`validates ${name} bid with sufficient doubles`, () => {
      const hand = HandBuilder.withDoubles(minDoubles);
      const state = StateBuilder.inBiddingPhase()
        .withBids([])
        .build();

      const testLayer = createTestLayer({
        isValidBid: () => false
      });
      const rules = composeRules([testLayer, layer]);
      const bid: Bid = { type: name as BidType, value: minValue, player: 0 };

      expect(rules.isValidBid(state, bid, hand)).toBe(true);
    });

    it(`rejects ${name} bid with insufficient doubles`, () => {
      const hand = HandBuilder.withDoubles(minDoubles - 1);
      const state = StateBuilder.inBiddingPhase().build();

      const testLayer = createTestLayer({
        isValidBid: () => true
      });
      const rules = composeRules([testLayer, layer]);
      const bid: Bid = { type: name as BidType, value: minValue, player: 0 };

      expect(rules.isValidBid(state, bid, hand)).toBe(false);
    });

    it(`rejects ${name} bid with value below minimum`, () => {
      const hand = HandBuilder.withDoubles(minDoubles);
      const state = StateBuilder.inBiddingPhase().build();

      const testLayer = createTestLayer();
      const rules = composeRules([testLayer, layer]);
      const bid: Bid = { type: name as BidType, value: minValue - 1, player: 0 };

      expect(rules.isValidBid(state, bid, hand)).toBe(false);
    });

    if (maxValue !== undefined) {
      it(`rejects ${name} bid with value above maximum`, () => {
        const hand = HandBuilder.withDoubles(minDoubles);
        const state = StateBuilder.inBiddingPhase().build();

        const testLayer = createTestLayer();
        const rules = composeRules([testLayer, layer]);
        const bid: Bid = { type: name as BidType, value: maxValue + 1, player: 0 };

        expect(rules.isValidBid(state, bid, hand)).toBe(false);
      });
    }
  });

  describe('calculateScore', () => {
    it('awards marks to bidding team on success (all tricks won)', () => {
      const state = StateBuilder.inBiddingPhase()
        .withWinningBid(0, { type: name as BidType, value: minValue, player: 0 })
        .withTeamMarks(0, 0)
        .with({
          players: [
            { id: 0, name: 'P0', teamId: 0, marks: 0, hand: [] },
            { id: 1, name: 'P1', teamId: 1, marks: 0, hand: [] },
            { id: 2, name: 'P2', teamId: 0, marks: 0, hand: [] },
            { id: 3, name: 'P3', teamId: 1, marks: 0, hand: [] }
          ],
          tricks: Array.from({ length: 7 }, () => ({
            plays: [
              { player: 0, domino: { id: '1-0', high: 1, low: 0 } },
              { player: 1, domino: { id: '2-0', high: 2, low: 0 } },
              { player: 2, domino: { id: '3-0', high: 3, low: 0 } },
              { player: 3, domino: { id: '4-0', high: 4, low: 0 } }
            ],
            winner: 0, // Team 0 wins all
            points: 5
          }))
        })
        .build();

      const testLayer = createTestLayer({
        calculateScore: () => [0, 0]
      });
      const rules = composeRules([testLayer, layer]);
      const [team0, team1] = rules.calculateScore(state);

      expect(team0).toBe(minValue);
      expect(team1).toBe(0);
    });

    it('awards marks to opponents on failure (lost a trick)', () => {
      const state = StateBuilder.inBiddingPhase()
        .withWinningBid(0, { type: name as BidType, value: minValue, player: 0 })
        .withTeamMarks(0, 0)
        .with({
          players: [
            { id: 0, name: 'P0', teamId: 0, marks: 0, hand: [] },
            { id: 1, name: 'P1', teamId: 1, marks: 0, hand: [] },
            { id: 2, name: 'P2', teamId: 0, marks: 0, hand: [] },
            { id: 3, name: 'P3', teamId: 1, marks: 0, hand: [] }
          ],
          tricks: [
            {
              plays: [
                { player: 0, domino: { id: '1-0', high: 1, low: 0 } },
                { player: 1, domino: { id: '6-0', high: 6, low: 0 } },
                { player: 2, domino: { id: '2-0', high: 2, low: 0 } },
                { player: 3, domino: { id: '3-0', high: 3, low: 0 } }
              ],
              winner: 1, // Team 1 wins - contract fails!
              points: 0
            }
          ]
        })
        .build();

      const testLayer = createTestLayer({
        calculateScore: () => [0, 0]
      });
      const rules = composeRules([testLayer, layer]);
      const [team0, team1] = rules.calculateScore(state);

      expect(team0).toBe(0);
      expect(team1).toBe(minValue);
    });

    it(`passes through for non-${name} bids`, () => {
      const state = StateBuilder.inBiddingPhase()
        .withWinningBid(0, { type: BID_TYPES.MARKS, value: 2, player: 0 })
        .with({
          players: [
            { id: 0, name: 'P0', teamId: 0, marks: 0, hand: [] },
            { id: 1, name: 'P1', teamId: 1, marks: 0, hand: [] },
            { id: 2, name: 'P2', teamId: 0, marks: 0, hand: [] },
            { id: 3, name: 'P3', teamId: 1, marks: 0, hand: [] }
          ],
          tricks: []
        })
        .build();

      const testLayer = createTestLayer({
        calculateScore: () => [999, 999]
      });
      const rules = composeRules([testLayer, layer]);
      const [team0, team1] = rules.calculateScore(state);

      expect(team0).toBe(999);
      expect(team1).toBe(999);
    });
  });
});
