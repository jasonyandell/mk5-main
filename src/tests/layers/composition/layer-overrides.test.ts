/**
 * Tests for layer override behavior.
 *
 * Verifies that layers correctly override base rules when appropriate:
 * - Nello overrides for 3-player gameplay
 * - Plunge/Splash override trump selector (partner)
 * - Sevens overrides trick winner calculation
 * - Overrides only apply when trump/bid type matches
 */

import { describe, it, expect } from 'vitest';
import { composeRules, baseLayer, nelloLayer, plungeLayer, splashLayer, sevensLayer } from '../../../game/layers';
import type { GameState, Bid, Play, Domino } from '../../../game/types';
import { createInitialState } from '../../../game/core/state';

describe('Layer Override Behavior', () => {
  function createTestState(overrides: Partial<GameState> = {}): GameState {
    const base = createInitialState();
    return {
      ...base,
      ...overrides
    };
  }

  describe('Nello layer overrides', () => {
    const rules = composeRules([baseLayer, nelloLayer]);

    describe('getNextPlayer (3-player vs 4-player)', () => {
      it('should skip partner in nello', () => {
        const state = createTestState({
          trump: { type: 'nello' },
          winningBidder: 0
        });

        // Partner of player 0 is player 2
        // From player 1, should skip 2 and go to 3
        expect(rules.getNextPlayer(state, 1)).toBe(3);

        // From player 3, should go to 0 (no skip)
        expect(rules.getNextPlayer(state, 3)).toBe(0);
      });

      it('should not skip partner when not nello', () => {
        const state = createTestState({
          trump: { type: 'suit', suit: 1 },
          winningBidder: 0
        });

        // Normal 4-player progression
        expect(rules.getNextPlayer(state, 1)).toBe(2);
        expect(rules.getNextPlayer(state, 2)).toBe(3);
      });
    });

    describe('isTrickComplete (3 vs 4 plays)', () => {
      it('should complete after 3 plays in nello', () => {
        const state = createTestState({
          trump: { type: 'nello' },
          currentTrick: [
            { player: 0, domino: { id: '0-1', high: 0, low: 1, points: 0 } },
            { player: 1, domino: { id: '1-2', high: 1, low: 2, points: 0 } },
            { player: 3, domino: { id: '3-4', high: 3, low: 4, points: 0 } }
          ]
        });

        expect(rules.isTrickComplete(state)).toBe(true);
      });

      it('should complete after 4 plays when not nello', () => {
        const state3Plays = createTestState({
          trump: { type: 'suit', suit: 1 },
          currentTrick: [
            { player: 0, domino: { id: '0-1', high: 0, low: 1, points: 0 } },
            { player: 1, domino: { id: '1-2', high: 1, low: 2, points: 0 } },
            { player: 2, domino: { id: '2-3', high: 2, low: 3, points: 0 } }
          ]
        });
        expect(rules.isTrickComplete(state3Plays)).toBe(false);

        const state4Plays = createTestState({
          trump: { type: 'suit', suit: 1 },
          currentTrick: [
            { player: 0, domino: { id: '0-1', high: 0, low: 1, points: 0 } },
            { player: 1, domino: { id: '1-2', high: 1, low: 2, points: 0 } },
            { player: 2, domino: { id: '2-3', high: 2, low: 3, points: 0 } },
            { player: 3, domino: { id: '3-4', high: 3, low: 4, points: 0 } }
          ]
        });
        expect(rules.isTrickComplete(state4Plays)).toBe(true);
      });
    });

    describe('getLedSuit (doubles = suit 7)', () => {
      it('should treat doubles as suit 7 in nello', () => {
        const state = createTestState({ trump: { type: 'nello' } });
        const double: Domino = { id: '3-3', high: 3, low: 3, points: 0 };

        expect(rules.getLedSuit(state, double)).toBe(7);
      });

      it('should use higher pip for non-doubles in nello', () => {
        const state = createTestState({ trump: { type: 'nello' } });
        const domino: Domino = { id: '2-5', high: 5, low: 2, points: 0 };

        expect(rules.getLedSuit(state, domino)).toBe(5);
      });

      it('should not treat doubles as suit 7 when not nello', () => {
        const state = createTestState({ trump: { type: 'suit', suit: 1 } });
        const double: Domino = { id: '3-3', high: 3, low: 3, points: 0 };

        // Base behavior: higher pip (which for doubles is same as low)
        expect(rules.getLedSuit(state, double)).toBe(3);
      });
    });

    describe('checkHandOutcome (early termination)', () => {
      it('should end early if bidding team wins any trick', () => {
        const state = createTestState({
          trump: { type: 'nello' },
          winningBidder: 0,
          tricks: [
            {
              plays: [
                { player: 0, domino: { id: '0-1', high: 0, low: 1, points: 0 } },
                { player: 1, domino: { id: '1-2', high: 1, low: 2, points: 0 } },
                { player: 3, domino: { id: '3-4', high: 3, low: 4, points: 0 } }
              ],
              winner: 0, // Bidding team (team 0) won
              points: 0
            }
          ]
        });

        const outcome = rules.checkHandOutcome(state);
        expect(outcome.isDetermined).toBe(true);
        expect((outcome as { isDetermined: true; decidedAtTrick: number }).decidedAtTrick).toBe(1);
      });

      it('should continue if defending team wins tricks', () => {
        const state = createTestState({
          trump: { type: 'nello' },
          winningBidder: 0,
          tricks: [
            {
              plays: [
                { player: 0, domino: { id: '0-1', high: 0, low: 1, points: 0 } },
                { player: 1, domino: { id: '1-2', high: 1, low: 2, points: 0 } },
                { player: 3, domino: { id: '3-4', high: 3, low: 4, points: 0 } }
              ],
              winner: 1, // Defending team (team 1) won
              points: 0
            }
          ]
        });

        const outcome = rules.checkHandOutcome(state);
        expect(outcome.isDetermined).toBe(false); // Not determined yet
      });

      it('should not apply early termination when not nello', () => {
        const state = createTestState({
          trump: { type: 'suit', suit: 1 },
          winningBidder: 0,
          tricks: [
            {
              plays: [
                { player: 0, domino: { id: '0-1', high: 0, low: 1, points: 0 } }
              ],
              winner: 1,
              points: 0
            }
          ]
        });

        const outcome = rules.checkHandOutcome(state);
        expect(outcome.isDetermined).toBe(false); // Base: continue until all tricks
      });
    });
  });

  describe('Plunge/Splash override getTrumpSelector (partner vs bidder)', () => {
    const rules = composeRules([baseLayer, plungeLayer, splashLayer]);

    it('should have partner select trump for plunge', () => {
      const state = createTestState();
      const bid: Bid = { type: 'plunge', value: 4, player: 1 };

      // Partner of player 1 is player 3
      expect(rules.getTrumpSelector(state, bid)).toBe(3);
    });

    it('should have partner select trump for splash', () => {
      const state = createTestState();
      const bid: Bid = { type: 'splash', value: 2, player: 0 };

      // Partner of player 0 is player 2
      expect(rules.getTrumpSelector(state, bid)).toBe(2);
    });

    it('should have bidder select trump for regular bids', () => {
      const state = createTestState();
      const marksBid: Bid = { type: 'marks', value: 2, player: 1 };
      const pointsBid: Bid = { type: 'points', value: 30, player: 2 };

      expect(rules.getTrumpSelector(state, marksBid)).toBe(1);
      expect(rules.getTrumpSelector(state, pointsBid)).toBe(2);
    });
  });

  describe('Plunge/Splash/Sevens override checkHandOutcome (early termination)', () => {
    const rules = composeRules([baseLayer, plungeLayer, splashLayer, sevensLayer]);

    it('should end plunge early if opponents win any trick', () => {
      const state = createTestState({
        winningBidder: 0,
        currentBid: { type: 'plunge', value: 4, player: 0 },
        tricks: [
          {
            plays: [
              { player: 0, domino: { id: '0-1', high: 0, low: 1, points: 0 } },
              { player: 1, domino: { id: '1-2', high: 1, low: 2, points: 0 } }
            ],
            winner: 1, // Opponent won
            points: 5
          }
        ]
      });

      const outcome = rules.checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(true);
      expect((outcome as { isDetermined: true; reason: string }).reason).toContain('Defending team won');
    });

    it('should end splash early if opponents win any trick', () => {
      const state = createTestState({
        winningBidder: 2,
        currentBid: { type: 'splash', value: 3, player: 2 },
        tricks: [
          {
            plays: [
              { player: 0, domino: { id: '0-1', high: 0, low: 1, points: 0 } }
            ],
            winner: 1, // Opponent won
            points: 5
          }
        ]
      });

      const outcome = rules.checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(true);
    });

    it('should end sevens early if opponents win any trick', () => {
      const state = createTestState({
        trump: { type: 'sevens' },
        winningBidder: 0,
        tricks: [
          {
            plays: [
              { player: 0, domino: { id: '0-1', high: 0, low: 1, points: 0 } }
            ],
            winner: 1, // Opponent won
            points: 5
          }
        ]
      });

      const outcome = rules.checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(true);
    });

    it('should continue if bidding team wins tricks', () => {
      const state = createTestState({
        winningBidder: 0,
        currentBid: { type: 'plunge', value: 4, player: 0 },
        tricks: [
          {
            plays: [
              { player: 0, domino: { id: '0-1', high: 0, low: 1, points: 0 } }
            ],
            winner: 0, // Bidding team won
            points: 5
          },
          {
            plays: [
              { player: 1, domino: { id: '1-2', high: 1, low: 2, points: 0 } }
            ],
            winner: 2, // Partner won
            points: 5
          }
        ]
      });

      const outcome = rules.checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(false); // Not determined yet
    });
  });

  describe('Sevens overrides calculateTrickWinner (closest to 7)', () => {
    const rules = composeRules([baseLayer, sevensLayer]);

    it('should pick domino closest to 7 total pips', () => {
      const state = createTestState({ trump: { type: 'sevens' } });
      const trick: Play[] = [
        { player: 0, domino: { id: '0-0', high: 0, low: 0, points: 0 } }, // 0 (distance 7)
        { player: 1, domino: { id: '3-4', high: 3, low: 4, points: 0 } }, // 7 (distance 0) - winner
        { player: 2, domino: { id: '6-6', high: 6, low: 6, points: 0 } }  // 12 (distance 5)
      ];

      expect(rules.calculateTrickWinner(state, trick)).toBe(1);
    });

    it('should handle ties (first played wins)', () => {
      const state = createTestState({ trump: { type: 'sevens' } });
      const trick: Play[] = [
        { player: 0, domino: { id: '2-5', high: 2, low: 5, points: 0 } }, // 7 (distance 0) - winner
        { player: 1, domino: { id: '3-4', high: 3, low: 4, points: 0 } }, // 7 (distance 0)
        { player: 2, domino: { id: '1-6', high: 1, low: 6, points: 0 } }  // 7 (distance 0)
      ];

      expect(rules.calculateTrickWinner(state, trick)).toBe(0); // First wins tie
    });

    it('should not use sevens logic when not sevens trump', () => {
      const state = createTestState({
        trump: { type: 'suit', suit: 1 },
        currentSuit: 2
      });
      const trick: Play[] = [
        { player: 0, domino: { id: '2-3', high: 2, low: 3, points: 0 } }, // Led suit
        { player: 1, domino: { id: '3-4', high: 3, low: 4, points: 10 } }, // 7 total, but not sevens trump
        { player: 2, domino: { id: '1-2', high: 1, low: 2, points: 0 } }  // Trump
      ];

      // Should use base logic: trump wins
      expect(rules.calculateTrickWinner(state, trick)).toBe(2);
    });
  });

  describe('Overrides only apply when trump type matches', () => {
    it('should not apply nello overrides when trump is not nello', () => {
      const rules = composeRules([baseLayer, nelloLayer]);

      const state = createTestState({
        trump: { type: 'suit', suit: 1 },
        winningBidder: 0,
        currentTrick: [
          { player: 0, domino: { id: '0-1', high: 0, low: 1, points: 0 } },
          { player: 1, domino: { id: '1-2', high: 1, low: 2, points: 0 } },
          { player: 2, domino: { id: '2-3', high: 2, low: 3, points: 0 } }
        ]
      });

      // Should use base behavior (4 plays for complete)
      expect(rules.isTrickComplete(state)).toBe(false);

      // Should use base getNextPlayer (no skip)
      expect(rules.getNextPlayer(state, 1)).toBe(2);
    });

    it('should not apply sevens overrides when trump is not sevens', () => {
      const rules = composeRules([baseLayer, sevensLayer]);

      const state = createTestState({
        trump: { type: 'doubles' },
        currentSuit: 2
      });
      const trick: Play[] = [
        { player: 0, domino: { id: '2-3', high: 2, low: 3, points: 0 } },
        { player: 1, domino: { id: '3-4', high: 3, low: 4, points: 10 } }, // 7 total
        { player: 2, domino: { id: '2-2', high: 2, low: 2, points: 0 } }  // Double (trump)
      ];

      // Should use base logic: trump (double) wins, not closest to 7
      expect(rules.calculateTrickWinner(state, trick)).toBe(2);
    });
  });

  describe('Overrides only apply when bid type matches', () => {
    it('should not apply plunge overrides when bid is not plunge', () => {
      const rules = composeRules([baseLayer, plungeLayer]);

      const state = createTestState();
      const marksBid: Bid = { type: 'marks', value: 2, player: 1 };

      // Should use base behavior (bidder selects trump)
      expect(rules.getTrumpSelector(state, marksBid)).toBe(1);
    });

    it('should not apply splash overrides when bid is not splash', () => {
      const rules = composeRules([baseLayer, splashLayer]);

      const state = createTestState();
      const pointsBid: Bid = { type: 'points', value: 30, player: 2 };

      // Should use base behavior (bidder selects trump)
      expect(rules.getTrumpSelector(state, pointsBid)).toBe(2);
    });

    it('should not end early for regular bids', () => {
      const rules = composeRules([baseLayer, plungeLayer, splashLayer]);

      const state = createTestState({
        winningBidder: 0,
        currentBid: { type: 'marks', value: 2, player: 0 },
        tricks: [
          {
            plays: [
              { player: 0, domino: { id: '0-1', high: 0, low: 1, points: 0 } }
            ],
            winner: 1, // Opponent won
            points: 5
          }
        ]
      });

      // Should use base behavior (no early termination)
      const outcome = rules.checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(false);
    });
  });
});
