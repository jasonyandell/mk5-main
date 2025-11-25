import { describe, it, expect } from 'vitest';
import { checkHandOutcome } from '../../game/core/handOutcome';
import { StateBuilder } from '../helpers';
import type { GameState, Domino, Trick, LedSuit } from '../../game/types';
import { BLANKS, SIXES, NO_BIDDER } from '../../game/types';

// Helper to create a playing phase state with common defaults
function createTestState(overrides?: Partial<GameState>): GameState {
  return StateBuilder
    .inPlayingPhase({ type: 'suit', suit: SIXES })
    .withWinningBid(0, { type: 'points', value: 35, player: 0 })
    .withSeed(1234)
    .with(overrides || {})
    .build();
}

function createDomino(high: number, low: number): Domino {
  return {
    high: Math.max(high, low),
    low: Math.min(high, low),
    id: `${Math.max(high, low)}-${Math.min(high, low)}`
  };
}

function createTrick(plays: Array<{player: number, domino: Domino}>, winner: number, points: number): Trick {
  return {
    plays: plays.map(p => ({ player: p.player, domino: p.domino })),
    winner,
    points,
    ledSuit: (plays[0]?.domino.high ?? BLANKS) as LedSuit
  };
}

describe('Hand Outcome Detection', () => {
  describe('Points Bids', () => {
    it('should detect when bidding team makes their bid', () => {
      const state = createTestState({
        currentBid: { type: 'points', value: 30, player: 0 },
        teamScores: [30, 12],
        tricks: [
          createTrick([
            {player: 0, domino: createDomino(5, 5)},
            {player: 1, domino: createDomino(5, 4)},
            {player: 2, domino: createDomino(5, 3)},
            {player: 3, domino: createDomino(5, 2)}
          ], 0, 10),
          createTrick([
            {player: 0, domino: createDomino(6, 4)},
            {player: 1, domino: createDomino(6, 3)},
            {player: 2, domino: createDomino(6, 2)},
            {player: 3, domino: createDomino(6, 1)}
          ], 0, 10),
          createTrick([
            {player: 0, domino: createDomino(4, 1)},
            {player: 1, domino: createDomino(4, 0)},
            {player: 2, domino: createDomino(3, 2)},
            {player: 3, domino: createDomino(3, 0)}
          ], 0, 10)
        ]
      });
      
      const outcome = checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(true);
      if (outcome.isDetermined) {
        expect(outcome.reason).toBe('Bidding team made their 30 bid');
        expect(outcome.decidedAtTrick).toBe(4);
      }
    });

    it('should detect when bidding team cannot possibly make their bid', () => {
      const state = createTestState({
        currentBid: { type: 'points', value: 40, player: 0 },
        teamScores: [15, 27],
        tricks: [
          createTrick([
            {player: 0, domino: createDomino(5, 5)},
            {player: 1, domino: createDomino(5, 4)},
            {player: 2, domino: createDomino(5, 3)},
            {player: 3, domino: createDomino(5, 2)}
          ], 1, 10),
          createTrick([
            {player: 1, domino: createDomino(6, 4)},
            {player: 2, domino: createDomino(6, 3)},
            {player: 3, domino: createDomino(6, 2)},
            {player: 0, domino: createDomino(6, 1)}
          ], 1, 10),
          createTrick([
            {player: 1, domino: createDomino(3, 2)},
            {player: 2, domino: createDomino(3, 1)},
            {player: 3, domino: createDomino(3, 0)},
            {player: 0, domino: createDomino(2, 1)}
          ], 1, 5),
          createTrick([
            {player: 1, domino: createDomino(4, 4)},
            {player: 2, domino: createDomino(4, 3)},
            {player: 3, domino: createDomino(4, 2)},
            {player: 0, domino: createDomino(4, 0)}
          ], 1, 0),
          createTrick([
            {player: 1, domino: createDomino(3, 3)},
            {player: 2, domino: createDomino(2, 2)},
            {player: 3, domino: createDomino(1, 1)},
            {player: 0, domino: createDomino(0, 0)}
          ], 1, 0)
        ]
      });
      
      const outcome = checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(true);
      if (outcome.isDetermined) {
        expect(outcome.reason).toContain('Bidding team cannot reach 40');
        expect(outcome.decidedAtTrick).toBe(6);
      }
    });

    it('should detect when defending team sets the bid', () => {
      const state = createTestState({
        currentBid: { type: 'points', value: 35, player: 0 },
        teamScores: [5, 8],
        tricks: [
          createTrick([
            {player: 0, domino: createDomino(5, 5)},
            {player: 1, domino: createDomino(5, 4)},
            {player: 2, domino: createDomino(5, 3)},
            {player: 3, domino: createDomino(5, 2)}
          ], 1, 10)
        ]
      });
      
      const outcome = checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(true);
      if (outcome.isDetermined) {
        expect(outcome.reason).toBe('Defending team set the 35 bid');
        expect(outcome.decidedAtTrick).toBe(2);
      }
    });
  });

  describe('Marks Bids', () => {
    it('should detect when defending team scores on marks bid', () => {
      const state = createTestState({
        currentBid: { type: 'marks', value: 1, player: 0 },
        teamScores: [40, 2],
        tricks: [
          createTrick([
            {player: 0, domino: createDomino(5, 5)},
            {player: 1, domino: createDomino(5, 4)},
            {player: 2, domino: createDomino(5, 3)},
            {player: 3, domino: createDomino(5, 2)}
          ], 1, 0)
        ]
      });
      
      const outcome = checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(true);
      if (outcome.isDetermined) {
        expect(outcome.reason).toBe('Defending team scored points on marks bid');
        expect(outcome.decidedAtTrick).toBe(2);
      }
    });

    it('should detect when defending team scores any points on marks bid', () => {
      const state = createTestState({
        currentBid: { type: 'marks', value: 2, player: 0 },
        teamScores: [25, 12],
        tricks: [
          createTrick([
            {player: 0, domino: createDomino(5, 5)},
            {player: 1, domino: createDomino(5, 4)},
            {player: 2, domino: createDomino(5, 3)},
            {player: 3, domino: createDomino(5, 2)}
          ], 1, 10),
          createTrick([
            {player: 1, domino: createDomino(6, 4)},
            {player: 2, domino: createDomino(6, 3)},
            {player: 3, domino: createDomino(6, 2)},
            {player: 0, domino: createDomino(6, 1)}
          ], 1, 10)
        ]
      });
      
      const outcome = checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(true);
      if (outcome.isDetermined) {
        expect(outcome.reason).toBe('Defending team scored points on marks bid');
        expect(outcome.decidedAtTrick).toBe(3);
      }
    });
  });

  // NOTE: Nello tests removed - nello is now a trump selection (not bid type)
  // and early termination logic is handled by nello layer's checkHandOutcome.
  // See src/tests/layers/unit/nello-layer.test.ts for nello-specific tests.

  // NOTE: Splash/Plunge handling has been moved to their respective layers
  // (splashLayer, plungeLayer). The core handOutcome function no longer handles
  // these special bid types directly - they are handled through the layer composition system.
  // See ADR-20251112-onehand-terminal-phase.md for details on this architectural change.

  describe('Edge Cases', () => {
    it('should not detect outcome during bidding phase', () => {
      const state = createTestState({
        phase: 'bidding',
        currentBid: { type: 'pass', player: NO_BIDDER },
        winningBidder: NO_BIDDER
      });
      
      const outcome = checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(false);
    });

    it('should return determined for scoring phase', () => {
      const state = createTestState({
        phase: 'scoring'
      });
      
      const outcome = checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(true);
      if (outcome.isDetermined) {
        expect(outcome.reason).toBe('Hand complete');
      }
    });

    it('should handle empty current trick correctly', () => {
      const state = createTestState({
        currentBid: { type: 'points', value: 30, player: 0 },
        teamScores: [25, 5],
        tricks: [
          createTrick([
            {player: 0, domino: createDomino(5, 5)},
            {player: 1, domino: createDomino(5, 4)},
            {player: 2, domino: createDomino(5, 3)},
            {player: 3, domino: createDomino(5, 2)}
          ], 0, 10),
          createTrick([
            {player: 0, domino: createDomino(6, 4)},
            {player: 1, domino: createDomino(6, 3)},
            {player: 2, domino: createDomino(6, 2)},
            {player: 3, domino: createDomino(6, 1)}
          ], 0, 10),
          createTrick([
            {player: 0, domino: createDomino(3, 2)},
            {player: 1, domino: createDomino(3, 1)},
            {player: 2, domino: createDomino(3, 0)},
            {player: 3, domino: createDomino(2, 1)}
          ], 0, 5)
        ],
        currentTrick: []
      });
      
      const outcome = checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(false); // Still need 5 more points
    });
  });
});