import { describe, it, expect } from 'vitest';
import { checkHandOutcome } from '../../game/core/handOutcome';
import { createInitialState } from '../../game/core/state';
import type { GameState, Domino, Trick } from '../../game/types';

function createTestState(overrides?: Partial<GameState>): GameState {
  const state = createInitialState({ shuffleSeed: 1234 });
  return {
    ...state,
    phase: 'playing',
    currentBid: { type: 'points', value: 35, player: 0 },
    winningBidder: 0,
    trump: { type: 'suit', suit: 6 },
    ...overrides
  };
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
    ledSuit: plays[0].domino.high
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
      expect(outcome.reason).toBe('Bidding team made their 30 bid');
      expect(outcome.decidedAtTrick).toBe(4);
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
      expect(outcome.reason).toContain('Bidding team cannot reach 40');
      expect(outcome.decidedAtTrick).toBe(6);
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
      expect(outcome.reason).toBe('Defending team set the 35 bid');
      expect(outcome.decidedAtTrick).toBe(2);
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
      expect(outcome.reason).toBe('Defending team scored points on marks bid');
      expect(outcome.decidedAtTrick).toBe(2);
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
      expect(outcome.reason).toBe('Defending team scored points on marks bid');
      expect(outcome.decidedAtTrick).toBe(3);
    });
  });

  describe('Nello', () => {
    it('should detect when bidding team wins a trick on nello', () => {
      const state = createTestState({
        currentBid: { type: 'nello', value: 1, player: 0 },
        teamScores: [1, 41],
        tricks: [
          createTrick([
            {player: 0, domino: createDomino(5, 5)},
            {player: 1, domino: createDomino(5, 4)},
            {player: 2, domino: createDomino(5, 3)},
            {player: 3, domino: createDomino(5, 2)}
          ], 0, 10)
        ]
      });
      
      const outcome = checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(true);
      expect(outcome.reason).toBe('Bidding team won a trick on nello');
      expect(outcome.decidedAtTrick).toBe(2);
    });

    it('should not end nello when bidding team has lost all tricks so far', () => {
      const state = createTestState({
        currentBid: { type: 'nello', value: 1, player: 0 },
        teamScores: [0, 25],
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
          ], 1, 5)
        ]
      });
      
      const outcome = checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(false);
    });
  });

  describe('Splash/Plunge', () => {
    it('should detect when defending team wins a trick on splash', () => {
      const state = createTestState({
        currentBid: { type: 'splash', value: 3, player: 0 },
        teamScores: [41, 1],
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
      expect(outcome.reason).toBe('Defending team won a trick on splash');
      expect(outcome.decidedAtTrick).toBe(2);
    });

    it('should detect when defending team wins a trick on plunge', () => {
      const state = createTestState({
        currentBid: { type: 'plunge', value: 4, player: 0 },
        teamScores: [40, 2],
        tricks: [
          createTrick([
            {player: 0, domino: createDomino(5, 5)},
            {player: 1, domino: createDomino(5, 4)},
            {player: 2, domino: createDomino(5, 3)},
            {player: 3, domino: createDomino(5, 2)}
          ], 1, 0),
          createTrick([
            {player: 1, domino: createDomino(0, 0)},
            {player: 2, domino: createDomino(1, 0)},
            {player: 3, domino: createDomino(2, 0)},
            {player: 0, domino: createDomino(3, 0)}
          ], 3, 0)
        ]
      });
      
      const outcome = checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(true);
      expect(outcome.reason).toBe('Defending team won a trick on plunge');
      expect(outcome.decidedAtTrick).toBe(3);
    });
  });

  describe('Edge Cases', () => {
    it('should not detect outcome during bidding phase', () => {
      const state = createTestState({
        phase: 'bidding',
        currentBid: { type: 'pass', player: -1 },
        winningBidder: -1
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
      expect(outcome.reason).toBe('Hand complete');
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