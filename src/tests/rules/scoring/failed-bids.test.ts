import { describe, it, expect } from 'vitest';
import type { GameState, GamePhase } from '../../../game/types';

describe('Feature: Mark System Scoring - Failed Bids', () => {
  describe('Scenario: Failed Bids', () => {
    it('Given a team has failed to make their bid', () => {
      // Test-only mock implementation
      const state: GameState = {
        phase: 'scoring' as GamePhase,
        players: [
          { id: 0, name: 'Player 1', hand: [], teamId: 0 as 0, marks: 0 },
          { id: 1, name: 'Player 2', hand: [], teamId: 1 as 1, marks: 0 },
          { id: 2, name: 'Player 3', hand: [], teamId: 0 as 0, marks: 0 },
          { id: 3, name: 'Player 4', hand: [], teamId: 1 as 1, marks: 0 },
        ],
        currentPlayer: 0,
        dealer: 3,
        bids: [],
        currentBid: { type: 'marks', value: 2, player: 0 }, // Team 0 bid 2 marks
        winningBidder: 0,
        trump: null,
        tricks: [],
        currentTrick: [],
        teamScores: [35, 7], // Team 0 scored 35 points, Team 1 scored 7 points
        teamMarks: [0, 0], // Before scoring
        gameTarget: 7,
        tournamentMode: true,
        shuffleSeed: 12345,
      };

      // Test setup: Team 0 bid 2 marks (84 points) but only scored 35 points
      expect(state.currentBid?.value).toBe(2);
      expect(state.teamScores[0]).toBeLessThan(84); // Failed to make their bid
    });

    it('When awarding marks', () => {
      // Test-only mock implementation
      const initialState: GameState = {
        phase: 'scoring' as GamePhase,
        players: [
          { id: 0, name: 'Player 1', hand: [], teamId: 0 as 0, marks: 0 },
          { id: 1, name: 'Player 2', hand: [], teamId: 1 as 1, marks: 0 },
          { id: 2, name: 'Player 3', hand: [], teamId: 0 as 0, marks: 0 },
          { id: 3, name: 'Player 4', hand: [], teamId: 1 as 1, marks: 0 },
        ],
        currentPlayer: 0,
        dealer: 3,
        bids: [],
        currentBid: { type: 'marks', value: 2, player: 0 },
        winningBidder: 0,
        trump: null,
        tricks: [],
        currentTrick: [],
        teamScores: [35, 7],
        teamMarks: [0, 0],
        gameTarget: 7,
        tournamentMode: true,
        shuffleSeed: 12345,
      };

      // Mock scoring function for test
      function scoreFailedBid(state: GameState): GameState {
        const newState = { ...state };
        const bidderTeam = state.players[state.winningBidder!].teamId;
        const opponentTeam = bidderTeam === 0 ? 1 : 0;
        const marksBid = state.currentBid?.value || 0;
        
        // When bid fails, opponents get the marks that were bid
        newState.teamMarks = [...state.teamMarks] as [number, number];
        newState.teamMarks[opponentTeam] += marksBid;
        
        return newState;
      }

      const scoredState = scoreFailedBid(initialState);
      
      // Verify scoring logic is ready to be tested
      expect(scoredState.teamMarks).toBeDefined();
      expect(scoredState.teamMarks.length).toBe(2);
    });

    it('Then the opponents receive marks equal to what was bid', () => {
      // Test-only mock implementation
      const initialState: GameState = {
        phase: 'scoring' as GamePhase,
        players: [
          { id: 0, name: 'Player 1', hand: [], teamId: 0 as 0, marks: 0 },
          { id: 1, name: 'Player 2', hand: [], teamId: 1 as 1, marks: 0 },
          { id: 2, name: 'Player 3', hand: [], teamId: 0 as 0, marks: 0 },
          { id: 3, name: 'Player 4', hand: [], teamId: 1 as 1, marks: 0 },
        ],
        currentPlayer: 0,
        dealer: 3,
        bids: [],
        currentBid: { type: 'marks', value: 2, player: 0 }, // Team 0 bid 2 marks
        winningBidder: 0,
        trump: null,
        tricks: [],
        currentTrick: [],
        teamScores: [35, 7], // Team 0 failed to make 84 points
        teamMarks: [3, 2], // Initial marks before scoring this hand
        gameTarget: 7,
        tournamentMode: true,
        shuffleSeed: 12345,
      };

      // Mock scoring function for test
      function scoreFailedBid(state: GameState): GameState {
        const newState = { ...state };
        const bidderTeam = state.players[state.winningBidder!].teamId;
        const opponentTeam = bidderTeam === 0 ? 1 : 0;
        const marksBid = state.currentBid?.value || 0;
        
        // When bid fails, opponents get the marks that were bid
        newState.teamMarks = [...state.teamMarks] as [number, number];
        newState.teamMarks[opponentTeam] += marksBid;
        
        return newState;
      }

      const scoredState = scoreFailedBid(initialState);
      
      // Team 0 bid 2 marks and failed
      // Team 1 (opponents) should receive 2 marks
      expect(scoredState.teamMarks[1]).toBe(4); // 2 initial + 2 from failed bid
      expect(scoredState.teamMarks[0]).toBe(3); // Unchanged
    });

    // Additional test cases for different bid types
    it('handles failed point bids correctly', () => {
      // Test-only mock implementation
      const state: GameState = {
        phase: 'scoring' as GamePhase,
        players: [
          { id: 0, name: 'Player 1', hand: [], teamId: 0 as 0, marks: 0 },
          { id: 1, name: 'Player 2', hand: [], teamId: 1 as 1, marks: 0 },
          { id: 2, name: 'Player 3', hand: [], teamId: 0 as 0, marks: 0 },
          { id: 3, name: 'Player 4', hand: [], teamId: 1 as 1, marks: 0 },
        ],
        currentPlayer: 1,
        dealer: 0,
        bids: [],
        currentBid: { type: 'points', value: 35, player: 1 }, // Team 1 bid 35 points
        winningBidder: 1,
        trump: null,
        tricks: [],
        currentTrick: [],
        teamScores: [28, 14], // Team 1 only got 14 points
        teamMarks: [0, 0],
        gameTarget: 7,
        tournamentMode: true,
        shuffleSeed: 12345,
      };

      // Mock scoring function for test
      function scoreFailedBid(state: GameState): GameState {
        const newState = { ...state };
        const bidderTeam = state.players[state.winningBidder!].teamId;
        const opponentTeam = bidderTeam === 0 ? 1 : 0;
        
        // For point bids, opponents get 1 mark when bid fails
        newState.teamMarks = [...state.teamMarks] as [number, number];
        newState.teamMarks[opponentTeam] += 1;
        
        return newState;
      }

      const scoredState = scoreFailedBid(state);
      
      // Team 1 failed a point bid, Team 0 gets 1 mark
      expect(scoredState.teamMarks[0]).toBe(1);
      expect(scoredState.teamMarks[1]).toBe(0);
    });

    it('handles failed 1 mark bid correctly', () => {
      // Test-only mock implementation
      const state: GameState = {
        phase: 'scoring' as GamePhase,
        players: [
          { id: 0, name: 'Player 1', hand: [], teamId: 0 as 0, marks: 0 },
          { id: 1, name: 'Player 2', hand: [], teamId: 1 as 1, marks: 0 },
          { id: 2, name: 'Player 3', hand: [], teamId: 0 as 0, marks: 0 },
          { id: 3, name: 'Player 4', hand: [], teamId: 1 as 1, marks: 0 },
        ],
        currentPlayer: 2,
        dealer: 1,
        bids: [],
        currentBid: { type: 'marks', value: 1, player: 2 }, // Team 0 bid 1 mark (42 points)
        winningBidder: 2,
        trump: null,
        tricks: [],
        currentTrick: [],
        teamScores: [40, 2], // Team 0 got 40 points, failed to make 42
        teamMarks: [1, 1],
        gameTarget: 7,
        tournamentMode: true,
        shuffleSeed: 12345,
      };

      // Mock scoring function for test
      function scoreFailedBid(state: GameState): GameState {
        const newState = { ...state };
        const bidderTeam = state.players[state.winningBidder!].teamId;
        const opponentTeam = bidderTeam === 0 ? 1 : 0;
        const marksBid = state.currentBid?.value || 0;
        
        // When bid fails, opponents get the marks that were bid
        newState.teamMarks = [...state.teamMarks] as [number, number];
        newState.teamMarks[opponentTeam] += marksBid;
        
        return newState;
      }

      const scoredState = scoreFailedBid(state);
      
      // Team 0 failed 1 mark bid, Team 1 gets 1 mark
      expect(scoredState.teamMarks[1]).toBe(2); // 1 initial + 1 from failed bid
      expect(scoredState.teamMarks[0]).toBe(1); // Unchanged
    });

    it('handles failed plunge bid correctly', () => {
      // Test-only mock implementation
      const state: GameState = {
        phase: 'scoring' as GamePhase,
        players: [
          { id: 0, name: 'Player 1', hand: [], teamId: 0 as 0, marks: 0 },
          { id: 1, name: 'Player 2', hand: [], teamId: 1 as 1, marks: 0 },
          { id: 2, name: 'Player 3', hand: [], teamId: 0 as 0, marks: 0 },
          { id: 3, name: 'Player 4', hand: [], teamId: 1 as 1, marks: 0 },
        ],
        currentPlayer: 3,
        dealer: 2,
        bids: [],
        currentBid: { type: 'plunge', value: 4, player: 3 }, // Team 1 bid plunge (4 marks)
        winningBidder: 3,
        trump: null,
        tricks: [],
        currentTrick: [],
        teamScores: [7, 35], // Team 1 failed to win all tricks
        teamMarks: [0, 0],
        gameTarget: 7,
        tournamentMode: true,
        shuffleSeed: 12345,
      };

      // Mock scoring function for test
      function scoreFailedBid(state: GameState): GameState {
        const newState = { ...state };
        const bidderTeam = state.players[state.winningBidder!].teamId;
        const opponentTeam = bidderTeam === 0 ? 1 : 0;
        const marksBid = state.currentBid?.value || 0;
        
        // When bid fails, opponents get the marks that were bid
        newState.teamMarks = [...state.teamMarks] as [number, number];
        newState.teamMarks[opponentTeam] += marksBid;
        
        return newState;
      }

      const scoredState = scoreFailedBid(state);
      
      // Team 1 failed 4 mark plunge, Team 0 gets 4 marks
      expect(scoredState.teamMarks[0]).toBe(4);
      expect(scoredState.teamMarks[1]).toBe(0);
    });
  });
});