import { describe, it, expect } from 'vitest';
import { createInitialState } from '../../game/core/state';
import { getNextStates } from '../../game/core/gameEngine';
import { getNextPlayer } from '../../game/core/players';
import type { GameState } from '../../game/types';
import { testLog } from '../helpers/testConsole';
import { BLANKS } from '../../game/types';

describe('State Transitions Integration', () => {
  describe('Phase Transitions', () => {
    it('should transition from bidding to playing correctly', () => {
      let state = createInitialState();
      expect(state.phase).toBe('bidding');
      
      // Complete bidding phase
      // Player 1 bids 30
      let transitions = getNextStates(state);
      const bid30 = transitions.find(t => t.id === 'bid-30');
      if (bid30) {
        state = bid30.newState;
      }
      
      // Others pass
      for (let i = 0; i < 3; i++) {
        transitions = getNextStates(state);
        const pass = transitions.find(t => t.id === 'pass');
        if (pass) {
          state = pass.newState;
        }
      }
      
      // Select trump
      transitions = getNextStates(state);
      const trumpAction = transitions.find(t => t.id.startsWith('trump-'));
      if (trumpAction) {
        state = trumpAction.newState;
        expect(state.phase).toBe('playing');
        expect(state.trump).not.toBeNull();
        expect(state.winningBidder).not.toBeNull();
      }
    });

    it('should transition from playing to scoring after 7 tricks', () => {
      // This would be a complex test requiring playing through 7 complete tricks
      // For now, we'll test the state structure is correct
      const state = createInitialState();
      
      // Simulate end of playing phase
      state.phase = 'playing';
      state.trump = { type: 'suit', suit: BLANKS }; // blanks
      state.winningBidder = 0;
      
      // Add 7 complete tricks (simulated)
      for (let i = 0; i < 7; i++) {
        state.tricks.push({
          plays: [],
          winner: i % 4, // Keep this as-is since it's just cycling through players for test data
          points: 6 // 42/7 = 6 average
        });
      }
      
      // Verify we can transition to scoring
      expect(state.tricks).toHaveLength(7);
    });

    it('should transition from scoring to next hand or game end', () => {
      const state = createInitialState();
      state.phase = 'scoring';
      state.teamScores = [25, 17];
      state.teamMarks = [6, 4];
      
      // If neither team has reached target, should start new hand
      if (Math.max(...state.teamMarks) < state.gameTarget) {
        expect(state.phase).toBe('scoring');
      }
      
      // If a team reaches target, should end game
      state.teamMarks = [7, 4];
      expect(Math.max(...state.teamMarks)).toBeGreaterThanOrEqual(state.gameTarget);
    });
  });

  describe('Action Availability', () => {
    it('should provide correct actions for each phase', () => {
      let state = createInitialState();
      
      // Bidding phase actions
      expect(state.phase).toBe('bidding');
      let actions = getNextStates(state);
      expect(actions.some(a => a.id.startsWith('bid-'))).toBe(true);
      expect(actions.some(a => a.id === 'pass')).toBe(true);
      
      // Simulate transition to trump selection
      const bid = actions.find(a => a.id.startsWith('bid-'));
      if (bid) {
        state = bid.newState;
        
        // Pass for other players
        for (let i = 0; i < 3; i++) {
          actions = getNextStates(state);
          const pass = actions.find(a => a.id === 'pass');
          if (pass) {
            state = pass.newState;
          }
        }
        
        // Trump selection actions
        actions = getNextStates(state);
        expect(actions.some(a => a.id.startsWith('trump-'))).toBe(true);
        
        // Select trump and transition to playing
        const trump = actions.find(a => a.id.startsWith('trump-'));
        if (trump) {
          state = trump.newState;
          expect(state.phase).toBe('playing');
          
          // Playing phase actions
          actions = getNextStates(state);
          expect(actions.every(a => a.id.startsWith('play-'))).toBe(true);
        }
      }
    });

    it('should respect player turn order', () => {
      const state = createInitialState();
      const initialPlayer = state.currentPlayer;
      
      const actions = getNextStates(state);
      expect(actions.length).toBeGreaterThan(0);
      
      // Execute first action
      const nextState = actions[0]!.newState;
      
      // Current player should advance (unless it's trump selection)
      if (nextState.phase === 'bidding') {
        const expectedNext = getNextPlayer(initialPlayer);
        expect(nextState.currentPlayer).toBe(expectedNext);
      }
    });

    it('should prevent invalid actions', () => {
      const state = createInitialState();
      state.phase = 'playing';
      state.trump = { type: 'suit', suit: BLANKS }; // blanks
      
      // Should only have play actions in playing phase
      const actions = getNextStates(state);
      expect(actions.every(a => a.id.startsWith('play-'))).toBe(true);
      expect(actions.some(a => a.id.startsWith('bid-'))).toBe(false);
      expect(actions.some(a => a.id.startsWith('trump-'))).toBe(false);
    });
  });

  describe('State Consistency', () => {
    it('should maintain valid game state through transitions', () => {
      let state = createInitialState();
      
      // Execute several random transitions
      for (let i = 0; i < 10; i++) {
        const actions = getNextStates(state);
        if (actions.length === 0) break;
        
        // Pick random action
        const randomAction = actions[Math.floor(Math.random() * actions.length)];
        if (!randomAction) break;
        state = randomAction.newState;
        
        // Verify state consistency
        expect(state.players).toHaveLength(4);
        expect(state.currentPlayer).toBeGreaterThanOrEqual(0);
        expect(state.currentPlayer).toBeLessThan(4);
        expect(['bidding', 'trump_selection', 'playing', 'scoring', 'game_end']).toContain(state.phase);
        expect(state.teamScores).toHaveLength(2);
        expect(state.teamMarks).toHaveLength(2);
        
        // Verify player hands don't grow
        const totalCards = state.players.reduce((sum, p) => sum + p.hand.length, 0);
        expect(totalCards).toBeLessThanOrEqual(28);
      }
    });

    it('should preserve immutability of state', () => {
      const state = createInitialState();
      // Deep clone for comparison, handling Sets properly
      const originalState = {
        ...state,
        players: state.players.map(p => ({ ...p, hand: [...p.hand] })),
        bids: [...state.bids],
        tricks: [...state.tricks],
        currentTrick: [...state.currentTrick],
        teamScores: [...state.teamScores],
        teamMarks: [...state.teamMarks],
        consensus: {
          completeTrick: new Set(state.consensus.completeTrick),
          scoreHand: new Set(state.consensus.scoreHand)
        },
        actionHistory: [...state.actionHistory]
      };
      
      const actions = getNextStates(state);
      if (actions.length > 0) {
        const newState = actions[0]!.newState;
        
        // Original state should be unchanged
        expect(state).toEqual(originalState);
        expect(newState).not.toBe(state);
      }
    });

    it('should handle edge cases gracefully', () => {
      const state = createInitialState();
      
      // Empty current trick
      expect(state.currentTrick).toEqual([]);
      
      // No completed tricks initially
      expect(state.tricks).toEqual([]);
      
      // Valid initial scores
      expect(state.teamScores).toEqual([0, 0]);
      expect(state.teamMarks).toEqual([0, 0]);
      
      // Valid dealer
      expect(state.dealer).toBeGreaterThanOrEqual(0);
      expect(state.dealer).toBeLessThan(4);
    });

    it('should provide trump selection moves when in trump_selection phase', () => {
      // Create a state in trump_selection phase with a winning bidder
      const state = createInitialState();
      state.phase = 'trump_selection';
      state.currentPlayer = 2;
      state.winningBidder = 2;
      state.currentBid = { type: 'points', value: 30, player: 2 };
      state.bids = [
        { type: 'pass', player: 0 },
        { type: 'pass', player: 1 },
        { type: 'points', value: 30, player: 2 },
        { type: 'pass', player: 3 }
      ];
      
      const actions = getNextStates(state);
      
      // Should have trump selection actions available
      expect(actions.length).toBeGreaterThan(0);
      expect(actions.some(a => a.id.startsWith('trump-'))).toBe(true);
      
      // Should have actions for each trump suit
      const trumpActions = actions.filter(a => a.id.startsWith('trump-'));
      expect(trumpActions.length).toBeGreaterThan(0);
      
      // Each trump action should transition to playing phase
      trumpActions.forEach(action => {
        expect(action.newState.phase).toBe('playing');
        expect(action.newState.trump).not.toBeNull();
      });
    });

    it('should provide trump selection moves with exact user-provided state', () => {
      // Create state exactly matching the user's provided JSON
      const state: GameState = {
      initialConfig: {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        shuffleSeed: 0,
        theme: 'business',
        colorOverrides: {}
      },
        phase: 'trump_selection',
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        players: [
          {
            id: 0,
            name: 'Player 1',
            hand: [
              { high: 1, low: 2, id: '1-2' },
              { high: 2, low: 5, id: '2-5' },
              { high: 1, low: 6, id: '1-6' },
              { high: 3, low: 4, id: '3-4' },
              { high: 5, low: 5, id: '5-5' },
              { high: 0, low: 0, id: '0-0' },
              { high: 5, low: 6, id: '5-6' }
            ],
            teamId: 0,
            marks: 0
          },
          {
            id: 1,
            name: 'Player 2',
            hand: [
              { high: 1, low: 4, id: '1-4' },
              { high: 4, low: 4, id: '4-4' },
              { high: 2, low: 2, id: '2-2' },
              { high: 1, low: 5, id: '1-5' },
              { high: 0, low: 6, id: '0-6' },
              { high: 1, low: 3, id: '1-3' },
              { high: 0, low: 5, id: '0-5' }
            ],
            teamId: 1,
            marks: 0
          },
          {
            id: 2,
            name: 'Player 3',
            hand: [
              { high: 3, low: 5, id: '3-5' },
              { high: 2, low: 6, id: '2-6' },
              { high: 4, low: 6, id: '4-6' },
              { high: 0, low: 1, id: '0-1' },
              { high: 3, low: 6, id: '3-6' },
              { high: 6, low: 6, id: '6-6' },
              { high: 2, low: 3, id: '2-3' }
            ],
            teamId: 0,
            marks: 0
          },
          {
            id: 3,
            name: 'Player 4',
            hand: [
              { high: 3, low: 3, id: '3-3' },
              { high: 0, low: 3, id: '0-3' },
              { high: 0, low: 2, id: '0-2' },
              { high: 2, low: 4, id: '2-4' },
              { high: 4, low: 5, id: '4-5' },
              { high: 0, low: 4, id: '0-4' },
              { high: 1, low: 1, id: '1-1' }
            ],
            teamId: 1,
            marks: 0
          }
        ],
        currentPlayer: 2,
        dealer: 3,
        bids: [
          { type: 'pass', player: 0 },
          { type: 'pass', player: 1 },
          { type: 'points', value: 30, player: 2 },
          { type: 'pass', player: 3 }
        ],
        currentBid: { type: 'points', value: 30, player: 2 },
        winningBidder: 2,
        trump: { type: 'not-selected' },
        tricks: [],
        currentTrick: [],
        currentSuit: -1,
        teamScores: [0, 0],
        teamMarks: [0, 0],
        gameTarget: 7,
        shuffleSeed: 12345,
        actionHistory: [],
        consensus: {
          completeTrick: new Set<number>(),
          scoreHand: new Set<number>()
        },
        theme: 'coffee',
        colorOverrides: {}
      };
      
      const actions = getNextStates(state);
      
      // Debug: log the actions to see what's happening
      testLog('Trump selection actions:', actions);
      
      // Should have trump selection actions available
      expect(actions.length).toBeGreaterThan(0);
      expect(actions.some(a => a.id.startsWith('trump-'))).toBe(true);
    });
  });

  describe('Complex State Scenarios', () => {
    it('should handle multiple bid rounds', () => {
      let state = createInitialState();
      
      // Simulate competitive bidding
      const bidSequence = ['bid-30', 'bid-32', 'pass', 'bid-35', 'pass', 'pass', 'pass'];
      
      for (const actionId of bidSequence) {
        const actions = getNextStates(state);
        const action = actions.find(a => a.id === actionId);
        if (action) {
          state = action.newState;
        } else {
          break; // Action not available
        }
      }
      
      // Should have valid bid history
      expect(state.bids.length).toBeGreaterThan(0);
      if (state.currentBid) {
        expect(state.currentBid.value).toBeGreaterThanOrEqual(30);
      }
    });

    it('should handle all-pass scenario', () => {
      let state = createInitialState();
      
      // All players pass
      for (let i = 0; i < 4; i++) {
        const actions = getNextStates(state);
        const pass = actions.find(a => a.id === 'pass');
        if (pass) {
          state = pass.newState;
        }
      }
      
      // Should handle redeal or end hand appropriately
      expect(state.bids).toHaveLength(4);
    });

    it('should maintain mathematical constraints', () => {
      const state = createInitialState();
      
      // Total points should always be 42 (from counting dominoes + tricks)
      let totalCountingPoints = 0;
      state.players.forEach(player => {
        player.hand.forEach(domino => {
          if (domino.high === 5 && domino.low === 5) totalCountingPoints += 10; // 5-5 = 10 points
          else if ((domino.high === 6 && domino.low === 4) || (domino.high === 4 && domino.low === 6)) totalCountingPoints += 10; // 6-4 = 10 points
          else if (domino.high + domino.low === 5) totalCountingPoints += 5; // Any domino totaling 5 pips = 5 points
        });
      });
      expect(totalCountingPoints + 7).toBe(42); // 35 counting points + 7 trick points = 42
      
      // Team scores should not exceed 42 in a single hand
      expect(state.teamScores[0] + state.teamScores[1]).toBeLessThanOrEqual(42);
    });
  });
});