import { describe, it, expect } from 'vitest';
import { createInitialState } from '../../game/core/state';
import { getNextStates } from '../../game/core/actions';
import { GameState, StateTransition } from '../../game/types';
import { BID_TYPES } from '../../game/constants';

describe('Full Game Scenarios', () => {
  function executeAction(state: GameState, actionId: string): GameState {
    const transitions = getNextStates(state);
    const transition = transitions.find(t => t.id === actionId);
    if (!transition) {
      throw new Error(`Action ${actionId} not available`);
    }
    return transition.newState;
  }

  describe('Complete Game Flow', () => {
    it('should handle a complete bidding and playing sequence', () => {
      let state = createInitialState();
      
      // Verify initial state
      expect(state.phase).toBe('bidding');
      expect(state.currentPlayer).toBe((state.dealer + 1) % 4);
      
      // First player bids 30
      const bidTransitions = getNextStates(state);
      const bid30 = bidTransitions.find(t => t.id === 'bid-30');
      expect(bid30).toBeDefined();
      
      state = bid30!.newState;
      expect(state.bids).toHaveLength(1);
      expect(state.bids[0]).toEqual({
        type: BID_TYPES.POINTS,
        value: 30,
        player: (state.dealer + 1) % 4
      });
      
      // Other players pass
      for (let i = 0; i < 3; i++) {
        const passTransitions = getNextStates(state);
        const pass = passTransitions.find(t => t.id === 'pass');
        expect(pass).toBeDefined();
        state = pass!.newState;
      }
      
      // Should transition to trump selection
      expect(state.winningBidder).toBe((state.dealer + 1) % 4);
      expect(state.currentBid).toEqual({
        type: BID_TYPES.POINTS,
        value: 30,
        player: (state.dealer + 1) % 4
      });
      
      // Select trump
      const trumpTransitions = getNextStates(state);
      const trumpBlanks = trumpTransitions.find(t => t.id === 'trump-blanks');
      expect(trumpBlanks).toBeDefined();
      
      state = trumpBlanks!.newState;
      expect(state.trump).toBe(0); // 0 represents blanks trump
      expect(state.phase).toBe('playing');
      
      // Verify game can proceed to playing
      expect(state.currentPlayer).toBe(state.winningBidder);
      const playTransitions = getNextStates(state);
      expect(playTransitions.length).toBeGreaterThan(0);
      expect(playTransitions.every(t => t.id.startsWith('play-'))).toBe(true);
    });

    it('should handle all players passing', () => {
      let state = createInitialState();
      
      // All players pass
      for (let i = 0; i < 4; i++) {
        const transitions = getNextStates(state);
        const pass = transitions.find(t => t.id === 'pass');
        expect(pass).toBeDefined();
        state = pass!.newState;
      }
      
      // Should redeal or handle all-pass scenario
      expect(state.bids).toHaveLength(4);
      expect(state.bids.every(bid => bid.type === BID_TYPES.PASS)).toBe(true);
    });

    it('should handle competitive bidding', () => {
      let state = createInitialState();
      
      // Player 1 bids 30
      state = executeAction(state, 'bid-30');
      
      // Player 2 bids 32
      const bidTransitions = getNextStates(state);
      const bid32 = bidTransitions.find(t => t.id === 'bid-32');
      if (bid32) {
        state = bid32.newState;
        expect(state.currentBid?.value).toBe(32);
      }
      
      // Player 3 passes
      state = executeAction(state, 'pass');
      
      // Player 4 bids 35
      const bid35Transitions = getNextStates(state);
      const bid35 = bid35Transitions.find(t => t.id === 'bid-35');
      if (bid35) {
        state = bid35.newState;
        expect(state.currentBid?.value).toBe(35);
      }
      
      // Verify bidding continues properly
      expect(state.bids.length).toBeGreaterThan(2);
      expect(state.phase).toBe('bidding');
    });
  });

  describe('Special Bid Scenarios', () => {
    it('should handle Nello bid', () => {
      let state = createInitialState();
      
      // Check if Nello is available (depends on hand)
      const transitions = getNextStates(state);
      const nello = transitions.find(t => t.id.startsWith('bid-nello'));
      
      if (nello) {
        state = nello.newState;
        expect(state.currentBid?.type).toBe(BID_TYPES.NELLO);
        
        // Others should pass or bid higher
        for (let i = 0; i < 3; i++) {
          const nextTransitions = getNextStates(state);
          const pass = nextTransitions.find(t => t.id === 'pass');
          if (pass) {
            state = pass.newState;
          }
        }
        
        // Should proceed to trump selection for Nello
        if (state.winningBidder !== null) {
          expect(state.currentBid?.type).toBe(BID_TYPES.NELLO);
        }
      }
    });

    it('should handle mark bids', () => {
      let state = createInitialState();
      
      // Try to bid marks
      const transitions = getNextStates(state);
      const markBid = transitions.find(t => t.id.startsWith('bid-mark'));
      
      if (markBid) {
        state = markBid.newState;
        expect(state.currentBid?.type).toBe(BID_TYPES.MARKS);
        expect(state.currentBid?.value).toBeGreaterThan(0);
      }
    });
  });

  describe('Game State Validation', () => {
    it('should maintain valid state throughout game', () => {
      let state = createInitialState();
      
      // Verify initial state constraints
      expect(state.players).toHaveLength(4);
      expect(state.players.every(p => p.hand.length === 7)).toBe(true);
      expect(state.teamScores).toEqual([0, 0]);
      expect(state.teamMarks).toEqual([0, 0]);
      
      // Execute several actions and verify state remains valid
      for (let i = 0; i < 8; i++) {
        const transitions = getNextStates(state);
        if (transitions.length === 0) break;
        
        // Take first available action
        state = transitions[0].newState;
        
        // Verify state constraints
        expect(state.players).toHaveLength(4);
        expect(state.currentPlayer).toBeGreaterThanOrEqual(0);
        expect(state.currentPlayer).toBeLessThan(4);
        expect(['bidding', 'playing', 'scoring', 'game_end']).toContain(state.phase);
      }
    });

    it('should preserve total domino count', () => {
      const state = createInitialState();
      
      const totalDominoes = state.players.reduce((sum, player) => sum + player.hand.length, 0);
      expect(totalDominoes).toBe(28); // All dominoes dealt
      
      // Verify no duplicate dominoes
      const allDominoes = state.players.flatMap(p => p.hand);
      const uniqueDominoes = new Set(allDominoes.map(d => d.id));
      expect(uniqueDominoes.size).toBe(28);
    });
  });
});