import { describe, it, expect } from 'vitest';
import { GameEngine } from '../../game/core/gameEngine';
import { executeAction } from '../../game/core/actions';
import { createInitialState } from '../../game/core/state';
import type { GameAction } from '../../game/types';

describe('Game Engine', () => {
  describe('Action-based state management', () => {
    it('should handle basic action application', () => {
      const initialState = createInitialState({ shuffleSeed: 12345 });
      const engine = new GameEngine(initialState);
      
      // Verify initial state
      const state = engine.getState();
      expect(state.phase).toBe('bidding');
      expect(state.winningBidder).toBe(-1); // Not null
      expect(state.trump.type).toBe('none'); // TrumpSelection, not null
      expect(state.currentSuit).toBe(-1); // Not null
    });
    
    it('should maintain action history', () => {
      const initialState = createInitialState({ shuffleSeed: 12345 });
      const engine = new GameEngine(initialState);
      
      const passAction: GameAction = { type: 'pass', player: 0 };
      engine.executeAction(passAction);
      
      const history = engine.getHistory();
      expect(history).toHaveLength(1);
      expect(history[0]).toEqual(passAction);
    });
    
    it('should support undo functionality', () => {
      const initialState = createInitialState({ shuffleSeed: 12345 });
      const engine = new GameEngine(initialState);
      
      const originalBidsLength = engine.getState().bids.length;
      
      // Execute action
      const passAction: GameAction = { type: 'pass', player: 0 };
      engine.executeAction(passAction);
      
      expect(engine.getState().bids.length).toBe(originalBidsLength + 1);
      
      // Undo
      const undoResult = engine.undo();
      expect(undoResult).toBe(true);
      expect(engine.getState().bids.length).toBe(originalBidsLength);
      expect(engine.getHistory()).toHaveLength(0);
    });
    
    it('should prevent undo when no history exists', () => {
      const initialState = createInitialState({ shuffleSeed: 12345 });
      const engine = new GameEngine(initialState);
      
      const undoResult = engine.undo();
      expect(undoResult).toBe(false);
    });
  });
  
  describe('Pure action functions', () => {
    it('should apply pass action correctly', () => {
      const initialState = createInitialState({ shuffleSeed: 12345 });
      const passAction: GameAction = { type: 'pass', player: 0 };
      
      const newState = executeAction(initialState, passAction);
      
      expect(newState.bids).toHaveLength(1);
      expect(newState.bids[0]!.type).toBe('pass');
      expect(newState.bids[0]!.player).toBe(0);
      expect(newState.currentPlayer).toBe(1); // Next player
    });
    
    it('should apply bid action correctly', () => {
      const initialState = createInitialState({ shuffleSeed: 12345 });
      const bidAction: GameAction = { 
        type: 'bid', 
        player: 0, 
        bid: 'points', 
        value: 30 
      };
      
      const newState = executeAction(initialState, bidAction);
      
      expect(newState.bids).toHaveLength(1);
      expect(newState.bids[0]!.type).toBe('points');
      expect(newState.bids[0]!.value).toBe(30);
      expect(newState.bids[0]!.player).toBe(0);
      expect(newState.currentBid).toEqual(newState.bids[0]!);
    });
    
    it('should apply trump selection correctly', () => {
      const initialState = createInitialState({ shuffleSeed: 12345 });
      // Set up state for trump selection
      const stateWithBid = { 
        ...initialState, 
        phase: 'trump_selection' as const,
        winningBidder: 0,
        currentPlayer: 0,
        actionHistory: [],
        consensus: {
          completeTrick: new Set<number>(),
          scoreHand: new Set<number>()
        }
      };
      
      const trumpAction: GameAction = {
        type: 'select-trump',
        player: 0,
        trump: { type: 'suit', suit: 2 }
      };
      
      const newState = executeAction(stateWithBid, trumpAction);
      
      expect(newState.phase).toBe('playing');
      expect(newState.trump).toEqual({ type: 'suit', suit: 2 });
      expect(newState.currentPlayer).toBe(0);
    });
  });
  
  describe('Non-nullable fields', () => {
    it('should never have null values for key fields', () => {
      const initialState = createInitialState({ shuffleSeed: 12345 });
      
      // Check initial state
      expect(initialState.winningBidder).toBe(-1);
      expect(initialState.trump.type).toBe('none');
      expect(initialState.currentSuit).toBe(-1);
      
      // Apply some actions and verify no nulls are introduced
      const engine = new GameEngine(initialState);
      
      const passAction: GameAction = { type: 'pass', player: 0 };
      engine.executeAction(passAction);
      
      const state = engine.getState();
      expect(state.winningBidder).toBe(-1);
      expect(state.trump.type).toBe('none');
      expect(state.currentSuit).toBe(-1);
    });
  });
});
