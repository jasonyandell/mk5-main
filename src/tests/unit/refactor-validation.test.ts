import { describe, it, expect } from 'vitest';
import { createInitialState } from '../../game/core/state';
import { GameEngine } from '../../game/core/gameEngine';
import { getNextStates } from '../../game/core/gameEngine';

describe('Refactor Validation', () => {
  it('should have no nullable fields in initial state', () => {
    const state = createInitialState({ shuffleSeed: 12345 });
    
    // Core assertions - these should never be null after refactor
    expect(state.winningBidder).toBe(-1);
    expect(state.trump).toEqual({ type: 'none' });
    expect(state.currentSuit).toBe(-1);
    expect(state.winner).toBe(-1);
    expect(state.bidWinner).toBe(-1);
    // New assertion for currentBid - should be EMPTY_BID instead of null
    expect(state.currentBid).toEqual({ type: 'pass', player: -1 });
  });
  
  it('should work with the action-based engine', () => {
    const state = createInitialState({ shuffleSeed: 12345 });
    const engine = new GameEngine(state);
    
    // Basic functionality
    expect(engine.getHistory()).toHaveLength(0);
    
    const validActions = engine.getValidActions();
    expect(validActions.length).toBeGreaterThan(0);
    
    // Execute first action
    const firstAction = validActions[0];
    engine.executeAction(firstAction);
    
    expect(engine.getHistory()).toHaveLength(1);
    expect(engine.getHistory()[0]).toEqual(firstAction);
    
    // Test undo
    const undoResult = engine.undo();
    expect(undoResult).toBe(true);
    expect(engine.getHistory()).toHaveLength(0);
  });
  
  it('should maintain backward compatibility with getNextStates', () => {
    const state = createInitialState({ shuffleSeed: 12345 });
    
    // This should still work as before
    const transitions = getNextStates(state);
    expect(transitions.length).toBeGreaterThan(0);
    
    // Each transition should have the expected properties
    transitions.forEach(transition => {
      expect(transition).toHaveProperty('id');
      expect(transition).toHaveProperty('label');
      expect(transition).toHaveProperty('newState');
      
      // New states should also follow the no-null rule
      expect(transition.newState.winningBidder).toBeGreaterThanOrEqual(-1);
      expect(transition.newState.trump).toBeDefined();
      expect(transition.newState.trump.type).toBeDefined();
      expect(transition.newState.currentSuit).toBeGreaterThanOrEqual(-1);
    });
  });
  
  it('should handle trump selection with new TrumpSelection type', () => {
    const state = createInitialState({ shuffleSeed: 12345 });
    
    // Verify initial trump state
    expect(state.trump.type).toBe('none');
    
    // Test trump selection through engine
    const engine = new GameEngine(state);
    
    // Fast-forward to trump selection phase by applying bids
    const bidAction = { type: 'bid' as const, player: 0, bidType: 'points' as const, value: 30 };
    engine.executeAction(bidAction);
    
    // Pass for other players
    for (let i = 1; i < 4; i++) {
      const passAction = { type: 'pass' as const, player: i };
      engine.executeAction(passAction);
    }
    
    // Should now be in trump selection
    const trumpState = engine.getState();
    expect(trumpState.phase).toBe('trump_selection');
    expect(trumpState.winningBidder).toBe(0);
    
    // Select trump
    const trumpAction = {
      type: 'select-trump' as const,
      player: 0,
      selection: { type: 'suit' as const, suit: 2 as const }
    };
    engine.executeAction(trumpAction);
    
    const finalState = engine.getState();
    expect(finalState.phase).toBe('playing');
    expect(finalState.trump).toEqual({ type: 'suit', suit: 2 });
  });
});
