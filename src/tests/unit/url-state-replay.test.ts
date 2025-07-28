import { describe, test, expect, beforeEach } from 'vitest';
import { gameActions, gameState, debugSnapshot, gameHistory } from '../../stores/gameStore';
import { get } from 'svelte/store';
import { createInitialState } from '../../game/core/state';
import { getNextStates } from '../../game/core/actions';

describe('URL State Replay', () => {
  beforeEach(() => {
    // Reset to clean state before each test
    gameActions.resetGame();
  });

  test('loadStateWithActionReplay should replay actions correctly', () => {
    // Create a base state
    const baseState = createInitialState();
    
    // Define a sequence of actions (simulating bidding)
    const actions = [
      { id: 'pass', label: 'Pass' },
      { id: 'bid-30', label: 'Bid 30 points' },
      { id: 'pass', label: 'Pass' },
      { id: 'pass', label: 'Pass' }
    ];

    // Load the state with action replay
    gameActions.loadStateWithActionReplay(baseState, actions);

    // Verify the final state reflects all actions
    const finalState = get(gameState);
    
    // Should be in trump_selection phase after bidding is complete
    expect(finalState.phase).toBe('trump_selection');
    expect(finalState.winningBidder).toBe(1); // Player 1 won with bid-30
    expect(finalState.currentPlayer).toBe(1); // Winning bidder selects trump
    
    // Verify bids were recorded
    expect(finalState.bids).toHaveLength(4);
    expect(finalState.bids[0].type).toBe('pass');
    expect(finalState.bids[1].type).toBe('points');
    expect(finalState.bids[1].value).toBe(30);
    expect(finalState.bids[2].type).toBe('pass');
    expect(finalState.bids[3].type).toBe('pass');

    // Verify debug snapshot was created correctly
    const snapshot = get(debugSnapshot);
    expect(snapshot).toBeDefined();
    expect(snapshot!.baseState).toEqual(baseState);
    expect(snapshot!.actions).toHaveLength(4);
    expect(snapshot!.actions.map(a => a.id)).toEqual(['pass', 'bid-30', 'pass', 'pass']);
  });

  test('loadStateWithActionReplay should throw error on invalid actions', () => {
    const baseState = createInitialState();
    
    // Mix of valid and invalid actions
    const actions = [
      { id: 'pass', label: 'Pass' },
      { id: 'invalid-action', label: 'Invalid Action' }, // This should cause an error
      { id: 'bid-30', label: 'Bid 30 points' } // This should not be reached
    ];

    // Should throw an error when encountering invalid action
    expect(() => {
      gameActions.loadStateWithActionReplay(baseState, actions);
    }).toThrow('Invalid action replay: Action "invalid-action" is not available in current state');

    // State should be left in the intermediate state (after pass, before invalid action)
    const finalState = get(gameState);
    expect(finalState.phase).toBe('bidding');  
    expect(finalState.currentPlayer).toBe(1); // Next player after pass
    expect(finalState.bids).toHaveLength(1);
    expect(finalState.bids[0].type).toBe('pass');

    // Debug snapshot should contain only the valid actions before the error
    const snapshot = get(debugSnapshot);
    expect(snapshot).toBeDefined();
    expect(snapshot!.actions).toHaveLength(1);
    expect(snapshot!.actions[0].id).toBe('pass');
  });

  test('loadStateWithActionReplay should work with complete game sequence', () => {
    const baseState = createInitialState();
    
    // A longer sequence through bidding and trump selection
    const actions = [
      { id: 'pass', label: 'Pass' },
      { id: 'bid-30', label: 'Bid 30 points' },
      { id: 'pass', label: 'Pass' },
      { id: 'pass', label: 'Pass' },
      { id: 'trump-fives', label: 'Declare FIVES trump' }
    ];

    gameActions.loadStateWithActionReplay(baseState, actions);

    const finalState = get(gameState);
    
    // Should be in playing phase now
    expect(finalState.phase).toBe('playing');
    expect(finalState.trump).toBe(5); // Fives are trump
    expect(finalState.winningBidder).toBe(1);
    expect(finalState.currentPlayer).toBe(1); // Bid winner leads

    // Verify all actions in debug snapshot
    const snapshot = get(debugSnapshot);
    expect(snapshot).toBeDefined();
    expect(snapshot!.actions).toHaveLength(5);
    expect(snapshot!.actions.map(a => a.id)).toEqual([
      'pass', 'bid-30', 'pass', 'pass', 'trump-fives'
    ]);
  });

  test('loadStateWithActionReplay with empty actions should just load base state', () => {
    const baseState = createInitialState();
    const actions: Array<{id: string, label: string}> = [];

    gameActions.loadStateWithActionReplay(baseState, actions);

    const finalState = get(gameState);
    expect(finalState).toEqual(baseState);

    // No debug snapshot should be created for empty actions
    const snapshot = get(debugSnapshot);
    expect(snapshot).toBeNull();
  });

  test('should maintain consistency between normal play and replay', () => {
    // Play a sequence normally using getNextStates to get valid transitions
    const initialState = createInitialState();
    
    // Get a valid pass action
    const transitions = getNextStates(initialState);
    const passTransition = transitions.find(t => t.id === 'pass');
    
    if (passTransition) {
      gameActions.executeAction(passTransition);
    }
    
    const currentState = get(gameState);
    const currentSnapshot = get(debugSnapshot);
    
    // Only test if we have a snapshot
    if (currentSnapshot && currentSnapshot.actions.length > 0) {
      // Reset and replay the same sequence
      const baseState = currentSnapshot.baseState;
      const actions = currentSnapshot.actions.map(a => ({ id: a.id, label: a.label }));
      
      gameActions.loadStateWithActionReplay(baseState, actions);
      
      const replayedState = get(gameState);
      const replayedSnapshot = get(debugSnapshot);
      
      // States should be identical
      expect(replayedState.phase).toBe(currentState.phase);
      expect(replayedState.currentPlayer).toBe(currentState.currentPlayer);
      expect(replayedSnapshot!.actions.map(a => a.id)).toEqual(
        currentSnapshot!.actions.map(a => a.id)
      );
    }
  });

  test('loadStateWithActionReplay should populate history for Undo functionality', () => {
    const baseState = createInitialState();
    
    // Actions that should build up history
    const actions = [
      { id: 'pass', label: 'Pass' },
      { id: 'bid-30', label: 'Bid 30 points' },
      { id: 'pass', label: 'Pass' }
    ];

    gameActions.loadStateWithActionReplay(baseState, actions);

    const finalState = get(gameState);
    const history = get(gameHistory);
    
    // History should have 3 entries (one before each action)
    expect(history).toHaveLength(3);
    
    // First history entry should be the base state
    expect(history[0]).toEqual(baseState);
    
    // Each history entry should be different (representing states before each action)
    expect(history[0].currentPlayer).toBe(0); // Base state
    expect(history[1].currentPlayer).toBe(1); // After first pass
    expect(history[2].currentPlayer).toBe(2); // After bid-30
    
    // Final state should be different from last history entry
    expect(finalState.currentPlayer).toBe(3); // After third pass
    
    // Should be able to undo step by step
    gameActions.undo();
    const afterFirstUndo = get(gameState);
    expect(afterFirstUndo.currentPlayer).toBe(2); // Back to before third pass
    
    gameActions.undo();
    const afterSecondUndo = get(gameState);
    expect(afterSecondUndo.currentPlayer).toBe(1); // Back to before bid-30
    
    gameActions.undo();  
    const afterThirdUndo = get(gameState);
    expect(afterThirdUndo).toEqual(baseState); // Back to initial state
  });
});