import { describe, it, expect } from 'vitest';
import { createInitialState, getNextStates } from '../../game';
import type { GameState, StateTransition } from '../../game/types';

describe('Deterministic Shuffle with Undo/Redo', () => {
  
  it('should maintain deterministic shuffling after undo/redo', () => {
    // Create initial state
    const initialState = createInitialState();
    const initialSeed = initialState.shuffleSeed;
    
    // Store initial hands for comparison
    const initialHands = initialState.players.map(p => [...p.hand]);
    
    // Make all players pass to trigger a redeal
    let currentState = initialState;
    const passActions: StateTransition[] = [];
    
    for (let i = 0; i < 4; i++) {
      const transitions = getNextStates(currentState);
      const passAction = transitions.find(t => t.label === 'Pass');
      expect(passAction).toBeDefined();
      
      passActions.push(passAction!);
      currentState = passAction!.newState;
    }
    
    // After 4 passes, we should have a redeal action available
    expect(currentState.phase).toBe('bidding');
    expect(currentState.bids.length).toBe(4);
    
    // Get the redeal action
    const redealTransitions = getNextStates(currentState);
    const redealAction = redealTransitions.find(t => t.id === 'redeal');
    expect(redealAction).toBeDefined();
    currentState = redealAction!.newState;
    passActions.push(redealAction!);
    
    // Now seed should be increased
    expect(currentState.shuffleSeed).toBe(initialSeed + 1000000);
    
    // Store the redealt hands
    const redealtHands = currentState.players.map(p => [...p.hand]);
    
    // Verify hands are different after redeal
    const handsChanged = redealtHands.some((hand, playerIdx) => 
      hand.some((domino, idx) => domino.id !== initialHands[playerIdx][idx].id)
    );
    expect(handsChanged).toBe(true);
    
    // Now replay from initial state - should get exact same results
    let replayState = initialState;
    for (const action of passActions) {
      const transitions = getNextStates(replayState);
      const matchingAction = transitions.find(t => t.id === action.id);
      expect(matchingAction).toBeDefined();
      replayState = matchingAction!.newState;
    }
    
    // Verify replay produces identical state
    expect(replayState.shuffleSeed).toBe(currentState.shuffleSeed);
    expect(replayState.players.map(p => p.hand)).toEqual(currentState.players.map(p => p.hand));
  });

  it('should produce same shuffle result with same seed', () => {
    // Create two states with same seed
    const state1 = createInitialState();
    const seed = 12345;
    
    // Create a state with specific seed
    const stateWithSeed: GameState = {
      ...state1,
      shuffleSeed: seed
    };
    
    // Simulate redeal by making all pass
    let current = stateWithSeed;
    for (let i = 0; i < 4; i++) {
      const transitions = getNextStates(current);
      const passAction = transitions.find(t => t.label === 'Pass');
      current = passAction!.newState;
    }
    
    const hands1 = current.players.map(p => p.hand.map(d => d.id));
    
    // Do it again with same starting seed
    current = stateWithSeed;
    for (let i = 0; i < 4; i++) {
      const transitions = getNextStates(current);
      const passAction = transitions.find(t => t.label === 'Pass');
      current = passAction!.newState;
    }
    
    const hands2 = current.players.map(p => p.hand.map(d => d.id));
    
    // Should produce identical hands
    expect(hands1).toEqual(hands2);
  });

  it('should handle multiple redeals deterministically', () => {
    const initialState = createInitialState();
    const states: GameState[] = [initialState];
    
    // Perform 3 rounds of all-pass redeals
    let currentState = initialState;
    
    for (let round = 0; round < 3; round++) {
      // All 4 players pass
      for (let player = 0; player < 4; player++) {
        const transitions = getNextStates(currentState);
        const passAction = transitions.find(t => t.label === 'Pass');
        currentState = passAction!.newState;
      }
      
      // Execute the redeal
      const redealTransitions = getNextStates(currentState);
      const redealAction = redealTransitions.find(t => t.id === 'redeal');
      currentState = redealAction!.newState;
      states.push(currentState);
      
      // Verify seed increments
      expect(currentState.shuffleSeed).toBeGreaterThan(
        states[states.length - 2].shuffleSeed
      );
    }
    
    // Replay the same sequence
    let replayState = initialState;
    const replayStates: GameState[] = [replayState];
    
    for (let round = 0; round < 3; round++) {
      for (let player = 0; player < 4; player++) {
        const transitions = getNextStates(replayState);
        const passAction = transitions.find(t => t.label === 'Pass');
        replayState = passAction!.newState;
      }
      
      // Execute the redeal
      const redealTransitions = getNextStates(replayState);
      const redealAction = redealTransitions.find(t => t.id === 'redeal');
      replayState = redealAction!.newState;
      replayStates.push(replayState);
    }
    
    // All states should match exactly
    for (let i = 0; i < states.length; i++) {
      expect(replayStates[i].shuffleSeed).toBe(states[i].shuffleSeed);
      expect(replayStates[i].players.map(p => p.hand)).toEqual(
        states[i].players.map(p => p.hand)
      );
    }
  });

  it('should maintain determinism through complete hand with scoring', () => {
    const initialState = createInitialState();
    
    // Play a complete hand: bid, set trump, play all tricks
    let currentState = initialState;
    const actions: StateTransition[] = [];
    
    // First player bids 30
    let transitions = getNextStates(currentState);
    let bidAction = transitions.find(t => t.label === 'Bid 30 points');
    expect(bidAction).toBeDefined();
    actions.push(bidAction!);
    currentState = bidAction!.newState;
    
    // Other players pass
    for (let i = 0; i < 3; i++) {
      transitions = getNextStates(currentState);
      const passAction = transitions.find(t => t.label === 'Pass');
      actions.push(passAction!);
      currentState = passAction!.newState;
    }
    
    // Set trump
    transitions = getNextStates(currentState);
    const trumpAction = transitions[0]; // Just pick first available trump
    actions.push(trumpAction);
    currentState = trumpAction.newState;
    
    // Play all tricks (just play first available domino each time)
    while (currentState.phase === 'playing') {
      transitions = getNextStates(currentState);
      if (transitions.length > 0) {
        actions.push(transitions[0]);
        currentState = transitions[0].newState;
      } else {
        break;
      }
    }
    
    // Should be in scoring phase
    expect(currentState.phase).toBe('scoring');
    
    // Score the hand
    transitions = getNextStates(currentState);
    const scoreAction = transitions[0];
    actions.push(scoreAction);
    currentState = scoreAction.newState;
    
    // Should be back in bidding with new seed
    expect(currentState.phase).toBe('bidding');
    const expectedSeed = initialState.shuffleSeed + 1000000;
    expect(currentState.shuffleSeed).toBe(expectedSeed);
    
    // Store the new hands
    const newHands = currentState.players.map(p => p.hand.map(d => d.id));
    
    // Replay entire sequence
    let replayState = initialState;
    for (const action of actions) {
      transitions = getNextStates(replayState);
      const matchingAction = transitions.find(t => t.id === action.id);
      expect(matchingAction).toBeDefined();
      replayState = matchingAction!.newState;
    }
    
    // Final state should match exactly
    expect(replayState.shuffleSeed).toBe(currentState.shuffleSeed);
    expect(replayState.players.map(p => p.hand.map(d => d.id))).toEqual(newHands);
  });

  it('should handle undo across redeals correctly', () => {
    const initialState = createInitialState();
    const stateHistory: GameState[] = [initialState];
    
    // Make moves and track states
    let currentState = initialState;
    
    // All pass to trigger redeal
    for (let i = 0; i < 4; i++) {
      const transitions = getNextStates(currentState);
      const passAction = transitions.find(t => t.label === 'Pass');
      currentState = passAction!.newState;
      stateHistory.push(currentState);
    }
    
    // Execute the redeal
    let transitions = getNextStates(currentState);
    const redealAction = transitions.find(t => t.id === 'redeal');
    expect(redealAction).toBeDefined();
    currentState = redealAction!.newState;
    stateHistory.push(currentState);
    
    const stateAfterRedeal = currentState;
    const handsAfterRedeal = currentState.players.map(p => p.hand.map(d => d.id));
    
    // Make a bid after redeal
    transitions = getNextStates(currentState);
    const bidAction = transitions.find(t => t.label === 'Bid 30 points');
    expect(bidAction).toBeDefined();
    currentState = bidAction!.newState;
    stateHistory.push(currentState);
    
    // Simulate undo back to before redeal
    const stateBeforeRedeal = stateHistory[3]; // State after 3 passes
    expect(stateBeforeRedeal.shuffleSeed).toBe(initialState.shuffleSeed);
    
    // Redo the last pass
    const redoTransitions = getNextStates(stateBeforeRedeal);
    const redoPass = redoTransitions.find(t => t.label === 'Pass');
    const afterFourthPass = redoPass!.newState;
    
    // Now get the redeal action
    const afterPassTransitions = getNextStates(afterFourthPass);
    const redoRedeal = afterPassTransitions.find(t => t.id === 'redeal');
    const redoState = redoRedeal!.newState;
    
    // Should match the state after redeal exactly
    expect(redoState.shuffleSeed).toBe(stateAfterRedeal.shuffleSeed);
    expect(redoState.players.map(p => p.hand.map(d => d.id))).toEqual(handsAfterRedeal);
  });
});