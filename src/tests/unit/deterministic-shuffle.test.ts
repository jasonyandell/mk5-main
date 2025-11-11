import { describe, it, expect } from 'vitest';
import { createInitialState, getNextStates } from '../../game';
import { createTestContext } from '../helpers/executionContext';
import type { GameState, StateTransition } from '../../game/types';

describe('Deterministic Shuffle with Undo/Redo', () => {
  
  it('should maintain deterministic shuffling after undo/redo', () => {
  
    const ctx = createTestContext();
    // Create initial state
    const initialState = createInitialState();
    const initialSeed = initialState.shuffleSeed;
    
    // Store initial hands for comparison
    const initialHands = initialState.players.map(p => [...p.hand]);
    
    // Make all players pass to trigger a redeal
    let currentState = initialState;
    const passActions: StateTransition[] = [];
    
    for (let i = 0; i < 4; i++) {
      const transitions = getNextStates(currentState, ctx);
      const passAction = transitions.find(t => t.label === 'Pass');
      expect(passAction).toBeDefined();
      
      passActions.push(passAction!);
      currentState = passAction!.newState;
    }
    
    // After 4 passes, we should have a redeal action available
    expect(currentState.phase).toBe('bidding');
    expect(currentState.bids.length).toBe(4);
    
    // Get the redeal action
    const redealTransitions = getNextStates(currentState, ctx);
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
      hand.some((domino, idx) => domino.id !== initialHands[playerIdx]?.[idx]?.id)
    );
    expect(handsChanged).toBe(true);
    
    // Now replay from initial state - should get exact same results
    let replayState = initialState;
    for (const action of passActions) {
      const transitions = getNextStates(replayState, ctx);
      const matchingAction = transitions.find(t => t.id === action.id);
      expect(matchingAction).toBeDefined();
      replayState = matchingAction!.newState;
    }
    
    // Verify replay produces identical state
    expect(replayState.shuffleSeed).toBe(currentState.shuffleSeed);
    expect(replayState.players.map(p => p.hand)).toEqual(currentState.players.map(p => p.hand));
  });

  it('should produce same shuffle result with same seed', () => {

    const ctx = createTestContext();
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
      const transitions = getNextStates(current, ctx);
      const passAction = transitions.find(t => t.label === 'Pass');
      current = passAction!.newState;
    }
    
    const hands1 = current.players.map(p => p.hand.map(d => d.id));
    
    // Do it again with same starting seed
    current = stateWithSeed;
    for (let i = 0; i < 4; i++) {
      const transitions = getNextStates(current, ctx);
      const passAction = transitions.find(t => t.label === 'Pass');
      if (!passAction) throw new Error('Pass action not found');
      current = passAction.newState;
    }
    
    const hands2 = current.players.map(p => p.hand.map(d => d.id));
    
    // Should produce identical hands
    expect(hands1).toEqual(hands2);
  });

  it('should handle multiple redeals deterministically', () => {

    const ctx = createTestContext();
    const initialState = createInitialState();
    const states: GameState[] = [initialState];
    
    // Perform 3 rounds of all-pass redeals
    let currentState = initialState;
    
    for (let round = 0; round < 3; round++) {
      // All 4 players pass
      for (let player = 0; player < 4; player++) {
        const transitions = getNextStates(currentState, ctx);
        const passAction = transitions.find(t => t.label === 'Pass');
        currentState = passAction!.newState;
      }
      
      // Execute the redeal
      const redealTransitions = getNextStates(currentState, ctx);
      const redealAction = redealTransitions.find(t => t.id === 'redeal');
      if (!redealAction) throw new Error('Redeal action not found');
      currentState = redealAction.newState;
      states.push(currentState);
      
      // Verify seed increments
      expect(currentState.shuffleSeed).toBeGreaterThan(
        states[states.length - 2]?.shuffleSeed ?? 0
      );
    }
    
    // Replay the same sequence
    let replayState = initialState;
    const replayStates: GameState[] = [replayState];
    
    for (let round = 0; round < 3; round++) {
      for (let player = 0; player < 4; player++) {
        const transitions = getNextStates(replayState, ctx);
        const passAction = transitions.find(t => t.label === 'Pass');
        replayState = passAction!.newState;
      }
      
      // Execute the redeal
      const redealTransitions = getNextStates(replayState, ctx);
      const redealAction = redealTransitions.find(t => t.id === 'redeal');
      if (!redealAction) throw new Error('Redeal action not found');
      replayState = redealAction.newState;
      replayStates.push(replayState);
    }
    
    // All states should match exactly
    for (let i = 0; i < states.length; i++) {
      expect(replayStates[i]?.shuffleSeed).toBe(states[i]?.shuffleSeed);
      expect(replayStates[i]?.players.map(p => p.hand)).toEqual(
        states[i]?.players.map(p => p.hand)
      );
    }
  });

  it('should maintain determinism through complete hand with scoring', () => {

    const ctx = createTestContext();
    const initialState = createInitialState();
    
    // Play a complete hand: bid, set trump, play all tricks
    let currentState = initialState;
    const actions: StateTransition[] = [];
    
    // First player bids 30
    let transitions = getNextStates(currentState, ctx);
    // console.log('Available transitions:', transitions.map(t => ({ id: t.id, label: t.label })));
    let bidAction = transitions.find(t => t.label === '30' || t.label === 'Bid 30' || t.label === 'Bid 30 points');
    expect(bidAction).toBeDefined();
    actions.push(bidAction!);
    currentState = bidAction!.newState;
    
    // Other players pass
    for (let i = 0; i < 3; i++) {
      transitions = getNextStates(currentState, ctx);
      const passAction = transitions.find(t => t.label === 'Pass');
      if (!passAction) throw new Error('Pass action not found');
      actions.push(passAction);
      currentState = passAction.newState;
    }
    
    // Set trump
    transitions = getNextStates(currentState, ctx);
    const trumpAction = transitions[0]!; // Just pick first available trump
    actions.push(trumpAction);
    currentState = trumpAction.newState;
    
    // Play all tricks (just play first available domino each time)
    while (currentState.phase === 'playing') {
      transitions = getNextStates(currentState, ctx);
      if (transitions.length > 0 && transitions[0]) {
        actions.push(transitions[0]);
        currentState = transitions[0].newState;
      } else {
        break;
      }
    }
    
    // Should be in scoring phase
    expect(currentState.phase).toBe('scoring');
    
    // Score the hand - all players must agree sequentially
    for (let i = 0; i < 4; i++) {
      transitions = getNextStates(currentState, ctx);
      const agreeAction = transitions.find(t => 
        t.action.type === 'agree-score-hand' &&
        t.action.player === currentState.currentPlayer
      );
      if (!agreeAction) throw new Error(`Agree score action for current player ${currentState.currentPlayer} not found`);
      actions.push(agreeAction);
      currentState = agreeAction.newState;
    }
    
    // Now the score-hand action should be available
    transitions = getNextStates(currentState, ctx);
    const scoreAction = transitions.find(t => t.id === 'score-hand');
    if (!scoreAction) throw new Error('Score hand action not found');
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
      transitions = getNextStates(replayState, ctx);
      const matchingAction = transitions.find(t => t.id === action.id);
      expect(matchingAction).toBeDefined();
      replayState = matchingAction!.newState;
    }
    
    // Final state should match exactly
    expect(replayState.shuffleSeed).toBe(currentState.shuffleSeed);
    expect(replayState.players.map(p => p.hand.map(d => d.id))).toEqual(newHands);
  });

  it('should handle undo across redeals correctly', () => {

    const ctx = createTestContext();
    const initialState = createInitialState();
    const stateHistory: GameState[] = [initialState];
    
    // Make moves and track states
    let currentState = initialState;
    
    // All pass to trigger redeal
    for (let i = 0; i < 4; i++) {
      const transitions = getNextStates(currentState, ctx);
      const passAction = transitions.find(t => t.label === 'Pass');
      currentState = passAction!.newState;
      stateHistory.push(currentState);
    }
    
    // Execute the redeal
    let transitions = getNextStates(currentState, ctx);
    const redealAction = transitions.find(t => t.id === 'redeal');
    expect(redealAction).toBeDefined();
    currentState = redealAction!.newState;
    stateHistory.push(currentState);
    
    const stateAfterRedeal = currentState;
    const handsAfterRedeal = currentState.players.map(p => p.hand.map(d => d.id));
    
    // Make a bid after redeal
    transitions = getNextStates(currentState, ctx);
    const bidAction = transitions.find(t => t.label === '30' || t.label === 'Bid 30' || t.label === 'Bid 30 points');
    expect(bidAction).toBeDefined();
    currentState = bidAction!.newState;
    stateHistory.push(currentState);
    
    // Simulate undo back to before redeal
    const stateBeforeRedeal = stateHistory[3]; // State after 3 passes
    if (!stateBeforeRedeal) throw new Error('State not found in history');
    expect(stateBeforeRedeal.shuffleSeed).toBe(initialState.shuffleSeed);
    
    // Redo the last pass
    const redoTransitions = getNextStates(stateBeforeRedeal, ctx);
    const redoPass = redoTransitions.find(t => t.label === 'Pass');
    if (!redoPass) throw new Error('Pass action not found');
    const afterFourthPass = redoPass.newState;
    
    // Now get the redeal action
    const afterPassTransitions = getNextStates(afterFourthPass, ctx);
    const redoRedeal = afterPassTransitions.find(t => t.id === 'redeal');
    if (!redoRedeal) throw new Error('Redeal action not found');
    const redoState = redoRedeal.newState;
    
    // Should match the state after redeal exactly
    expect(redoState.shuffleSeed).toBe(stateAfterRedeal.shuffleSeed);
    expect(redoState.players.map((p) => p.hand.map((d) => d.id))).toEqual(handsAfterRedeal);
  });
});