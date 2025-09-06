import { describe, test, expect } from 'vitest';
import { getNextStates } from '../../game/core/gameEngine';
import { executeAction } from '../../game/core/actions';
import { createInitialState } from '../../game/core/state';
import { prepareDeterministicHand } from '../../stores/utils/sectionHelpers';
import type { GameAction, StateTransition } from '../../game/types';

// Type guard for agree-complete-trick transitions
function isAgreeCompleteTrick(
  t: StateTransition
): t is StateTransition & { action: Extract<GameAction, { type: 'agree-complete-trick' }> } {
  return t.action.type === 'agree-complete-trick';
}

describe('Consensus Action Flow', () => {
  test('engine offers consensus actions when trick is complete', async () => {
    // Start from a deterministic state
    let state = await prepareDeterministicHand(424242);
    
    // Play cards to complete a trick
    // This simulates the actual game flow
    while (state.currentTrick.length < 4 && state.phase === 'playing') {
      const transitions = getNextStates(state);
      const playAction = transitions.find(t => t.action.type === 'play');
      if (playAction) {
        state = executeAction(state, playAction.action);
      } else {
        break;
      }
    }
    
    // When trick has 4 cards, engine should offer agree actions
    expect(state.currentTrick).toHaveLength(4);
    
    const transitions = getNextStates(state);
    const agreeActions = transitions.filter(isAgreeCompleteTrick);
    
    // Should offer 4 agree actions (one per player)
    expect(agreeActions).toHaveLength(4);
    expect(agreeActions.map(a => a.action.player).sort()).toEqual([0, 1, 2, 3]);
  });

  test('consensus actions are processed one at a time', async () => {
    let state = await prepareDeterministicHand(424242);
    
    // Complete a trick
    while (state.currentTrick.length < 4 && state.phase === 'playing') {
      const transitions = getNextStates(state);
      const playAction = transitions.find(t => t.action.type === 'play');
      if (playAction) {
        state = executeAction(state, playAction.action);
      } else {
        break;
      }
    }
    
    const executedAgrees: number[] = [];
    
    // Process agrees one by one
    while (state.consensus.completeTrick.size < 4) {
      const transitions = getNextStates(state);
      const agreeAction = transitions
        .filter(isAgreeCompleteTrick)
        .find(t => !executedAgrees.includes(t.action.player));
      
      if (!agreeAction) break;
      
      executedAgrees.push(agreeAction.action.player);
      state = executeAction(state, agreeAction.action);
      
      // Verify state is updated correctly
      expect(state.consensus.completeTrick.has(agreeAction.action.player)).toBe(true);
    }
    
    // All 4 players should have agreed
    expect(state.consensus.completeTrick.size).toBe(4);
    expect(executedAgrees).toHaveLength(4);
  });

  test('complete-trick action available only after all agrees', async () => {
    let state = await prepareDeterministicHand(424242);
    
    // Complete a trick
    while (state.currentTrick.length < 4 && state.phase === 'playing') {
      const transitions = getNextStates(state);
      const playAction = transitions.find(t => t.action.type === 'play');
      if (playAction) {
        state = executeAction(state, playAction.action);
      } else {
        break;
      }
    }
    
    // Before any agrees, no complete-trick action
    let transitions = getNextStates(state);
    expect(transitions.find(t => t.action.type === 'complete-trick')).toBeUndefined();
    
    // Add 3 agrees
    for (let i = 0; i < 3; i++) {
      transitions = getNextStates(state);
      const agreeAction = transitions
        .filter(isAgreeCompleteTrick)
        .find(t => !state.consensus.completeTrick.has(t.action.player));
      if (agreeAction) {
        state = executeAction(state, agreeAction.action);
      }
    }
    
    // Still no complete-trick (only 3 agrees)
    transitions = getNextStates(state);
    expect(transitions.find(t => t.action.type === 'complete-trick')).toBeUndefined();
    
    // Add 4th agree
    const lastAgree = transitions
      .filter(isAgreeCompleteTrick)
      .find(t => !state.consensus.completeTrick.has(t.action.player));
    if (lastAgree) {
      state = executeAction(state, lastAgree.action);
    }
    
    // Now complete-trick should be available
    transitions = getNextStates(state);
    const completeTrick = transitions.find(t => t.action.type === 'complete-trick');
    expect(completeTrick).toBeDefined();
    
    // Execute complete-trick
    if (completeTrick) {
      state = executeAction(state, completeTrick.action);
      expect(state.currentTrick).toHaveLength(0);
      expect(state.consensus.completeTrick.size).toBe(0);
    }
  });

  test('duplicate agrees are an error', async () => {
    let state = await prepareDeterministicHand(424242);
    
    // Complete a trick
    while (state.currentTrick.length < 4 && state.phase === 'playing') {
      const transitions = getNextStates(state);
      const playAction = transitions.find(t => t.action.type === 'play');
      if (playAction) {
        state = executeAction(state, playAction.action);
      } else {
        break;
      }
    }
    
    // Add player 0's agree
    const transitions = getNextStates(state);
    const agree0 = transitions.find(t => 
      t.action.type === 'agree-complete-trick' && 
      t.action.player === 0
    );
    
    if (agree0) {
      state = executeAction(state, agree0.action);
      expect(state.consensus.completeTrick.has(0)).toBe(true);
      expect(state.consensus.completeTrick.size).toBe(1);
      
      // Try to add player 0's agree again
      const duplicateAgree: GameAction = {
        type: 'agree-complete-trick',
        player: 0
      };
      
      // should throw
      expect(() => executeAction(state, duplicateAgree)).toThrowError();
    }
  });

  test('consensus flow for scoring phase', async () => {
    // Create a state in scoring phase
    let state = createInitialState();
    state.phase = 'scoring';
    state.trump = { type: 'suit', suit: 3 };
    
    // Get available actions in scoring
    const transitions = getNextStates(state);
    const scoreAgrees = transitions.filter(t => t.action.type === 'agree-score-hand');
    
    // Should offer 4 agree-score-hand actions
    expect(scoreAgrees).toHaveLength(4);
    
    // Process all agrees
    for (const agree of scoreAgrees) {
      state = executeAction(state, agree.action);
    }
    
    // After all agrees, score-hand should be available
    const finalTransitions = getNextStates(state);
    const scoreHand = finalTransitions.find(t => t.action.type === 'score-hand');
    expect(scoreHand).toBeDefined();
    
    // Execute score-hand
    if (scoreHand) {
      state = executeAction(state, scoreHand.action);
      // Should advance to next hand or game_end
      expect(['bidding', 'game_end']).toContain(state.phase);
    }
  });

  test('section runner consensus helper pattern', async () => {
    let state = await prepareDeterministicHand(424242);
    
    // Complete a trick
    while (state.currentTrick.length < 4 && state.phase === 'playing') {
      const transitions = getNextStates(state);
      const playAction = transitions.find(t => t.action.type === 'play');
      if (playAction) {
        state = executeAction(state, playAction.action);
      } else {
        break;
      }
    }
    
    // Simulate what section runner should do
    const humanPlayers = new Set([0]);
    const processedAgrees = new Set<number>();
    
    // Process non-human agrees one at a time
    while (processedAgrees.size < 3) { // 3 non-human players
      const transitions = getNextStates(state);
      
      // Find next non-human agree that hasn't been processed
      const nextAgree = transitions
        .filter(isAgreeCompleteTrick)
        .find(t => !humanPlayers.has(t.action.player) && !processedAgrees.has(t.action.player));
      
      if (!nextAgree) break;
      
      processedAgrees.add(nextAgree.action.player);
      state = executeAction(state, nextAgree.action);
    }
    
    // Should have processed 3 non-human agrees
    expect(processedAgrees.size).toBe(3);
    expect(state.consensus.completeTrick.size).toBe(3);
    
    // Human's agree should still be available
    const finalTransitions = getNextStates(state);
    const humanAgree = finalTransitions.find(t => 
      t.action.type === 'agree-complete-trick' && 
      t.action.player === 0
    );
    expect(humanAgree).toBeDefined();
  });
});
