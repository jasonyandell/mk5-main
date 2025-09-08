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
    
    // Should offer only 1 agree action (for current player)
    expect(agreeActions).toHaveLength(1);
    expect(agreeActions[0]!.action.player).toBe(state.currentPlayer);
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
    
    // Process agrees sequentially (each player in turn)
    while (state.consensus.completeTrick.size < 4) {
      const transitions = getNextStates(state);
      const agreeAction = transitions.find(t => 
        t.action.type === 'agree-complete-trick' &&
        'player' in t.action &&
        t.action.player === state.currentPlayer
      );
      
      if (!agreeAction) break;
      if (!('player' in agreeAction.action)) break;
      
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
    
    // Add 3 agrees sequentially
    for (let i = 0; i < 3; i++) {
      transitions = getNextStates(state);
      const agreeAction = transitions.find(t => 
        t.action.type === 'agree-complete-trick' &&
        t.action.player === state.currentPlayer
      );
      if (agreeAction) {
        state = executeAction(state, agreeAction.action);
      }
    }
    
    // Still no complete-trick (only 3 agrees)
    transitions = getNextStates(state);
    expect(transitions.find(t => t.action.type === 'complete-trick')).toBeUndefined();
    
    // Add 4th agree
    const lastAgree = transitions.find(t => 
      t.action.type === 'agree-complete-trick' &&
      t.action.player === state.currentPlayer
    );
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
    
    // Process agrees until we get to player 0
    let processedPlayers = 0;
    while (state.currentPlayer !== 0 && processedPlayers < 4) {
      const transitions = getNextStates(state);
      const agreeAction = transitions.find(t => 
        t.action.type === 'agree-complete-trick' &&
        t.action.player === state.currentPlayer
      );
      if (agreeAction) {
        state = executeAction(state, agreeAction.action);
        processedPlayers++;
      } else {
        break;
      }
    }
    
    // Now player 0 should be current player
    expect(state.currentPlayer).toBe(0);
    
    const transitions = getNextStates(state);
    const agree0 = transitions.find(t => 
      t.action.type === 'agree-complete-trick' && 
      t.action.player === 0
    );
    
    expect(agree0).toBeDefined();
    if (agree0) {
      // Record how many players agreed before player 0
      const agreeCountBefore = state.consensus.completeTrick.size;
      
      state = executeAction(state, agree0.action);
      expect(state.consensus.completeTrick.has(0)).toBe(true);
      expect(state.consensus.completeTrick.size).toBe(agreeCountBefore + 1);
      
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
    
    // Should offer only 1 agree-score-hand action (for current player)
    expect(scoreAgrees).toHaveLength(1);
    expect('player' in scoreAgrees[0]!.action && scoreAgrees[0]!.action.player).toBe(state.currentPlayer);
    
    // Process all agrees sequentially
    for (let i = 0; i < 4; i++) {
      const transitions = getNextStates(state);
      const agree = transitions.find(t => 
        t.action.type === 'agree-score-hand' &&
        t.action.player === state.currentPlayer
      );
      if (agree) {
        state = executeAction(state, agree.action);
      }
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
    
    // Simulate what section runner should do - process agrees sequentially
    const humanPlayers = new Set([0]);
    let processedCount = 0;
    
    // Process agrees sequentially until we reach human player
    while (state.consensus.completeTrick.size < 4) {
      const transitions = getNextStates(state);
      const agreeAction = transitions.find(t => 
        t.action.type === 'agree-complete-trick' &&
        t.action.player === state.currentPlayer
      );
      
      if (!agreeAction) break;
      
      // If current player is human, stop (let human decide)
      if (humanPlayers.has(state.currentPlayer)) {
        break;
      }
      
      // Non-human player agrees
      state = executeAction(state, agreeAction.action);
      processedCount++;
    }
    
    // Should have processed some non-human agrees
    expect(processedCount).toBeGreaterThan(0);
    expect(state.consensus.completeTrick.size).toBe(processedCount);
    
    // If we stopped at human player, their agree should be available
    if (state.currentPlayer === 0) {
      const finalTransitions = getNextStates(state);
      const humanAgree = finalTransitions.find(t => 
        t.action.type === 'agree-complete-trick' && 
        t.action.player === 0
      );
      expect(humanAgree).toBeDefined();
    }
  });
});
