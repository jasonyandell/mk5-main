import { describe, test, expect } from 'vitest';
import { createTestContext } from '../helpers/executionContext';
import { getNextStates } from '../../game/core/state';
import { StateBuilder } from '../helpers';
import { ACES, DEUCES, TRES, FOURS, FIVES } from '../../game/types';

describe('Trick Winner Leads Next Trick', () => {
  // Use all human players so consensus layer generates agree actions for all
  const ctx = createTestContext({ layers: ['consensus'], playerTypes: ['human', 'human', 'human', 'human'] });

  test('trick winner becomes current player for next trick', () => {
    // Create a state with a completed 4-domino trick (all human players)
    const state = StateBuilder.inPlayingPhase({ type: 'suit', suit: FIVES })
      .withConfig({ playerTypes: ['human', 'human', 'human', 'human'] })
      .withCurrentTrick([
        { player: 0, domino: { id: '6-4', high: 6, low: 4, points: 10 } }, // lead with 6s
        { player: 1, domino: { id: '6-3', high: 6, low: 3 } }, // follow suit
        { player: 2, domino: { id: '5-2', high: 5, low: 2 } }, // play trump - should win
        { player: 3, domino: { id: '6-1', high: 6, low: 1 } }  // follow suit
      ])
      .withCurrentPlayer(0) // doesn't matter, trick is complete
      .build();

    // Get transitions - should include agreement actions first
    let transitions = getNextStates(state, ctx);

    // All 4 human players must agree to complete the trick
    let currentState = state;
    for (let i = 0; i < 4; i++) {
      const agreeAction = transitions.find(t =>
        t.action.type === 'agree-trick' &&
        t.action.player === i
      );
      expect(agreeAction).toBeDefined();
      currentState = agreeAction!.newState;
      transitions = getNextStates(currentState, ctx);
    }

    // Now the complete-trick action should be available
    const completeTrickTransition = transitions.find(t => t.id === 'complete-trick');
    expect(completeTrickTransition).toBeDefined();

    // Verify that player 2 (who played trump) wins the trick
    const newState = completeTrickTransition!.newState;
    expect(newState.tricks).toHaveLength(1);
    expect(newState.tricks[0]!.winner).toBe(2); // Player 2 played trump

    // Verify that the trick winner (player 2) is now the current player
    expect(newState.currentPlayer).toBe(2);

    // Verify the current trick is cleared
    expect(newState.currentTrick).toHaveLength(0);
  });

  test('highest domino of led suit wins when no trump played', () => {
    const state = StateBuilder.inPlayingPhase({ type: 'suit', suit: ACES })
      .withConfig({ playerTypes: ['human', 'human', 'human', 'human'] })
      .withCurrentTrick([
        { player: 0, domino: { id: '6-4', high: 6, low: 4, points: 10 } }, // lead with 6s - 10 points
        { player: 1, domino: { id: '6-5', high: 6, low: 5 } }, // higher 6 - should win
        { player: 2, domino: { id: '3-2', high: 3, low: 2, points: 5 } }, // off suit
        { player: 3, domino: { id: '6-2', high: 6, low: 2 } }  // lower 6
      ])
      .withCurrentPlayer(0)
      .build();

    // First, all players must agree to complete the trick sequentially
    let transitions = getNextStates(state, ctx);
    let currentState = state;
    for (let i = 0; i < 4; i++) {
      const agreeAction = transitions.find(t => 
        t.action.type === 'agree-trick' &&
        t.action.player === i
      );
      expect(agreeAction).toBeDefined();
      currentState = agreeAction!.newState;
      transitions = getNextStates(currentState, ctx);
    }
    
    const completeTrickTransition = transitions.find(t => t.id === 'complete-trick');
    expect(completeTrickTransition).toBeDefined();

    const newState = completeTrickTransition!.newState;
    expect(newState.tricks[0]!.winner).toBe(1); // Player 1 had highest 6
    expect(newState.currentPlayer).toBe(1); // Player 1 leads next trick
  });

  test('first trump played wins over non-trump', () => {
    const state = StateBuilder.inPlayingPhase({ type: 'suit', suit: DEUCES })
      .withConfig({ playerTypes: ['human', 'human', 'human', 'human'] })
      .withCurrentTrick([
        { player: 0, domino: { id: '6-4', high: 6, low: 4, points: 10 } }, // lead with 6s
        { player: 1, domino: { id: '2-0', high: 2, low: 0 } }, // play trump - should win
        { player: 2, domino: { id: '6-5', high: 6, low: 5 } }, // higher 6, but not trump
        { player: 3, domino: { id: '3-1', high: 3, low: 1 } }  // off suit
      ])
      .withCurrentPlayer(0)
      .build();

    // First, all players must agree to complete the trick sequentially
    let transitions = getNextStates(state, ctx);
    let currentState = state;
    for (let i = 0; i < 4; i++) {
      const agreeAction = transitions.find(t => 
        t.action.type === 'agree-trick' &&
        t.action.player === i
      );
      expect(agreeAction).toBeDefined();
      currentState = agreeAction!.newState;
      transitions = getNextStates(currentState, ctx);
    }
    
    const completeTrickTransition = transitions.find(t => t.id === 'complete-trick');
    expect(completeTrickTransition).toBeDefined();

    const newState = completeTrickTransition!.newState;
    expect(newState.tricks[0]!.winner).toBe(1); // Player 1 played trump
    expect(newState.currentPlayer).toBe(1); // Player 1 leads next trick
  });

  test('higher trump beats lower trump', () => {
    const state = StateBuilder.inPlayingPhase({ type: 'suit', suit: TRES })
      .withConfig({ playerTypes: ['human', 'human', 'human', 'human'] })
      .withCurrentTrick([
        { player: 0, domino: { id: '6-4', high: 6, low: 4, points: 10 } }, // lead with 6s
        { player: 1, domino: { id: '3-0', high: 3, low: 0 } }, // low trump
        { player: 2, domino: { id: '3-5', high: 3, low: 5 } }, // higher trump - should win
        { player: 3, domino: { id: '3-1', high: 3, low: 1 } }  // medium trump
      ])
      .withCurrentPlayer(0)
      .build();

    // First, all players must agree to complete the trick sequentially
    let transitions = getNextStates(state, ctx);
    let currentState = state;
    for (let i = 0; i < 4; i++) {
      const agreeAction = transitions.find(t => 
        t.action.type === 'agree-trick' &&
        t.action.player === i
      );
      expect(agreeAction).toBeDefined();
      currentState = agreeAction!.newState;
      transitions = getNextStates(currentState, ctx);
    }
    
    const completeTrickTransition = transitions.find(t => t.id === 'complete-trick');
    expect(completeTrickTransition).toBeDefined();

    const newState = completeTrickTransition!.newState;
    expect(newState.tricks[0]!.winner).toBe(2); // Player 2 had highest trump
    expect(newState.currentPlayer).toBe(2); // Player 2 leads next trick
  });

  test('doubles trump works correctly', () => {
    const state = StateBuilder.inPlayingPhase({ type: 'doubles' })
      .withConfig({ playerTypes: ['human', 'human', 'human', 'human'] })
      .withCurrentTrick([
        { player: 0, domino: { id: '6-4', high: 6, low: 4, points: 10 } }, // lead with 6s
        { player: 1, domino: { id: '3-3', high: 3, low: 3 } }, // double (trump) - should win
        { player: 2, domino: { id: '6-5', high: 6, low: 5 } }, // higher 6, but not trump
        { player: 3, domino: { id: '2-1', high: 2, low: 1 } }  // off suit
      ])
      .withCurrentPlayer(0)
      .build();

    // First, all players must agree to complete the trick sequentially
    let transitions = getNextStates(state, ctx);
    let currentState = state;
    for (let i = 0; i < 4; i++) {
      const agreeAction = transitions.find(t => 
        t.action.type === 'agree-trick' &&
        t.action.player === i
      );
      expect(agreeAction).toBeDefined();
      currentState = agreeAction!.newState;
      transitions = getNextStates(currentState, ctx);
    }
    
    const completeTrickTransition = transitions.find(t => t.id === 'complete-trick');
    expect(completeTrickTransition).toBeDefined();

    const newState = completeTrickTransition!.newState;
    expect(newState.tricks[0]!.winner).toBe(1); // Player 1 played doubles trump
    expect(newState.currentPlayer).toBe(1); // Player 1 leads next trick
  });

  test('multiple tricks - winner leads each time', () => {
    // Start with empty state (all human players)
    let state = StateBuilder.inPlayingPhase({ type: 'suit', suit: FOURS })
      .withConfig({ playerTypes: ['human', 'human', 'human', 'human'] })
      .withCurrentPlayer(2) // Player 2 starts
      .withTricks([])
      .build();

    // Play first trick where player 3 wins
    state.currentTrick = [
      { player: 2, domino: { id: '6-5', high: 6, low: 5 } }, // lead
      { player: 3, domino: { id: '4-1', high: 4, low: 1 } }, // trump - wins
      { player: 0, domino: { id: '6-2', high: 6, low: 2 } }, // follow
      { player: 1, domino: { id: '3-0', high: 3, low: 0 } }  // off
    ];

    // First, all players must agree to complete the trick sequentially
    let transitions = getNextStates(state, ctx);
    let currentState = state;
    for (let i = 0; i < 4; i++) {
      const agreeAction = transitions.find(t => 
        t.action.type === 'agree-trick' &&
        t.action.player === i
      );
      expect(agreeAction).toBeDefined();
      currentState = agreeAction!.newState;
      transitions = getNextStates(currentState, ctx);
    }
    
    let completeTrickTransition = transitions.find(t => t.id === 'complete-trick');
    expect(completeTrickTransition).toBeDefined();

    let newState = completeTrickTransition!.newState;
    expect(newState.tricks[0]!.winner).toBe(3);
    expect(newState.currentPlayer).toBe(3); // Player 3 leads next trick

    // Play second trick where player 0 wins
    newState.currentTrick = [
      { player: 3, domino: { id: '5-2', high: 5, low: 2, points: 5 } }, // lead
      { player: 0, domino: { id: '4-3', high: 4, low: 3 } }, // trump - wins
      { player: 1, domino: { id: '5-1', high: 5, low: 1 } }, // follow
      { player: 2, domino: { id: '2-0', high: 2, low: 0 } }  // off
    ];

    // Second trick: all players agree again sequentially
    transitions = getNextStates(newState, ctx);
    currentState = newState;
    for (let i = 0; i < 4; i++) {
      const agreeAction = transitions.find(t => 
        t.action.type === 'agree-trick' &&
        t.action.player === i
      );
      expect(agreeAction).toBeDefined();
      currentState = agreeAction!.newState;
      transitions = getNextStates(currentState, ctx);
    }
    
    completeTrickTransition = transitions.find(t => t.id === 'complete-trick');
    expect(completeTrickTransition).toBeDefined();

    const finalState = completeTrickTransition!.newState;
    expect(finalState.tricks).toHaveLength(2);
    expect(finalState.tricks[1]!.winner).toBe(0);
    expect(finalState.currentPlayer).toBe(0); // Player 0 leads next trick
  });
});