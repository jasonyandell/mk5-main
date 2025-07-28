import { describe, test, expect } from 'vitest';
import { getNextStates } from '../../game/core/actions';
import { createTestState } from '../helpers/gameTestHelper';

describe('Trick Winner Leads Next Trick', () => {
  test('trick winner becomes current player for next trick', () => {
    // Create a state with a completed 4-domino trick
    const state = createTestState({
      phase: 'playing',
      trump: 5, // 5s are trump
      currentTrick: [
        { player: 0, domino: { id: '6-4', high: 6, low: 4, points: 10 } }, // lead with 6s
        { player: 1, domino: { id: '6-3', high: 6, low: 3 } }, // follow suit
        { player: 2, domino: { id: '5-2', high: 5, low: 2 } }, // play trump - should win
        { player: 3, domino: { id: '6-1', high: 6, low: 1 } }  // follow suit
      ],
      currentPlayer: 0 // doesn't matter, trick is complete
    });

    // Get transitions - should include trick completion
    const transitions = getNextStates(state);
    
    // Find the complete-trick transition
    const completeTrickTransition = transitions.find(t => t.id === 'complete-trick');
    expect(completeTrickTransition).toBeDefined();

    // Verify that player 2 (who played trump) wins the trick
    const newState = completeTrickTransition!.newState;
    expect(newState.tricks).toHaveLength(1);
    expect(newState.tricks[0].winner).toBe(2); // Player 2 played trump
    
    // Verify that the trick winner (player 2) is now the current player
    expect(newState.currentPlayer).toBe(2);
    
    // Verify the current trick is cleared
    expect(newState.currentTrick).toHaveLength(0);
  });

  test('highest domino of led suit wins when no trump played', () => {
    const state = createTestState({
      phase: 'playing',
      trump: 1, // 1s are trump, but none played
      currentTrick: [
        { player: 0, domino: { id: '6-4', high: 6, low: 4, points: 10 } }, // lead with 6s - 10 points
        { player: 1, domino: { id: '6-5', high: 6, low: 5 } }, // higher 6 - should win
        { player: 2, domino: { id: '3-2', high: 3, low: 2, points: 5 } }, // off suit
        { player: 3, domino: { id: '6-2', high: 6, low: 2 } }  // lower 6
      ],
      currentPlayer: 0
    });

    const transitions = getNextStates(state);
    const completeTrickTransition = transitions.find(t => t.id === 'complete-trick');
    expect(completeTrickTransition).toBeDefined();

    const newState = completeTrickTransition!.newState;
    expect(newState.tricks[0].winner).toBe(1); // Player 1 had highest 6
    expect(newState.currentPlayer).toBe(1); // Player 1 leads next trick
  });

  test('first trump played wins over non-trump', () => {
    const state = createTestState({
      phase: 'playing',
      trump: 2, // 2s are trump
      currentTrick: [
        { player: 0, domino: { id: '6-4', high: 6, low: 4, points: 10 } }, // lead with 6s
        { player: 1, domino: { id: '2-0', high: 2, low: 0 } }, // play trump - should win
        { player: 2, domino: { id: '6-5', high: 6, low: 5 } }, // higher 6, but not trump
        { player: 3, domino: { id: '3-1', high: 3, low: 1 } }  // off suit
      ],
      currentPlayer: 0
    });

    const transitions = getNextStates(state);
    const completeTrickTransition = transitions.find(t => t.id === 'complete-trick');
    expect(completeTrickTransition).toBeDefined();

    const newState = completeTrickTransition!.newState;
    expect(newState.tricks[0].winner).toBe(1); // Player 1 played trump
    expect(newState.currentPlayer).toBe(1); // Player 1 leads next trick
  });

  test('higher trump beats lower trump', () => {
    const state = createTestState({
      phase: 'playing',
      trump: 3, // 3s are trump
      currentTrick: [
        { player: 0, domino: { id: '6-4', high: 6, low: 4, points: 10 } }, // lead with 6s
        { player: 1, domino: { id: '3-0', high: 3, low: 0 } }, // low trump
        { player: 2, domino: { id: '3-5', high: 3, low: 5 } }, // higher trump - should win
        { player: 3, domino: { id: '3-1', high: 3, low: 1 } }  // medium trump
      ],
      currentPlayer: 0
    });

    const transitions = getNextStates(state);
    const completeTrickTransition = transitions.find(t => t.id === 'complete-trick');
    expect(completeTrickTransition).toBeDefined();

    const newState = completeTrickTransition!.newState;
    expect(newState.tricks[0].winner).toBe(2); // Player 2 had highest trump
    expect(newState.currentPlayer).toBe(2); // Player 2 leads next trick
  });

  test('doubles trump works correctly', () => {
    const state = createTestState({
      phase: 'playing',
      trump: 7, // doubles are trump
      currentTrick: [
        { player: 0, domino: { id: '6-4', high: 6, low: 4, points: 10 } }, // lead with 6s
        { player: 1, domino: { id: '3-3', high: 3, low: 3 } }, // double (trump) - should win
        { player: 2, domino: { id: '6-5', high: 6, low: 5 } }, // higher 6, but not trump
        { player: 3, domino: { id: '2-1', high: 2, low: 1 } }  // off suit
      ],
      currentPlayer: 0
    });

    const transitions = getNextStates(state);
    const completeTrickTransition = transitions.find(t => t.id === 'complete-trick');
    expect(completeTrickTransition).toBeDefined();

    const newState = completeTrickTransition!.newState;
    expect(newState.tricks[0].winner).toBe(1); // Player 1 played doubles trump
    expect(newState.currentPlayer).toBe(1); // Player 1 leads next trick
  });

  test('multiple tricks - winner leads each time', () => {
    // Start with empty state
    let state = createTestState({
      phase: 'playing',
      trump: 4, // 4s are trump
      currentPlayer: 2, // Player 2 starts
      tricks: []
    });

    // Play first trick where player 3 wins
    state.currentTrick = [
      { player: 2, domino: { id: '6-5', high: 6, low: 5 } }, // lead
      { player: 3, domino: { id: '4-1', high: 4, low: 1 } }, // trump - wins
      { player: 0, domino: { id: '6-2', high: 6, low: 2 } }, // follow
      { player: 1, domino: { id: '3-0', high: 3, low: 0 } }  // off
    ];

    let transitions = getNextStates(state);
    let completeTrickTransition = transitions.find(t => t.id === 'complete-trick');
    expect(completeTrickTransition).toBeDefined();

    let newState = completeTrickTransition!.newState;
    expect(newState.tricks[0].winner).toBe(3);
    expect(newState.currentPlayer).toBe(3); // Player 3 leads next trick

    // Play second trick where player 0 wins
    newState.currentTrick = [
      { player: 3, domino: { id: '5-2', high: 5, low: 2, points: 5 } }, // lead
      { player: 0, domino: { id: '4-3', high: 4, low: 3 } }, // trump - wins
      { player: 1, domino: { id: '5-1', high: 5, low: 1 } }, // follow
      { player: 2, domino: { id: '2-0', high: 2, low: 0 } }  // off
    ];

    transitions = getNextStates(newState);
    completeTrickTransition = transitions.find(t => t.id === 'complete-trick');
    expect(completeTrickTransition).toBeDefined();

    const finalState = completeTrickTransition!.newState;
    expect(finalState.tricks).toHaveLength(2);
    expect(finalState.tricks[1].winner).toBe(0);
    expect(finalState.currentPlayer).toBe(0); // Player 0 leads next trick
  });
});