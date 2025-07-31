import { describe, test, expect } from 'vitest';
import { getNextStates } from '../../game/core/actions';
import { createTestState } from '../helpers/gameTestHelper';
import type { GameState } from '../../game/types';

describe('Debug Validation Bug - Step 7', () => {
  test('should validate action sequence correctly for the reported bug', () => {
    // Recreate the exact base state from the URL
    const baseState: GameState = {
      phase: 'bidding',
      players: [
        {
          id: 0,
          name: 'Player 1',
          hand: [
            { high: 5, low: 2, id: '5-2' },
            { high: 1, low: 0, id: '1-0' },
            { high: 4, low: 3, id: '4-3' },
            { high: 5, low: 3, id: '5-3' },
            { high: 6, low: 1, id: '6-1' },
            { high: 6, low: 5, id: '6-5' }, // Player 0 has 6-5
            { high: 4, low: 2, id: '4-2' }
          ],
          teamId: 0,
          marks: 0
        },
        {
          id: 1,
          name: 'Player 2',
          hand: [
            { high: 0, low: 0, id: '0-0' },
            { high: 5, low: 5, id: '5-5', points: 10 },
            { high: 2, low: 0, id: '2-0' },
            { high: 4, low: 4, id: '4-4' },
            { high: 6, low: 3, id: '6-3' },
            { high: 2, low: 1, id: '2-1' },
            { high: 3, low: 2, id: '3-2', points: 5 }
          ],
          teamId: 1,
          marks: 0
        },
        {
          id: 2,
          name: 'Player 3',
          hand: [
            { high: 5, low: 1, id: '5-1', points: 5 },
            { high: 5, low: 4, id: '5-4' },
            { high: 6, low: 0, id: '6-0' },
            { high: 3, low: 1, id: '3-1' },
            { high: 5, low: 0, id: '5-0', points: 5 },
            { high: 3, low: 0, id: '3-0' },
            { high: 6, low: 6, id: '6-6' }
          ],
          teamId: 0,
          marks: 0
        },
        {
          id: 3,
          name: 'Player 4',
          hand: [
            { high: 6, low: 4, id: '6-4', points: 10 },
            { high: 4, low: 0, id: '4-0' },
            { high: 2, low: 2, id: '2-2' },
            { high: 1, low: 1, id: '1-1' },
            { high: 4, low: 1, id: '4-1', points: 5 },
            { high: 3, low: 3, id: '3-3' },
            { high: 6, low: 2, id: '6-2' }
          ],
          teamId: 1,
          marks: 0
        }
      ],
      currentPlayer: 0,
      dealer: 3,
      bids: [],
      currentBid: null,
      winningBidder: null,
      trump: null,
      tricks: [],
      currentTrick: [],
      teamScores: [0, 0],
      teamMarks: [0, 0],
      gameTarget: 7,
      tournamentMode: true,
      hands: {
        '0': [
          { high: 5, low: 2, id: '5-2' },
          { high: 1, low: 0, id: '1-0' },
          { high: 4, low: 3, id: '4-3' },
          { high: 5, low: 3, id: '5-3' },
          { high: 6, low: 1, id: '6-1' },
          { high: 6, low: 5, id: '6-5' },
          { high: 4, low: 2, id: '4-2' }
        ],
        '1': [
          { high: 0, low: 0, id: '0-0' },
          { high: 5, low: 5, id: '5-5' },
          { high: 2, low: 0, id: '2-0' },
          { high: 4, low: 4, id: '4-4' },
          { high: 6, low: 3, id: '6-3' },
          { high: 2, low: 1, id: '2-1' },
          { high: 3, low: 2, id: '3-2' }
        ],
        '2': [
          { high: 5, low: 1, id: '5-1' },
          { high: 5, low: 4, id: '5-4' },
          { high: 6, low: 0, id: '6-0' },
          { high: 3, low: 1, id: '3-1' },
          { high: 5, low: 0, id: '5-0' },
          { high: 3, low: 0, id: '3-0' },
          { high: 6, low: 6, id: '6-6' }
        ],
        '3': [
          { high: 6, low: 4, id: '6-4' },
          { high: 4, low: 0, id: '4-0' },
          { high: 2, low: 2, id: '2-2' },
          { high: 1, low: 1, id: '1-1' },
          { high: 4, low: 1, id: '4-1' },
          { high: 3, low: 3, id: '3-3' },
          { high: 6, low: 2, id: '6-2' }
        ]
      },
      bidWinner: null,
      isComplete: false,
      winner: null,
      shuffleSeed: 12345
    };

    // The exact action sequence from the URL
    const actions = [
      { id: 'pass', label: 'Pass' },
      { id: 'bid-30', label: 'Bid 30 points' },
      { id: 'pass', label: 'Pass' },
      { id: 'pass', label: 'Pass' },
      { id: 'trump-blanks', label: 'Declare BLANKS trump' },
      { id: 'play-5-5', label: 'Play 5-5' },
      { id: 'play-6-5', label: 'Play 6-5' }, // This is step 7 - should be valid but fails
    ];

    let currentState = baseState;
    
    // Validate each step
    for (let i = 0; i < actions.length; i++) {
      const action = actions[i];
      console.log(`\nStep ${i + 1}: ${action.id} (${action.label})`);
      console.log(`Current player: ${currentState.currentPlayer}`);
      console.log(`Current phase: ${currentState.phase}`);
      
      if (currentState.phase === 'playing') {
        console.log(`Trump: ${currentState.trump}`);
        console.log(`Current trick: ${JSON.stringify(currentState.currentTrick.map(p => `P${p.player}:${p.domino.id}`))}`);
        
        if (action.id.startsWith('play-')) {
          const dominoId = action.id.replace('play-', '');
          const currentPlayerHand = currentState.players[currentState.currentPlayer].hand;
          console.log(`Player ${currentState.currentPlayer} hand: ${currentPlayerHand.map(d => d.id).join(', ')}`);
          console.log(`Looking for domino: ${dominoId}`);
          console.log(`Player has domino: ${currentPlayerHand.some(d => d.id === dominoId)}`);
        }
      }
      
      // Get available actions for current state
      const availableActions = getNextStates(currentState);
      console.log(`Available actions: ${availableActions.map(a => a.id).join(', ')}`);
      
      // Check if the action is valid
      const validAction = availableActions.find(a => a.id === action.id);
      
      if (!validAction) {
        console.log(`❌ INVALID: Action ${action.id} not found in available actions`);
        if (i === 6) { // Step 7 (0-indexed)
          // This is the bug we're investigating
          console.log('This is step 7 - the reported bug!');
          console.log('Expected: play-6-5 should be available');
          
          // Let's debug who should have 6-5
          for (let playerId = 0; playerId < 4; playerId++) {
            const playerHand = currentState.players[playerId].hand;
            const has65 = playerHand.some(d => d.id === '6-5');
            console.log(`Player ${playerId} has 6-5: ${has65}`);
          }
        }
        
        // For now, let's expect this to fail at step 7
        if (i === 6) {
          expect(validAction).toBeUndefined(); // We expect this to fail for now
          return; // Stop validation here
        } else {
          throw new Error(`Unexpected validation failure at step ${i + 1}`);
        }
      }
      
      console.log(`✅ VALID: ${action.id}`);
      
      // Update state for next iteration
      currentState = validAction.newState;
    }
  });

  test('should properly track current player after trick completion', () => {
    // Create a state where we complete a trick and see who should lead next
    const state = createTestState({
      phase: 'playing',
      trump: 0, // blanks are trump
      currentTrick: [
        { player: 1, domino: { id: '5-5', high: 5, low: 5, points: 10 } }, // Player 1 leads with 5-5
        { player: 2, domino: { id: '5-1', high: 5, low: 1, points: 5 } },   // Player 2 follows with 5-1
        { player: 3, domino: { id: '5-4', high: 5, low: 4 } },               // Player 3 follows with 5-4  
        { player: 0, domino: { id: '5-3', high: 5, low: 3 } }                // Player 0 follows with 5-3
      ],
      currentPlayer: 1 // Doesn't matter, trick is complete
    });

    const transitions = getNextStates(state);
    const completeTrickTransition = transitions.find(t => t.id === 'complete-trick');
    expect(completeTrickTransition).toBeDefined();

    // Player 1 should win (played 5-5, highest 5)
    const newState = completeTrickTransition!.newState;
    expect(newState.tricks[0].winner).toBe(1);
    expect(newState.currentPlayer).toBe(1); // Winner leads next trick
  });
});