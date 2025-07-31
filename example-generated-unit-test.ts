// Example of generated unit test for bug reports
import { test, expect } from 'vitest';
import { getNextStates } from '../game';
import type { GameState } from '../game/types';

test('Bug report - 2025-07-31T23-55-00-000Z', () => {
  // Bug report with efficient action array
  // Base state for reproduction
  const baseState: GameState = {
    "phase": "bidding",
    "players": [
      {
        "id": 0,
        "name": "Player 1",
        "hand": [
          { "high": 6, "low": 4, "id": "6-4" },
          { "high": 5, "low": 0, "id": "5-0" },
          { "high": 3, "low": 2, "id": "3-2" },
          { "high": 4, "low": 1, "id": "4-1" },
          { "high": 2, "low": 2, "id": "2-2" },
          { "high": 1, "low": 0, "id": "1-0" },
          { "high": 6, "low": 1, "id": "6-1" }
        ],
        "teamId": 0,
        "marks": 0
      }
      // ... other players
    ],
    "currentPlayer": 0,
    "dealer": 3,
    "winningBidder": null,
    "trump": null,
    "currentTrick": [],
    "tricks": [],
    "teamScores": [0, 0],
    "teamMarks": [0, 0],
    "gameTarget": 7,
    "tournamentMode": true,
    "bids": [],
    "currentBid": null,
    "shuffleSeed": 12345
  };
  
  // Action sequence from action history
  const actionIds = [
    "bid-30",
    "pass", 
    "pass",
    "pass",
    "trump-doubles",
    "play-6-4"
  ];
  
  // Replay actions step by step using game logic
  let currentState = baseState;
  
  for (let i = 0; i < actionIds.length; i++) {
    const actionId = actionIds[i];
    console.log(`Step ${i + 1}: Executing action "${actionId}"`);
    
    // Get available transitions from current state
    const availableTransitions = getNextStates(currentState);
    const matchingTransition = availableTransitions.find(t => t.id === actionId);
    
    // Verify action is available
    if (!matchingTransition) {
      const availableActions = availableTransitions.map(t => t.id).join(', ');
      throw new Error(`Action "${actionId}" not available at step ${i + 1}. Available: [${availableActions}]`);
    }
    
    // Execute the action
    currentState = matchingTransition.newState;
    
    // Verify state is valid after transition
    expect(currentState).toBeDefined();
    expect(currentState.phase).toBeTruthy();
  }
  
  // Verify final state matches expected
  expect(currentState.phase).toBe('playing');
  expect(currentState.currentPlayer).toBe(1);
  expect(currentState.trump).toBe(7); // Doubles trump
  expect(currentState.winningBidder).toBe(0);
  
  // Team state verification
  expect(currentState.teamScores).toEqual([0, 0]);
  expect(currentState.teamMarks).toEqual([0, 0]);
  
  // Verify specific game state properties
  expect(currentState.players).toHaveLength(4);
  expect(currentState.bids).toHaveLength(4);
  expect(currentState.tricks).toHaveLength(0);
  expect(currentState.currentTrick).toHaveLength(1);
  
  // Add your specific bug assertions here
  // Example: expect(specificBugCondition).toBe(expectedValue);
});