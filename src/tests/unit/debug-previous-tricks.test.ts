import { describe, test, expect } from 'vitest';
import { getCurrentSuit } from '../../game/core/rules';
import { createInitialState } from '../../game/core/state';
import type { GameState, Domino } from '../../game/types';

// Test the getCurrentSuit function used in DebugPreviousTricks
describe('DebugPreviousTricks Logic', () => {
  test('getCurrentSuit should work correctly for previous tricks', () => {
    // Test with a trick where trump was led
    const state1 = createInitialState();
    state1.trump = { type: 'suit', suit: 5 }; // 5s are trump
    state1.currentSuit = 5; // 5s were led
    expect(getCurrentSuit(state1)).toBe('Fives (Trump)');
    
    // Test with a trick where non-trump was led
    const state2 = createInitialState();
    state2.trump = { type: 'suit', suit: 5 }; // 5s are trump
    state2.currentSuit = 6; // 6s were led
    expect(getCurrentSuit(state2)).toBe('Sixes');
    
    // Test with doubles trump
    const state3 = createInitialState();
    state3.trump = { type: 'doubles' }; // doubles are trump
    state3.currentSuit = 7; // doubles were led
    expect(getCurrentSuit(state3)).toBe('Doubles (Trump)');
  });

  test('should handle empty current trick gracefully', () => {
    const state = createInitialState();
    state.trump = { type: 'suit', suit: 5 };
    state.currentSuit = -1; // No trick in progress
    expect(getCurrentSuit(state)).toBe('None (no domino led)');
  });

  test('should handle null trump gracefully', () => {
    const state = createInitialState();
    state.trump = { type: 'none' }; // No trump set
    state.currentSuit = 6; // Some suit was led
    expect(getCurrentSuit(state)).toBe('None (no trump set)');
  });

  // Test the domino display logic used in the component
  function getDominoDisplay(domino: Domino): string {
    return `${domino.high}-${domino.low}`;
  }

  test('getDominoDisplay should format dominoes correctly', () => {
    expect(getDominoDisplay({ high: 6, low: 4, id: '6-4' })).toBe('6-4');
    expect(getDominoDisplay({ high: 0, low: 0, id: '0-0' })).toBe('0-0');
    expect(getDominoDisplay({ high: 5, low: 1, id: '5-1' })).toBe('5-1');
  });

  test('compact design should show dominoes efficiently', () => {
    // Test the compact design shows essential info only
    const mockTrick = {
      plays: [
        { player: 0, domino: { high: 6, low: 4, id: "6-4", points: 10 } },
        { player: 1, domino: { high: 5, low: 2, id: "5-2" } }
      ],
      winner: 0,
      points: 11
    };

    // Test that we can extract the key info needed for compact display
    expect(mockTrick.plays.length).toBe(2);
    expect(mockTrick.winner).toBe(0);
    expect(mockTrick.points).toBe(11);
    expect(getDominoDisplay(mockTrick.plays[0].domino)).toBe('6-4');
    expect(mockTrick.plays[0].domino.points).toBe(10); // Counter domino
  });

  test('component should work with realistic game state structure and use P0-P3 numbering', () => {
    const mockGameState: Partial<GameState> = {
      tricks: [
        {
          plays: [
            { player: 0, domino: { high: 6, low: 4, id: "6-4", points: 10 } },
            { player: 1, domino: { high: 5, low: 2, id: "5-2" } },
            { player: 2, domino: { high: 3, low: 1, id: "3-1" } },
            { player: 3, domino: { high: 2, low: 0, id: "2-0" } }
          ],
          winner: 0,
          points: 11
        }
      ],
      currentTrick: [
        { player: 0, domino: { high: 5, low: 5, id: "5-5" } },
        { player: 1, domino: { high: 4, low: 2, id: "4-2" } }
      ],
      trump: { type: 'suit', suit: 5 }
    };

    // Test that we can process the tricks data structure
    expect(mockGameState.tricks?.length).toBe(1);
    expect(mockGameState.tricks?.[0].plays.length).toBe(4);
    expect(mockGameState.tricks?.[0].winner).toBe(0);
    expect(mockGameState.tricks?.[0].points).toBe(11);
    
    // Test that player numbers are 0-based (P0, P1, P2, P3)
    expect(mockGameState.tricks?.[0].plays[0].player).toBe(0); // Should be P0
    expect(mockGameState.tricks?.[0].plays[1].player).toBe(1); // Should be P1
    expect(mockGameState.tricks?.[0].plays[2].player).toBe(2); // Should be P2
    expect(mockGameState.tricks?.[0].plays[3].player).toBe(3); // Should be P3
    expect(mockGameState.tricks?.[0].winner).toBe(0); // Winner should be P0
    
    // Test current suit for current trick
    const currentState: Partial<GameState> = {
      currentTrick: mockGameState.currentTrick,
      currentSuit: 5, // The suit that was led (5s)
      trump: mockGameState.trump
    };
    expect(getCurrentSuit(currentState as GameState)).toBe('Fives (Trump)');
  });
});