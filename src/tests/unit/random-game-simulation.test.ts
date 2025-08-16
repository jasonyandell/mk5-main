import { describe, test, expect } from 'vitest';
import { createInitialState } from '../../game/core/state';
import { getNextStates } from '../../game/core/gameEngine';

describe('Random Game Simulation', () => {
  test('should play 100 random games without getting stuck (no available actions)', () => {
    let stuckCount = 0;

    for (let gameNum = 1; gameNum <= 100; gameNum++) {
      let currentState = createInitialState();
      let actionCount = 0;

      while (actionCount < 40 && currentState.phase !== 'game_end') {
        const transitions = getNextStates(currentState);

        if (transitions.length === 0) {
          stuckCount++;
          break;
        }

        // Take random action
        const randomIndex = Math.floor(Math.random() * transitions.length);
        const selectedTransition = transitions[randomIndex];
        if (!selectedTransition) throw new Error('Selected transition is undefined');
        currentState = selectedTransition.newState;
        actionCount++;
      }
    }

    expect(stuckCount).toBe(0); // No games should get stuck with no actions available
  });
});