/**
 * Test helpers for creating GameState objects
 */

import type { GameState } from '../game/types';
import type { GameConfig } from '../shared/multiplayer/protocol';

/**
 * Default initialConfig for tests
 */
export const defaultTestConfig: GameConfig = {
  playerTypes: ['human', 'ai', 'ai', 'ai'],
  shuffleSeed: 0,
  theme: 'business',
  colorOverrides: {}
};

/**
 * Adds initialConfig to a partial GameState object.
 * Useful for updating old tests that create GameState literals.
 */
export function withInitialConfig(
  state: Omit<GameState, 'initialConfig'>,
  config?: Partial<GameConfig>
): GameState {
  return {
    initialConfig: { ...defaultTestConfig, ...config },
    ...state
  };
}
