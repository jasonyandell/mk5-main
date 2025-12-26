import type { GameState, StateTransition } from '../types';
import type { ValidAction } from '../../multiplayer/types';

/**
 * Interface for player controllers - both human and AI.
 * Controllers decide when and how to emit actions.
 */
export interface PlayerController {
  readonly playerId: number;

  /**
   * Called whenever game state changes.
   * Controller decides if it needs to take action.
   */
  onStateChange(state: GameState, availableTransitions: StateTransition[]): void;

  /**
   * Cleanup when controller is removed
   */
  destroy?(): void;
}

/**
 * Configuration for a player
 */
export interface PlayerConfig {
  type: 'human' | 'ai';
  name?: string;
  aiStrategy?: 'beginner' | 'minimax' | 'random';
}

/**
 * AI strategy interface
 */
export interface AIStrategy {
  /**
   * Choose an action from available valid actions
   */
  chooseAction(state: GameState, validActions: ValidAction[]): ValidAction;
}