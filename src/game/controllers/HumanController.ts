import type { PlayerController } from './types';
import type { GameState, StateTransition } from '../types';

/**
 * Controller for human players.
 * Doesn't automatically emit actions - waits for UI interaction.
 */
export class HumanController implements PlayerController {
  constructor(
    public readonly playerId: number,
    private executeTransition: (transition: StateTransition) => void
  ) {}
  
  onStateChange(state: GameState, availableTransitions: StateTransition[]): void {
    // Human controller doesn't auto-emit actions
    // The UI will show available transitions and human will click when ready
    // This method could be used for notifications, sounds, etc.
  }
  
  /**
   * Called by UI when human clicks an action
   */
  handleUserAction(transition: StateTransition): void {
    // In test mode, allow controlling any player (for deterministic tests)
    const urlParams = typeof window !== 'undefined' ? new URLSearchParams(window.location.search) : null;
    const testMode = urlParams?.get('testMode') === 'true';
    
    // Validate this transition is for this player (unless in test mode)
    if (!testMode && 'player' in transition.action && transition.action.player !== this.playerId) {
      console.warn(`Player ${this.playerId} tried to execute action for player ${transition.action.player}`);
      return;
    }
    
    // Execute through the same mechanism as AI
    this.executeTransition(transition);
  }
}