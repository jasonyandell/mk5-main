import type { PlayerController, AIStrategy } from './types';
import type { GameState, StateTransition } from '../types';
import { SimpleAIStrategy } from './strategies';

/**
 * Controller for AI players.
 * Emits actions automatically using the same mechanism as humans.
 */
export class AIController implements PlayerController {
  private timeoutId?: ReturnType<typeof setTimeout>;
  
  constructor(
    public readonly playerId: number,
    private executeTransition: (transition: StateTransition) => void,
    private strategy: AIStrategy = new SimpleAIStrategy()
  ) {}
  
  onStateChange(state: GameState, availableTransitions: StateTransition[]): void {
    // Clear any pending action
    if (this.timeoutId) {
      clearTimeout(this.timeoutId);
      this.timeoutId = undefined;
    }
    
    // Filter to only this player's actions
    const myTransitions = availableTransitions.filter(t => {
      // Actions without a player field are available to everyone
      if (!('player' in t.action)) {
        return true;
      }
      // Actions with a player field are only for that player
      return t.action.player === this.playerId;
    });
    
    if (myTransitions.length === 0) return;
    
    // Choose an action
    const choice = this.strategy.chooseAction(state, myTransitions);
    if (!choice) return;
    
    // AI "thinks" then emits action just like a human would
    const thinkingTime = this.strategy.getThinkingTime(choice.action.type);
    
    this.timeoutId = setTimeout(() => {
      // Emit action through exact same mechanism as human
      this.executeTransition(choice);
      this.timeoutId = undefined;
    }, thinkingTime);
  }
  
  destroy(): void {
    // Clean up any pending timeouts
    if (this.timeoutId) {
      clearTimeout(this.timeoutId);
      this.timeoutId = undefined;
    }
  }
}