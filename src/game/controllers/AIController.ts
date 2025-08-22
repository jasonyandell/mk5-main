import type { PlayerController, AIStrategy } from './types';
import type { GameState, StateTransition } from '../types';
import { SimpleAIStrategy } from './strategies';

/**
 * Controller for AI players.
 * Registers decisions immediately in state for deterministic execution.
 */
export class AIController implements PlayerController {
  private pendingDecision?: {
    transition: StateTransition;
    decidedAt: number;
    minDelay: number;
  };
  
  constructor(
    public readonly playerId: number,
    private executeTransition: (transition: StateTransition) => void,
    private strategy: AIStrategy = new SimpleAIStrategy()
  ) {}
  
  onStateChange(state: GameState, availableTransitions: StateTransition[]): void {
    // Check if we have a pending decision to execute
    if (this.pendingDecision) {
      const elapsed = Date.now() - this.pendingDecision.decidedAt;
      // Execute if enough time passed
      if (elapsed >= this.pendingDecision.minDelay) {
        const transition = this.pendingDecision.transition;
        delete this.pendingDecision;
        this.executeTransition(transition);
        return;
      }
      // Still waiting, check again on next state change
      return;
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
    
    // Get thinking time
    const thinkingTime = this.strategy.getThinkingTime(choice.action.type);
    
    // If thinking time is 0, execute immediately
    if (thinkingTime === 0) {
      this.executeTransition(choice);
    } else {
      // Store decision for later execution
      this.pendingDecision = {
        transition: choice,
        decidedAt: Date.now(),
        minDelay: thinkingTime
      };
      // The decision will be executed on the next state change
      // when enough time has passed
    }
  }
  
  executeNow(_state: GameState): void {
    // Execute immediately if there's a pending decision
    if (this.pendingDecision) {
      const transition = this.pendingDecision.transition;
      delete this.pendingDecision;
      this.executeTransition(transition);
    }
  }
  
  destroy(): void {
    // No timers to clean up anymore!
  }
}