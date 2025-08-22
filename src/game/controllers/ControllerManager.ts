import type { PlayerController, PlayerConfig } from './types';
import type { GameState, StateTransition } from '../types';
import { HumanController } from './HumanController';
import { getNextStates } from '../core/gameEngine';

/**
 * Manages player controllers for the game.
 * Handles switching between human and AI control.
 */
export class ControllerManager {
  private controllers = new Map<number, PlayerController>();
  private executeTransitionCallback: (transition: StateTransition) => void;
  private currentState?: GameState;
  
  constructor(executeTransition: (transition: StateTransition) => void) {
    this.executeTransitionCallback = executeTransition;
  }
  
  /**
   * Set up controllers for a local game
   * Note: AI is now handled via pure functions in the game state
   */
  setupLocalGame(config: PlayerConfig[] = [
    { type: 'human' },
    { type: 'ai' },
    { type: 'ai' },
    { type: 'ai' }
  ]): void {
    // Clear existing controllers
    this.clearControllers();
    
    // Set up new controllers based on config
    // Only human controllers are managed here now
    // AI is handled by pure functions in ai-scheduler.ts
    config.forEach((cfg, playerId) => {
      if (cfg.type === 'human') {
        this.controllers.set(playerId, 
          new HumanController(playerId, this.executeTransitionCallback)
        );
      }
      // AI players don't need controllers anymore - handled by pure functions
    });
  }
  
  /**
   * Called whenever game state changes
   */
  onStateChange(state: GameState): void {
    this.currentState = state;
    const transitions = getNextStates(state);
    
    // Notify ALL controllers - they decide if they need to act
    for (const controller of this.controllers.values()) {
      controller.onStateChange(state, transitions);
    }
  }
  
  /**
   * Switch a player to AI control
   * Note: AI is now handled via pure functions, so we just remove the human controller
   */
  switchToAI(playerId: number, strategy?: 'simple' | 'smart' | 'random'): void {
    const old = this.controllers.get(playerId);
    if (old?.destroy) {
      old.destroy();
    }
    
    // Remove the controller - AI is handled by pure functions
    this.controllers.delete(playerId);
  }
  
  /**
   * Switch a player to human control
   */
  switchToHuman(playerId: number): HumanController {
    const old = this.controllers.get(playerId);
    if (old?.destroy) {
      old.destroy();
    }
    
    const controller = new HumanController(playerId, this.executeTransitionCallback);
    this.controllers.set(playerId, controller);
    return controller;
  }
  
  /**
   * Get the controller for a specific player
   */
  getController(playerId: number): PlayerController | undefined {
    return this.controllers.get(playerId);
  }
  
  /**
   * Get human controller if it exists
   */
  getHumanController(playerId: number): HumanController | undefined {
    const controller = this.controllers.get(playerId);
    if (controller instanceof HumanController) {
      return controller;
    }
    return undefined;
  }
  
  /**
   * Check if a player is controlled by AI
   * Note: AI is now handled via pure functions, so check if no human controller exists
   */
  isAIControlled(playerId: number): boolean {
    // If there's no human controller, it's AI-controlled
    return !this.controllers.has(playerId);
  }
  
  /**
   * Check if a player is controlled by human
   */
  isHumanControlled(playerId: number): boolean {
    const controller = this.controllers.get(playerId);
    return controller instanceof HumanController;
  }
  
  /**
   * Get all human-controlled player IDs
   */
  getHumanPlayers(): Set<number> {
    const humanPlayers = new Set<number>();
    for (const [playerId, controller] of this.controllers) {
      if (controller instanceof HumanController) {
        humanPlayers.add(playerId);
      }
    }
    return humanPlayers;
  }
  
  /**
   * Skip all AI delays and execute pending actions immediately
   * Note: With pure AI scheduling, this now updates state directly
   */
  skipAIDelays(): void {
    // This will be handled by the pure skipAIDelays function in the store
    // The ControllerManager no longer manages AI timing
  }
  
  /**
   * Clean up all controllers
   */
  clearControllers(): void {
    for (const controller of this.controllers.values()) {
      if (controller.destroy) {
        controller.destroy();
      }
    }
    this.controllers.clear();
  }
  
  /**
   * Destroy the manager and clean up
   */
  destroy(): void {
    this.clearControllers();
  }
}