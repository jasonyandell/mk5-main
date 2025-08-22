import type { PlayerController, PlayerConfig } from './types';
import type { GameState, StateTransition } from '../types';
import { HumanController } from './HumanController';
import { AIController } from './AIController';
import { SimpleAIStrategy, SmartAIStrategy, RandomAIStrategy } from './strategies';
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
    config.forEach((cfg, playerId) => {
      if (cfg.type === 'human') {
        this.controllers.set(playerId, 
          new HumanController(playerId, this.executeTransitionCallback)
        );
      } else {
        // Choose AI strategy - default to smart
        let strategy;
        switch (cfg.aiStrategy) {
          case 'simple':
            strategy = new SimpleAIStrategy();
            break;
          case 'random':
            strategy = new RandomAIStrategy();
            break;
          default:
            strategy = new SmartAIStrategy();
        }
        
        this.controllers.set(playerId,
          new AIController(playerId, this.executeTransitionCallback, strategy)
        );
      }
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
   */
  switchToAI(playerId: number, strategy?: 'simple' | 'smart' | 'random'): void {
    const old = this.controllers.get(playerId);
    if (old?.destroy) {
      old.destroy();
    }
    
    let aiStrategy;
    switch (strategy) {
      case 'simple':
        aiStrategy = new SimpleAIStrategy();
        break;
      case 'random':
        aiStrategy = new RandomAIStrategy();
        break;
      default:
        aiStrategy = new SmartAIStrategy();
    }
    
    this.controllers.set(playerId,
      new AIController(playerId, this.executeTransitionCallback, aiStrategy)
    );
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
   */
  isAIControlled(playerId: number): boolean {
    const controller = this.controllers.get(playerId);
    return controller instanceof AIController;
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
   */
  skipAIDelays(): void {
    // Execute any pending AI decisions immediately
    if (!this.currentState) {
      return; // No state to process
    }
    
    for (const controller of this.controllers.values()) {
      if (controller instanceof AIController) {
        controller.executeNow(this.currentState);
      }
    }
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