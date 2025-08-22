import type { AIStrategy } from './types';
import type { GameState, StateTransition } from '../types';

/**
 * Simple AI strategy - picks first available action
 */
export class SimpleAIStrategy implements AIStrategy {
  chooseAction(state: GameState, transitions: StateTransition[]): StateTransition | null {
    if (transitions.length === 0) return null;
    
    // Prioritize consensus actions for quick agreement
    const consensusAction = transitions.find(t =>
      t.action.type === 'agree-complete-trick' ||
      t.action.type === 'agree-score-hand'
    );
    
    if (consensusAction) {
      return consensusAction;
    }
    
    // Otherwise pick the first available action
    return transitions[0];
  }
  
  getThinkingTime(actionType: string): number {
    // Quick response for consensus actions
    if (actionType === 'agree-complete-trick' || actionType === 'agree-score-hand') {
      return 100 + Math.random() * 200;
    }
    
    // Normal thinking time for game actions
    return 500 + Math.random() * 1500;
  }
}

/**
 * Random AI strategy - picks random action
 */
export class RandomAIStrategy implements AIStrategy {
  chooseAction(state: GameState, transitions: StateTransition[]): StateTransition | null {
    if (transitions.length === 0) return null;
    
    // Still prioritize consensus for game flow
    const consensusAction = transitions.find(t =>
      t.action.type === 'agree-complete-trick' ||
      t.action.type === 'agree-score-hand'
    );
    
    if (consensusAction) {
      return consensusAction;
    }
    
    // Random choice from available actions
    const index = Math.floor(Math.random() * transitions.length);
    return transitions[index];
  }
  
  getThinkingTime(actionType: string): number {
    // Slightly faster "thinking" for random play
    if (actionType === 'agree-complete-trick' || actionType === 'agree-score-hand') {
      return 50 + Math.random() * 150;
    }
    
    return 300 + Math.random() * 1000;
  }
}

/**
 * Smart AI strategy - makes intelligent decisions
 * TODO: Implement actual game logic evaluation
 */
export class SmartAIStrategy implements AIStrategy {
  chooseAction(state: GameState, transitions: StateTransition[]): StateTransition | null {
    if (transitions.length === 0) return null;
    
    // Prioritize consensus
    const consensusAction = transitions.find(t =>
      t.action.type === 'agree-complete-trick' ||
      t.action.type === 'agree-score-hand'
    );
    
    if (consensusAction) {
      return consensusAction;
    }
    
    // TODO: Implement smart logic based on game state
    // For now, use simple strategy
    return transitions[0];
  }
  
  getThinkingTime(actionType: string): number {
    // Smart AI "thinks" a bit longer
    if (actionType === 'agree-complete-trick' || actionType === 'agree-score-hand') {
      return 150 + Math.random() * 250;
    }
    
    return 800 + Math.random() * 2000;
  }
}