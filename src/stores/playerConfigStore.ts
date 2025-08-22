import { writable } from 'svelte/store';
import { controllerManager } from './gameStore';
import type { PlayerConfig } from '../game/controllers/types';

/**
 * Store for managing player configurations
 */
export const playerConfigs = writable<PlayerConfig[]>([
  { type: 'human', name: 'Player 1' },
  { type: 'ai', name: 'Player 2', aiStrategy: 'simple' },
  { type: 'ai', name: 'Player 3', aiStrategy: 'simple' },
  { type: 'ai', name: 'Player 4', aiStrategy: 'simple' }
]);

/**
 * Preset configurations for easy testing
 */
export const presetConfigs = {
  singlePlayer: [
    { type: 'human', name: 'You' },
    { type: 'ai', name: 'AI 1', aiStrategy: 'simple' },
    { type: 'ai', name: 'AI 2', aiStrategy: 'simple' },
    { type: 'ai', name: 'AI 3', aiStrategy: 'simple' }
  ] as PlayerConfig[],
  
  twoHumans: [
    { type: 'human', name: 'Player 1' },
    { type: 'ai', name: 'AI 1', aiStrategy: 'simple' },
    { type: 'human', name: 'Player 2' },
    { type: 'ai', name: 'AI 2', aiStrategy: 'simple' }
  ] as PlayerConfig[],
  
  allHumans: [
    { type: 'human', name: 'Player 1' },
    { type: 'human', name: 'Player 2' },
    { type: 'human', name: 'Player 3' },
    { type: 'human', name: 'Player 4' }
  ] as PlayerConfig[],
  
  allAI: [
    { type: 'ai', name: 'AI 1', aiStrategy: 'smart' },
    { type: 'ai', name: 'AI 2', aiStrategy: 'smart' },
    { type: 'ai', name: 'AI 3', aiStrategy: 'smart' },
    { type: 'ai', name: 'AI 4', aiStrategy: 'smart' }
  ] as PlayerConfig[],
  
  mixed: [
    { type: 'human', name: 'You' },
    { type: 'ai', name: 'Smart AI', aiStrategy: 'smart' },
    { type: 'ai', name: 'Random AI', aiStrategy: 'random' },
    { type: 'human', name: 'Friend' }
  ] as PlayerConfig[]
};

/**
 * Apply a configuration
 */
export function applyConfiguration(config: PlayerConfig[]): void {
  playerConfigs.set(config);
  controllerManager.setupLocalGame(config);
}

/**
 * Switch a specific player between human and AI
 */
export function togglePlayerControl(playerId: number): void {
  const isHuman = controllerManager.isHumanControlled(playerId);
  
  if (isHuman) {
    controllerManager.switchToAI(playerId, 'simple');
  } else {
    controllerManager.switchToHuman(playerId);
  }
  
  // Update the store
  playerConfigs.update(configs => {
    const newConfigs = [...configs];
    newConfigs[playerId] = {
      ...newConfigs[playerId],
      type: isHuman ? 'ai' : 'human'
    };
    return newConfigs;
  });
}