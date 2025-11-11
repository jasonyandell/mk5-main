import { writable } from 'svelte/store';
import { game, getInternalClient } from './gameStore';

/**
 * Player configuration type for setup UI
 */
export interface PlayerConfig {
  type: 'human' | 'ai';
  name: string;
  aiStrategy?: 'random' | 'beginner';
}

/**
 * Store for managing player configurations
 */
export const playerConfigs = writable<PlayerConfig[]>([
  { type: 'human', name: 'Player 1' },
  { type: 'ai', name: 'Player 2', aiStrategy: 'beginner' },
  { type: 'ai', name: 'Player 3', aiStrategy: 'beginner' },
  { type: 'ai', name: 'Player 4', aiStrategy: 'beginner' }
]);

/**
 * Preset configurations for easy testing
 */
export const presetConfigs = {
  singlePlayer: [
    { type: 'human', name: 'You' },
    { type: 'ai', name: 'AI 1', aiStrategy: 'beginner' },
    { type: 'ai', name: 'AI 2', aiStrategy: 'beginner' },
    { type: 'ai', name: 'AI 3', aiStrategy: 'beginner' }
  ] as PlayerConfig[],
  
  twoHumans: [
    { type: 'human', name: 'Player 1' },
    { type: 'ai', name: 'AI 1', aiStrategy: 'beginner' },
    { type: 'human', name: 'Player 2' },
    { type: 'ai', name: 'AI 2', aiStrategy: 'beginner' }
  ] as PlayerConfig[],
  
  allHumans: [
    { type: 'human', name: 'Player 1' },
    { type: 'human', name: 'Player 2' },
    { type: 'human', name: 'Player 3' },
    { type: 'human', name: 'Player 4' }
  ] as PlayerConfig[],
  
  allAI: [
    { type: 'ai', name: 'AI 1', aiStrategy: 'beginner' },
    { type: 'ai', name: 'AI 2', aiStrategy: 'beginner' },
    { type: 'ai', name: 'AI 3', aiStrategy: 'beginner' },
    { type: 'ai', name: 'AI 4', aiStrategy: 'beginner' }
  ] as PlayerConfig[],
  
  mixed: [
    { type: 'human', name: 'You' },
    { type: 'ai', name: 'Beginner AI', aiStrategy: 'beginner' },
    { type: 'ai', name: 'Random AI', aiStrategy: 'random' },
    { type: 'human', name: 'Friend' }
  ] as PlayerConfig[]
};

/**
 * Apply a configuration
 */
export async function applyConfiguration(config: PlayerConfig[]): Promise<void> {
  playerConfigs.set(config);
  // Update player control types in the game
  for (let i = 0; i < config.length; i++) {
    const playerConfig = config[i];
    if (playerConfig) {
      await game.setPlayerControl(i, playerConfig.type);
    }
  }
}

/**
 * Switch a specific player between human and AI
 */
export async function togglePlayerControl(playerId: number): Promise<void> {
  const client = getInternalClient();
  if (!client) return;
  const currentView = await client.getView();
  const currentPlayer = currentView.players.find(p => p.playerId === playerId);
  const isHuman = currentPlayer?.controlType === 'human';

  const newType = isHuman ? 'ai' : 'human';
  await game.setPlayerControl(playerId, newType);

  // Update the store
  playerConfigs.update(configs => {
    const newConfigs = [...configs];
    const existingConfig = newConfigs[playerId];
    if (existingConfig) {
      newConfigs[playerId] = {
        ...existingConfig,
        type: newType,
        ...(newType === 'ai' ? { aiStrategy: 'beginner' as const } : {})
      };
    }
    return newConfigs;
  });
}
