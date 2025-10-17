import type { GameState } from '../game/types';
import type { Writable } from 'svelte/store';

declare global {
  interface Window {
    getGameState: () => GameState;
    gameActions: {
      executeAction: (action: import('../game/types').GameAction) => void;
      loadFromURL: () => void;
      startNewGame: (seed?: number) => void;
    };
    controllerManager?: {
      skipAIDelays: () => void;
    };
  }
}

export {};