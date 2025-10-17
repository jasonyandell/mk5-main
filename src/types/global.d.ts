import type { GameState } from '../game/types';
import type { Writable } from 'svelte/store';
import type { GameView } from '../shared/multiplayer/protocol';

declare global {
  interface Window {
    getGameView: () => GameView;
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