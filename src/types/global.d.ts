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