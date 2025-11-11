import type { GameView } from '../shared/multiplayer/protocol';
import type { game } from '../stores/gameStore';

declare global {
  interface Window {
    getGameView: () => GameView;
    game: typeof game;
    controllerManager?: {
      skipAIDelays: () => void;
    };
  }
}

export {};